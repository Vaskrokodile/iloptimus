import { resolve, sep } from "node:path"

import type { ModelDef, Provider } from "./config.ts"
import { checkPermission, DEFAULT_PERMISSIONS, type PermissionConfig } from "./permissions.ts"
import { createProviderClient, type ChatMessage, type ProviderClient } from "./providers.ts"
import { SessionManager, type SessionEntry, type SessionEntryPayload } from "./session.ts"
import { BUILTIN_TOOLS, executeBuiltinTool, type ToolDef, type ToolResult } from "./tools.ts"

export type HarnessEventType =
  | "started"
  | "assistant_delta"
  | "assistant_message"
  | "tool_call"
  | "tool_result"
  | "approval_required"
  | "controller_retry"
  | "completed"
  | "failed"

export interface HarnessEvent {
  type: HarnessEventType
  sequence: number
  timestamp: number
  panelId: string
  data: Record<string, unknown>
}

export interface HarnessOptions {
  panelId: string
  workspace: string
  provider: Provider
  model: ModelDef
  systemPrompt: string
  effort?: "low" | "medium" | "high"
  maxSteps?: number
  tools?: ToolDef[]
  permissions?: PermissionConfig
  client?: ProviderClient
  session?: HarnessSession
  onEvent?: (event: HarnessEvent) => void
}

export interface HarnessSession {
  id: string
  append(entry: SessionEntryPayload & { parentId?: string | null }): string
  getEntries(): SessionEntry[]
}

export interface HarnessRunResult {
  text: string
  steps: number
  completed: boolean
  sessionId: string
}

const DEFAULT_TOOL_NAMES = new Set([
  "read_file",
  "write_file",
  "edit_file",
  "list_directory",
  "run_command",
  "create_directory",
  "file_info",
  "glob_search",
  "grep_search",
])

function actionRequirements(prompt: string): {
  mutate: boolean
  execute: boolean
  requestedPaths: string[]
  expectedOutput: string | null
  requiredSymbols: string[]
} {
  const normalized = prompt.toLowerCase()
  const requestedPaths = Array.from(
    prompt.matchAll(/\b[\w.-]+(?:\/[\w.-]+)*\.(?:py|js|ts|tsx|jsx|json|md|txt|html|css|sh)\b/gi),
    (match) => match[0],
  )
  const expectedOutput = prompt.match(/output\s+(?:is\s+)?exactly\s+([^\s.,;]+)/i)?.[1] ?? null
  const requiredSymbols = Array.from(
    prompt.matchAll(/\bimplement(?:ing)?\s+([A-Za-z_]\w*)\s*\(/gi),
    (match) => match[1],
  )
  return {
    mutate: /\b(create|write|edit|modify|fix|build|implement|folder|file|code)\b/.test(normalized),
    execute: /\b(run|execute|test|verify|check|compile|build)\b/.test(normalized),
    requestedPaths,
    expectedOutput,
    requiredSymbols,
  }
}

export const HEADLESS_TOOLS = BUILTIN_TOOLS.filter((tool) => DEFAULT_TOOL_NAMES.has(tool.function.name))

function insideWorkspace(path: string, workspace: string): boolean {
  const root = resolve(workspace)
  const candidate = resolve(path)
  return candidate === root || candidate.startsWith(root + sep)
}

function workspaceArgs(name: string, args: Record<string, unknown>, workspace: string): Record<string, unknown> {
  const normalized = { ...args }
  const pathKeys = ["path", "source", "destination"]
  for (const key of pathKeys) {
    if (typeof normalized[key] !== "string") continue
    const value = resolve(workspace, String(normalized[key]))
    if (!insideWorkspace(value, workspace)) {
      throw new Error(`${name} cannot access paths outside the panel workspace`)
    }
    normalized[key] = value
  }
  if (name === "run_command") {
    const cwd = resolve(workspace, String(normalized.cwd ?? normalized.path ?? "."))
    if (!insideWorkspace(cwd, workspace)) {
      throw new Error("run_command cwd must stay inside the panel workspace")
    }
    normalized.cwd = cwd
    delete normalized.path
  } else if ((name === "list_directory" || name === "glob_search" || name === "grep_search") && !normalized.path) {
    normalized.path = workspace
  }
  return normalized
}

export class AgentHarness {
  private readonly options: HarnessOptions
  private readonly client: ProviderClient
  private readonly session: HarnessSession
  private readonly messages: ChatMessage[]
  private sequence = 0
  private running = false

  constructor(options: HarnessOptions) {
    this.options = {
      effort: "medium",
      maxSteps: 20,
      tools: HEADLESS_TOOLS,
      permissions: DEFAULT_PERMISSIONS,
      ...options,
      workspace: resolve(options.workspace),
    }
    this.client = options.client ?? createProviderClient(options.provider)
    this.session = options.session ?? new SessionManager().create(
      this.options.workspace,
      options.provider.id,
      options.model.id,
    )
    this.messages = [{ role: "system", content: options.systemPrompt }]
    this.restoreMessages()
  }

  get sessionId(): string {
    return this.session.id
  }

  private restoreMessages(): void {
    for (const entry of this.session.getEntries()) {
      if (entry.type === "user_message") this.messages.push({ role: "user", content: entry.content })
      if (entry.type === "assistant_message") {
        this.messages.push({ role: "assistant", content: entry.content, toolCalls: entry.toolCalls })
      }
      if (entry.type === "tool_result") {
        this.messages.push({
          role: "tool",
          content: entry.content,
          toolCallId: entry.parentId ?? entry.id,
          toolName: entry.toolName,
        })
      }
    }
  }

  private emit(type: HarnessEventType, data: Record<string, unknown> = {}): void {
    this.options.onEvent?.({
      type,
      sequence: ++this.sequence,
      timestamp: Date.now(),
      panelId: this.options.panelId,
      data,
    })
  }

  private async runTool(name: string, args: Record<string, unknown>): Promise<ToolResult> {
    const normalized = workspaceArgs(name, args, this.options.workspace)
    const permission = checkPermission(name, normalized, this.options.permissions)
    if (permission === "deny") return { name, isError: true, content: `Permission denied for ${name}` }
    if (permission === "ask") {
      this.emit("approval_required", { name, arguments: normalized })
      return { name, isError: true, content: `Approval is required for ${name}; no action was taken.` }
    }
    return executeBuiltinTool(name, normalized)
  }

  async run(prompt: string, signal?: AbortSignal): Promise<HarnessRunResult> {
    if (this.running) throw new Error("This RSI panel is already running")
    this.running = true
    this.messages.push({ role: "user", content: prompt })
    this.session.append({ type: "user_message", content: prompt })
    this.emit("started", { prompt, sessionId: this.session.id, workspace: this.options.workspace })
    let finalText = ""
    let steps = 0
    let controllerRetries = 0
    const successfulTools = new Set<string>()
    const requirements = actionRequirements(prompt)
    const requestedToolNames = requirements.mutate || requirements.execute
      ? new Set(["create_directory", "write_file", "edit_file", "read_file", "run_command"])
      : null
    const admittedTools = requestedToolNames
      ? this.options.tools!.filter((tool) => requestedToolNames.has(tool.function.name))
      : this.options.tools!
    let lastRunFailed = false

    try {
      for (let step = 0; step < this.options.maxSteps!; step++) {
        if (signal?.aborted) throw new Error("Panel run aborted")
        steps++
        const mutated = ["write_file", "edit_file", "create_directory"].some((name) => successfulTools.has(name))
        let runTools = admittedTools
        if (requirements.mutate && !mutated) {
          runTools = admittedTools.filter((tool) => tool.function.name === "write_file")
        } else if (lastRunFailed) {
          runTools = admittedTools.filter((tool) => tool.function.name === "write_file")
        } else if (requirements.execute && !successfulTools.has("run_command") && !lastRunFailed) {
          runTools = admittedTools.filter((tool) => tool.function.name === "run_command")
        }
        const result = await this.client.stream(
          this.messages,
          this.options.model,
          runTools,
          this.options.effort!,
          { onText: (delta) => this.emit("assistant_delta", { delta }) },
          signal,
        )
        finalText = result.text
        this.messages.push({
          role: "assistant",
          content: result.text,
          toolCalls: result.toolCalls.length ? result.toolCalls : undefined,
        })
        this.session.append({
          type: "assistant_message",
          content: result.text,
          toolCalls: result.toolCalls.length ? result.toolCalls : undefined,
        })
        this.emit("assistant_message", { text: result.text, toolCalls: result.toolCalls })

        if (!result.toolCalls.length) {
          const missing: string[] = []
          if (requirements.mutate && !mutated) missing.push("a successful filesystem mutation")
          if (requirements.execute && !successfulTools.has("run_command")) missing.push("a successful run_command verification")
          if (missing.length && controllerRetries < 3) {
            controllerRetries++
            const correction =
              `CONTROLLER: The task is not complete because it still needs ${missing.join(" and ")}. ` +
              "Do not answer or simulate results. Call exactly one available tool now using its required arguments."
            this.messages.push({ role: "user", content: correction })
            this.session.append({ type: "user_message", content: correction })
            this.emit("controller_retry", { retry: controllerRetries, missing })
            continue
          }
          if (missing.length) {
            const error = `The model stopped without ${missing.join(" and ")}`
            this.emit("failed", { error, steps })
            return { text: finalText, steps, completed: false, sessionId: this.session.id }
          }
          const verified = []
          if (requirements.mutate) verified.push("the requested file mutation succeeded")
          if (requirements.execute) verified.push("the command completed with exit code 0")
          if (requirements.expectedOutput) verified.push(`output contained the exact line ${requirements.expectedOutput}`)
          const summary = verified.length ? `Verified: ${verified.join("; ")}.` : finalText
          this.emit("completed", { text: summary, steps, sessionId: this.session.id })
          return { text: summary, steps, completed: true, sessionId: this.session.id }
        }

        for (const call of result.toolCalls) {
          this.emit("tool_call", { id: call.id, name: call.name, arguments: call.args })
          let toolResult: ToolResult
          if (!runTools.some((tool) => tool.function.name === call.name)) {
            toolResult = { name: call.name, isError: true, content: `${call.name} was not admitted for this step.` }
          } else try {
            toolResult = await this.runTool(call.name, call.args)
          } catch (error) {
            toolResult = { name: call.name, isError: true, content: String(error) }
          }
          if (!toolResult.isError && ["write_file", "edit_file"].includes(call.name) && requirements.requestedPaths.length) {
            const actualPath = String(call.args.path ?? "").replaceAll("\\", "/")
            if (!requirements.requestedPaths.some((path) => actualPath.endsWith(path))) {
              toolResult = {
                name: call.name,
                isError: true,
                content: `Verification failed: write the requested path ${requirements.requestedPaths.join(" or ")}, not ${actualPath}.`,
              }
            }
          }
          if (!toolResult.isError && call.name === "write_file" && requirements.requiredSymbols.length) {
            const content = String(call.args.content ?? "")
            const missingSymbols = requirements.requiredSymbols.filter(
              (symbol) => !new RegExp(`\\bdef\\s+${symbol}\\s*\\(`).test(content),
            )
            if (missingSymbols.length) {
              toolResult = {
                name: call.name,
                isError: true,
                content: `Verification failed: the requested function ${missingSymbols.join(", ")} is missing. Rewrite the file with that exact callable name and signature.`,
              }
            }
          }
          if (!toolResult.isError && call.name === "write_file" && requirements.execute && String(call.args.path ?? "").endsWith(".py")) {
            if (!String(call.args.content ?? "").includes("print(")) {
              toolResult = {
                name: call.name,
                isError: true,
                content: "Verification failed: the Python file must print the requested result when executed.",
              }
            }
          }
          if (!toolResult.isError && call.name === "run_command" && requirements.expectedOutput) {
            const outputLines = toolResult.content.split(/\r?\n/).map((line) => line.trim())
            if (!outputLines.includes(requirements.expectedOutput)) {
              toolResult = {
                name: call.name,
                isError: true,
                content: `Verification failed: expected an output line exactly equal to ${requirements.expectedOutput}. Actual result:\n${toolResult.content}`,
              }
            }
          }
          this.emit("tool_result", {
            id: call.id,
            name: call.name,
            content: toolResult.content,
            isError: Boolean(toolResult.isError),
          })
          if (!toolResult.isError) successfulTools.add(call.name)
          if (!toolResult.isError && ["write_file", "edit_file"].includes(call.name) && lastRunFailed) {
            lastRunFailed = false
          }
          if (call.name === "run_command") lastRunFailed = Boolean(toolResult.isError)
          this.messages.push({
            role: "tool",
            content: toolResult.content,
            toolCallId: call.id,
            toolName: call.name,
          })
          this.session.append({
            type: "tool_result",
            parentId: call.id,
            toolName: call.name,
            content: toolResult.content,
            isError: Boolean(toolResult.isError),
          })
        }
      }
      this.emit("failed", { error: "Maximum agent steps reached", steps })
      return { text: finalText, steps, completed: false, sessionId: this.session.id }
    } catch (error) {
      this.emit("failed", { error: String(error), steps })
      throw error
    } finally {
      this.running = false
    }
  }
}
