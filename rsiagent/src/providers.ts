// Provider clients with streaming and tool-calling support.
// Supports OpenAI-compatible chat completions and Anthropic messages API.
// Includes provider fallback chain — retries on alternate providers on failure.
import type { Provider, ModelDef } from "./config.ts"
import { BUILTIN_TOOLS, toOpenAITools, toAnthropicTools, type ToolDef } from "./tools.ts"

export type Role = "system" | "user" | "assistant" | "tool"

export interface ChatMessage {
  role: Role
  content: string
  /** OpenAI-style tool calls emitted by the assistant. */
  toolCalls?: { id: string; name: string; args: Record<string, unknown> }[]
  /** Tool call id this message answers (for role: tool). */
  toolCallId?: string
  /** Name of the tool (for role: tool). */
  toolName?: string
}

export interface StreamCallbacks {
  onText: (delta: string) => void
  onToolCall?: (call: { id: string; name: string; args: Record<string, unknown> }) => void
  onDone?: (full: { text: string; toolCalls: { id: string; name: string; args: Record<string, unknown> }[] }) => void
}

interface StreamResult {
  text: string
  toolCalls: { id: string; name: string; args: Record<string, unknown> }[]
}

export interface ProviderClient {
  stream(
    messages: ChatMessage[],
    model: ModelDef,
    tools: ToolDef[],
    effort: "low" | "medium" | "high",
    cb: StreamCallbacks,
    signal?: AbortSignal,
  ): Promise<StreamResult>
}

function parseSSELine(line: string): string | null {
  if (line.startsWith("data:")) return line.slice(5).trim()
  return null
}

async function readSSEStream(
  res: Response,
  onData: (data: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const reader = res.body?.getReader()
  if (!reader) throw new Error("No response body")
  const decoder = new TextDecoder()
  let buffer = ""
  while (true) {
    if (signal?.aborted) {
      reader.cancel().catch(() => {})
      break
    }
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).trim()
      buffer = buffer.slice(idx + 1)
      if (!line) continue
      const data = parseSSELine(line)
      if (data != null) {
        if (data === "[DONE]") return
        onData(data)
      }
    }
  }
}

// ---- OpenAI-compatible ----
class OpenAIProviderClient implements ProviderClient {
  constructor(private provider: Provider) {}

  async stream(
    messages: ChatMessage[],
    model: ModelDef,
    tools: ToolDef[],
    effort: "low" | "medium" | "high",
    cb: StreamCallbacks,
    signal?: AbortSignal,
  ): Promise<StreamResult> {
    const url = `${this.provider.baseURL.replace(/\/$/, "")}/chat/completions`
    const apiMessages: any[] = []

    for (const m of messages) {
      if (m.role === "system") {
        apiMessages.push({ role: "system", content: m.content })
      } else if (m.role === "user") {
        apiMessages.push({ role: "user", content: m.content })
      } else if (m.role === "assistant") {
        const entry: any = { role: "assistant", content: m.content || null }
        if (m.toolCalls?.length) {
          entry.tool_calls = m.toolCalls.map((tc) => ({
            id: tc.id,
            type: "function",
            function: { name: tc.name, arguments: JSON.stringify(tc.args) },
          }))
        }
        apiMessages.push(entry)
      } else if (m.role === "tool") {
        apiMessages.push({
          role: "tool",
          tool_call_id: m.toolCallId,
          content: m.content,
        })
      }
    }

    const body: any = {
      model: model.id,
      messages: apiMessages,
      stream: true,
      max_tokens: 16384,
    }
    if (tools.length) {
      body.tools = toOpenAITools(tools)
      body.tool_choice = "auto"
    }
    // Reasoning effort: only send if the model explicitly supports it via
    // effortParam, or if it's an o-series / reasoning model. Sending
    // reasoning_effort to non-reasoning models causes 400 errors.
    if (model.effortParam) {
      body[model.effortParam] = effort
    } else if (/^o\d|reason|think|deep/i.test(model.id)) {
      body.reasoning_effort = effort
    }

    // Retry logic for transient errors (429, 500, 502, 503, 504)
    let lastError: Error | null = null
    let res: Response | null = null
    for (let attempt = 0; attempt < 3; attempt++) {
      if (signal?.aborted) throw new Error("Aborted")
      try {
        res = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.provider.apiKey}`,
          },
          body: JSON.stringify(body),
          signal,
        })
        if (res.ok && res.body) break
        // Don't retry on 4xx (except 429) — these are client errors
        if (res.status >= 400 && res.status < 500 && res.status !== 429) {
          const errText = await res.text().catch(() => "")
          throw new Error(`Provider error ${res.status}: ${errText.slice(0, 500)}`)
        }
        lastError = new Error(`Provider error ${res.status}`)
        // Exponential backoff: 1s, 2s, 4s
        if (attempt < 2) await new Promise((r) => setTimeout(r, 1000 * Math.pow(2, attempt)))
      } catch (e: any) {
        if (signal?.aborted) throw new Error("Aborted")
        lastError = e
        if (attempt < 2) await new Promise((r) => setTimeout(r, 1000 * Math.pow(2, attempt)))
      }
    }
    if (!res || !res.ok || !res.body) {
      const errText = res ? await res.text().catch(() => "") : ""
      throw new Error(`Provider error after retries: ${lastError?.message ?? "unknown"}${errText ? ` — ${errText.slice(0, 300)}` : ""}`)
    }

    let text = ""
    const toolCallsMap = new Map<number, { id: string; name: string; argStr: string }>()

    await readSSEStream(
      res,
      (data) => {
        try {
          const json = JSON.parse(data)
          const choice = json.choices?.[0]
          if (!choice) return
          const delta = choice.delta
          if (delta?.content) {
            text += delta.content
            cb.onText(delta.content)
          }
          if (delta?.tool_calls) {
            for (const tc of delta.tool_calls) {
              const idx = tc.index ?? 0
              const existing = toolCallsMap.get(idx) ?? { id: tc.id ?? "", name: "", argStr: "" }
              if (tc.id) existing.id = tc.id
              if (tc.function?.name) existing.name += tc.function.name
              if (tc.function?.arguments) existing.argStr += tc.function.arguments
              toolCallsMap.set(idx, existing)
            }
          }
        } catch {
          // ignore parse errors on keepalive lines
        }
      },
      signal,
    )

    const toolCalls = Array.from(toolCallsMap.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([, v]) => {
        let args: Record<string, unknown> = {}
        try {
          args = v.argStr ? JSON.parse(v.argStr) : {}
        } catch {
          args = { _raw: v.argStr }
        }
        const call = { id: v.id, name: v.name, args }
        cb.onToolCall?.(call)
        return call
      })

    const result = { text, toolCalls }
    cb.onDone?.(result)
    return result
  }
}

// ---- Anthropic ----
class AnthropicProviderClient implements ProviderClient {
  constructor(private provider: Provider) {}

  async stream(
    messages: ChatMessage[],
    model: ModelDef,
    tools: ToolDef[],
    effort: "low" | "medium" | "high",
    cb: StreamCallbacks,
    signal?: AbortSignal,
  ): Promise<StreamResult> {
    const url = `${this.provider.baseURL.replace(/\/$/, "")}/v1/messages`
    let system = ""
    const apiMessages: any[] = []
    for (const m of messages) {
      if (m.role === "system") {
        system += (system ? "\n\n" : "") + m.content
        continue
      }
      if (m.role === "user") {
        apiMessages.push({ role: "user", content: m.content })
      } else if (m.role === "assistant") {
        const content: any[] = []
        if (m.content) content.push({ type: "text", text: m.content })
        if (m.toolCalls?.length) {
          for (const tc of m.toolCalls) {
            content.push({
              type: "tool_use",
              id: tc.id,
              name: tc.name,
              input: tc.args,
            })
          }
        }
        apiMessages.push({ role: "assistant", content })
      } else if (m.role === "tool") {
        apiMessages.push({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: m.toolCallId,
              content: m.content,
            },
          ],
        })
      }
    }

    const body: any = {
      model: model.id,
      max_tokens: 16384,
      stream: true,
      messages: apiMessages,
    }
    if (system) body.system = system
    if (tools.length) body.tools = toAnthropicTools(tools)
    // Anthropic reasoning effort via thinking budget heuristic.
    // Only enable thinking for models that support it (claude-3.5+, claude-4+).
    if (/claude.*[3-9]|claude.*opus|claude.*sonnet/i.test(model.id)) {
      if (effort === "high") {
        body.thinking = { type: "enabled", budget_tokens: 4096 }
      } else if (effort === "medium") {
        body.thinking = { type: "enabled", budget_tokens: 2048 }
      }
    }

    // Retry logic for transient errors
    let lastError: Error | null = null
    let res: Response | null = null
    for (let attempt = 0; attempt < 3; attempt++) {
      if (signal?.aborted) throw new Error("Aborted")
      try {
        res = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key": this.provider.apiKey,
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
          },
          body: JSON.stringify(body),
          signal,
        })
        if (res.ok && res.body) break
        if (res.status >= 400 && res.status < 500 && res.status !== 429) {
          const errText = await res.text().catch(() => "")
          throw new Error(`Provider error ${res.status}: ${errText.slice(0, 500)}`)
        }
        lastError = new Error(`Provider error ${res.status}`)
        if (attempt < 2) await new Promise((r) => setTimeout(r, 1000 * Math.pow(2, attempt)))
      } catch (e: any) {
        if (signal?.aborted) throw new Error("Aborted")
        lastError = e
        if (attempt < 2) await new Promise((r) => setTimeout(r, 1000 * Math.pow(2, attempt)))
      }
    }
    if (!res || !res.ok || !res.body) {
      const errText = res ? await res.text().catch(() => "") : ""
      throw new Error(`Provider error after retries: ${lastError?.message ?? "unknown"}${errText ? ` — ${errText.slice(0, 300)}` : ""}`)
    }

    let text = ""
    const toolCalls: { id: string; name: string; args: Record<string, unknown> }[] = []
    let currentTool: { id: string; name: string; input: string } | null = null

    await readSSEStream(
      res,
      (data) => {
        try {
          const evt = JSON.parse(data)
          const t = evt.type
          if (t === "content_block_start") {
            const block = evt.content_block
            if (block?.type === "tool_use") {
              currentTool = { id: block.id, name: block.name, input: "" }
            }
          } else if (t === "content_block_delta") {
            const delta = evt.delta
            if (delta?.type === "text_delta") {
              text += delta.text
              cb.onText(delta.text)
            } else if (delta?.type === "input_json_delta" && currentTool) {
              currentTool.input += delta.partial_json
            } else if (delta?.type === "thinking_delta") {
              // reasoning text; surface as dim stream too
              cb.onText(delta.thinking)
            }
          } else if (t === "content_block_stop") {
            if (currentTool) {
              let args: Record<string, unknown> = {}
              try {
                args = currentTool.input ? JSON.parse(currentTool.input) : {}
              } catch {
                args = { _raw: currentTool.input }
              }
              const call = { id: currentTool.id, name: currentTool.name, args }
              toolCalls.push(call)
              cb.onToolCall?.(call)
              currentTool = null
            }
          }
        } catch {
          // ignore
        }
      },
      signal,
    )

    const result = { text, toolCalls }
    cb.onDone?.(result)
    return result
  }
}

export function createProviderClient(provider: Provider): ProviderClient {
  if (provider.type === "anthropic") return new AnthropicProviderClient(provider)
  return new OpenAIProviderClient(provider)
}

// ---- Provider fallback chain ----
// Wraps a primary provider client with fallback providers. On retryable errors
// (rate limit, 5xx, overload), tries the next provider in the chain.

export interface FallbackConfig {
  /** Cooldown in ms before retrying a failed provider (default 300000 = 5min). */
  cooldownMs: number
  /** Whether to respect retry-after headers (default true). */
  respectRetryAfter: boolean
}

const DEFAULT_FALLBACK: FallbackConfig = {
  cooldownMs: 300_000,
  respectRetryAfter: true,
}

interface ProviderState {
  client: ProviderClient
  provider: Provider
  model: ModelDef
  cooldownUntil: number
}

export class FallbackProviderClient implements ProviderClient {
  private states: ProviderState[]
  private config: FallbackConfig

  constructor(
    primary: { provider: Provider; model: ModelDef },
    fallbacks: { provider: Provider; model: ModelDef }[] = [],
    config?: Partial<FallbackConfig>,
  ) {
    this.config = { ...DEFAULT_FALLBACK, ...config }
    this.states = [
      { client: createProviderClient(primary.provider), provider: primary.provider, model: primary.model, cooldownUntil: 0 },
      ...fallbacks.map((f) => ({
        client: createProviderClient(f.provider),
        provider: f.provider,
        model: f.model,
        cooldownUntil: 0,
      })),
    ]
  }

  /** Get the list of providers in order, skipping cooled-down ones. */
  private availableProviders(): ProviderState[] {
    const now = Date.now()
    return this.states.filter((s) => s.cooldownUntil <= now)
  }

  /** Mark a provider as cooled down after a failure. */
  private cooldown(state: ProviderState, retryAfterMs?: number) {
    const cd = retryAfterMs && this.config.respectRetryAfter ? retryAfterMs : this.config.cooldownMs
    state.cooldownUntil = Date.now() + cd
  }

  async stream(
    messages: ChatMessage[],
    _model: ModelDef,
    tools: ToolDef[],
    effort: "low" | "medium" | "high",
    cb: StreamCallbacks,
    signal?: AbortSignal,
  ): Promise<StreamResult> {
    let lastError: Error | null = null

    for (let i = 0; i < this.states.length; i++) {
      const state = this.states[i]
      if (state.cooldownUntil > Date.now()) continue

      try {
        const result = await state.client.stream(messages, state.model, tools, effort, cb, signal)
        // Success — clear cooldown
        state.cooldownUntil = 0
        return result
      } catch (e: any) {
        if (signal?.aborted) throw e
        lastError = e

        // Check if error is retryable (rate limit, 5xx, overload)
        const errStr = String(e?.message ?? e)
        const isRetryable =
          /429|rate.?limit|overload|502|503|504|5\d\d|quota|capacity|temporarily/i.test(errStr)

        if (isRetryable) {
          this.cooldown(state)
          continue // try next provider
        }

        // Non-retryable error — throw immediately
        throw e
      }
    }

    // All providers exhausted — try one last time with the primary (ignoring cooldown)
    if (lastError) {
      const primary = this.states[0]
      primary.cooldownUntil = 0
      return primary.client.stream(messages, primary.model, tools, effort, cb, signal)
    }

    throw new Error("No providers available")
  }
}

export { BUILTIN_TOOLS }
