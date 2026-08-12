// Subagent infrastructure for RSI — inspired by opencode's agent system.
// Subagents are isolated agent loops that can be spawned for parallel or
// independent work. They have their own message context, can use tools, and
// return a final result to the parent agent.

import type { ChatMessage, ProviderClient } from "./providers.ts"
import type { ModelDef } from "./config.ts"
import type { ToolDef, ToolResult } from "./tools.ts"
import { executeBuiltinTool } from "./tools.ts"

export interface SubagentOptions {
  /** The system prompt for the subagent. */
  systemPrompt: string
  /** The task/prompt to execute. */
  task: string
  /** Provider client to use. */
  client: ProviderClient
  /** Model to use. */
  model: ModelDef
  /** Tools available to the subagent (defaults to built-in tools). */
  tools: ToolDef[]
  /** Reasoning effort. */
  effort: "low" | "medium" | "high"
  /** Max agent loop iterations. */
  maxSteps?: number
  /** Optional callback for streaming text output. */
  onText?: (delta: string) => void
  /** Optional callback for tool calls. */
  onToolCall?: (name: string, args: Record<string, unknown>) => void
  /** Optional callback for tool results. */
  onToolResult?: (name: string, result: string) => void
  /** Abort signal. */
  signal?: AbortSignal
}

export interface SubagentResult {
  /** The final text response from the subagent. */
  text: string
  /** Number of agent loop steps taken. */
  steps: number
  /** Whether the subagent completed its task within maxSteps. */
  completed: boolean
}

/** Run a subagent loop to completion. The subagent has its own isolated
 *  message context and can use tools autonomously. */
export async function runSubagent(opts: SubagentOptions): Promise<SubagentResult> {
  const maxSteps = opts.maxSteps ?? 15
  const messages: ChatMessage[] = [
    { role: "system", content: opts.systemPrompt },
    { role: "user", content: opts.task },
  ]

  let lastText = ""
  let steps = 0
  let completed = false

  for (let step = 0; step < maxSteps; step++) {
    steps++
    if (opts.signal?.aborted) break

    const result = await opts.client.stream(
      messages,
      opts.model,
      opts.tools,
      opts.effort,
      {
        onText: (delta) => {
          lastText += delta
          opts.onText?.(delta)
        },
      },
      opts.signal,
    )

    // Record assistant message
    const assistantMsg: ChatMessage = {
      role: "assistant",
      content: result.text,
      toolCalls: result.toolCalls.length ? result.toolCalls : undefined,
    }
    messages.push(assistantMsg)

    if (!result.toolCalls.length) {
      completed = true
      lastText = result.text
      break
    }

    // Execute tool calls
    for (const call of result.toolCalls) {
      opts.onToolCall?.(call.name, call.args)
      let resultStr: string
      try {
        const r: ToolResult = await executeBuiltinTool(call.name, call.args)
        resultStr = r.content
        if (r.isError) resultStr = `ERROR: ${resultStr}`
      } catch (e: any) {
        resultStr = `ERROR: ${String(e?.message ?? e)}`
      }
      opts.onToolResult?.(call.name, resultStr)
      messages.push({
        role: "tool",
        content: resultStr.slice(0, 30000),
        toolCallId: call.id,
        toolName: call.name,
      })
    }
  }

  return { text: lastText, steps, completed }
}

/** Default subagent system prompt — focused, autonomous, returns results. */
export const SUBAGENT_SYSTEM_PROMPT = `You are a focused subagent working inside RSI. You have been given a specific task to complete autonomously.

Rules:
- Use your tools to actually do the work, don't just describe what you would do.
- Be efficient: take the most direct path to completing the task.
- When you are done, provide a clear summary of what you did and what you found.
- If you encounter errors, try to fix them. If you cannot, report the error clearly.
- Do not ask questions — make reasonable assumptions and proceed.`
