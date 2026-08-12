// Context compaction for RSI — inspired by opencode's session compaction.
// When the conversation grows too large for the model's context window, older
// messages are summarized into a compact "conversation summary" that preserves
// key context (decisions, file changes, goals) while freeing token budget.
//
// Improvements based on opencode's approach:
// - Tail preservation: keep recent N turns intact for continuity
// - Tool output pruning: erase old tool outputs to free space (keep tool call metadata)
// - Token-budget aware: estimate tokens precisely and compact based on budget

import type { ChatMessage } from "./providers.ts"

export interface CompactionConfig {
  /** Approximate token threshold before compaction kicks in. */
  maxTokens: number
  /** Number of recent messages to always keep (never compact these). */
  keepRecent: number
  /** Number of recent turns to preserve as "tail" (never compact). */
  tailTurns: number
  /** Minimum tokens of tool output to protect from pruning. */
  pruneProtectTokens: number
  /** Minimum tokens to preserve as recent context. */
  minPreserveRecentTokens: number
  /** Maximum tokens to preserve as recent context. */
  maxPreserveRecentTokens: number
  /** Buffer tokens below maxTokens to trigger compaction early. */
  buffer: number
}

export const DEFAULT_COMPACTION: CompactionConfig = {
  maxTokens: 100_000,
  keepRecent: 10,
  tailTurns: 2,
  pruneProtectTokens: 40_000,
  minPreserveRecentTokens: 2_000,
  maxPreserveRecentTokens: 8_000,
  buffer: 20_000,
}

/** Rough token estimate: ~4 chars per token for English text. */
export function estimateTokens(messages: ChatMessage[]): number {
  let chars = 0
  for (const m of messages) {
    chars += m.content.length
    if (m.toolCalls) {
      for (const tc of m.toolCalls) {
        chars += JSON.stringify(tc.args).length
      }
    }
  }
  return Math.ceil(chars / 4)
}

/** Check if the conversation needs compaction. */
export function needsCompaction(messages: ChatMessage[], config: CompactionConfig = DEFAULT_COMPACTION): boolean {
  return estimateTokens(messages) > config.maxTokens - config.buffer
}

/** Prune old tool outputs to free context space. Keeps recent tool outputs
 *  (up to pruneProtectTokens) and preserves tool call metadata. Tools in the
 *  protected list are never pruned. */
const PRUNE_PROTECTED_TOOLS = new Set(["create_skill", "invoke_skill", "memory_save", "memory_update"])

export function pruneToolOutputs(messages: ChatMessage[], config: CompactionConfig = DEFAULT_COMPACTION): ChatMessage[] {
  // Walk backwards through tool messages, keeping recent ones up to pruneProtectTokens
  let protectedTokens = 0
  const result = [...messages]

  for (let i = result.length - 1; i >= 0; i--) {
    const m = result[i]
    if (m.role !== "tool") continue
    if (m.toolName && PRUNE_PROTECTED_TOOLS.has(m.toolName)) continue

    const tokens = Math.ceil(m.content.length / 4)
    if (protectedTokens + tokens > config.pruneProtectTokens) {
      // Prune this tool output — replace with placeholder but keep the call metadata
      result[i] = {
        ...m,
        content: `[Tool output pruned to save context — ${m.content.length} chars removed]`,
      }
    } else {
      protectedTokens += tokens
    }
  }
  return result
}

/** Identify the "tail" — recent turns to preserve during compaction.
 *  Returns the index where the tail starts. */
export function findTailStart(messages: ChatMessage[], config: CompactionConfig = DEFAULT_COMPACTION): number {
  const nonSystem = messages.filter((m) => m.role !== "system")
  let turnCount = 0
  let tailStartIdx = messages.length

  // Walk backwards counting turns (a turn = user message + assistant response + tool results)
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user" && i < messages.length - 1) {
      turnCount++
      if (turnCount >= config.tailTurns) {
        tailStartIdx = i
        break
      }
    }
  }
  return tailStartIdx
}

/** Build the compaction prompt — asks the model to summarize the conversation. */
export function buildCompactionPrompt(messages: ChatMessage[]): string {
  const lines: string[] = [
    "Summarize the conversation so far, preserving all important context for continuing the work.",
    "Include:",
    "- The user's goals and what has been accomplished",
    "- Key decisions made and their rationale",
    "- Files created, modified, or deleted (with paths)",
    "- Any errors encountered and how they were resolved",
    "- Pending tasks or next steps",
    "- Important code patterns or conventions discovered",
    "",
    "Be concise but complete. This summary will replace the older conversation history.",
    "",
    "--- Conversation to summarize ---",
    "",
  ]
  for (const m of messages) {
    if (m.role === "system") continue
    if (m.role === "user") {
      lines.push(`[USER]: ${m.content}`)
    } else if (m.role === "assistant") {
      lines.push(`[ASSISTANT]: ${m.content}`)
      if (m.toolCalls) {
        for (const tc of m.toolCalls) {
          lines.push(`  [TOOL CALL: ${tc.name}(${JSON.stringify(tc.args).slice(0, 200)})]`)
        }
      }
    } else if (m.role === "tool") {
      lines.push(`[TOOL RESULT (${m.toolName})]: ${m.content.slice(0, 500)}`)
    }
  }
  return lines.join("\n")
}

/** Compact the conversation: keep the system prompt + summary + tail (recent turns).
 *  Uses tail preservation to keep recent turns intact for continuity.
 *  Returns the new message array. The caller is responsible for generating the summary via the LLM. */
export function compactMessages(
  messages: ChatMessage[],
  summary: string,
  config: CompactionConfig = DEFAULT_COMPACTION,
): ChatMessage[] {
  const systemMsg = messages.find((m) => m.role === "system")

  // Find the tail start — preserve recent turns
  const tailStart = findTailStart(messages, config)
  const tail = messages.slice(tailStart).filter((m) => m.role !== "system")

  // Also keep a minimum number of recent messages as fallback
  const nonSystem = messages.filter((m) => m.role !== "system")
  const recent = nonSystem.slice(-config.keepRecent)

  // Use whichever is larger (tail or keepRecent) for better continuity
  const preserved = tail.length >= recent.length ? tail : recent

  // Prune tool outputs in the preserved section if still over budget
  let result: ChatMessage[] = []

  if (systemMsg) {
    result.push({
      role: "system",
      content: systemMsg.content + "\n\n## Conversation Summary\n" + summary,
    })
  }
  // Add a marker so the model knows older context was compacted
  result.push({
    role: "user",
    content: "[Context note: The earlier conversation has been summarized above. The recent messages follow below.]",
  })
  result.push({
    role: "assistant",
    content: "Understood. I have the conversation summary and will continue from the recent messages.",
  })

  // Prune tool outputs in the preserved tail if needed
  const prunedTail = pruneToolOutputs([...result, ...preserved], config)
  // The first 3 messages are system+markers, the rest is the preserved tail
  result = prunedTail.slice(0, 3).concat(prunedTail.slice(3))

  return result
}
