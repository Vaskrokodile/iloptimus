// Persistent memory system for RSI — inspired by codex CLI's memories and
// opencode's plugin-based memory. Memories are markdown files with YAML
// frontmatter stored in ~/.local/share/rsi/memories/. They persist learned
// context across sessions and can be project-scoped or global.
//
// The system exposes:
//   - MemoryManager: CRUD + search + context-building for memories
//   - MEMORY_TOOLS: ToolDef[] for the LLM to save/search/update/delete memories
//   - executeMemoryTool: dispatch function for memory tool calls
//   - extractMemoryPrompt: builds a prompt to extract memories from a conversation

import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, unlinkSync, statSync } from "node:fs"
import { homedir } from "node:os"
import { join, basename } from "node:path"
import { createHash } from "node:crypto"
import { randomUUID } from "node:crypto"
import type { ToolDef, ToolResult } from "./tools.ts"
import type { ChatMessage } from "./providers.ts"

// ============================================================
//  Types
// ============================================================

export interface Memory {
  /** UUID identifier. */
  id: string
  /** Short human-readable title. */
  title: string
  /** Markdown body — the actual memory content. */
  content: string
  /** Tags for categorization and search. */
  tags: string[]
  /** ISO timestamp of creation. */
  created: string
  /** ISO timestamp of last update. */
  updated: string
  /** Project scope: "global" or a cwd hash. */
  project: string
  /** Relevance score 0–1, used for search ranking. */
  relevance: number
}

export interface MemoryInput {
  title: string
  content: string
  tags?: string[]
  project?: string
}

export interface MemoryUpdateInput {
  title?: string
  content?: string
  tags?: string[]
  relevance?: number
}

// ============================================================
//  Constants & helpers
// ============================================================

const MEMORIES_DIR = join(homedir(), ".local", "share", "rsi", "memories")

/** Derive a stable project ID from a cwd path via SHA-256 hash (first 12 hex chars). */
export function projectHash(cwd: string = process.cwd()): string {
  return createHash("sha256").update(cwd).digest("hex").slice(0, 12)
}

/** Generate a new UUID. Falls back to a crypto-based random string if randomUUID
 *  is unavailable (shouldn't happen on modern Bun/Node). */
function newId(): string {
  try {
    return randomUUID()
  } catch {
    return createHash("sha256")
      .update(`${Date.now()}-${Math.random()}`)
      .digest("hex")
      .slice(0, 32)
  }
}

/** Current ISO timestamp. */
function now(): string {
  return new Date().toISOString()
}

// ============================================================
//  Frontmatter parsing & serialization
// ============================================================

interface MemoryFrontmatter {
  id?: string
  title?: string
  tags?: string[]
  created?: string
  updated?: string
  project?: string
  relevance?: number
}

/** Parse YAML-like frontmatter from a markdown string. Handles the subset of
 *  fields used by memories: scalar keys and a `tags` inline array. */
function parseFrontmatter(raw: string): { frontmatter: MemoryFrontmatter; body: string } {
  const fmMatch = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/)
  if (!fmMatch) return { frontmatter: {}, body: raw }
  const fmText = fmMatch[1]
  const body = fmMatch[2]
  const fm: MemoryFrontmatter = {}

  for (const line of fmText.split("\n")) {
    const m = line.match(/^(\w+)\s*:\s*(.*)$/)
    if (!m) continue
    const key = m[1].trim()
    let val = m[2].trim()

    // Strip surrounding quotes
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1)
    }

    if (key === "tags") {
      // Inline array: [tag1, tag2] or ["tag1", "tag2"]
      const arrMatch = val.match(/^\[(.*)\]$/)
      if (arrMatch) {
        fm.tags = arrMatch[1]
          .split(",")
          .map((s) => s.trim().replace(/^["']|["']$/g, ""))
          .filter(Boolean)
      } else if (val) {
        fm.tags = [val]
      } else {
        fm.tags = []
      }
    } else if (key === "relevance") {
      const n = parseFloat(val)
      fm.relevance = isNaN(n) ? undefined : n
    } else if (key === "id" || key === "title" || key === "created" || key === "updated" || key === "project") {
      ;(fm as any)[key] = val
    }
  }

  return { frontmatter: fm, body }
}

/** Serialize a Memory into a markdown file with YAML frontmatter. */
function serializeMemory(mem: Memory): string {
  const tagsStr = mem.tags.length > 0 ? `[${mem.tags.map((t) => `"${t.replace(/"/g, '\\"')}"`).join(", ")}]` : "[]"
  const lines = [
    "---",
    `id: "${mem.id}"`,
    `title: "${mem.title.replace(/"/g, '\\"')}"`,
    `tags: ${tagsStr}`,
    `created: "${mem.created}"`,
    `updated: "${mem.updated}"`,
    `project: "${mem.project}"`,
    `relevance: ${mem.relevance}`,
    "---",
    "",
    mem.content.trim(),
    "",
  ]
  return lines.join("\n")
}

/** Parse a memory markdown file from raw text. Returns null if invalid. */
function parseMemoryFile(raw: string): Memory | null {
  const { frontmatter, body } = parseFrontmatter(raw)
  if (!frontmatter.id || !frontmatter.title) return null
  return {
    id: frontmatter.id,
    title: frontmatter.title,
    content: body.trim(),
    tags: frontmatter.tags ?? [],
    created: frontmatter.created ?? now(),
    updated: frontmatter.updated ?? now(),
    project: frontmatter.project ?? "global",
    relevance: frontmatter.relevance ?? 1.0,
  }
}

// ============================================================
//  MemoryManager
// ============================================================

export class MemoryManager {
  private memories = new Map<string, Memory>()
  private dir: string

  constructor(dir?: string) {
    this.dir = dir ?? MEMORIES_DIR
  }

  /** Scan the memory directory and reload all memories into memory. */
  reload(): void {
    this.memories.clear()
    if (!existsSync(this.dir)) return
    let entries: string[]
    try {
      entries = readdirSync(this.dir)
    } catch {
      return
    }
    for (const entry of entries) {
      if (!entry.endsWith(".md")) continue
      const filepath = join(this.dir, entry)
      try {
        const st = statSync(filepath)
        if (!st.isFile()) continue
        const raw = readFileSync(filepath, "utf8")
        const mem = parseMemoryFile(raw)
        if (mem) this.memories.set(mem.id, mem)
      } catch {
        // skip unreadable / invalid files
      }
    }
  }

  /** Ensure the memory directory exists. */
  private ensureDir(): void {
    mkdirSync(this.dir, { recursive: true })
  }

  /** File path for a given memory id. */
  private filePath(id: string): string {
    return join(this.dir, `${id}.md`)
  }

  /** Persist a memory to disk. */
  private persist(mem: Memory): void {
    this.ensureDir()
    writeFileSync(this.filePath(mem.id), serializeMemory(mem))
  }

  /** Remove a memory file from disk. */
  private removeFile(id: string): void {
    try {
      unlinkSync(this.filePath(id))
    } catch {
      // ignore — file may already be gone
    }
  }

  /** List all memories, optionally filtered by project.
   *  When a project is specified, global memories ("global") are always included
   *  alongside project-specific ones. Results are sorted by updated desc. */
  list(project?: string): Memory[] {
    let items = Array.from(this.memories.values())
    if (project) {
      items = items.filter((m) => m.project === project || m.project === "global")
    }
    return items.sort((a, b) => b.updated.localeCompare(a.updated))
  }

  /** Get a single memory by id. */
  get(id: string): Memory | null {
    return this.memories.get(id) ?? null
  }

  /** Create a new memory. */
  create(input: MemoryInput): Memory {
    const ts = now()
    const mem: Memory = {
      id: newId(),
      title: input.title,
      content: input.content,
      tags: input.tags ?? [],
      created: ts,
      updated: ts,
      project: input.project ?? "global",
      relevance: 1.0,
    }
    this.persist(mem)
    this.memories.set(mem.id, mem)
    return mem
  }

  /** Update an existing memory. Returns null if not found. */
  update(id: string, input: MemoryUpdateInput): Memory | null {
    const mem = this.memories.get(id)
    if (!mem) return null
    if (input.title !== undefined) mem.title = input.title
    if (input.content !== undefined) mem.content = input.content
    if (input.tags !== undefined) mem.tags = input.tags
    if (input.relevance !== undefined) mem.relevance = input.relevance
    mem.updated = now()
    this.persist(mem)
    return mem
  }

  /** Delete a memory by id. Returns true if deleted, false if not found. */
  delete(id: string): boolean {
    const mem = this.memories.get(id)
    if (!mem) return false
    this.removeFile(id)
    this.memories.delete(id)
    return true
  }

  /** Simple text search: substring match on title + content + tags,
   *  sorted by relevance (desc) then updated (desc). */
  search(query: string, project?: string): Memory[] {
    const q = query.trim().toLowerCase()
    if (!q) return this.list(project)
    const tokens = q.split(/\s+/).filter(Boolean)
    let candidates = Array.from(this.memories.values())
    if (project) {
      candidates = candidates.filter((m) => m.project === project || m.project === "global")
    }
    const scored: { mem: Memory; score: number }[] = []
    for (const mem of candidates) {
      const haystack = `${mem.title} ${mem.content} ${mem.tags.join(" ")}`.toLowerCase()
      // Every token must match somewhere (AND semantics) for a result to qualify
      const allMatch = tokens.every((tok) => haystack.includes(tok))
      if (!allMatch) continue
      // Score: relevance + bonus for title matches
      let score = mem.relevance
      const titleLower = mem.title.toLowerCase()
      for (const tok of tokens) {
        if (titleLower.includes(tok)) score += 0.1
      }
      scored.push({ mem, score })
    }
    return scored
      .sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score
        return b.mem.updated.localeCompare(a.mem.updated)
      })
      .map((s) => s.mem)
  }

  /** Build a context string for injection into the system prompt.
   *  Includes all relevant memories (project-specific + global), formatted
   *  as markdown. Returns empty string if no memories exist. */
  buildMemoryContext(project?: string): string {
    const memories = this.list(project)
    if (memories.length === 0) return ""
    const lines: string[] = [
      "",
      "## Memories",
      "These are persisted memories from previous sessions. Use them as context but verify against current state when relevant.",
      "",
    ]
    for (const mem of memories) {
      const tags = mem.tags.length > 0 ? ` _[${mem.tags.join(", ")}]_` : ""
      const scope = mem.project === "global" ? "global" : "project"
      lines.push(`### ${mem.title}${tags}`)
      lines.push(`_id: ${mem.id} · scope: ${scope} · relevance: ${mem.relevance.toFixed(1)}_`)
      lines.push("")
      lines.push(mem.content)
      lines.push("")
    }
    return lines.join("\n")
  }

  /** Get the memory directory path. */
  getDir(): string {
    return this.dir
  }
}

// ============================================================
//  Memory tools (for the LLM)
// ============================================================

/** A shared default MemoryManager instance. Callers can also construct their own. */
const defaultManager = new MemoryManager()

/** Ensure the default manager has loaded memories from disk at least once. */
let defaultLoaded = false
function ensureLoaded(): MemoryManager {
  if (!defaultLoaded) {
    defaultManager.reload()
    defaultLoaded = true
  }
  return defaultManager
}

export const MEMORY_TOOLS: ToolDef[] = [
  {
    type: "function",
    function: {
      name: "memory_save",
      description:
        "Save a new memory to persistent storage. Memories persist across sessions and are injected into future conversations as context. Use this when you learn something worth remembering: user preferences, project conventions, important decisions, key file locations, etc.",
      parameters: {
        type: "object",
        properties: {
          title: { type: "string", description: "A short, descriptive title for the memory." },
          content: { type: "string", description: "The full memory content in markdown." },
          tags: {
            type: "array",
            items: { type: "string" },
            description: "Optional tags for categorization and search.",
          },
          project: {
            type: "string",
            description:
              "Project scope. Use 'global' for cross-project memories (default), or omit to auto-scope to the current project.",
          },
        },
        required: ["title", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "memory_search",
      description:
        "Search persisted memories by text query. Matches against title, content, and tags. Returns ranked results sorted by relevance.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "The search query." },
          project: {
            type: "string",
            description: "Optional project scope to filter by. If omitted, searches all memories.",
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "memory_update",
      description:
        "Update an existing memory by id. You can change the title, content, tags, or relevance score. Use memory_search first if you need to find the id.",
      parameters: {
        type: "object",
        properties: {
          id: { type: "string", description: "The id of the memory to update." },
          title: { type: "string", description: "New title (optional)." },
          content: { type: "string", description: "New content in markdown (optional)." },
          tags: {
            type: "array",
            items: { type: "string" },
            description: "New tags (optional, replaces existing).",
          },
          relevance: {
            type: "number",
            description: "New relevance score 0–1 (optional).",
          },
        },
        required: ["id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "memory_delete",
      description: "Delete a memory by id. This is irreversible. Use memory_search first if you need to find the id.",
      parameters: {
        type: "object",
        properties: {
          id: { type: "string", description: "The id of the memory to delete." },
        },
        required: ["id"],
      },
    },
  },
]

/** Set of memory tool names, for quick membership checks. */
export const MEMORY_TOOL_NAMES = new Set(MEMORY_TOOLS.map((t) => t.function.name))

/** Dispatch a memory tool call. Uses the shared default MemoryManager. */
export function executeMemoryTool(name: string, args: Record<string, unknown>): ToolResult {
  try {
    const mgr = ensureLoaded()
    switch (name) {
      case "memory_save": {
        const title = String(args.title ?? "")
        const content = String(args.content ?? "")
        if (!title || !content) {
          return { name, isError: true, content: "title and content are required." }
        }
        const tags = Array.isArray(args.tags) ? (args.tags as string[]).map(String) : []
        const project = args.project ? String(args.project) : "global"
        const mem = mgr.create({ title, content, tags, project })
        return {
          name,
          content: `Memory saved: "${mem.title}" (id: ${mem.id}, project: ${mem.project})`,
        }
      }
      case "memory_search": {
        const query = String(args.query ?? "")
        if (!query) return { name, isError: true, content: "query is required." }
        const project = args.project ? String(args.project) : undefined
        const results = mgr.search(query, project)
        if (results.length === 0) {
          return { name, content: "No memories found matching the query." }
        }
        const lines = results.map((m) => {
          const tags = m.tags.length > 0 ? ` [${m.tags.join(", ")}]` : ""
          return `- ${m.title}${tags} (id: ${m.id}, relevance: ${m.relevance.toFixed(1)}, project: ${m.project})\n  ${m.content.slice(0, 200)}${m.content.length > 200 ? "..." : ""}`
        })
        return {
          name,
          content: `Found ${results.length} memor${results.length === 1 ? "y" : "ies"}:\n${lines.join("\n")}`,
        }
      }
      case "memory_update": {
        const id = String(args.id ?? "")
        if (!id) return { name, isError: true, content: "id is required." }
        const input: MemoryUpdateInput = {}
        if (args.title !== undefined) input.title = String(args.title)
        if (args.content !== undefined) input.content = String(args.content)
        if (args.tags !== undefined) input.tags = Array.isArray(args.tags) ? (args.tags as string[]).map(String) : []
        if (args.relevance !== undefined) {
          const r = Number(args.relevance)
          if (!isNaN(r)) input.relevance = Math.max(0, Math.min(1, r))
        }
        const mem = mgr.update(id, input)
        if (!mem) return { name, isError: true, content: `Memory not found: ${id}` }
        return { name, content: `Memory updated: "${mem.title}" (id: ${mem.id})` }
      }
      case "memory_delete": {
        const id = String(args.id ?? "")
        if (!id) return { name, isError: true, content: "id is required." }
        const ok = mgr.delete(id)
        if (!ok) return { name, isError: true, content: `Memory not found: ${id}` }
        return { name, content: `Memory deleted: ${id}` }
      }
      default:
        return { name, isError: true, content: `Unknown memory tool: ${name}` }
    }
  } catch (e: any) {
    return { name, isError: true, content: String(e?.message ?? e) }
  }
}

// ============================================================
//  Auto-extraction
// ============================================================

/** Build a prompt asking the LLM to extract noteworthy memories from a
 *  conversation. The conversation messages are rendered in a compact format.
 *  The resulting prompt instructs the model to output memories as a JSON array
 *  (or "none" if nothing is worth saving). */
export function extractMemoryPrompt(messages: ChatMessage[]): string {
  const lines: string[] = [
    "You are a memory extraction assistant. Review the conversation below and extract noteworthy memories worth persisting across sessions.",
    "",
    "Extract memories for things like:",
    "- User preferences, working style, or environment details",
    "- Project conventions, architecture decisions, or important file locations",
    "- Key facts, decisions, or outcomes that would be useful in future sessions",
    "- Errors and their resolutions (if likely to recur)",
    "",
    "Do NOT extract memories for:",
    "- Trivial or ephemeral details (current task progress, temporary state)",
    "- Things already captured in existing skills or documentation",
    "- Sensitive data (passwords, API keys, secrets)",
    "",
    'Output format: a JSON array of objects, or the single word "none" if nothing is worth saving.',
    "Each object should have:",
    '  {"title": "short title", "content": "markdown body", "tags": ["tag1", "tag2"], "project": "global"}',
    'Use "global" for the project field unless the memory is specific to a particular project.',
    "",
    "--- Conversation ---",
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

/** Parse the output of the memory extraction prompt into a list of memory
 *  inputs. Handles the JSON array format or "none". Returns empty array if
 *  nothing parseable is found. */
export function parseExtractedMemories(output: string): MemoryInput[] {
  const trimmed = output.trim()
  if (!trimmed || /^none$/i.test(trimmed)) return []

  // Try to find a JSON array in the output (the model may wrap it in markdown)
  const jsonMatch = trimmed.match(/\[[\s\S]*\]/)
  if (!jsonMatch) return []

  try {
    const parsed = JSON.parse(jsonMatch[0])
    if (!Array.isArray(parsed)) return []
    const results: MemoryInput[] = []
    for (const item of parsed) {
      if (typeof item !== "object" || item === null) continue
      const title = String(item.title ?? "").trim()
      const content = String(item.content ?? "").trim()
      if (!title || !content) continue
      const tags = Array.isArray(item.tags) ? (item.tags as string[]).map(String) : []
      const project = item.project ? String(item.project) : "global"
      results.push({ title, content, tags, project })
    }
    return results
  } catch {
    return []
  }
}
