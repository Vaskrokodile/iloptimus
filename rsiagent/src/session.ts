// Session persistence and resume for RSI — JSONL-based conversation storage.
// Inspired by prime-agent, opencode, and codex CLI session formats.
//
// Each session is a single append-only JSONL file stored under
// ~/.local/share/rsi/sessions/YYYY/MM/DD/<sessionId>.jsonl
//
// The first line is always a `session_header` entry containing the session
// metadata (id, cwd, provider, model, createdAt). Subsequent lines are
// user_message / assistant_message / tool_result / compaction_entry records.
// Every entry has a unique id, a parentId linking it into the conversation
// tree, and a timestamp. Corrupt or partial lines are skipped gracefully.

import { mkdirSync, readdirSync, statSync, appendFileSync, existsSync, readFileSync } from "node:fs"
import { homedir } from "node:os"
import { join, dirname } from "node:path"

// ---------------------------------------------------------------------------
//  Types
// ---------------------------------------------------------------------------

export interface SessionHeader {
  type: "session_header"
  id: string
  cwd: string
  providerId: string
  modelId: string
  createdAt: number
}

export interface UserMessageEntry {
  type: "user_message"
  id: string
  parentId: string | null
  content: string
  timestamp: number
}

export interface AssistantMessageEntry {
  type: "assistant_message"
  id: string
  parentId: string | null
  content: string
  toolCalls?: { id: string; name: string; args: Record<string, unknown> }[]
  timestamp: number
}

export interface ToolResultEntry {
  type: "tool_result"
  id: string
  parentId: string | null
  toolName: string
  content: string
  isError: boolean
  timestamp: number
}

export interface CompactionEntry {
  type: "compaction_entry"
  id: string
  parentId: string | null
  summary: string
  tokensBefore: number
  timestamp: number
}

export type SessionEntry =
  | SessionHeader
  | UserMessageEntry
  | AssistantMessageEntry
  | ToolResultEntry
  | CompactionEntry

/** Discriminated union of the non-header entry types (what callers append). */
export type SessionEntryPayload =
  | Omit<UserMessageEntry, "id" | "parentId" | "timestamp">
  | Omit<AssistantMessageEntry, "id" | "parentId" | "timestamp">
  | Omit<ToolResultEntry, "id" | "parentId" | "timestamp">
  | Omit<CompactionEntry, "id" | "parentId" | "timestamp">

export interface SessionInfo {
  id: string
  cwd: string
  providerId: string
  modelId: string
  createdAt: number
  entryCount: number
  lastActivity: number
}

// ---------------------------------------------------------------------------
//  Paths
// ---------------------------------------------------------------------------

const SESSIONS_DIR = join(homedir(), ".local", "share", "rsi", "sessions")

/** Format a timestamp into YYYY/MM/DD path segments. */
function dateSegments(ts: number): [string, string, string] {
  const d = new Date(ts)
  const y = String(d.getFullYear())
  const m = String(d.getMonth() + 1).padStart(2, "0")
  const day = String(d.getDate()).padStart(2, "0")
  return [y, m, day]
}

/** Resolve the JSONL file path for a session id + creation timestamp. */
function sessionFilePath(id: string, createdAt: number): string {
  const [y, m, d] = dateSegments(createdAt)
  return join(SESSIONS_DIR, y, m, d, `${id}.jsonl`)
}

/** Recursively walk the sessions directory and yield JSONL file paths. */
function* walkSessionFiles(dir: string): Generator<string> {
  let entries: string[]
  try {
    entries = readdirSync(dir)
  } catch {
    return
  }
  for (const entry of entries) {
    const full = join(dir, entry)
    let st: ReturnType<typeof statSync>
    try {
      st = statSync(full)
    } catch {
      continue
    }
    if (st.isDirectory()) {
      yield* walkSessionFiles(full)
    } else if (st.isFile() && entry.endsWith(".jsonl")) {
      yield full
    }
  }
}

// ---------------------------------------------------------------------------
//  ID generation
// ---------------------------------------------------------------------------

function newId(): string {
  // Prefer crypto.randomUUID when available (Bun / Node 19+).
  try {
    return crypto.randomUUID()
  } catch {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
  }
}

// ---------------------------------------------------------------------------
//  JSONL parsing (resilient)
// ---------------------------------------------------------------------------

/** Parse a JSONL string into entries, skipping corrupt/partial lines. */
function parseJsonl(text: string): SessionEntry[] {
  const out: SessionEntry[] = []
  const lines = text.split("\n")
  for (const raw of lines) {
    const line = raw.trim()
    if (!line) continue
    try {
      const obj = JSON.parse(line)
      if (obj && typeof obj === "object" && typeof obj.type === "string") {
        out.push(obj as SessionEntry)
      }
    } catch {
      // Skip malformed/partial line — resilient to crashes mid-write.
    }
  }
  return out
}

// ---------------------------------------------------------------------------
//  Session
// ---------------------------------------------------------------------------

export class Session {
  readonly id: string
  readonly cwd: string
  readonly providerId: string
  readonly modelId: string
  readonly createdAt: number
  private readonly path: string

  constructor(header: SessionHeader, path: string) {
    this.id = header.id
    this.cwd = header.cwd
    this.providerId = header.providerId
    this.modelId = header.modelId
    this.createdAt = header.createdAt
    this.path = path
  }

  /** Get the JSONL file path on disk. */
  getPath(): string {
    return this.path
  }

  /**
   * Append a new entry to the session. The entry is written to disk
   * immediately (append-only, not buffered). Returns the new entry id.
   */
  append(
    entry: SessionEntryPayload & { parentId?: string | null },
  ): string {
    const id = newId()
    const parentId = entry.parentId ?? this.lastEntryId()
    const timestamp = Date.now()

    const full: SessionEntry = {
      ...(entry as any),
      id,
      parentId: parentId ?? null,
      timestamp,
    } as SessionEntry

    const line = JSON.stringify(full) + "\n"
    appendFileSync(this.path, line)
    return id
  }

  /** Read all entries from the JSONL file (resilient to corruption). */
  getEntries(): SessionEntry[] {
    if (!existsSync(this.path)) return []
    // Synchronous read — getEntries is a sync API. Bun.file().text() is
    // async, so we use node:fs readFileSync here for the sync contract.
    return parseJsonl(readFileSync(this.path, "utf8"))
  }

  /**
   * Create a fork of this session starting from a specific entry (inclusive).
   * If fromEntryId is omitted, forks from the beginning (full copy of the
   * conversation tree up to now). The fork is a brand-new session with its
   * own id and JSONL file, sharing the same cwd/provider/model.
   */
  fork(fromEntryId?: string): Session {
    const entries = this.getEntries()
    const header = entries.find((e) => e.type === "session_header") as SessionHeader | undefined

    const forkId = newId()
    const now = Date.now()
    const forkHeader: SessionHeader = {
      type: "session_header",
      id: forkId,
      cwd: this.cwd,
      providerId: this.providerId,
      modelId: this.modelId,
      createdAt: now,
    }
    const forkPath = sessionFilePath(forkId, now)
    mkdirSync(dirname(forkPath), { recursive: true })

    const lines: string[] = [JSON.stringify(forkHeader)]

    // Determine slice start index.
    let startIdx = 0
    if (fromEntryId) {
      const idx = entries.findIndex((e) => e.id === fromEntryId)
      if (idx >= 0) startIdx = idx
    }

    // Remap old ids to new ids so the parent chain stays consistent.
    const idMap = new Map<string, string>()
    for (const entry of entries.slice(startIdx)) {
      if (entry.type === "session_header") continue
      const newEid = newId()
      idMap.set(entry.id, newEid)
      const mappedParent = entry.parentId ? idMap.get(entry.parentId) ?? null : null
      const forked: SessionEntry = {
        ...(entry as any),
        id: newEid,
        parentId: mappedParent,
      } as SessionEntry
      lines.push(JSON.stringify(forked))
    }

    Bun.write(forkPath, lines.join("\n") + "\n")
    return new Session(forkHeader, forkPath)
  }

  /** Id of the most recently appended entry (for chaining parentId). */
  private lastEntryId(): string | null {
    const entries = this.getEntries()
    for (let i = entries.length - 1; i >= 0; i--) {
      if (entries[i].type !== "session_header") return entries[i].id
    }
    return null
  }
}

// ---------------------------------------------------------------------------
//  SessionManager
// ---------------------------------------------------------------------------

export class SessionManager {
  /** Create a new session and write its header to disk. */
  create(cwd: string, providerId: string, modelId: string): Session {
    const id = newId()
    const createdAt = Date.now()
    const header: SessionHeader = {
      type: "session_header",
      id,
      cwd,
      providerId,
      modelId,
      createdAt,
    }
    const path = sessionFilePath(id, createdAt)
    mkdirSync(dirname(path), { recursive: true })
    Bun.write(path, JSON.stringify(header) + "\n")
    return new Session(header, path)
  }

  /** Open an existing session by id. Returns null if not found or invalid. */
  open(sessionId: string): Session | null {
    // Search all session files for one whose header matches the id.
    for (const file of walkSessionFiles(SESSIONS_DIR)) {
      const header = readHeader(file)
      if (header && header.id === sessionId) {
        return new Session(header, file)
      }
    }
    return null
  }

  /** List all sessions, optionally filtered by cwd. */
  list(cwd?: string): SessionInfo[] {
    return this.listAll().filter((s) => !cwd || s.cwd === cwd)
  }

  /** List every session on disk. */
  listAll(): SessionInfo[] {
    const infos: SessionInfo[] = []
    for (const file of walkSessionFiles(SESSIONS_DIR)) {
      const info = readSessionInfo(file)
      if (info) infos.push(info)
    }
    // Sort by lastActivity descending (most recent first).
    infos.sort((a, b) => b.lastActivity - a.lastActivity)
    return infos
  }

  /** Get the most recent session for a cwd (or globally if cwd omitted). */
  recent(cwd?: string): Session | null {
    const infos = this.list(cwd)
    if (infos.length === 0) return null
    return this.open(infos[0].id)
  }
}

// ---------------------------------------------------------------------------
//  Helpers — read header / info from a JSONL file
// ---------------------------------------------------------------------------

/** Read just the first valid line (session_header) of a JSONL file. */
function readHeader(path: string): SessionHeader | null {
  try {
    const text = readFileSync(path, "utf8")
    const entries = parseJsonl(text)
    return (entries.find((e) => e.type === "session_header") as SessionHeader) ?? null
  } catch {
    return null
  }
}

/** Build a SessionInfo from a JSONL file by scanning all entries. */
function readSessionInfo(path: string): SessionInfo | null {
  try {
    const text = readFileSync(path, "utf8")
    const entries = parseJsonl(text)
    const header = entries.find((e) => e.type === "session_header") as SessionHeader | undefined
    if (!header) return null

    let lastActivity = header.createdAt
    let entryCount = 0
    for (const e of entries) {
      if (e.type === "session_header") continue
      entryCount++
      if (typeof e.timestamp === "number" && e.timestamp > lastActivity) {
        lastActivity = e.timestamp
      }
    }

    return {
      id: header.id,
      cwd: header.cwd,
      providerId: header.providerId,
      modelId: header.modelId,
      createdAt: header.createdAt,
      entryCount,
      lastActivity,
    }
  } catch {
    return null
  }
}
