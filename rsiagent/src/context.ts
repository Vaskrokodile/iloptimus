// Context file discovery for RSI — inspired by codex CLI's AGENTS.md system
// and opencode's context file loading. Walks up from cwd looking for AGENTS.md
// or CLAUDE.md files, plus checks global locations.

import { existsSync, readFileSync } from "node:fs"
import { join, resolve, dirname } from "node:path"
import { homedir } from "node:os"

export interface ContextFile {
  path: string
  content: string
  scope: "global" | "project"
}

const FALLBACK_NAMES = ["AGENTS.md", "CLAUDE.md"]
const GLOBAL_DIR = join(homedir(), ".config", "rsi")
const CLAUDE_GLOBAL = join(homedir(), ".claude", "CLAUDE.md")

/** Max bytes to read from a context file (prevents huge files from bloating context). */
const MAX_BYTES = 32_768

/** Walk up from cwd looking for context files. Returns merged content in order:
 *  global first, then project files from root down to cwd (closer = later = higher precedence). */
export function discoverContextFiles(cwd: string = process.cwd()): ContextFile[] {
  const found: ContextFile[] = []
  const seen = new Set<string>()

  // 1. Global context files
  for (const name of FALLBACK_NAMES) {
    const p = join(GLOBAL_DIR, name)
    if (existsSync(p) && !seen.has(p)) {
      seen.add(p)
      found.push({ path: p, content: readFileTruncated(p), scope: "global" })
    }
  }
  // Claude compatibility
  if (existsSync(CLAUDE_GLOBAL) && !seen.has(CLAUDE_GLOBAL)) {
    seen.add(CLAUDE_GLOBAL)
    found.push({ path: CLAUDE_GLOBAL, content: readFileTruncated(CLAUDE_GLOBAL), scope: "global" })
  }

  // 2. Project context files — walk from git root down to cwd
  const projectFiles: ContextFile[] = []
  const checked = new Set<string>()
  let dir = resolve(cwd)
  const dirs: string[] = []
  for (let i = 0; i < 20; i++) {
    dirs.push(dir)
    const parent = dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  // Reverse so we go root -> cwd (closer files override earlier ones)
  for (const d of dirs.reverse()) {
    for (const name of FALLBACK_NAMES) {
      const p = join(d, name)
      if (existsSync(p) && !seen.has(p)) {
        seen.add(p)
        projectFiles.push({ path: p, content: readFileTruncated(p), scope: "project" })
      }
    }
    // Also check .rsi/AGENTS.md
    const rsiCtx = join(d, ".rsi", "AGENTS.md")
    if (existsSync(rsiCtx) && !seen.has(rsiCtx)) {
      seen.add(rsiCtx)
      projectFiles.push({ path: rsiCtx, content: readFileTruncated(rsiCtx), scope: "project" })
    }
  }

  found.push(...projectFiles)
  return found
}

function readFileTruncated(path: string): string {
  try {
    const content = readFileSync(path, "utf8")
    if (content.length > MAX_BYTES) {
      return content.slice(0, MAX_BYTES) + "\n...[truncated, context file too large]"
    }
    return content
  } catch {
    return ""
  }
}

/** Build a combined context string from discovered files, for injection into system prompt. */
export function buildContextFromFiles(cwd: string = process.cwd()): string {
  const files = discoverContextFiles(cwd)
  if (files.length === 0) return ""

  const sections: string[] = []
  for (const f of files) {
    const label = f.scope === "global" ? "Global" : "Project"
    sections.push(`## ${label} context (${f.path})\n\n${f.content}`)
  }
  return sections.join("\n\n")
}

/** Get just the paths of discovered context files (for status display). */
export function listContextFiles(cwd: string = process.cwd()): string[] {
  return discoverContextFiles(cwd).map((f) => f.path)
}
