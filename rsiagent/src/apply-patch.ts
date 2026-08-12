// apply_patch: a structured patch format for safe multi-file edits.
// Inspired by OpenAI Codex CLI's apply_patch format. Self-contained,
// well-typed, and ready to import from src/index.ts and src/tools.ts.
//
// Patch format:
//   *** Begin Patch
//   *** Add File: path/to/file.ts
//   +new file content here
//   +line by line
//   *** End File
//   *** Delete File: path/to/file.ts
//   *** End File
//   *** Update File: path/to/file.ts
//   @@ context line before
//   -removed line
//   +added line
//    context line after
//   *** End File
//   *** Move File: path/old.ts -> path/new.ts
//   *** End File
//   *** End Patch
//
// Only node:fs/promises, node:fs, node:path, and Bun APIs are used.

import { readFile, writeFile, mkdir, unlink, rename, stat } from "node:fs/promises"
import { existsSync } from "node:fs"
import { dirname, join, isAbsolute } from "node:path"
import type { ToolDef } from "./tools.ts"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single line within an UpdateFile chunk: context, removed, or added. */
export type PatchLine =
  | { type: "context"; text: string }
  | { type: "remove"; text: string }
  | { type: "add"; text: string }

/** A contiguous group of context/removed/added lines within an UpdateFile. */
export interface ContextChunk {
  lines: PatchLine[]
}

export type PatchHunk =
  | { type: "add"; path: string; content: string }
  | { type: "delete"; path: string }
  | { type: "update"; path: string; moveTo?: string; chunks: ContextChunk[] }
  | { type: "move"; from: string; to: string }

export interface Patch {
  hunks: PatchHunk[]
}

export interface ApplyResult {
  success: boolean
  applied: string[]
  errors: string[]
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/**
 * Parse a patch text string into a structured Patch object.
 * Throws on malformed input.
 */
export function parsePatch(patchText: string): Patch {
  const rawLines = patchText.split(/\r?\n/)
  const hunks: PatchHunk[] = []

  let i = 0
  // Skip leading blank lines / whitespace before Begin Patch
  while (i < rawLines.length && rawLines[i].trim() === "") i++

  if (i >= rawLines.length || rawLines[i].trim() !== "*** Begin Patch") {
    throw new Error("Patch must start with '*** Begin Patch'")
  }
  i++

  while (i < rawLines.length) {
    const line = rawLines[i]

    // End Patch — stop parsing
    if (line.trim() === "*** End Patch") {
      break
    }

    // Blank lines between hunks are allowed
    if (line.trim() === "") {
      i++
      continue
    }

    // Add File
    if (line.startsWith("*** Add File: ")) {
      const path = line.slice("*** Add File: ".length).trim()
      i++
      const contentLines: string[] = []
      while (i < rawLines.length && rawLines[i].trim() !== "*** End File") {
        const cl = rawLines[i]
        if (cl.startsWith("+")) {
          contentLines.push(cl.slice(1))
        } else if (cl.startsWith(" ")) {
          // tolerate leading-space-prefixed lines in add content
          contentLines.push(cl.slice(1))
        } else if (cl === "") {
          contentLines.push("")
        } else {
          throw new Error(`Add File: unexpected line without '+' prefix: ${JSON.stringify(cl)}`)
        }
        i++
      }
      if (i >= rawLines.length) throw new Error(`Add File: missing '*** End File' for ${path}`)
      i++ // consume *** End File
      hunks.push({ type: "add", path, content: contentLines.join("\n") })
      continue
    }

    // Delete File
    if (line.startsWith("*** Delete File: ")) {
      const path = line.slice("*** Delete File: ".length).trim()
      i++
      // Optional End File
      if (i < rawLines.length && rawLines[i].trim() === "*** End File") i++
      hunks.push({ type: "delete", path })
      continue
    }

    // Update File (with optional " -> newPath" move)
    if (line.startsWith("*** Update File: ")) {
      const rest = line.slice("*** Update File: ".length)
      let path = rest
      let moveTo: string | undefined
      const arrowIdx = rest.indexOf(" -> ")
      if (arrowIdx !== -1) {
        path = rest.slice(0, arrowIdx).trim()
        moveTo = rest.slice(arrowIdx + 4).trim()
      } else {
        path = rest.trim()
      }
      i++
      const chunks: ContextChunk[] = []
      let current: ContextChunk | null = null

      while (i < rawLines.length && rawLines[i].trim() !== "*** End File") {
        const cl = rawLines[i]
        if (cl.startsWith("@@")) {
          // Start a new chunk; the @@ line itself is a context line
          if (current && current.lines.length > 0) chunks.push(current)
          current = { lines: [] }
          const ctxText = cl.slice(2)
          current.lines.push({ type: "context", text: ctxText })
          i++
          continue
        }
        if (cl.startsWith("-")) {
          if (!current) current = { lines: [] }
          current.lines.push({ type: "remove", text: cl.slice(1) })
          i++
          continue
        }
        if (cl.startsWith("+")) {
          if (!current) current = { lines: [] }
          current.lines.push({ type: "add", text: cl.slice(1) })
          i++
          continue
        }
        if (cl.startsWith(" ")) {
          if (!current) current = { lines: [] }
          current.lines.push({ type: "context", text: cl.slice(1) })
          i++
          continue
        }
        if (cl === "") {
          // An empty line is treated as a context line with empty text
          if (!current) current = { lines: [] }
          current.lines.push({ type: "context", text: "" })
          i++
          continue
        }
        throw new Error(`Update File: unrecognized patch line: ${JSON.stringify(cl)}`)
      }
      if (i >= rawLines.length) throw new Error(`Update File: missing '*** End File' for ${path}`)
      i++ // consume *** End File
      if (current && current.lines.length > 0) chunks.push(current)
      hunks.push({ type: "update", path, moveTo, chunks })
      continue
    }

    // Move File
    if (line.startsWith("*** Move File: ")) {
      const rest = line.slice("*** Move File: ".length)
      const arrowIdx = rest.indexOf(" -> ")
      if (arrowIdx === -1) {
        throw new Error(`Move File: missing ' -> ' separator in: ${line}`)
      }
      const from = rest.slice(0, arrowIdx).trim()
      const to = rest.slice(arrowIdx + 4).trim()
      i++
      // Optional End File
      if (i < rawLines.length && rawLines[i].trim() === "*** End File") i++
      hunks.push({ type: "move", from, to })
      continue
    }

    throw new Error(`Unrecognized patch directive: ${JSON.stringify(line)}`)
  }

  return { hunks }
}

// ---------------------------------------------------------------------------
// Fuzzy matching
// ---------------------------------------------------------------------------

/**
 * Four-level progressive line matching to tolerate minor formatting
 * differences from the LLM.
 *
 *  1. Exact match
 *  2. Trim trailing whitespace
 *  3. Trim all whitespace
 *  4. Unicode normalize (NFC) + trim
 */
export function linesMatch(a: string, b: string): boolean {
  // Level 1: exact
  if (a === b) return true
  // Level 2: trim trailing whitespace
  if (a.replace(/\s+$/u, "") === b.replace(/\s+$/u, "")) return true
  // Level 3: trim all whitespace
  if (a.replace(/\s+/gu, "") === b.replace(/\s+/gu, "")) return true
  // Level 4: unicode normalize
  try {
    const na = a.normalize("NFC")
    const nb = b.normalize("NFC")
    if (na === nb) return true
    if (na.replace(/\s+/gu, "") === nb.replace(/\s+/gu, "")) return true
  } catch {
    // normalize may not be available in rare runtimes; ignore
  }
  return false
}

/**
 * Find the best index in `fileLines` where the sequence of context+remove
 * lines from a chunk matches, using progressive fuzzy matching.
 * Returns the starting index, or -1 if no match found.
 */
function findChunkMatch(
  fileLines: string[],
  searchFrom: number,
  chunk: ContextChunk,
): number {
  // Build the "search pattern": context + remove lines in order (added lines
  // are not part of the existing file, so we skip them).
  const pattern: { text: string; isRemove: boolean }[] = []
  for (const pl of chunk.lines) {
    if (pl.type === "context") pattern.push({ text: pl.text, isRemove: false })
    else if (pl.type === "remove") pattern.push({ text: pl.text, isRemove: true })
    // add lines skipped
  }

  if (pattern.length === 0) {
    // Pure-addition chunk: anchor on the last searchFrom position.
    // If there are no context/remove lines, we insert at searchFrom.
    return searchFrom
  }

  for (let start = searchFrom; start <= fileLines.length - pattern.length; start++) {
    let ok = true
    for (let k = 0; k < pattern.length; k++) {
      if (!linesMatch(fileLines[start + k], pattern[k].text)) {
        ok = false
        break
      }
    }
    if (ok) return start
  }
  return -1
}

// ---------------------------------------------------------------------------
// Applier
// ---------------------------------------------------------------------------

function toAbs(p: string, cwd: string): string {
  return isAbsolute(p) ? p : join(cwd, p)
}

/**
 * Apply a single UpdateFile hunk to the given file content (as lines).
 * Returns the new file content as a string. Throws on match failure.
 */
function applyUpdateHunk(content: string, hunk: Extract<PatchHunk, { type: "update" }>): string {
  const fileLines = content.split(/\r?\n/)
  // Detect trailing newline: if content ends with \n, split produces a trailing
  // "" element. We track this to re-append on join.
  const hadTrailingNewline = content.endsWith("\n")
  // If there's a trailing "" from split and the file had a trailing newline,
  // we keep it as a real empty line for matching purposes.
  const workLines = hadTrailingNewline && fileLines[fileLines.length - 1] === "" ? fileLines.slice(0, -1) : fileLines

  let searchFrom = 0
  const result: string[] = []

  for (const chunk of hunk.chunks) {
    const matchIdx = findChunkMatch(workLines, searchFrom, chunk)
    if (matchIdx === -1) {
      // Build a helpful error message
      const ctxPreview = chunk.lines
        .map((l) => `${l.type === "context" ? " " : l.type === "remove" ? "-" : "+"}${l.text}`)
        .join("\n")
      throw new Error(
        `Could not find context for patch chunk in ${hunk.path}.\n` +
          `Searched from line ${searchFrom + 1} onward.\n` +
          `Chunk:\n${ctxPreview}`,
      )
    }

    // Determine how many pattern (context+remove) lines this chunk consumes
    const patternLen = chunk.lines.filter((l) => l.type === "context" || l.type === "remove").length

    // Copy unchanged lines from searchFrom up to matchIdx
    for (let k = searchFrom; k < matchIdx; k++) result.push(workLines[k])

    // Now walk the chunk lines, emitting context/add lines and skipping remove lines
    let fileIdx = matchIdx
    for (const pl of chunk.lines) {
      if (pl.type === "context") {
        // Use the actual file line (preserves original formatting) if it matches
        if (fileIdx < workLines.length && linesMatch(workLines[fileIdx], pl.text)) {
          result.push(workLines[fileIdx])
        } else {
          result.push(pl.text)
        }
        fileIdx++
      } else if (pl.type === "remove") {
        // Validate the removed line actually exists (fuzzy match)
        if (fileIdx >= workLines.length || !linesMatch(workLines[fileIdx], pl.text)) {
          throw new Error(
            `Removed line not found in ${hunk.path} at line ${fileIdx + 1}: ${JSON.stringify(pl.text)}`,
          )
        }
        fileIdx++ // skip the removed line
      } else if (pl.type === "add") {
        result.push(pl.text)
      }
    }

    searchFrom = matchIdx + patternLen
  }

  // Copy any remaining lines after the last chunk
  for (let k = searchFrom; k < workLines.length; k++) result.push(workLines[k])

  let out = result.join("\n")
  if (hadTrailingNewline) out += "\n"
  return out
}

/**
 * Apply a parsed patch to the filesystem.
 *
 * - Applies all hunks sequentially.
 * - For UpdateFile: finds context lines in the existing file and replaces
 *   removed lines with added lines (fuzzy matching).
 * - For AddFile: creates the file (and parent dirs). Errors if file exists.
 * - For DeleteFile: deletes the file.
 * - For MoveFile: renames the file.
 * - If any hunk fails, the error is recorded but already-applied hunks are
 *   NOT rolled back. The result reports what was applied and what failed.
 */
export async function applyPatch(patchText: string, cwd: string = process.cwd()): Promise<ApplyResult> {
  let patch: Patch
  try {
    patch = parsePatch(patchText)
  } catch (e: any) {
    return { success: false, applied: [], errors: [`Parse error: ${String(e?.message ?? e)}`] }
  }

  const applied: string[] = []
  const errors: string[] = []

  for (const hunk of patch.hunks) {
    try {
      switch (hunk.type) {
        case "add": {
          const absPath = toAbs(hunk.path, cwd)
          if (existsSync(absPath)) {
            throw new Error(`Add File failed: file already exists: ${absPath}`)
          }
          await mkdir(dirname(absPath), { recursive: true }).catch(() => {})
          await writeFile(absPath, hunk.content, "utf8")
          applied.push(`add ${hunk.path}`)
          break
        }
        case "delete": {
          const absPath = toAbs(hunk.path, cwd)
          if (!existsSync(absPath)) {
            throw new Error(`Delete File failed: file not found: ${absPath}`)
          }
          const s = await stat(absPath)
          if (s.isDirectory()) {
            throw new Error(`Delete File failed: path is a directory: ${absPath}`)
          }
          await unlink(absPath)
          applied.push(`delete ${hunk.path}`)
          break
        }
        case "update": {
          const absPath = toAbs(hunk.path, cwd)
          if (!existsSync(absPath)) {
            throw new Error(`Update File failed: file not found: ${absPath}`)
          }
          const original = await readFile(absPath, "utf8")
          const updated = applyUpdateHunk(original, hunk)
          // If moveTo is specified, write to new path and remove old
          if (hunk.moveTo) {
            const newAbs = toAbs(hunk.moveTo, cwd)
            await mkdir(dirname(newAbs), { recursive: true }).catch(() => {})
            await writeFile(newAbs, updated, "utf8")
            if (absPath !== newAbs) await unlink(absPath).catch(() => {})
            applied.push(`update ${hunk.path} -> ${hunk.moveTo}`)
          } else {
            await writeFile(absPath, updated, "utf8")
            applied.push(`update ${hunk.path}`)
          }
          break
        }
        case "move": {
          const fromAbs = toAbs(hunk.from, cwd)
          const destAbs = toAbs(hunk.to, cwd)
          if (!existsSync(fromAbs)) {
            throw new Error(`Move File failed: source not found: ${fromAbs}`)
          }
          await mkdir(dirname(destAbs), { recursive: true }).catch(() => {})
          await rename(fromAbs, destAbs)
          applied.push(`move ${hunk.from} -> ${hunk.to}`)
          break
        }
      }
    } catch (e: any) {
      errors.push(`${hunk.type} ${"path" in hunk ? hunk.path : "from" in hunk ? hunk.from : ""}: ${String(e?.message ?? e)}`)
    }
  }

  return {
    success: errors.length === 0,
    applied,
    errors,
  }
}

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

export const APPLY_PATCH_TOOL: ToolDef = {
  type: "function",
  function: {
    name: "apply_patch",
    description:
      "Apply a structured patch to modify multiple files at once. Use this for multi-file edits.",
    parameters: {
      type: "object",
      properties: {
        patch: { type: "string", description: "The patch text in apply_patch format." },
      },
      required: ["patch"],
    },
  },
}

/**
 * Execute the apply_patch tool given parsed arguments.
 * Returns a ToolResult-compatible object { content, isError? }.
 */
export async function executeApplyPatch(
  args: Record<string, unknown>,
  cwd: string = process.cwd(),
): Promise<{ content: string; isError?: boolean }> {
  const patchText = String(args.patch ?? "")
  if (!patchText.trim()) {
    return { isError: true, content: "apply_patch: 'patch' argument is empty" }
  }
  const result = await applyPatch(patchText, cwd)
  const parts: string[] = []
  if (result.applied.length > 0) {
    parts.push(`Applied (${result.applied.length}):\n` + result.applied.map((a) => `  - ${a}`).join("\n"))
  }
  if (result.errors.length > 0) {
    parts.push(`Errors (${result.errors.length}):\n` + result.errors.map((e) => `  - ${e}`).join("\n"))
  }
  if (parts.length === 0) parts.push("No hunks in patch.")
  return {
    content: parts.join("\n\n"),
    isError: !result.success,
  }
}
