// Custom commands for RSI — inspired by opencode's custom command system.
// Users can define custom slash commands as markdown files in:
//   .rsi/commands/*.md       (project-scoped)
//   ~/.config/rsi/commands/*.md  (global)
// Each file's name (without .md) becomes the command name.
// The file content becomes the prompt template sent to the AI.
// Supports `!command` for bash output injection.

import { existsSync, readFileSync, readdirSync } from "node:fs"
import { join } from "node:path"
import { homedir } from "node:os"
import { execSync } from "node:child_process"

export interface CustomCommand {
  name: string
  description: string
  template: string
  source: "project" | "global"
  path: string
}

const PROJECT_DIR = ".rsi/commands"
const GLOBAL_DIR = join(homedir(), ".config", "rsi", "commands")

/** Discover all custom commands from project and global directories. */
export function loadCustomCommands(cwd: string = process.cwd()): CustomCommand[] {
  const commands: CustomCommand[] = []
  const seen = new Set<string>()

  // Global first (lower precedence)
  for (const cmd of scanDir(GLOBAL_DIR, "global")) {
    if (!seen.has(cmd.name)) {
      seen.add(cmd.name)
      commands.push(cmd)
    }
  }

  // Project (higher precedence — overrides global with same name)
  for (const cmd of scanDir(join(cwd, PROJECT_DIR), "project")) {
    const idx = commands.findIndex((c) => c.name === cmd.name)
    if (idx >= 0) {
      commands[idx] = cmd
    } else {
      commands.push(cmd)
    }
    seen.add(cmd.name)
  }

  return commands
}

function scanDir(dir: string, source: "project" | "global"): CustomCommand[] {
  const commands: CustomCommand[] = []
  if (!existsSync(dir)) return commands

  let entries: string[]
  try {
    entries = readdirSync(dir)
  } catch {
    return commands
  }

  for (const entry of entries) {
    if (!entry.endsWith(".md")) continue
    const name = entry.slice(0, -3)
    const path = join(dir, entry)
    let content: string
    try {
      content = readFileSync(path, "utf8")
    } catch {
      continue
    }

    // Extract description from first line if it starts with <!-- description: ... -->
    let description = ""
    const descMatch = content.match(/<!--\s*description:\s*(.+?)\s*-->/)
    if (descMatch) description = descMatch[1]

    commands.push({
      name,
      description: description || `custom command from ${path}`,
      template: content,
      source,
      path,
    })
  }

  return commands
}

/** Expand a command template, replacing `!command` with bash output. */
export function expandTemplate(template: string): string {
  // Replace `!{command}` or lines starting with `!` with bash output
  return template.replace(/^!(.+)$/gm, (_, cmd) => {
    try {
      return execSync(cmd.trim(), { encoding: "utf8", timeout: 10000 }).trim()
    } catch (e: any) {
      return `[error running: ${cmd.trim()}]`
    }
  })
}
