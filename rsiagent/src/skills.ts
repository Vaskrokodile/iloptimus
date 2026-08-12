// Skills system for RSI — inspired by opencode's skill architecture.
// Skills are markdown files with YAML frontmatter that provide domain-specific
// instructions to the agent. They are loaded from:
//   1. Project-local:  .rsi/skills/<name>/SKILL.md  (or  .rsi/skills/<name>.md)
//   2. Global:         ~/.config/rsi/skills/<name>/SKILL.md
// The agent can create new skills via the create_skill tool, and the user can
// manage them via the /skills command.

import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, statSync, rmSync } from "node:fs"
import { homedir } from "node:os"
import { join, dirname, basename } from "node:path"

export interface Skill {
  name: string
  description: string
  /** Whether this skill can be invoked as a slash command (/skill <name>). */
  slash: boolean
  /** Absolute path to the skill markdown file. */
  path: string
  /** The markdown body (without frontmatter) — injected into context when invoked. */
  content: string
  /** Where this skill was loaded from. */
  source: "project" | "global"
}

export interface SkillFrontmatter {
  name?: string
  description?: string
  slash?: boolean
}

const PROJECT_SKILLS_DIR = join(process.cwd(), ".rsi", "skills")
const GLOBAL_SKILLS_DIR = join(homedir(), ".config", "rsi", "skills")

/** Parse YAML-like frontmatter from a markdown string. Lightweight parser
 *  that handles the common key: value pairs used by skills. */
function parseFrontmatter(raw: string): { frontmatter: SkillFrontmatter; body: string } {
  const fmMatch = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/)
  if (!fmMatch) return { frontmatter: {}, body: raw }
  const fmText = fmMatch[1]
  const body = fmMatch[2]
  const fm: SkillFrontmatter = {}
  for (const line of fmText.split("\n")) {
    const m = line.match(/^(\w+)\s*:\s*(.*)$/)
    if (!m) continue
    const key = m[1].trim()
    let val = m[2].trim()
    // strip surrounding quotes
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1)
    }
    if (key === "slash") {
      fm.slash = val === "true" || val === "yes"
    } else if (key === "name") {
      fm.name = val
    } else if (key === "description") {
      fm.description = val
    }
  }
  return { frontmatter: fm, body }
}

/** Load a single skill from a markdown file path. */
function loadSkillFile(filepath: string, source: "project" | "global"): Skill | null {
  try {
    const raw = readFileSync(filepath, "utf8")
    const { frontmatter, body } = parseFrontmatter(raw)
    // Derive name: frontmatter > directory name > filename without extension
    let name = frontmatter.name
    if (!name) {
      const dir = dirname(filepath)
      if (basename(dir) !== "skills") {
        name = basename(dir)
      } else {
        name = basename(filepath, ".md")
      }
    }
    if (!name) return null
    return {
      name,
      description: frontmatter.description ?? "",
      slash: frontmatter.slash ?? true,
      path: filepath,
      content: body.trim(),
      source,
    }
  } catch {
    return null
  }
}

/** Scan a skills directory for skill markdown files. Supports both
 *  .rsi/skills/<name>/SKILL.md and .rsi/skills/<name>.md layouts. */
function scanSkillsDir(dir: string, source: "project" | "global"): Skill[] {
  const skills: Skill[] = []
  if (!existsSync(dir)) return skills
  let entries: string[]
  try {
    entries = readdirSync(dir)
  } catch {
    return skills
  }
  for (const entry of entries) {
    const full = join(dir, entry)
    let st
    try {
      st = statSync(full)
    } catch {
      continue
    }
    if (st.isDirectory()) {
      // Look for SKILL.md inside the directory
      const skillFile = join(full, "SKILL.md")
      if (existsSync(skillFile)) {
        const s = loadSkillFile(skillFile, source)
        if (s) skills.push(s)
      } else {
        // Also try <dirname>.md inside the directory
        const altFile = join(full, `${entry}.md`)
        if (existsSync(altFile)) {
          const s = loadSkillFile(altFile, source)
          if (s) skills.push(s)
        }
      }
    } else if (st.isFile() && entry.endsWith(".md")) {
      const s = loadSkillFile(full, source)
      if (s) skills.push(s)
    }
  }
  return skills
}

export class SkillManager {
  private skills = new Map<string, Skill>()
  private projectDir: string
  private globalDir: string

  constructor(projectDir?: string, globalDir?: string) {
    this.projectDir = projectDir ?? PROJECT_SKILLS_DIR
    this.globalDir = globalDir ?? GLOBAL_SKILLS_DIR
  }

  /** Reload all skills from disk. Project skills override global skills with
   *  the same name. */
  reload(): Skill[] {
    this.skills.clear()
    // Load global first, then project (project overrides)
    for (const s of scanSkillsDir(this.globalDir, "global")) {
      this.skills.set(s.name, s)
    }
    for (const s of scanSkillsDir(this.projectDir, "project")) {
      this.skills.set(s.name, s)
    }
    return this.list()
  }

  list(): Skill[] {
    return Array.from(this.skills.values()).sort((a, b) => a.name.localeCompare(b.name))
  }

  get(name: string): Skill | undefined {
    return this.skills.get(name)
  }

  /** Create a new skill on disk and register it. */
  create(name: string, description: string, content: string, slash = true): Skill {
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, "-").toLowerCase()
    const skillDir = join(this.projectDir, safeName)
    mkdirSync(skillDir, { recursive: true })
    const filepath = join(skillDir, "SKILL.md")
    const fm = [
      "---",
      `name: "${safeName}"`,
      `description: "${description.replace(/"/g, '\\"')}"`,
      `slash: ${slash}`,
      "---",
      "",
    ].join("\n")
    writeFileSync(filepath, fm + content + "\n")
    const skill: Skill = {
      name: safeName,
      description,
      slash,
      path: filepath,
      content: content.trim(),
      source: "project",
    }
    this.skills.set(safeName, skill)
    return skill
  }

  /** Delete a skill from disk and unregister it. */
  delete(name: string): boolean {
    const skill = this.skills.get(name)
    if (!skill) return false
    try {
      const dir = dirname(skill.path)
      // If the skill is in its own directory, remove the whole directory
      if (basename(dir) === skill.name) {
        rmSync(dir, { recursive: true, force: true })
      } else {
        rmSync(skill.path, { force: true })
      }
    } catch {
      // ignore
    }
    this.skills.delete(name)
    return true
  }

  /** Build the system-prompt section that lists available skills. */
  buildSkillContext(): string {
    const skills = this.list()
    if (skills.length === 0) return ""
    const lines = [
      "",
      "## Available Skills",
      "Skills are specialized instruction sets you can invoke for specific tasks. Use the invoke_skill tool to activate a skill, or the user can invoke it via /skill <name>.",
      "",
    ]
    for (const s of skills) {
      const desc = s.description ? ` — ${s.description}` : ""
      lines.push(`- **${s.name}**${desc}`)
    }
    lines.push("")
    lines.push("You can create new skills with the create_skill tool when you identify a reusable workflow or domain-specific instruction set.")
    return lines.join("\n")
  }

  /** Get the full instruction content for a skill (to inject when invoked). */
  getSkillContent(name: string): string | null {
    const s = this.skills.get(name)
    if (!s) return null
    return s.content
  }

  getProjectDir(): string {
    return this.projectDir
  }

  getGlobalDir(): string {
    return this.globalDir
  }
}
