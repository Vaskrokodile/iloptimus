// Built-in tools the agent can call. These give the model full access to the
// user's machine: filesystem (read/write/list), shell execution, and the web.
import { spawn } from "node:child_process"
import { readFile, writeFile, readdir, stat, mkdir, rm, rename, copyFile, unlink } from "node:fs/promises"
import { existsSync } from "node:fs"
import { join, resolve, isAbsolute, dirname } from "node:path"

export interface ToolDef {
  type: "function"
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

export interface ToolResult {
  name: string
  content: string
  isError?: boolean
}

export const BUILTIN_TOOLS: ToolDef[] = [
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read the full contents of a file at the given absolute or relative path.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to the file to read." },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description:
        "Write text content to a file, creating it (and parent directories) if it does not exist. Overwrites existing content.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to the file to write." },
          content: { type: "string", description: "The full text content to write." },
        },
        required: ["path", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "edit_file",
      description:
        "Replace a single exact occurrence of old_string with new_string in the file at path. Fails if old_string is not found or is not unique.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
          old_string: { type: "string" },
          new_string: { type: "string" },
        },
        required: ["path", "old_string", "new_string"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_directory",
      description: "List the entries of a directory with type indicators.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Directory path. Defaults to cwd." },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_command",
      description:
        "Execute a shell command and return stdout+stderr. Use this to run builds, tests, git, install packages, etc. Commands run in the project cwd by default.",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "The shell command to execute." },
          cwd: { type: "string", description: "Optional working directory." },
          timeout: { type: "number", description: "Optional timeout in ms (default 60000)." },
        },
        required: ["command"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_fetch",
      description: "Fetch a URL and return its response body as text (HTML stripped to readable text).",
      parameters: {
        type: "object",
        properties: {
          url: { type: "string", description: "The full URL to fetch." },
        },
        required: ["url"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web via DuckDuckGo HTML and return the top results (title, url, snippet).",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "The search query." },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "delete_file",
      description: "Delete a file at the given path. Use with caution — this is irreversible.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to the file to delete." },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "delete_directory",
      description: "Recursively delete a directory and all its contents. Use with caution — this is irreversible.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to the directory to delete." },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "move_file",
      description: "Move or rename a file/directory from source to destination. Creates parent directories of the destination if needed.",
      parameters: {
        type: "object",
        properties: {
          source: { type: "string", description: "Path to the file/directory to move." },
          destination: { type: "string", description: "The new path." },
        },
        required: ["source", "destination"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "copy_file",
      description: "Copy a file from source to destination. Creates parent directories of the destination if needed.",
      parameters: {
        type: "object",
        properties: {
          source: { type: "string", description: "Path to the source file." },
          destination: { type: "string", description: "Path to the destination." },
        },
        required: ["source", "destination"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "create_directory",
      description: "Create a directory (and any missing parent directories) at the given path.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to the directory to create." },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "file_info",
      description: "Get detailed information about a file or directory: size, type, timestamps, permissions.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to inspect." },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "glob_search",
      description: "Find files matching a glob pattern (e.g. **/*.ts) within a directory. Returns matching file paths.",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string", description: "Glob pattern to match (e.g. **/*.ts, *.json)." },
          path: { type: "string", description: "Base directory to search in. Defaults to cwd." },
        },
        required: ["pattern"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "grep_search",
      description: "Search for a text pattern (regex) within files in a directory. Returns matching lines with file paths and line numbers.",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string", description: "The regex pattern to search for." },
          path: { type: "string", description: "Base directory to search in. Defaults to cwd." },
          include: { type: "string", description: "Optional file glob to filter (e.g. *.ts)." },
        },
        required: ["pattern"],
      },
    },
  },
]

function abs(p: string): string {
  return isAbsolute(p) ? p : resolve(process.cwd(), p)
}

function truncate(s: string, max = 20000): string {
  if (s.length <= max) return s
  return s.slice(0, max) + `\n...[truncated ${s.length - max} chars]`
}

function stripHtml(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim()
}

export async function executeBuiltinTool(
  name: string,
  args: Record<string, unknown>,
): Promise<ToolResult> {
  try {
    switch (name) {
      case "read_file": {
        const p = abs(String(args.path ?? ""))
        const data = await readFile(p, "utf8")
        return { name, content: truncate(data) }
      }
      case "write_file": {
        const p = abs(String(args.path ?? ""))
        await mkdir(dirname(p), { recursive: true }).catch(() => {})
        await writeFile(p, String(args.content ?? ""))
        return { name, content: `Wrote ${p} (${String(args.content ?? "").length} bytes)` }
      }
      case "edit_file": {
        const p = abs(String(args.path ?? ""))
        const oldStr = String(args.old_string ?? "")
        const newStr = String(args.new_string ?? "")
        const orig = await readFile(p, "utf8")
        const count = orig.split(oldStr).length - 1
        if (count === 0) return { name, isError: true, content: `old_string not found in ${p}` }
        if (count > 1) return { name, isError: true, content: `old_string not unique (${count} matches) in ${p}` }
        const updated = orig.replace(oldStr, newStr)
        await writeFile(p, updated)
        return { name, content: `Edited ${p}` }
      }
      case "list_directory": {
        const p = abs(String(args.path ?? process.cwd()))
        const entries = await readdir(p, { withFileTypes: true })
        const lines = entries.map((e) => {
          const t = e.isDirectory() ? "dir " : e.isSymbolicLink() ? "link" : "file"
          return `${t}  ${e.name}`
        })
        return { name, content: lines.join("\n") || "(empty)" }
      }
      case "run_command": {
        const cmd = String(args.command ?? "")
        const cwd = args.cwd ? abs(String(args.cwd)) : process.cwd()
        const timeout = Number(args.timeout ?? 60000)
        return await new Promise<ToolResult>((resolveP) => {
          const child = spawn(cmd, { shell: true, cwd, env: process.env })
          let out = ""
          const timer = setTimeout(() => {
            child.kill("SIGTERM")
            out += "\n[timeout reached]"
            resolveP({ name, content: truncate(out) })
          }, timeout)
          child.stdout.on("data", (d) => (out += d.toString()))
          child.stderr.on("data", (d) => (out += d.toString()))
          child.on("close", (code) => {
            clearTimeout(timer)
            out += `\n[exit code ${code}]`
            resolveP({ name, isError: code !== 0, content: truncate(out) })
          })
          child.on("error", (e) => {
            clearTimeout(timer)
            resolveP({ name, isError: true, content: String(e) })
          })
        })
      }
      case "web_fetch": {
        const url = String(args.url ?? "")
        const res = await fetch(url, { redirect: "follow", headers: { "User-Agent": "rsi/1.0" } })
        const text = await res.text()
        const ct = res.headers.get("content-type") ?? ""
        const body = ct.includes("html") ? stripHtml(text) : text
        return { name, content: truncate(body, 30000) }
      }
      case "web_search": {
        const query = String(args.query ?? "")
        const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`
        const res = await fetch(url, { headers: { "User-Agent": "rsi/1.0" } })
        const html = await res.text()
        const results: string[] = []
        const re =
          /<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g
        let m: RegExpExecArray | null
        let i = 0
        while ((m = re.exec(html)) && i < 8) {
          results.push(
            `${i + 1}. ${stripHtml(m[2])}\n   ${stripHtml(m[3])}\n   ${m[1]}`,
          )
          i++
        }
        return { name, content: results.join("\n\n") || "No results." }
      }
      case "delete_file": {
        const p = abs(String(args.path ?? ""))
        if (!existsSync(p)) return { name, isError: true, content: `File not found: ${p}` }
        const s = await stat(p)
        if (s.isDirectory()) return { name, isError: true, content: `Path is a directory, use delete_directory instead: ${p}` }
        await unlink(p)
        return { name, content: `Deleted file ${p}` }
      }
      case "delete_directory": {
        const p = abs(String(args.path ?? ""))
        if (!existsSync(p)) return { name, isError: true, content: `Directory not found: ${p}` }
        const s = await stat(p)
        if (!s.isDirectory()) return { name, isError: true, content: `Path is a file, use delete_file instead: ${p}` }
        await rm(p, { recursive: true, force: true })
        return { name, content: `Deleted directory ${p}` }
      }
      case "move_file": {
        const src = abs(String(args.source ?? ""))
        const dst = abs(String(args.destination ?? ""))
        if (!existsSync(src)) return { name, isError: true, content: `Source not found: ${src}` }
        await mkdir(dirname(dst), { recursive: true }).catch(() => {})
        await rename(src, dst)
        return { name, content: `Moved ${src} → ${dst}` }
      }
      case "copy_file": {
        const src = abs(String(args.source ?? ""))
        const dst = abs(String(args.destination ?? ""))
        if (!existsSync(src)) return { name, isError: true, content: `Source not found: ${src}` }
        await mkdir(dirname(dst), { recursive: true }).catch(() => {})
        await copyFile(src, dst)
        return { name, content: `Copied ${src} → ${dst}` }
      }
      case "create_directory": {
        const p = abs(String(args.path ?? ""))
        await mkdir(p, { recursive: true })
        return { name, content: `Created directory ${p}` }
      }
      case "file_info": {
        const p = abs(String(args.path ?? ""))
        if (!existsSync(p)) return { name, isError: true, content: `Path not found: ${p}` }
        const s = await stat(p)
        const info = [
          `path: ${p}`,
          `type: ${s.isDirectory() ? "directory" : s.isSymbolicLink() ? "symlink" : "file"}`,
          `size: ${s.size} bytes`,
          `modified: ${s.mtime.toISOString()}`,
          `created: ${s.birthtime.toISOString()}`,
          `permissions: ${s.mode.toString(8).slice(-3)}`,
        ]
        return { name, content: info.join("\n") }
      }
      case "glob_search": {
        const pattern = String(args.pattern ?? "")
        const basePath = abs(String(args.path ?? process.cwd()))
        const matches = await globMatch(pattern, basePath)
        return { name, content: matches.length > 0 ? matches.join("\n") : "No matches." }
      }
      case "grep_search": {
        const pattern = String(args.pattern ?? "")
        const basePath = abs(String(args.path ?? process.cwd()))
        const include = args.include ? String(args.include) : undefined
        const matches = await grepMatch(pattern, basePath, include)
        return { name, content: matches.length > 0 ? matches.join("\n") : "No matches." }
      }
      default:
        return { name, isError: true, content: `Unknown builtin tool: ${name}` }
    }
  } catch (e: any) {
    return { name, isError: true, content: String(e?.message ?? e) }
  }
}

// ---- Skill tools (definitions only — execution handled in index.ts) ----
export const SKILL_TOOLS: ToolDef[] = [
  {
    type: "function",
    function: {
      name: "list_skills",
      description:
        "List all available skills with their names and descriptions. Skills are specialized instruction sets that can be invoked for specific tasks.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "invoke_skill",
      description:
        "Invoke a skill by name to activate its specialized instructions. The skill's content is injected into the conversation context. Use this when the user's task matches a skill's domain.",
      parameters: {
        type: "object",
        properties: {
          name: { type: "string", description: "The name of the skill to invoke." },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "create_skill",
      description:
        "Create a new skill with specialized instructions. Use this when you identify a reusable workflow, domain-specific knowledge, or a pattern the user would benefit from having as a named skill. The skill is saved to .rsi/skills/<name>/SKILL.md and becomes available immediately.",
      parameters: {
        type: "object",
        properties: {
          name: { type: "string", description: "Skill name (lowercase, hyphens ok, e.g. 'python-testing')." },
          description: { type: "string", description: "A short description of what the skill does." },
          content: { type: "string", description: "The full markdown instructions for the skill." },
          slash: { type: "boolean", description: "Whether this skill can be invoked as a slash command. Default true." },
        },
        required: ["name", "description", "content"],
      },
    },
  },
]

// ---- Subagent tool (definition only — execution handled in index.ts) ----
export const SUBAGENT_TOOLS: ToolDef[] = [
  {
    type: "function",
    function: {
      name: "spawn_subagent",
      description:
        "Spawn a subagent to handle a self-contained task autonomously. The subagent has its own isolated context, can use built-in tools (read, write, run_command, etc.), and returns a final result. Use this for tasks that can be parallelized or should be isolated from the main conversation. The subagent cannot access MCP tools or skills.",
      parameters: {
        type: "object",
        properties: {
          task: { type: "string", description: "The specific task for the subagent to complete." },
          system_prompt: { type: "string", description: "Optional custom system prompt for the subagent." },
        },
        required: ["task"],
      },
    },
  },
]

// Convert a ToolDef list to the OpenAI tools array shape.
export function toOpenAITools(tools: ToolDef[]): any[] {
  return tools
}

// Convert to Anthropic tool shape.
export function toAnthropicTools(tools: ToolDef[]): any[] {
  return tools.map((t) => ({
    name: t.function.name,
    description: t.function.description,
    input_schema: t.function.parameters,
  }))
}

// ---- glob search ----
async function globMatch(pattern: string, basePath: string): Promise<string[]> {
  try {
    const glob = new Bun.Glob(pattern)
    const matches: string[] = []
    for await (const path of glob.scan({ cwd: basePath, absolute: true })) {
      matches.push(path)
      if (matches.length >= 200) break
    }
    return matches
  } catch {
    return simpleGlob(pattern, basePath)
  }
}

async function simpleGlob(pattern: string, basePath: string): Promise<string[]> {
  // Convert simple glob to regex: support * and ** and ?
  const results: string[] = []
  const regex = globToRegex(pattern)
  async function walk(dir: string, depth: number) {
    if (depth > 15 || results.length > 200) return
    let entries: import("node:fs").Dirent[]
    try {
      entries = await readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      if (e.name.startsWith(".") && !pattern.startsWith(".")) continue
      const full = join(dir, e.name)
      const rel = full.slice(basePath.length + 1).replace(/\\/g, "/")
      if (regex.test(rel)) results.push(full)
      if (e.isDirectory()) await walk(full, depth + 1)
    }
  }
  await walk(basePath, 0)
  return results.slice(0, 200)
}

function globToRegex(pattern: string): RegExp {
  let re = ""
  let i = 0
  while (i < pattern.length) {
    const c = pattern[i]
    if (c === "*" && pattern[i + 1] === "*") {
      re += ".*"
      i += 2
      if (pattern[i] === "/") i++
    } else if (c === "*") {
      re += "[^/]*"
      i++
    } else if (c === "?") {
      re += "[^/]"
      i++
    } else if (".+^$(){}|[]".includes(c)) {
      re += "\\" + c
      i++
    } else {
      re += c
      i++
    }
  }
  return new RegExp("^" + re + "$")
}

// ---- grep search ----
async function grepMatch(pattern: string, basePath: string, include?: string): Promise<string[]> {
  const regex = new RegExp(pattern)
  const results: string[] = []
  const includeRegex = include ? globToRegex(include) : null
  async function walk(dir: string, depth: number) {
    if (depth > 15 || results.length > 100) return
    let entries: import("node:fs").Dirent[]
    try {
      entries = await readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      if (e.name.startsWith(".")) continue
      if (e.name === "node_modules") continue
      const full = join(dir, e.name)
      if (e.isDirectory()) {
        await walk(full, depth + 1)
      } else if (e.isFile()) {
        const rel = e.name
        if (includeRegex && !includeRegex.test(rel)) continue
        try {
          const content = await readFile(full, "utf8")
          const lines = content.split("\n")
          for (let li = 0; li < lines.length; li++) {
            if (regex.test(lines[li])) {
              results.push(`${full}:${li + 1}: ${lines[li].trim().slice(0, 200)}`)
              if (results.length >= 100) return
            }
          }
        } catch {
          // skip unreadable files
        }
      }
    }
  }
  await walk(basePath, 0)
  return results
}
