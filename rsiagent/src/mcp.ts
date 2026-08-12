// MCP (Model Context Protocol) client over stdio JSON-RPC.
// Supports tools, resources, and prompts discovery. Includes auto-reconnection
// and robust error handling. Inspired by opencode's MCP integration.
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process"
import type { McpServerDef } from "./config.ts"
import type { ToolDef } from "./tools.ts"

interface McpTool {
  serverId: string
  name: string
  description: string
  inputSchema: Record<string, unknown>
}

interface McpResource {
  serverId: string
  uri: string
  name: string
  description: string
  mimeType: string
}

interface McpPrompt {
  serverId: string
  name: string
  description: string
  arguments: { name: string; description: string; required: boolean }[]
}

const REQUEST_TIMEOUT = 30000

class McpConnection {
  private proc: ChildProcessWithoutNullStreams | null = null
  private buffer = ""
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void; timer: any }>()
  private id = 0
  private ready = false
  private tools: McpTool[] = []
  private resources: McpResource[] = []
  private prompts: McpPrompt[] = []
  private initPromise: Promise<void> | null = null
  private serverCapabilities: Record<string, any> = {}
  private reconnectAttempts = 0
  private maxReconnectAttempts = 3
  private shouldReconnect = true

  constructor(private def: McpServerDef) {}

  async start(): Promise<void> {
    if (this.initPromise) return this.initPromise
    this.initPromise = this._start()
    return this.initPromise
  }

  private async _start(): Promise<void> {
    const env = { ...process.env, ...this.def.env }
    this.proc = spawn(this.def.command, this.def.args, { env, stdio: ["pipe", "pipe", "pipe"] })
    this.proc.stdout.setEncoding("utf8")
    this.proc.stdout.on("data", (chunk: string) => {
      this.buffer += chunk
      let idx: number
      while ((idx = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, idx).trim()
        this.buffer = this.buffer.slice(idx + 1)
        if (!line) continue
        try {
          const msg = JSON.parse(line)
          this.handleMessage(msg)
        } catch {
          // ignore non-JSON lines (server logs)
        }
      }
    })
    this.proc.stderr.on("data", () => {
      // swallow server stderr
    })
    this.proc.on("error", (e) => {
      this.ready = false
      this.failAllPending(e)
    })
    this.proc.on("close", () => {
      this.ready = false
      this.failAllPending(new Error("MCP server closed"))
      // Auto-reconnect if enabled and not intentionally stopped
      if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++
        this.initPromise = null
        setTimeout(() => {
          this._start().catch(() => {})
        }, 1000 * this.reconnectAttempts)
      }
    })

    // initialize
    const initResp = await this.request("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "rsi", version: "1.0.0" },
    })
    this.serverCapabilities = initResp?.capabilities ?? {}
    this.notify("notifications/initialized", {})
    this.ready = true
    this.reconnectAttempts = 0

    // list tools (if supported)
    try {
      const toolsResp = await this.request("tools/list", {})
      const tools = toolsResp?.tools ?? []
      this.tools = tools.map((t: any) => ({
        serverId: this.def.id,
        name: t.name,
        description: t.description ?? "",
        inputSchema: t.inputSchema ?? { type: "object", properties: {} },
      }))
    } catch {
      this.tools = []
    }

    // list resources (if supported)
    try {
      if (this.serverCapabilities?.resources) {
        const resResp = await this.request("resources/list", {})
        const resources = resResp?.resources ?? []
        this.resources = resources.map((r: any) => ({
          serverId: this.def.id,
          uri: r.uri,
          name: r.name ?? r.uri,
          description: r.description ?? "",
          mimeType: r.mimeType ?? "",
        }))
      }
    } catch {
      this.resources = []
    }

    // list prompts (if supported)
    try {
      if (this.serverCapabilities?.prompts) {
        const promptResp = await this.request("prompts/list", {})
        const prompts = promptResp?.prompts ?? []
        this.prompts = prompts.map((p: any) => ({
          serverId: this.def.id,
          name: p.name,
          description: p.description ?? "",
          arguments: (p.arguments ?? []).map((a: any) => ({
            name: a.name,
            description: a.description ?? "",
            required: a.required ?? false,
          })),
        }))
      }
    } catch {
      this.prompts = []
    }
  }

  private failAllPending(e: any): void {
    for (const [id, p] of this.pending) {
      clearTimeout(p.timer)
      p.reject(e)
    }
    this.pending.clear()
  }

  private handleMessage(msg: any): void {
    if (msg.id != null && (msg.result !== undefined || msg.error !== undefined)) {
      const p = this.pending.get(msg.id)
      if (p) {
        clearTimeout(p.timer)
        this.pending.delete(msg.id)
        if (msg.error) p.reject(new Error(msg.error.message ?? "MCP error"))
        else p.resolve(msg.result)
      }
    }
  }

  private request(method: string, params: any): Promise<any> {
    if (!this.proc || !this.ready) {
      // Allow initialize to proceed even if not ready yet
      if (method !== "initialize") {
        return Promise.reject(new Error(`MCP server not ready: ${method}`))
      }
    }
    if (!this.proc) return Promise.reject(new Error("MCP not started"))
    const id = ++this.id
    const msg = JSON.stringify({ jsonrpc: "2.0", id, method, params })
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id)
          reject(new Error(`MCP request timed out (${REQUEST_TIMEOUT}ms): ${method}`))
        }
      }, REQUEST_TIMEOUT)
      this.pending.set(id, { resolve, reject, timer })
      try {
        this.proc!.stdin.write(msg + "\n")
      } catch (e) {
        clearTimeout(timer)
        this.pending.delete(id)
        reject(new Error(`Failed to send MCP request: ${e}`))
      }
    })
  }

  private notify(method: string, params: any): void {
    if (!this.proc) return
    const msg = JSON.stringify({ jsonrpc: "2.0", method, params })
    try {
      this.proc.stdin.write(msg + "\n")
    } catch {}
  }

  getTools(): McpTool[] {
    return this.tools
  }

  getResources(): McpResource[] {
    return this.resources
  }

  getPrompts(): McpPrompt[] {
    return this.prompts
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<string> {
    const resp = await this.request("tools/call", { name, arguments: args })
    if (resp?.content && Array.isArray(resp.content)) {
      return resp.content
        .map((c: any) => {
          if (c.type === "text") return c.text
          if (c.type === "image") return `[image: ${c.mimeType}]`
          if (c.type === "resource") return `[resource: ${c.resource?.uri}]`
          return JSON.stringify(c)
        })
        .join("\n")
    }
    return JSON.stringify(resp)
  }

  async readResource(uri: string): Promise<string> {
    const resp = await this.request("resources/read", { uri })
    if (resp?.contents && Array.isArray(resp.contents)) {
      return resp.contents
        .map((c: any) => {
          if (c.text) return c.text
          if (c.blob) return `[base64 blob: ${c.mimeType}]`
          return JSON.stringify(c)
        })
        .join("\n")
    }
    return JSON.stringify(resp)
  }

  async getPrompt(name: string, args: Record<string, string>): Promise<string> {
    const resp = await this.request("prompts/get", { name, arguments: args })
    if (resp?.messages && Array.isArray(resp.messages)) {
      return resp.messages
        .map((m: any) => {
          const role = m.role ?? "unknown"
          const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content)
          return `[${role}]: ${content}`
        })
        .join("\n")
    }
    return JSON.stringify(resp)
  }

  stop(): void {
    this.shouldReconnect = false
    try {
      this.proc?.stdin.end()
      this.proc?.kill("SIGTERM")
    } catch {}
    this.ready = false
  }

  isReady(): boolean {
    return this.ready
  }

  getServerName(): string {
    return this.def.name
  }
}

export class McpManager {
  private connections = new Map<string, McpConnection>()

  async connect(def: McpServerDef): Promise<{ ok: boolean; error?: string; toolCount: number; resourceCount: number; promptCount: number }> {
    try {
      const conn = new McpConnection(def)
      await conn.start()
      this.connections.set(def.id, conn)
      return {
        ok: true,
        toolCount: conn.getTools().length,
        resourceCount: conn.getResources().length,
        promptCount: conn.getPrompts().length,
      }
    } catch (e: any) {
      return { ok: false, error: String(e?.message ?? e), toolCount: 0, resourceCount: 0, promptCount: 0 }
    }
  }

  disconnect(id: string): void {
    const c = this.connections.get(id)
    if (c) {
      c.stop()
      this.connections.delete(id)
    }
  }

  disconnectAll(): void {
    for (const c of this.connections.values()) c.stop()
    this.connections.clear()
  }

  isConnected(id: string): boolean {
    return this.connections.get(id)?.isReady() ?? false
  }

  // All tools from all connected servers, namespaced as mcp__<server>__<tool>.
  getToolDefs(): ToolDef[] {
    const defs: ToolDef[] = []
    for (const conn of this.connections.values()) {
      if (!conn.isReady()) continue
      for (const t of conn.getTools()) {
        defs.push({
          type: "function",
          function: {
            name: `mcp__${t.serverId}__${t.name}`,
            description: `[MCP:${t.serverId}] ${t.description}`,
            parameters: t.inputSchema,
          },
        })
      }
    }
    return defs
  }

  async executeTool(namespacedName: string, args: Record<string, unknown>): Promise<string> {
    const parts = namespacedName.split("__")
    if (parts.length < 3 || parts[0] !== "mcp") {
      throw new Error(`Invalid MCP tool name: ${namespacedName}`)
    }
    const serverId = parts[1]
    const toolName = parts.slice(2).join("__")
    const conn = this.connections.get(serverId)
    if (!conn) throw new Error(`MCP server not connected: ${serverId}`)
    if (!conn.isReady()) throw new Error(`MCP server not ready: ${serverId}`)
    return conn.callTool(toolName, args)
  }

  /** Read a resource from a connected MCP server. */
  async readResource(serverId: string, uri: string): Promise<string> {
    const conn = this.connections.get(serverId)
    if (!conn) throw new Error(`MCP server not connected: ${serverId}`)
    return conn.readResource(uri)
  }

  /** Get a prompt from a connected MCP server. */
  async getPrompt(serverId: string, name: string, args: Record<string, string>): Promise<string> {
    const conn = this.connections.get(serverId)
    if (!conn) throw new Error(`MCP server not connected: ${serverId}`)
    return conn.getPrompt(name, args)
  }

  /** Get all resources from all connected servers. */
  getAllResources(): { serverId: string; serverName: string; uri: string; name: string; description: string }[] {
    const all: { serverId: string; serverName: string; uri: string; name: string; description: string }[] = []
    for (const conn of this.connections.values()) {
      if (!conn.isReady()) continue
      for (const r of conn.getResources()) {
        all.push({
          serverId: r.serverId,
          serverName: conn.getServerName(),
          uri: r.uri,
          name: r.name,
          description: r.description,
        })
      }
    }
    return all
  }

  /** Get all prompts from all connected servers. */
  getAllPrompts(): { serverId: string; serverName: string; name: string; description: string }[] {
    const all: { serverId: string; serverName: string; name: string; description: string }[] = []
    for (const conn of this.connections.values()) {
      if (!conn.isReady()) continue
      for (const p of conn.getPrompts()) {
        all.push({
          serverId: p.serverId,
          serverName: conn.getServerName(),
          name: p.name,
          description: p.description,
        })
      }
    }
    return all
  }

  status(): { id: string; ready: boolean; tools: number; resources: number; prompts: number }[] {
    return Array.from(this.connections.entries()).map(([id, c]) => ({
      id,
      ready: c.isReady(),
      tools: c.getTools().length,
      resources: c.getResources().length,
      prompts: c.getPrompts().length,
    }))
  }
}
