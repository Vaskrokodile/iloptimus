// Persistent configuration for RSI: providers, models, effort, tools, mcp servers.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { homedir } from "node:os"
import { dirname, join } from "node:path"

export type ProviderType = "openai" | "anthropic"

export interface ModelDef {
  id: string
  name: string
  /** Optional override reasoning-effort field name supported by the provider. */
  effortParam?: string
}

export interface Provider {
  id: string
  name: string
  type: ProviderType
  baseURL: string
  apiKey: string
  models: ModelDef[]
}

export interface McpServerDef {
  id: string
  name: string
  command: string
  args: string[]
  env?: Record<string, string>
  enabled: boolean
}

export interface RsiConfig {
  providers: Provider[]
  currentProviderId: string | null
  currentModelId: string | null
  effort: "low" | "medium" | "high"
  toolsEnabled: boolean
  mcpServers: McpServerDef[]
  systemPrompt: string
}

const CONFIG_DIR = join(homedir(), ".config", "rsi")
const CONFIG_PATH = join(CONFIG_DIR, "config.json")

export const DEFAULT_SYSTEM_PROMPT =
  "You are RSI, a terminal coding agent restricted to one admitted workspace. " +
  "Use the provided tools to inspect, create, edit, and execute files; never claim an action happened unless its tool succeeded. " +
  "Work in small verified steps. Call exactly one tool at a time and wait for the result. " +
  "For write_file, provide a complete filename in path and only the file contents in content. " +
  "For run_command, provide the command and the correct workspace-relative cwd. " +
  "Do not invent tool names or output a simulated tool transcript. Finish with a short summary of real results."

const DEFAULT_CONFIG: RsiConfig = {
  providers: [],
  currentProviderId: null,
  currentModelId: null,
  effort: "high",
  toolsEnabled: true,
  mcpServers: [],
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
}

export function loadConfig(): RsiConfig {
  try {
    if (!existsSync(CONFIG_PATH)) return { ...DEFAULT_CONFIG }
    const raw = readFileSync(CONFIG_PATH, "utf8")
    const parsed = JSON.parse(raw)
    return { ...DEFAULT_CONFIG, ...parsed }
  } catch {
    return { ...DEFAULT_CONFIG }
  }
}

export function saveConfig(cfg: RsiConfig): void {
  try {
    mkdirSync(CONFIG_DIR, { recursive: true })
    writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2))
  } catch (e) {
    // non-fatal
  }
}

export function configPath(): string {
  return CONFIG_PATH
}

export function getCurrentProvider(cfg: RsiConfig): Provider | null {
  if (!cfg.currentProviderId) return null
  return cfg.providers.find((p) => p.id === cfg.currentProviderId) ?? null
}

export function getCurrentModel(cfg: RsiConfig): ModelDef | null {
  const p = getCurrentProvider(cfg)
  if (!p || !cfg.currentModelId) return null
  return p.models.find((m) => m.id === cfg.currentModelId) ?? p.models[0] ?? null
}

export function genId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 8)}`
}
