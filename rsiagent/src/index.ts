#!/usr/bin/env bun
// RSI - a full terminal AI harness built on OpenTUI.
// Usage: bun src/index.ts   (or `bun start`)
//
// Slash commands: /help /model /providers /clear /goal /review /effort /tools /mcp /skills /skill /config /exit
// Submit input: Enter. Newline: Shift+Enter. Abort streaming / close popups: Esc.

import {
  createCliRenderer,
  createTimeline,
  BoxRenderable,
  TextRenderable,
  ScrollBoxRenderable,
  TextareaRenderable,
  SelectRenderable,
  SelectRenderableEvents,
  InputRenderable,
  InputRenderableEvents,
  TextAttributes,
  t,
  fg,
  bold,
  dim,
  italic,
  underline,
} from "@opentui/core"
import {
  loadConfig,
  saveConfig,
  configPath,
  getCurrentProvider,
  getCurrentModel,
  genId,
  type RsiConfig,
  type Provider,
  type ModelDef,
  type ProviderType,
  type McpServerDef,
} from "./config.ts"
import { createProviderClient, type ChatMessage, type ProviderClient } from "./providers.ts"
import { BUILTIN_TOOLS, SKILL_TOOLS, SUBAGENT_TOOLS, executeBuiltinTool, type ToolDef, type ToolResult } from "./tools.ts"
import { McpManager } from "./mcp.ts"
import { SkillManager, type Skill } from "./skills.ts"
import { runSubagent, SUBAGENT_SYSTEM_PROMPT } from "./agent.ts"
import { needsCompaction, buildCompactionPrompt, compactMessages, estimateTokens, DEFAULT_COMPACTION } from "./compaction.ts"
import { buildContextFromFiles, listContextFiles } from "./context.ts"
import { loadCustomCommands, expandTemplate, type CustomCommand } from "./commands.ts"
import { checkPermission, DEFAULT_PERMISSIONS, loadPermissions, type PermissionEffect } from "./permissions.ts"
import { MemoryManager, MEMORY_TOOLS, MEMORY_TOOL_NAMES, executeMemoryTool } from "./memory.ts"
import { SessionManager, type Session, type SessionEntry } from "./session.ts"
import { APPLY_PATCH_TOOL, applyPatch, executeApplyPatch } from "./apply-patch.ts"

// ---------- theme ----------
const C = {
  bg: "#0a0e14",
  panel: "#11161f",
  panelAlt: "#0d1117",
  border: "#2a3447",
  borderActive: "#7aa2f7",
  text: "#c0caf5",
  dim: "#565f89",
  user: "#7aa2f7",
  assistant: "#9ece6a",
  tool: "#e0af68",
  toolResult: "#73daca",
  error: "#f7768e",
  accent: "#bb9af7",
  neon1: "#ff5d9e",
  neon2: "#00e5ff",
  neon3: "#7ee787",
  neon4: "#ffd93d",
  neon5: "#b388ff",
  // golden gradient + white for banner
  gold1: "#ffffff",
  gold2: "#ffe55c",
  gold3: "#ffd700",
  gold4: "#ffc107",
  gold5: "#ffb300",
  gold6: "#ffa000",
  gold7: "#ff8f00",
  goldDim: "#b8860b",
}

// ---------- global state ----------
const cfg: RsiConfig = loadConfig()
const mcp = new McpManager()
const skills = new SkillManager()
skills.reload()
const customCommands = loadCustomCommands()
const permConfig = loadPermissions((cfg as any).permissions)
const memory = new MemoryManager()
memory.reload()
const sessions = new SessionManager()
let currentSession: Session | null = null
let planMode = false
let reviewMode = false
let busy = false
let abortController: AbortController | null = null
let popup: any = null
const r = (await createCliRenderer({ exitOnCtrlC: false })) as any
r.root.backgroundColor = C.bg

// ---------- system prompt builder ----------
// Centralizes system prompt construction: base prompt + skills + context files + plan/review mode
function buildSystemPrompt(): string {
  let prompt = cfg.systemPrompt + skills.buildSkillContext()
  // Add AGENTS.md / CLAUDE.md context files
  const ctxFiles = buildContextFromFiles()
  if (ctxFiles) prompt += "\n\n" + ctxFiles
  // Add persistent memories
  const memCtx = memory.buildMemoryContext()
  if (memCtx) prompt += "\n\n" + memCtx
  if (planMode) prompt += "\n\n" + PLAN_MODE_SUFFIX
  if (reviewMode) prompt += "\n\n" + reviewSuffix()
  return prompt
}

let messages: ChatMessage[] = [{ role: "system", content: buildSystemPrompt() }]

const PLAN_MODE_SUFFIX = `[PLAN MODE] You are in plan mode. Explore the codebase first before making any changes.
- Use read-only tools (read_file, list_directory, glob_search, grep_search) to understand the codebase
- Do NOT make any file changes (write_file, edit_file, delete_file, run_command with writes, etc.)
- When you have enough context, provide a TL;DR plan with 3-5 bullet points
- Wait for user approval before implementing
- After approval, the user will switch you out of plan mode`

// ---------- helpers ----------
function sysLine(s: any, color = C.dim) {
  addChatLine(s, color, true)
}

function addChatText(content: any, color = C.text, italicFlag = false) {
  const txt = new TextRenderable(r, {
    content,
    fg: color,
    attributes: italicFlag ? TextAttributes.ITALIC : 0,
    wrapMode: "word",
  } as any)
  chatBox.add(txt)
  chatBox.scrollTo(Infinity)
  return txt
}

function addChatLine(content: any, color = C.text, italicFlag = false) {
  return addChatText(content, color, italicFlag)
}

function addMessageBlock(role: string, roleColor: string, body: any) {
  const wrap = new BoxRenderable(r, { width: "100%", flexDirection: "column", gap: 0 } as any)
  const label = new TextRenderable(r, {
    content: t`${bold(fg(roleColor)(role))}`,
    fg: roleColor,
  } as any)
  wrap.add(label)
  const bodyTxt = new TextRenderable(r, { content: body, fg: C.text, wrapMode: "word" } as any)
  wrap.add(bodyTxt)
  chatBox.add(wrap)
  chatBox.scrollTo(Infinity)
  return bodyTxt
}

// ---------- banner ----------
// Clean "RSI" in ANSI Shadow font — recognizable, renders well in all terminals
const BANNER = [
  "██████╗  ███████╗ ██╗",
  "██╔═══██╗ ██╔════╝ ██║",
  "██████╔╝ ███████╗ ██║",
  "██╔═══╝  ╚════██║ ██║",
  "██║      ███████║ ██║",
  "╚═╝      ╚══════╝ ╚═╝",
]
// White → golden gradient (top to bottom)
const BANNER_COLORS = [C.gold1, C.gold2, C.gold3, C.gold4, C.gold5, C.gold6]

const bannerBox = new BoxRenderable(r, {
  width: "100%",
  height: 9,
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  padding: 0,
  paddingTop: 1,
  paddingBottom: 1,
} as any)
bannerBox.backgroundColor = C.bg

const bannerLines: TextRenderable[] = []
for (let i = 0; i < BANNER.length; i++) {
  const line = new TextRenderable(r, {
    content: BANNER[i],
    fg: BANNER_COLORS[i],
    attributes: TextAttributes.BOLD,
    opacity: 0,
  } as any)
  bannerLines.push(line)
  bannerBox.add(line)
}
const tagline = new TextRenderable(r, {
  content: t`${dim(italic("a terminal AI agent with full machine access"))}`,
  fg: C.goldDim,
  opacity: 0,
} as any)
bannerBox.add(tagline)
r.root.add(bannerBox)

// ---------- blackhole-style banner animation ----------
// Characters emerge from center outward, golden glow fades in with outExpo easing
const bannerTimeline = createTimeline({
  duration: 1200,
  onComplete: () => {
    // ensure all visible at end
    for (const l of bannerLines) (l as any).opacity = 1
    ;(tagline as any).opacity = 1
  },
})

// center lines appear first, then radiate outward (blackhole reverse)
const stagger = [300, 200, 0, 0, 200, 300] // ms delays per line (center-out)
for (let i = 0; i < bannerLines.length; i++) {
  bannerTimeline.add(
    bannerLines[i],
    {
      duration: 500,
      ease: "outExpo",
      onUpdate: (anim: any) => {
        ;(bannerLines[i] as any).opacity = anim.progress
      },
    },
    stagger[i],
  )
}
// tagline fades in after the banner
bannerTimeline.add(
  tagline,
  {
    duration: 400,
    ease: "outExpo",
    onUpdate: (anim: any) => {
      ;(tagline as any).opacity = anim.progress * 0.7
    },
  },
  600,
)
bannerTimeline.play()

// ---------- model header (model + reasoning effort, above chat) ----------
const headerBar = new BoxRenderable(r, {
  width: "100%",
  height: 1,
  flexDirection: "row",
  paddingLeft: 1,
  paddingRight: 1,
  backgroundColor: C.panel,
} as any)
const headerText = new TextRenderable(r, { content: "", fg: C.text } as any)
headerBar.add(headerText)
r.root.add(headerBar)

// ---------- chat ----------
const chatBox = new ScrollBoxRenderable(r, {
  id: "chat",
  width: "100%",
  flexGrow: 1,
  stickyScroll: true,
  stickyStart: "bottom",
  viewportCulling: false,
  rootOptions: { backgroundColor: C.panelAlt },
  wrapperOptions: { backgroundColor: C.panelAlt },
  viewportOptions: { backgroundColor: C.panelAlt },
  contentOptions: { backgroundColor: C.panelAlt },
  scrollbarOptions: { trackOptions: { foregroundColor: C.border, backgroundColor: C.panelAlt } },
} as any)
chatBox.backgroundColor = C.panelAlt
r.root.add(chatBox)

// ---------- input ----------
const inputBox = new BoxRenderable(r, {
  width: "100%",
  height: 5,
  borderStyle: "rounded",
  borderColor: C.border,
  backgroundColor: C.panel,
  paddingLeft: 1,
  paddingRight: 1,
} as any)
const input = new TextareaRenderable(r, {
  id: "input",
  width: "100%",
  height: 3,
  placeholder: t`${dim("Send a message (Enter)  ·  Shift+Enter for newline  ·  / for commands  ·  Esc to abort")}`,
  placeholderColor: C.dim,
  backgroundColor: "transparent",
  textColor: C.text,
  focusedTextColor: C.text,
  cursorColor: C.neon2,
  wrapMode: "word",
  onSubmit: () => handleSubmit(),
  keyBindings: [
    { name: "return", action: "submit" },
    { name: "kpenter", action: "submit" },
    { name: "return", shift: true, action: "newline" },
    { name: "kpenter", shift: true, action: "newline" },
  ],
} as any)
inputBox.add(input)
r.root.add(inputBox)

// ---------- status bar ----------
const statusBar = new BoxRenderable(r, {
  width: "100%",
  height: 1,
  flexDirection: "row",
  paddingLeft: 1,
  paddingRight: 1,
  backgroundColor: C.panel,
} as any)
const statusText = new TextRenderable(r, { content: "", fg: C.dim } as any)
statusBar.add(statusText)
r.root.add(statusBar)

input.focus()

// ---------- status / header refresh ----------
function refreshChrome() {
  const p = getCurrentProvider(cfg)
  const m = getCurrentModel(cfg)
  const modelName = m ? m.name : "none"
  const provName = p ? p.name : "no provider"
  headerText.content = t`${fg(C.accent)("model")} ${fg(C.text)(modelName)}   ${fg(C.accent)("effort")} ${fg(C.neon4)(cfg.effort)}   ${fg(C.accent)("provider")} ${fg(C.text)(provName)}`
  const toolsState = cfg.toolsEnabled ? fg(C.neon3)("on") : fg(C.error)("off")
  const mcpCount = mcp.status().filter((s) => s.ready).length
  const mcpState = mcpCount > 0 ? fg(C.neon3)(`${mcpCount} connected`) : fg(C.dim)("none")
  const skillCount = skills.list().length
  const skillState = skillCount > 0 ? fg(C.neon3)(`${skillCount} loaded`) : fg(C.dim)("none")
  const tokenEst = estimateTokens(messages)
  const tokenState = tokenEst > 50000 ? fg(C.error)(`${(tokenEst / 1000).toFixed(0)}k`) : fg(C.dim)(`${(tokenEst / 1000).toFixed(0)}k`)
  statusText.content = t`${fg(C.dim)("rsi")}  ${fg(C.dim)("│")}  tools ${toolsState}  ${fg(C.dim)("│")}  mcp ${mcpState}  ${fg(C.dim)("│")}  skills ${skillState}  ${fg(C.dim)("│")}  ctx ${tokenState}  ${fg(C.dim)("│")}  ${fg(C.dim)(planMode ? "plan mode" : reviewMode ? "review mode" : "agent mode")}  ${fg(C.dim)("│")}  ${fg(C.dim)(configPath())}`
}
refreshChrome()

// ---------- welcome ----------
sysLine("Welcome to RSI. Type / for command autocomplete, /help for full list, /providers to connect an AI provider.", C.dim)
if (!getCurrentProvider(cfg)) {
  sysLine("No provider configured yet. Use /providers to add your API key and base URL.", C.error)
}
if (skills.list().length > 0) {
  sysLine(`${skills.list().length} skill(s) loaded. Use /skills to manage, /skill <name> to invoke.`, C.dim)
} else {
  sysLine("No skills loaded. Place markdown files in .rsi/skills/ or ask the AI to create one.", C.dim)
}
const ctxFiles = listContextFiles()
if (ctxFiles.length > 0) {
  sysLine(`${ctxFiles.length} context file(s) loaded: ${ctxFiles.map((f) => f.split("/").pop()).join(", ")}`, C.dim)
}
if (memory.list().length > 0) {
  sysLine(`${memory.list().length} memory(s) loaded. Use /config to see details.`, C.dim)
}
if (customCommands.length > 0) {
  sysLine(`${customCommands.length} custom command(s): /${customCommands.map((c) => c.name).join(" /")}`, C.dim)
}
const recentSession = sessions.recent(process.cwd())
if (recentSession) {
  sysLine(`Recent session found (${recentSession.getEntries().length} entries). Use /sessions to resume.`, C.dim)
}

// ============================================================
//  Popups
// ============================================================
function closePopup() {
  if (popup) {
    try {
      r.root.remove(popup)
    } catch {}
    popup = null
    input.focus()
  }
}

function openOverlay(width: number, height: number, title: string): any {
  const overlay = new BoxRenderable(r, {
    position: "absolute",
    left: "50%",
    top: "50%",
    marginLeft: -Math.floor(width / 2),
    marginTop: -Math.floor(height / 2),
    width,
    height,
    borderStyle: "rounded",
    borderColor: C.borderActive,
    backgroundColor: C.panel,
    title,
    titleColor: C.borderActive,
    padding: 1,
    flexDirection: "column",
  } as any)
  overlay.zIndex = 100
  r.root.add(overlay)
  popup = overlay
  return overlay
}

// ---- model picker ----
function openModelPopup() {
  if (popup) closePopup()
  const all: { provider: Provider; model: ModelDef }[] = []
  for (const p of cfg.providers) for (const m of p.models) all.push({ provider: p, model: m })
  if (all.length === 0) {
    sysLine("No models available. Add a provider first with /providers.", C.error)
    return
  }
  const overlay = openOverlay(56, Math.min(20, 6 + all.length), "Select model")
  const options = all.map(({ provider, model }) => ({
    name: `${model.name}`,
    description: `${provider.name} · ${provider.type}`,
    value: { providerId: provider.id, modelId: model.id },
  }))
  const sel = new SelectRenderable(r, {
    width: "100%",
    height: overlay.height - 4,
    options,
    backgroundColor: "transparent",
    textColor: C.text,
    focusedBackgroundColor: C.panelAlt,
    focusedTextColor: C.text,
    selectedBackgroundColor: C.borderActive,
    selectedTextColor: "#0a0e14",
    descriptionColor: C.dim,
  } as any)
  const curIdx = all.findIndex(
    (a) => a.provider.id === cfg.currentProviderId && a.model.id === cfg.currentModelId,
  )
  if (curIdx >= 0) sel.setSelectedIndex(curIdx)
  overlay.add(sel)
  sel.on(SelectRenderableEvents.ITEM_SELECTED, (_i: number, opt: any) => {
    cfg.currentProviderId = opt.value.providerId
    cfg.currentModelId = opt.value.modelId
    saveConfig(cfg)
    refreshChrome()
    const p = getCurrentProvider(cfg)!
    sysLine(`Switched to ${getCurrentModel(cfg)?.name} via ${p.name}.`, C.neon3)
    closePopup()
  })
  sel.focus()
}

// ---- providers popup ----
function openProvidersPopup() {
  if (popup) closePopup()
  const overlay = openOverlay(60, 18, "Providers")
  const list = new SelectRenderable(r, {
    width: "100%",
    height: 6,
    options:
      cfg.providers.length > 0
        ? cfg.providers.map((p) => ({
            name: `${p.name}  [${p.type}]`,
            description: p.baseURL,
            value: p.id,
          }))
        : [{ name: "(no providers yet)", description: "use the add option below", value: null }],
    backgroundColor: "transparent",
    textColor: C.text,
    focusedBackgroundColor: C.panelAlt,
    selectedBackgroundColor: C.borderActive,
    selectedTextColor: "#0a0e14",
    descriptionColor: C.dim,
  } as any)
  overlay.add(list)

  const actions = new SelectRenderable(r, {
    width: "100%",
    height: 6,
    options: [
      { name: "Add new provider…", description: "name, type, base URL, API key", value: "add" },
      { name: "Add model to selected…", description: "model id + display name", value: "addmodel" },
      { name: "Set active", description: "use the highlighted provider", value: "setactive" },
      { name: "Delete selected", description: "remove the highlighted provider", value: "delete" },
      { name: "Close", description: "Esc", value: "close" },
    ],
    backgroundColor: "transparent",
    textColor: C.text,
    focusedBackgroundColor: C.panelAlt,
    selectedBackgroundColor: C.borderActive,
    selectedTextColor: "#0a0e14",
    descriptionColor: C.dim,
  } as any)
  overlay.add(actions)

  actions.on(SelectRenderableEvents.ITEM_SELECTED, (_i: number, opt: any) => {
    const action = opt.value
    if (action === "close") return closePopup()
    if (action === "add") {
      closePopup()
      return openAddProviderForm()
    }
    const selectedId = list.getSelectedOption().value
    if (!selectedId) {
      sysLine("No provider selected.", C.error)
      return closePopup()
    }
    const provider = cfg.providers.find((p) => p.id === selectedId)!
    if (action === "setactive") {
      cfg.currentProviderId = provider.id
      if (provider.models.length && !provider.models.find((m) => m.id === cfg.currentModelId)) {
        cfg.currentModelId = provider.models[0].id
      }
      saveConfig(cfg)
      refreshChrome()
      sysLine(`Active provider: ${provider.name}.`, C.neon3)
      closePopup()
    } else if (action === "delete") {
      cfg.providers = cfg.providers.filter((p) => p.id !== provider.id)
      if (cfg.currentProviderId === provider.id) {
        cfg.currentProviderId = cfg.providers[0]?.id ?? null
        cfg.currentModelId = cfg.providers[0]?.models[0]?.id ?? null
      }
      saveConfig(cfg)
      refreshChrome()
      sysLine(`Deleted provider ${provider.name}.`, C.tool)
      closePopup()
    } else if (action === "addmodel") {
      closePopup()
      openAddModelForm(provider)
    }
  })
  list.focus()
}

function openAddProviderForm() {
  const fields = [
    { key: "name", label: "Name", placeholder: "My OpenAI", value: "" },
    { key: "type", label: "Type", placeholder: "openai | anthropic", value: "openai" },
    { key: "baseURL", label: "Base URL", placeholder: "https://api.openai.com/v1", value: "https://api.openai.com/v1" },
    { key: "apiKey", label: "API Key", placeholder: "sk-...", value: "" },
  ]
  openForm("Add provider", fields, (vals) => {
    const type = (vals.type || "openai").trim() as ProviderType
    if (type !== "openai" && type !== "anthropic") {
      sysLine("Type must be 'openai' or 'anthropic'.", C.error)
      return
    }
    if (!vals.apiKey.trim()) {
      sysLine("API key is required.", C.error)
      return
    }
    const provider: Provider = {
      id: genId("prov"),
      name: vals.name.trim() || "provider",
      type,
      baseURL: vals.baseURL.trim(),
      apiKey: vals.apiKey.trim(),
      models: [],
    }
    cfg.providers.push(provider)
    cfg.currentProviderId = provider.id
    saveConfig(cfg)
    refreshChrome()
    sysLine(`Added provider ${provider.name}. Now add a model with /providers → Add model.`, C.neon3)
    openAddModelForm(provider)
  })
}

function openAddModelForm(provider: Provider) {
  const fields = [
    { key: "id", label: "Model ID", placeholder: "gpt-4o", value: "" },
    { key: "name", label: "Display name", placeholder: "GPT-4o", value: "" },
  ]
  openForm(`Add model to ${provider.name}`, fields, (vals) => {
    if (!vals.id.trim()) {
      sysLine("Model ID is required.", C.error)
      return
    }
    const model: ModelDef = {
      id: vals.id.trim(),
      name: vals.name.trim() || vals.id.trim(),
    }
    provider.models.push(model)
    cfg.currentProviderId = provider.id
    cfg.currentModelId = model.id
    saveConfig(cfg)
    refreshChrome()
    sysLine(`Added model ${model.name}. Ready to chat.`, C.neon3)
  })
}

// ---- generic form popup ----
function openForm(
  title: string,
  fields: { key: string; label: string; placeholder: string; value: string }[],
  onSubmit: (vals: Record<string, string>) => void,
) {
  if (popup) closePopup()
  const height = 4 + fields.length * 2
  const overlay = openOverlay(64, height, title)
  const inputs: InputRenderable[] = []
  let idx = 0
  for (const f of fields) {
    const row = new BoxRenderable(r, { width: "100%", flexDirection: "row", height: 1 } as any)
    const lab = new TextRenderable(r, {
      content: f.label.padEnd(12),
      fg: C.accent,
    } as any)
    row.add(lab)
    const inp = new InputRenderable(r, {
      width: "100%",
      placeholder: f.placeholder,
      value: f.value,
      backgroundColor: C.panelAlt,
      focusedBackgroundColor: C.bg,
      textColor: C.text,
      cursorColor: C.neon2,
    } as any)
    row.add(inp)
    overlay.add(row)
    inputs.push(inp)
    inp.on(InputRenderableEvents.ENTER, () => {
      // Enter on last field submits; otherwise tab down
      if (idx === inputs.length - 1) {
        const vals: Record<string, string> = {}
        fields.forEach((fd, i) => (vals[fd.key] = inputs[i].value))
        closePopup()
        onSubmit(vals)
      } else {
        inputs[idx + 1].focus()
      }
    })
    idx++
  }
  const hint = new TextRenderable(r, {
    content: t`${dim("Enter to submit / next field · Tab to move · Esc to cancel")}`,
    fg: C.dim,
  } as any)
  overlay.add(hint)
  inputs[0].focus()
  // store current form inputs for tab handling
  ;(overlay as any)._formInputs = inputs
  ;(overlay as any)._formFields = fields
  ;(overlay as any)._formOnSubmit = onSubmit
}

// ---- mcp popup ----
function openMcpPopup() {
  if (popup) closePopup()
  const overlay = openOverlay(60, 18, "MCP servers")
  const list = new SelectRenderable(r, {
    width: "100%",
    height: 7,
    options:
      cfg.mcpServers.length > 0
        ? cfg.mcpServers.map((s) => ({
            name: `${s.enabled ? "●" : "○"} ${s.name}  [${mcp.isConnected(s.id) ? "connected" : "off"}]`,
            description: `${s.command} ${s.args.join(" ")}`,
            value: s.id,
          }))
        : [{ name: "(no MCP servers)", description: "add one below", value: null }],
    backgroundColor: "transparent",
    textColor: C.text,
    focusedBackgroundColor: C.panelAlt,
    selectedBackgroundColor: C.borderActive,
    selectedTextColor: "#0a0e14",
    descriptionColor: C.dim,
  } as any)
  overlay.add(list)
  const actions = new SelectRenderable(r, {
    width: "100%",
    height: 6,
    options: [
      { name: "Connect selected", description: "spawn + initialize", value: "connect" },
      { name: "Disconnect selected", description: "stop server", value: "disconnect" },
      { name: "Add server…", description: "command + args", value: "add" },
      { name: "Delete selected", description: "remove from config", value: "delete" },
      { name: "Close", description: "Esc", value: "close" },
    ],
    backgroundColor: "transparent",
    textColor: C.text,
    focusedBackgroundColor: C.panelAlt,
    selectedBackgroundColor: C.borderActive,
    selectedTextColor: "#0a0e14",
    descriptionColor: C.dim,
  } as any)
  overlay.add(actions)
  actions.on(SelectRenderableEvents.ITEM_SELECTED, (_i: number, opt: any) => {
    const action = opt.value
    if (action === "close") return closePopup()
    if (action === "add") {
      closePopup()
      return openAddMcpForm()
    }
    const sid = list.getSelectedOption().value
    if (!sid) {
      sysLine("No server selected.", C.error)
      return closePopup()
    }
    const sdef = cfg.mcpServers.find((s) => s.id === sid)!
    if (action === "connect") {
      sysLine(`Connecting to MCP server ${sdef.name}…`, C.dim)
      mcp.connect(sdef).then((res) => {
        if (res.ok) sysLine(`MCP ${sdef.name} connected (${res.toolCount} tools).`, C.neon3)
        else sysLine(`MCP ${sdef.name} failed: ${res.error}`, C.error)
        refreshChrome()
      })
      closePopup()
    } else if (action === "disconnect") {
      mcp.disconnect(sdef.id)
      sysLine(`Disconnected MCP ${sdef.name}.`, C.tool)
      refreshChrome()
      closePopup()
    } else if (action === "delete") {
      mcp.disconnect(sdef.id)
      cfg.mcpServers = cfg.mcpServers.filter((s) => s.id !== sdef.id)
      saveConfig(cfg)
      refreshChrome()
      sysLine(`Deleted MCP server ${sdef.name}.`, C.tool)
      closePopup()
    }
  })
  list.focus()
}

function openAddMcpForm() {
  const fields = [
    { key: "name", label: "Name", placeholder: "filesystem", value: "" },
    { key: "command", label: "Command", placeholder: "npx", value: "npx" },
    { key: "args", label: "Args", placeholder: "-y @modelcontextprotocol/server-filesystem /", value: "" },
  ]
  openForm("Add MCP server", fields, (vals) => {
    if (!vals.command.trim()) {
      sysLine("Command is required.", C.error)
      return
    }
    const sdef: McpServerDef = {
      id: genId("mcp"),
      name: vals.name.trim() || "mcp",
      command: vals.command.trim(),
      args: vals.args.trim() ? vals.args.trim().split(/\s+/) : [],
      enabled: true,
    }
    cfg.mcpServers.push(sdef)
    saveConfig(cfg)
    sysLine(`Added MCP server ${sdef.name}. Connecting…`, C.dim)
    mcp.connect(sdef).then((res) => {
      if (res.ok) sysLine(`MCP ${sdef.name} connected (${res.toolCount} tools${res.resourceCount ? `, ${res.resourceCount} resources` : ""}${res.promptCount ? `, ${res.promptCount} prompts` : ""}).`, C.neon3)
      else sysLine(`MCP ${sdef.name} failed: ${res.error}`, C.error)
      refreshChrome()
    })
  })
}

// ---- skills popup ----
function openSkillsPopup() {
  if (popup) closePopup()
  skills.reload()
  const skillList = skills.list()
  const overlay = openOverlay(60, 18, "Skills")
  const list = new SelectRenderable(r, {
    width: "100%",
    height: 7,
    options:
      skillList.length > 0
        ? skillList.map((s) => ({
            name: `${s.name}  [${s.source}]`,
            description: s.description || "(no description)",
            value: s.name,
          }))
        : [{ name: "(no skills yet)", description: "create one below or ask the AI to create one", value: null }],
    backgroundColor: "transparent",
    textColor: C.text,
    focusedBackgroundColor: C.panelAlt,
    selectedBackgroundColor: C.borderActive,
    selectedTextColor: "#0a0e14",
    descriptionColor: C.dim,
  } as any)
  overlay.add(list)
  const actions = new SelectRenderable(r, {
    width: "100%",
    height: 6,
    options: [
      { name: "Invoke selected", description: "activate skill instructions", value: "invoke" },
      { name: "Reload skills", description: "rescan skill directories", value: "reload" },
      { name: "Delete selected", description: "remove skill from disk", value: "delete" },
      { name: "Close", description: "Esc", value: "close" },
    ],
    backgroundColor: "transparent",
    textColor: C.text,
    focusedBackgroundColor: C.panelAlt,
    selectedBackgroundColor: C.borderActive,
    selectedTextColor: "#0a0e14",
    descriptionColor: C.dim,
  } as any)
  overlay.add(actions)
  actions.on(SelectRenderableEvents.ITEM_SELECTED, (_i: number, opt: any) => {
    const action = opt.value
    if (action === "close") return closePopup()
    if (action === "reload") {
      skills.reload()
      messages[0] = { role: "system", content: buildSystemPrompt() }
      refreshChrome()
      sysLine(`Reloaded ${skills.list().length} skills.`, C.neon3)
      closePopup()
      openSkillsPopup()
      return
    }
    const skillName = list.getSelectedOption().value
    if (!skillName) {
      sysLine("No skill selected.", C.error)
      return closePopup()
    }
    if (action === "invoke") {
      const content = skills.getSkillContent(skillName)
      if (content) {
        messages.push({ role: "user", content: `[Skill: ${skillName}]\n${content}` })
        addMessageBlock("you", C.user, `[Skill: ${skillName}]`)
        sysLine(`Skill "${skillName}" invoked. The AI will now follow its instructions.`, C.neon3)
      } else {
        sysLine(`Skill not found: ${skillName}`, C.error)
      }
      closePopup()
    } else if (action === "delete") {
      if (skills.delete(skillName)) {
        messages[0] = { role: "system", content: buildSystemPrompt() }
        refreshChrome()
        sysLine(`Deleted skill "${skillName}".`, C.tool)
      } else {
        sysLine(`Could not delete skill "${skillName}".`, C.error)
      }
      closePopup()
    }
  })
  list.focus()
}

// ---- skill invocation (slash command) ----
function invokeSkillByName(name: string) {
  const content = skills.getSkillContent(name)
  if (!content) {
    sysLine(`Skill not found: ${name}. Use /skills to list available skills.`, C.error)
    return
  }
  messages.push({ role: "user", content: `[Skill: ${name}]\n${content}` })
  addMessageBlock("you", C.user, `[Skill: ${name}]`)
  sysLine(`Skill "${name}" invoked. The AI will now follow its instructions.`, C.neon3)
}

// ---- sessions popup ----
function openSessionsPopup() {
  if (popup) closePopup()
  const allSessions = sessions.list(process.cwd())
  if (allSessions.length === 0) {
    sysLine("No sessions found for this directory.", C.dim)
    return
  }
  const overlay = openOverlay(70, Math.min(24, 6 + allSessions.length), "Sessions — resume")
  const options = allSessions.slice(0, 18).map((s) => ({
    name: `${new Date(s.lastActivity).toLocaleString()}  ${s.entryCount} msgs`,
    description: `${s.providerId}/${s.modelId}`,
    value: s.id,
  }))
  const sel = new SelectRenderable(r, {
    width: "100%",
    height: overlay.height - 4,
    options,
    backgroundColor: "transparent",
    textColor: C.text,
    focusedBackgroundColor: C.panelAlt,
    focusedTextColor: C.text,
    selectedBackgroundColor: C.borderActive,
    selectedTextColor: "#0a0e14",
    descriptionColor: C.dim,
  } as any)
  overlay.add(sel)
  sel.on(SelectRenderableEvents.ITEM_SELECTED, (_i: number, opt: any) => {
    const sess = sessions.open(opt.value)
    if (!sess) {
      sysLine("Failed to open session.", C.error)
      closePopup()
      return
    }
    // Rebuild messages from session entries
    const entries = sess.getEntries()
    const newMsgs: ChatMessage[] = [{ role: "system", content: buildSystemPrompt() }]
    for (const e of entries) {
      if (e.type === "user_message") newMsgs.push({ role: "user", content: e.content })
      else if (e.type === "assistant_message") newMsgs.push({ role: "assistant", content: e.content, toolCalls: (e as any).toolCalls })
      else if (e.type === "tool_result") newMsgs.push({ role: "tool", content: e.content, toolName: e.toolName, toolCallId: "" })
    }
    messages = newMsgs
    currentSession = sess
    // Re-render chat
    chatBox.getChildren().forEach((c: any) => { try { chatBox.remove(c) } catch {} })
    sysLine(`Resumed session ${sess.id.slice(0, 8)} (${entries.length} entries).`, C.neon3)
    for (let i = 1; i < newMsgs.length; i++) {
      const m = newMsgs[i]
      if (m.role === "user") addMessageBlock("you", C.user, m.content)
      else if (m.role === "assistant") addMessageBlock("rsi", C.assistant, m.content)
      else if (m.role === "tool") sysLine(`  [tool: ${m.toolName}] ${truncate(m.content, 200)}`, C.dim)
    }
    refreshChrome()
    closePopup()
  })
  sel.focus()
}

// ---- help popup ----
function openHelpPopup() {
  if (popup) closePopup()
  const overlay = openOverlay(64, 28, "Help — RSI commands")
  const lines = [
    ["/help", "show this help"],
    ["/model", "pick a model (popup)"],
    ["/providers", "connect & manage providers (popup)"],
    ["/mcp", "manage MCP servers (popup)"],
    ["/skills", "manage skills (popup)"],
    ["/skill <name>", "invoke a skill by name"],
    ["/effort <low|medium|high>", "set reasoning effort"],
    ["/tools", "toggle built-in tools on/off"],
    ["/goal <text>", "set a goal / extra system instruction"],
    ["/review", "toggle review mode (code review focus)"],
    ["/plan", "toggle plan mode (explore first, no writes)"],
    ["/sessions", "list & resume past sessions"],
    ["/clear", "clear the conversation"],
    ["/config", "show current configuration"],
    ["/exit  or  /quit", "exit RSI"],
    ["", ""],
    ["Skills", "loaded from .rsi/skills/ and ~/.config/rsi/skills/"],
    ["Custom commands", ".rsi/commands/*.md and ~/.config/rsi/commands/*.md"],
    ["Context files", "AGENTS.md / CLAUDE.md auto-discovered"],
    ["Memories", "persisted in ~/.local/share/rsi/memories/"],
    ["AI can create skills", "via the create_skill tool"],
    ["AI can save memories", "via the memory_save tool"],
    ["Subagents", "spawned via the spawn_subagent tool"],
    ["Permissions", "allow/ask/deny rules in config.json"],
    ["Context compaction", "auto-compacts long conversations"],
    ["", ""],
    ["/", "type / for command autocomplete"],
    ["Enter", "send the input"],
    ["Shift+Enter", "insert a newline"],
    ["↑ / ↓", "navigate slash dropdown"],
    ["Tab", "complete highlighted command"],
    ["Esc", "abort streaming or close popups"],
  ]
  for (const [cmd, desc] of lines) {
    const line = new TextRenderable(r, {
      content: cmd ? t`${fg(C.neon2)(cmd.padEnd(28))} ${fg(C.dim)(desc)}` : " ",
      fg: C.text,
    } as any)
    overlay.add(line)
  }
  overlay.add(
    new TextRenderable(r, {
      content: t`${dim("Press Esc to close")}`,
      fg: C.dim,
    } as any),
  )
}

// ============================================================
//  Slash command dropdown (autocomplete above input)
// ============================================================
const SLASH_COMMANDS: { name: string; desc: string }[] = [
  { name: "help", desc: "show this help" },
  { name: "model", desc: "pick a model" },
  { name: "models", desc: "pick a model" },
  { name: "providers", desc: "manage providers" },
  { name: "provider", desc: "manage providers" },
  { name: "mcp", desc: "manage MCP servers" },
  { name: "skills", desc: "manage skills" },
  { name: "skill", desc: "invoke a skill" },
  { name: "effort", desc: "set reasoning effort" },
  { name: "tools", desc: "toggle built-in tools" },
  { name: "goal", desc: "set a goal / instruction" },
  { name: "review", desc: "toggle review mode" },
  { name: "plan", desc: "toggle plan mode (explore first)" },
  { name: "sessions", desc: "list & resume past sessions" },
  { name: "clear", desc: "clear conversation" },
  { name: "config", desc: "show configuration" },
  { name: "exit", desc: "exit RSI" },
  { name: "quit", desc: "exit RSI" },
  // custom commands from .rsi/commands/*.md and ~/.config/rsi/commands/*.md
  ...customCommands.map((c) => ({ name: c.name, desc: c.description })),
]

let slashDropdown: BoxRenderable | null = null
let slashSelectedIndex = 0
let slashFiltered: { name: string; desc: string }[] = []

function closeSlashDropdown() {
  if (slashDropdown) {
    try {
      r.root.remove(slashDropdown)
    } catch {}
    slashDropdown = null
    slashSelectedIndex = 0
    slashFiltered = []
  }
}

function isSlashDropdownOpen(): boolean {
  return slashDropdown !== null
}

function updateSlashDropdown() {
  const text = (input as any).plainText as string
  if (!text.startsWith("/") || text.includes(" ") || text.length <= 1) {
    closeSlashDropdown()
    return
  }
  const query = text.slice(1).toLowerCase()
  // deduplicate by name while filtering
  const seen = new Set<string>()
  slashFiltered = SLASH_COMMANDS.filter((c) => {
    if (!c.name.startsWith(query)) return false
    if (seen.has(c.name)) return false
    seen.add(c.name)
    return true
  })
  if (slashFiltered.length === 0) {
    closeSlashDropdown()
    return
  }
  if (slashSelectedIndex >= slashFiltered.length) slashSelectedIndex = 0
  renderSlashDropdown()
}

function renderSlashDropdown() {
  if (slashDropdown) {
    try {
      r.root.remove(slashDropdown)
    } catch {}
    slashDropdown = null
  }
  if (slashFiltered.length === 0) return

  const ddWidth = 58
  const visibleCount = Math.min(slashFiltered.length, 12)
  const ddHeight = visibleCount + 2 // border
  const dd = new BoxRenderable(r, {
    position: "absolute",
    bottom: 6,
    left: "50%",
    marginLeft: -Math.floor(ddWidth / 2),
    width: ddWidth,
    height: ddHeight,
    borderStyle: "rounded",
    borderColor: C.borderActive,
    backgroundColor: C.panel,
    flexDirection: "column",
    alignItems: "center",
    padding: 0,
  } as any)
  dd.zIndex = 200

  for (let i = 0; i < slashFiltered.length && i < 12; i++) {
    const cmd = slashFiltered[i]
    const isSelected = i === slashSelectedIndex
    // center the command name in a fixed-width field, description follows
    const cmdStr = "/" + cmd.name
    const nameField = cmdStr.padEnd(16)
    const row = new BoxRenderable(r, {
      width: "100%",
      height: 1,
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "center",
      backgroundColor: isSelected ? C.panelAlt : "transparent",
    } as any)
    const txt = new TextRenderable(r, {
      content: t`${isSelected ? fg(C.gold3)("▸") : " "} ${fg(isSelected ? C.gold3 : C.gold1)(nameField)}${fg(C.dim)(cmd.desc)}`,
      fg: isSelected ? C.gold3 : C.text,
      backgroundColor: isSelected ? C.panelAlt : "transparent",
    } as any)
    row.add(txt)
    dd.add(row)
  }

  r.root.add(dd)
  slashDropdown = dd
}

function slashSelectUp() {
  if (slashFiltered.length === 0) return
  slashSelectedIndex = (slashSelectedIndex - 1 + slashFiltered.length) % slashFiltered.length
  renderSlashDropdown()
}

function slashSelectDown() {
  if (slashFiltered.length === 0) return
  slashSelectedIndex = (slashSelectedIndex + 1) % slashFiltered.length
  renderSlashDropdown()
}

function slashExecuteSelected() {
  if (slashFiltered.length === 0 || slashSelectedIndex >= slashFiltered.length) return
  const cmd = slashFiltered[slashSelectedIndex]
  closeSlashDropdown()
  ;(input as any).clear()
  handleCommand("/" + cmd.name)
}

function slashCompleteSelected() {
  if (slashFiltered.length === 0 || slashSelectedIndex >= slashFiltered.length) return
  const cmd = slashFiltered[slashSelectedIndex]
  ;(input as any).setText("/" + cmd.name + " ")
  // keep dropdown closed after tab-completion since we added a space
  closeSlashDropdown()
  input.focus()
}

// monitor input content changes to show/hide/filter the dropdown
;(input as any).onContentChange = () => {
  updateSlashDropdown()
}

// ============================================================
//  Commands
// ============================================================
const COMMANDS: Record<string, (args: string) => void> = {
  help: () => openHelpPopup(),
  "?": () => openHelpPopup(),
  model: () => openModelPopup(),
  models: () => openModelPopup(),
  providers: () => openProvidersPopup(),
  provider: () => openProvidersPopup(),
  mcp: () => openMcpPopup(),
  skills: () => openSkillsPopup(),
  skill: (args) => {
    const name = args.trim()
    if (!name) {
      sysLine("Usage: /skill <name> — use /skills to list available skills.", C.error)
      return
    }
    invokeSkillByName(name)
  },
  clear: () => {
    messages = [{ role: "system", content: buildSystemPrompt() }]
    currentSession = null
    chatBox.getChildren().forEach((c: any) => {
      try {
        chatBox.remove(c)
      } catch {}
    })
    sysLine("Conversation cleared.", C.dim)
  },
  effort: (args) => {
    const v = args.trim().toLowerCase()
    if (v === "low" || v === "medium" || v === "high") {
      cfg.effort = v
      saveConfig(cfg)
      refreshChrome()
      sysLine(`Reasoning effort set to ${v}.`, C.neon3)
    } else {
      sysLine("Usage: /effort <low|medium|high>", C.error)
    }
  },
  tools: () => {
    cfg.toolsEnabled = !cfg.toolsEnabled
    saveConfig(cfg)
    refreshChrome()
    sysLine(`Tools ${cfg.toolsEnabled ? "enabled" : "disabled"}.`, C.tool)
  },
  goal: (args) => {
    const text = args.trim()
    if (!text) {
      sysLine("Usage: /goal <text>", C.error)
      return
    }
    cfg.systemPrompt = cfg.systemPrompt.replace(/\n\[GOAL\][\s\S]*$/, "")
    cfg.systemPrompt += `\n[GOAL] ${text}`
    messages[0] = { role: "system", content: buildSystemPrompt() }
    saveConfig(cfg)
    sysLine(`Goal set: ${text}`, C.neon3)
  },
  review: () => {
    reviewMode = !reviewMode
    messages[0] = { role: "system", content: buildSystemPrompt() }
    refreshChrome()
    sysLine(`Review mode ${reviewMode ? "on" : "off"}.`, C.tool)
  },
  plan: () => {
    planMode = !planMode
    messages[0] = { role: "system", content: buildSystemPrompt() }
    refreshChrome()
    sysLine(`Plan mode ${planMode ? "on" : "off"}. ${planMode ? "Explore first, then propose a plan. No file changes." : "Full agent mode restored."}`, C.tool)
  },
  sessions: () => openSessionsPopup(),
  config: () => {
    const p = getCurrentProvider(cfg)
    const m = getCurrentModel(cfg)
    const ctxFiles = listContextFiles()
    sysLine(
      t`provider: ${fg(C.text)(p ? p.name : "none")}  ·  model: ${fg(C.text)(m ? m.name : "none")}  ·  effort: ${fg(C.text)(cfg.effort)}  ·  tools: ${fg(C.text)(String(cfg.toolsEnabled))}`,
      C.accent,
    )
    sysLine(`providers: ${cfg.providers.length}  ·  mcp servers: ${cfg.mcpServers.length}  ·  skills: ${skills.list().length}  ·  custom commands: ${customCommands.length}  ·  ctx: ${estimateTokens(messages)} tokens  ·  config: ${configPath()}`, C.dim)
    if (ctxFiles.length > 0) sysLine(`context files: ${ctxFiles.join(", ")}`, C.dim)
  },
  exit: () => shutdown(),
  quit: () => shutdown(),
}

// Custom commands: /<name> sends the expanded template as a user message
for (const cmd of customCommands) {
  if (!COMMANDS[cmd.name]) {
    COMMANDS[cmd.name] = (args) => {
      const template = expandTemplate(cmd.template)
      const fullPrompt = args.trim() ? `${template}\n\nAdditional context: ${args}` : template
      ;(input as any).setText(fullPrompt)
      handleSubmit()
    }
  }
}

function reviewSuffix(): string {
  return "\n[MODE] You are in REVIEW mode. Focus on reviewing code the user provides: read files, run checks, and give concrete, prioritized feedback with file:line references. Use tools to inspect."
}

function shutdown() {
  mcp.disconnectAll()
  try {
    r.destroy()
  } catch {}
  process.exit(0)
}

function handleCommand(raw: string) {
  const trimmed = raw.trim()
  const [name, ...rest] = trimmed.slice(1).split(/\s+/)
  const args = trimmed.slice(1 + name.length).trim()
  const cmd = COMMANDS[name.toLowerCase()]
  if (cmd) cmd(args)
  else sysLine(`Unknown command: /${name}. Try /help.`, C.error)
}

// ============================================================
//  Agent loop
// ============================================================

// Tool names that are handled specially (need external state like SkillManager)
const SKILL_TOOL_NAMES = new Set(["list_skills", "invoke_skill", "create_skill"])
const SUBAGENT_TOOL_NAMES = new Set(["spawn_subagent"])

function gatherTools(): ToolDef[] {
  const tools: ToolDef[] = []
  if (cfg.toolsEnabled) {
    tools.push(...BUILTIN_TOOLS)
    tools.push(APPLY_PATCH_TOOL)
    tools.push(...SKILL_TOOLS)
    tools.push(...SUBAGENT_TOOLS)
    tools.push(...MEMORY_TOOLS)
    tools.push(...mcp.getToolDefs())
  }
  return tools
}

/** Execute a skill tool call using the SkillManager. */
function executeSkillTool(name: string, args: Record<string, unknown>): ToolResult {
  try {
    switch (name) {
      case "list_skills": {
        const list = skills.list()
        if (list.length === 0) return { name, content: "No skills available. You can create one with the create_skill tool." }
        const lines = list.map((s) => `- ${s.name}: ${s.description || "(no description)"} [${s.source}]`)
        return { name, content: `Available skills (${list.length}):\n${lines.join("\n")}` }
      }
      case "invoke_skill": {
        const skillName = String(args.name ?? "")
        const content = skills.getSkillContent(skillName)
        if (!content) return { name, isError: true, content: `Skill not found: ${skillName}. Use list_skills to see available skills.` }
        return { name, content: `Skill "${skillName}" invoked. Instructions:\n\n${content}` }
      }
      case "create_skill": {
        const skillName = String(args.name ?? "")
        const description = String(args.description ?? "")
        const content = String(args.content ?? "")
        const slash = args.slash !== false
        if (!skillName || !content) return { name, isError: true, content: "name and content are required." }
        const skill = skills.create(skillName, description, content, slash)
        // Update system prompt with new skill context
        messages[0] = { role: "system", content: buildSystemPrompt() }
        refreshChrome()
        return { name, content: `Skill "${skill.name}" created at ${skill.path}. It is now available via invoke_skill and /skill ${skill.name}.` }
      }
      default:
        return { name, isError: true, content: `Unknown skill tool: ${name}` }
    }
  } catch (e: any) {
    return { name, isError: true, content: String(e?.message ?? e) }
  }
}

/** Execute a subagent tool call. */
async function executeSubagentTool(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<ToolResult> {
  if (name !== "spawn_subagent") return { name, isError: true, content: `Unknown subagent tool: ${name}` }
  const task = String(args.task ?? "")
  const systemPrompt = String(args.system_prompt ?? "") || SUBAGENT_SYSTEM_PROMPT
  if (!task) return { name, isError: true, content: "task is required." }

  const provider = getCurrentProvider(cfg)
  const model = getCurrentModel(cfg)
  if (!provider || !model) return { name, isError: true, content: "No provider/model configured for subagent." }

  const client = createProviderClient(provider)
  const subagentTools = [...BUILTIN_TOOLS] // subagents get built-in tools only

  sysLine(`  spawning subagent for: ${truncate(task, 80)}`, C.dim)

  const result = await runSubagent({
    systemPrompt,
    task,
    client,
    model,
    tools: subagentTools,
    effort: cfg.effort,
    maxSteps: 15,
    onToolCall: (toolName, toolArgs) => {
      const line = new TextRenderable(r, {
        content: t`  ${fg(C.tool)("▸ subagent tool")} ${fg(C.neon4)(toolName)} ${fg(C.dim)(formatArgs(toolArgs))}`,
        fg: C.text,
        wrapMode: "word",
      } as any)
      chatBox.add(line)
      chatBox.scrollTo(Infinity)
    },
    onToolResult: (toolName, resultStr) => {
      const line = new TextRenderable(r, {
        content: t`  ${fg(C.toolResult)("◂ subagent result")} ${fg(C.dim)(truncate(resultStr, 600))}`,
        fg: C.toolResult,
        wrapMode: "word",
      } as any)
      chatBox.add(line)
      chatBox.scrollTo(Infinity)
    },
    signal,
  })

  const status = result.completed ? "completed" : `stopped after ${result.steps} steps`
  return { name, content: `Subagent ${status} (${result.steps} steps).\n\nResult:\n${result.text}` }
}

/** Check and perform context compaction if needed. */
async function maybeCompact(client: ProviderClient, model: ModelDef, signal?: AbortSignal): Promise<boolean> {
  if (!needsCompaction(messages)) return false
  sysLine("Context limit approaching — compacting conversation…", C.dim)

  // Build the compaction prompt from older messages (exclude system + recent)
  const nonSystem = messages.filter((m) => m.role !== "system")
  const toSummarize = nonSystem.slice(0, -DEFAULT_COMPACTION.keepRecent)
  if (toSummarize.length < 4) return false // not enough to compact

  const prompt = buildCompactionPrompt(toSummarize)
  let summary = ""

  // Use a non-streaming call to get the summary
  const compactMessages_list: ChatMessage[] = [
    { role: "system", content: "You are a conversation summarizer. Provide a concise but complete summary." },
    { role: "user", content: prompt },
  ]

  try {
    const result = await client.stream(
      compactMessages_list,
      model,
      [],
      "low",
      { onText: (delta) => { summary += delta } },
      signal,
    )
    summary = result.text
  } catch (e: any) {
    sysLine(`Compaction failed: ${String(e?.message ?? e)}`, C.error)
    return false
  }

  if (!summary.trim()) return false

  messages = compactMessages(messages, summary, DEFAULT_COMPACTION)
  sysLine(`Context compacted: ${estimateTokens(messages)} tokens remaining.`, C.neon3)
  refreshChrome()
  return true
}

// ============================================================
//  Thinking spinner — animated shiny logo with rotating text
// ============================================================
const SPINNER_WORDS = [
  "thinking", "brewing", "cooking", "processing", "pondering",
  "vibing", "computing", "dreaming", "forging", "weaving",
  "distilling", "cracking", "spinning", "marinating", "percolating",
]
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
// Shiny golden gradient cycle for the spinner
const SPINNER_COLORS = [C.gold2, C.gold3, C.gold4, C.gold5, C.gold4, C.gold3]

let thinkingSpinner: { stop: () => void } | null = null

function startThinkingSpinner() {
  if (thinkingSpinner) thinkingSpinner.stop()

  // Create the spinner line in the chat — no "rsi" label, just the shiny spinner
  const spinnerBox = new BoxRenderable(r, {
    width: "100%",
    height: 1,
    flexDirection: "row",
    alignItems: "center",
    paddingLeft: 1,
  } as any)
  const spinnerText = new TextRenderable(r, {
    content: t`${fg(C.gold3)("⠋")} ${fg(C.gold2)("thinking")} ${fg(C.gold4)("·")}`,
    fg: C.gold3,
  } as any)
  spinnerBox.add(spinnerText)
  chatBox.add(spinnerBox)
  chatBox.scrollTo(Infinity)

  let frameIdx = 0
  let wordIdx = 0
  let colorIdx = 0
  let tick = 0
  let stopped = false

  const interval = setInterval(() => {
    if (stopped) return
    tick++

    // Advance spinner frame every tick (80ms)
    frameIdx = (frameIdx + 1) % SPINNER_FRAMES.length

    // Change word every ~2.5s (every 30 ticks at 80ms)
    if (tick % 30 === 0) {
      wordIdx = (wordIdx + 1) % SPINNER_WORDS.length
    }

    // Cycle golden gradient color every 2 ticks — shiny shimmer effect
    colorIdx = (colorIdx + 1) % SPINNER_COLORS.length

    // Offset colors for each part so they shimmer in wave
    const frameColor = SPINNER_COLORS[colorIdx]
    const wordColor = SPINNER_COLORS[(colorIdx + 1) % SPINNER_COLORS.length]
    const dotsColor = SPINNER_COLORS[(colorIdx + 2) % SPINNER_COLORS.length]

    const frame = SPINNER_FRAMES[frameIdx]
    const word = SPINNER_WORDS[wordIdx]
    const dots = "·".repeat((tick % 3) + 1)

    // Everything shiny — braille, word, and dots all in animated golden gradient
    spinnerText.content = t`${fg(frameColor)(frame)} ${fg(wordColor)(word)} ${fg(dotsColor)(dots)}`
  }, 80)

  thinkingSpinner = {
    stop: () => {
      stopped = true
      clearInterval(interval)
      try {
        chatBox.remove(spinnerBox)
      } catch {}
      thinkingSpinner = null
    },
  }
}

function stopThinkingSpinner() {
  if (thinkingSpinner) thinkingSpinner.stop()
}

async function runAgent(userText: string) {
  const provider = getCurrentProvider(cfg)
  const model = getCurrentModel(cfg)
  if (!provider || !model) {
    sysLine("No provider/model configured. Use /providers then /model.", C.error)
    return
  }
  if (!provider.apiKey) {
    sysLine(`Provider ${provider.name} has no API key. Edit ${configPath()}.`, C.error)
    return
  }

  // Create a session on first message
  if (!currentSession) {
    currentSession = sessions.create(process.cwd(), provider.id, model.id)
  }

  messages.push({ role: "user", content: userText })
  addMessageBlock("you", C.user, userText)
  currentSession.append({ type: "user_message", content: userText } as any)

  const client: ProviderClient = createProviderClient(provider)
  const tools = gatherTools()
  busy = true
  refreshChrome()

  try {
    for (let step = 0; step < 25; step++) {
      abortController = new AbortController()

      // Check for context compaction before each step
      if (step > 0) {
        await maybeCompact(client, model, abortController.signal)
      }

      let assistantText = ""
      let bodyTxt: TextRenderable | null = null
      let firstDelta = true

      // Start the thinking spinner while waiting for first token
      startThinkingSpinner()

      const result = await client.stream(
        messages,
        model,
        tools,
        cfg.effort,
        {
          onText: (delta) => {
            if (firstDelta) {
              firstDelta = false
              stopThinkingSpinner()
              // Only create the "rsi" message block when text actually arrives
              bodyTxt = addMessageBlock("rsi", C.assistant, "")
            }
            assistantText += delta
            if (bodyTxt) bodyTxt.content = assistantText
            chatBox.scrollTo(Infinity)
          },
        },
        abortController.signal,
      )

      // Stop spinner in case no text was received (tool-only response)
      stopThinkingSpinner()
      // If we got text but no block was created (edge case), create it now
      if (!bodyTxt && result.text) {
        bodyTxt = addMessageBlock("rsi", C.assistant, result.text)
      }

      // record assistant message
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: result.text,
        toolCalls: result.toolCalls.length ? result.toolCalls : undefined,
      }
      messages.push(assistantMsg)
      currentSession?.append({
        type: "assistant_message",
        content: result.text,
        toolCalls: result.toolCalls.length ? result.toolCalls : undefined,
      } as any)

      if (!result.toolCalls.length) {
        // done
        if (!result.text.trim()) {
          if (bodyTxt) bodyTxt.content = t`${dim(italic("(no text response)"))}`
        }
        break
      }

      // execute tool calls
      for (const call of result.toolCalls) {
        const isMcp = call.name.startsWith("mcp__")
        const isSkill = SKILL_TOOL_NAMES.has(call.name)
        const isSubagent = SUBAGENT_TOOL_NAMES.has(call.name)
        const isMemory = MEMORY_TOOL_NAMES.has(call.name)
        const isApplyPatch = call.name === "apply_patch"
        const callLine = new TextRenderable(r, {
          content: t`${fg(C.tool)("▸ tool")} ${fg(C.neon4)(call.name)} ${fg(C.dim)(formatArgs(call.args))}`,
          fg: C.text,
          wrapMode: "word",
        } as any)
        chatBox.add(callLine)
        chatBox.scrollTo(Infinity)

        // Permission check for builtin tools (not skill/subagent/memory/mcp/apply_patch)
        if (!isMcp && !isSkill && !isSubagent && !isMemory && !isApplyPatch) {
          const perm = checkPermission(call.name, call.args, permConfig)
          if (perm === "deny") {
            const resultStr = `ERROR: Permission denied for ${call.name}. This operation is blocked by your permission config.`
            const resLine = new TextRenderable(r, {
              content: t`${fg(C.error)("◂ denied")} ${fg(C.dim)(resultStr)}`,
              fg: C.error,
              wrapMode: "word",
            } as any)
            chatBox.add(resLine)
            chatBox.scrollTo(Infinity)
            messages.push({ role: "tool", content: resultStr, toolCallId: call.id, toolName: call.name })
            continue
          }
          if (perm === "ask") {
            // In a TUI we can't easily prompt mid-stream; log and proceed.
            // The user can configure permissions to allow or deny explicitly.
            sysLine(`  [permission: ask] ${call.name} — auto-allowing (configure to deny if needed)`, C.dim)
          }
        }

        // Plan mode: block write tools
        if (planMode && !isSkill && !isMemory && !isApplyPatch) {
          const writeTools = new Set(["write_file", "edit_file", "delete_file", "delete_directory", "move_file", "run_command", "create_directory"])
          if (writeTools.has(call.name)) {
            const resultStr = `ERROR: Plan mode is active. File changes are blocked. Use read-only tools to explore, then propose a plan.`
            const resLine = new TextRenderable(r, {
              content: t`${fg(C.error)("◂ blocked")} ${fg(C.dim)(resultStr)}`,
              fg: C.error,
              wrapMode: "word",
            } as any)
            chatBox.add(resLine)
            chatBox.scrollTo(Infinity)
            messages.push({ role: "tool", content: resultStr, toolCallId: call.id, toolName: call.name })
            continue
          }
        }

        let resultStr: string
        try {
          if (isMcp) {
            resultStr = await mcp.executeTool(call.name, call.args)
          } else if (isSkill) {
            const r2 = executeSkillTool(call.name, call.args)
            resultStr = r2.content
            if (r2.isError) resultStr = `ERROR: ${resultStr}`
          } else if (isSubagent) {
            const r2 = await executeSubagentTool(call.name, call.args, abortController.signal)
            resultStr = r2.content
            if (r2.isError) resultStr = `ERROR: ${resultStr}`
          } else if (isMemory) {
            const r2 = executeMemoryTool(call.name, call.args)
            resultStr = r2.content
            if (r2.isError) resultStr = `ERROR: ${resultStr}`
            // Refresh system prompt with updated memory context
            messages[0] = { role: "system", content: buildSystemPrompt() }
          } else if (isApplyPatch) {
            const r2 = await executeApplyPatch(call.args)
            resultStr = r2.content
            if (r2.isError) resultStr = `ERROR: ${resultStr}`
          } else {
            const r2: ToolResult = await executeBuiltinTool(call.name, call.args)
            resultStr = r2.content
            if (r2.isError) resultStr = `ERROR: ${resultStr}`
          }
        } catch (e: any) {
          resultStr = `ERROR: ${String(e?.message ?? e)}`
        }

        const resLine = new TextRenderable(r, {
          content: t`${fg(C.toolResult)("◂ result")} ${fg(C.dim)(truncate(resultStr, 1200))}`,
          fg: C.toolResult,
          wrapMode: "word",
        } as any)
        chatBox.add(resLine)
        chatBox.scrollTo(Infinity)

        messages.push({
          role: "tool",
          content: truncate(resultStr, 30000),
          toolCallId: call.id,
          toolName: call.name,
        })
        currentSession?.append({
          type: "tool_result",
          toolName: call.name,
          content: truncate(resultStr, 30000),
          isError: resultStr.startsWith("ERROR:"),
        } as any)
      }
      // loop again for the model to continue
    }
  } catch (e: any) {
    stopThinkingSpinner()
    if (abortController?.signal.aborted) {
      sysLine("(aborted)", C.dim)
    } else {
      sysLine(`Error: ${String(e?.message ?? e)}`, C.error)
    }
  } finally {
    stopThinkingSpinner()
    busy = false
    abortController = null
    refreshChrome()
    input.focus()
  }
}

function formatArgs(args: Record<string, unknown>): string {
  try {
    const s = JSON.stringify(args)
    return truncate(s, 160)
  } catch {
    return ""
  }
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s
  return s.slice(0, max) + "…"
}

// ============================================================
//  Input handling
// ============================================================
function handleSubmit() {
  const text = (input as any).plainText as string
  if (!text.trim()) return
  if (busy) {
    sysLine("Already working — press Esc to abort.", C.dim)
    return
  }
  ;(input as any).clear()
  closeSlashDropdown()
  if (text.trim().startsWith("/")) {
    handleCommand(text.trim())
    return
  }
  // Check provider before attempting to send
  const provider = getCurrentProvider(cfg)
  if (!provider) {
    sysLine(t`${fg(C.error)("✗ No provider configured. Use ")}${fg(C.neon2)("/providers")}${fg(C.error)(" to add your API key and base URL, then ")}${fg(C.neon2)("/model")}${fg(C.error)(" to pick a model.")}`, C.error)
    return
  }
  const model = getCurrentModel(cfg)
  if (!model) {
    sysLine(t`${fg(C.error)("✗ No model selected. Use ")}${fg(C.neon2)("/model")}${fg(C.error)(" to pick one, or ")}${fg(C.neon2)("/providers")}${fg(C.error)(" to add a model to your provider.")}`, C.error)
    return
  }
  if (!provider.apiKey) {
    sysLine(t`${fg(C.error)("✗ Provider ")}${fg(C.text)(provider.name)}${fg(C.error)(" has no API key. Use ")}${fg(C.neon2)("/providers")}${fg(C.error)(" to set it, or edit ")}${fg(C.dim)(configPath())}`, C.error)
    return
  }
  runAgent(text)
}

// global key handling: Esc closes popups / aborts; Tab in forms; slash dropdown nav
r.keyInput.on("keypress", (key: any) => {
  // Slash dropdown key interception (runs before textarea processes the key)
  if (isSlashDropdownOpen()) {
    if (key.name === "up") {
      slashSelectUp()
      key.preventDefault()
      return
    }
    if (key.name === "down") {
      slashSelectDown()
      key.preventDefault()
      return
    }
    if (key.name === "return" || key.name === "kpenter") {
      if (!key.shift) {
        slashExecuteSelected()
        key.preventDefault()
        return
      }
    }
    if (key.name === "tab") {
      slashCompleteSelected()
      key.preventDefault()
      return
    }
    if (key.name === "escape") {
      closeSlashDropdown()
      key.preventDefault()
      return
    }
  }

  if (key.name === "escape") {
    if (popup) {
      closePopup()
      return
    }
    if (isSlashDropdownOpen()) {
      closeSlashDropdown()
      return
    }
    if (busy && abortController) {
      abortController.abort()
      sysLine("Aborting…", C.dim)
      return
    }
  }
  if (key.name === "tab" && popup && (popup as any)._formInputs) {
    const inputs: InputRenderable[] = (popup as any)._formInputs
    const cur = inputs.findIndex((i) => (i as any).focused)
    const next = (cur + 1) % inputs.length
    inputs[next].focus()
    key.preventDefault()
  }
  // Ctrl+C hard exit if not busy with popup
  if (key.ctrl && key.name === "c" && !popup) {
    shutdown()
  }
})

// auto-connect enabled MCP servers on startup
for (const s of cfg.mcpServers.filter((s) => s.enabled)) {
  mcp.connect(s).then((res) => {
    if (res.ok) sysLine(`MCP ${s.name} connected (${res.toolCount} tools${res.resourceCount ? `, ${res.resourceCount} resources` : ""}${res.promptCount ? `, ${res.promptCount} prompts` : ""}).`, C.dim)
    refreshChrome()
  })
}

console.log = () => {} // silence any stray logs into the renderer overlay
