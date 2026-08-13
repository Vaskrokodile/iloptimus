#!/usr/bin/env bun

import { createInterface } from "node:readline"

import { DEFAULT_SYSTEM_PROMPT, type ModelDef, type Provider } from "./config.ts"
import { AgentHarness, type HarnessEvent } from "./harness.ts"

const workspace = process.env.RSI_WORKSPACE || process.cwd()
const panelId = process.env.RSI_PANEL_ID || crypto.randomUUID()
const baseURL = process.env.RSI_PROVIDER_BASE_URL || "http://127.0.0.1:7860/v1"
const modelId = process.env.RSI_MODEL_ID || "deepseek-r1-distill-qwen-1.5b"
const model: ModelDef = { id: modelId, name: modelId }
const provider: Provider = {
  id: "iloptimus-local",
  name: "IL Optimus local model",
  type: "openai",
  baseURL,
  apiKey: "local",
  models: [model],
}

function output(payload: Record<string, unknown> | HarnessEvent): void {
  process.stdout.write(JSON.stringify(payload) + "\n")
}

const harness = new AgentHarness({
  panelId,
  workspace,
  provider,
  model,
  systemPrompt: process.env.RSI_SYSTEM_PROMPT || DEFAULT_SYSTEM_PROMPT,
  maxSteps: Number(process.env.RSI_MAX_STEPS || 20),
  onEvent: output,
})

output({ type: "ready", panelId, sessionId: harness.sessionId, workspace, model: modelId })

const input = createInterface({ input: process.stdin, crlfDelay: Infinity })
for await (const line of input) {
  if (!line.trim()) continue
  let command: Record<string, unknown>
  try {
    command = JSON.parse(line)
  } catch {
    output({ type: "protocol_error", error: "Input must be one JSON object per line" })
    continue
  }
  if (command.type === "shutdown") break
  if (command.type !== "prompt" || typeof command.prompt !== "string") {
    output({ type: "protocol_error", error: "Expected {type:'prompt', prompt:'...'}" })
    continue
  }
  try {
    await harness.run(command.prompt)
  } catch (error) {
    output({ type: "failed", panelId, error: String(error) })
  }
}

output({ type: "stopped", panelId, sessionId: harness.sessionId })
