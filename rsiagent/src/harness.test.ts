import { afterEach, describe, expect, test } from "bun:test"
import { mkdtemp, readFile, rm, stat } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"

import type { ModelDef, Provider } from "./config.ts"
import { AgentHarness, type HarnessEvent, type HarnessSession } from "./harness.ts"
import type { ChatMessage, ProviderClient, StreamCallbacks } from "./providers.ts"
import type { SessionEntry, SessionEntryPayload } from "./session.ts"
import type { ToolDef } from "./tools.ts"

const temporaryDirectories: string[] = []

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })))
})

class MemorySession implements HarnessSession {
  id = "test-session"
  entries: SessionEntry[] = []

  append(entry: SessionEntryPayload & { parentId?: string | null }): string {
    const id = `entry-${this.entries.length + 1}`
    this.entries.push({ ...entry, id, parentId: entry.parentId ?? null, timestamp: Date.now() } as SessionEntry)
    return id
  }

  getEntries(): SessionEntry[] {
    return this.entries
  }
}

class ScriptedProvider implements ProviderClient {
  private step = 0

  async stream(
    _messages: ChatMessage[],
    _model: ModelDef,
    _tools: ToolDef[],
    _effort: "low" | "medium" | "high",
    callbacks: StreamCallbacks,
  ) {
    const scripts = [
      {
        text: "", toolCalls: [{ id: "1", name: "write_file", args: { path: "demo/add.js", content: "console.log(2 + 3)\n" } }],
      },
      { text: "", toolCalls: [{ id: "2", name: "run_command", args: { command: "bun demo/add.js" } }] },
      { text: "Created the program and verified that it prints 5.", toolCalls: [] },
    ]
    const result = scripts[this.step++]
    if (result.text) callbacks.onText(result.text)
    callbacks.onDone?.(result)
    return result
  }
}

describe("headless AgentHarness", () => {
  test("creates files, runs commands, emits events, and persists its transcript", async () => {
    const workspace = await mkdtemp(join(tmpdir(), "rsi-harness-"))
    temporaryDirectories.push(workspace)
    const events: HarnessEvent[] = []
    const provider: Provider = {
      id: "test",
      name: "Test",
      type: "openai",
      baseURL: "http://unused",
      apiKey: "test",
      models: [{ id: "test-model", name: "Test model" }],
    }
    const session = new MemorySession()
    const harness = new AgentHarness({
      panelId: "panel-1",
      workspace,
      provider,
      model: provider.models[0],
      systemPrompt: "Use tools and verify your work.",
      client: new ScriptedProvider(),
      session,
      onEvent: (event) => events.push(event),
    })

    const result = await harness.run("Create a tiny program and test it")

    expect(result.completed).toBe(true)
    expect(result.steps).toBe(3)
    expect(await readFile(join(workspace, "demo/add.js"), "utf8")).toContain("2 + 3")
    expect((await stat(join(workspace, "demo"))).isDirectory()).toBe(true)
    expect(events.some((event) => event.type === "tool_result" && String(event.data.content).includes("exit code 0"))).toBe(true)
    expect(events.at(-1)?.type).toBe("completed")
    expect(session.entries.filter((entry) => entry.type === "tool_result")).toHaveLength(2)
  })

  test("blocks paths outside the admitted workspace", async () => {
    const workspace = await mkdtemp(join(tmpdir(), "rsi-harness-"))
    temporaryDirectories.push(workspace)
    const events: HarnessEvent[] = []
    const provider: Provider = {
      id: "test",
      name: "Test",
      type: "openai",
      baseURL: "http://unused",
      apiKey: "test",
      models: [{ id: "test-model", name: "Test model" }],
    }
    class EscapeProvider extends ScriptedProvider {
      override async stream() {
        return { text: "", toolCalls: [{ id: "escape", name: "write_file", args: { path: "../escape.txt", content: "no" } }] }
      }
    }
    const harness = new AgentHarness({
      panelId: "panel-2",
      workspace,
      provider,
      model: provider.models[0],
      systemPrompt: "test",
      client: new EscapeProvider(),
      session: new MemorySession(),
      maxSteps: 1,
      onEvent: (event) => events.push(event),
    })

    const result = await harness.run("escape")
    expect(result.completed).toBe(false)
    expect(events.some((event) => event.type === "tool_result" && event.data.isError === true)).toBe(true)
  })

  test("rejects an unverified claim and forces a small model back into tools", async () => {
    const workspace = await mkdtemp(join(tmpdir(), "rsi-harness-"))
    temporaryDirectories.push(workspace)
    const events: HarnessEvent[] = []
    class RecoveringProvider extends ScriptedProvider {
      private recoveryStep = 0
      override async stream() {
        const scripts = [
          { text: "I created and ran it.", toolCalls: [] },
          { text: "", toolCalls: [{ id: "write", name: "write_file", args: { path: "proof/a.js", content: "console.log(42)\n" } }] },
          { text: "", toolCalls: [{ id: "run", name: "run_command", args: { command: "bun a.js", cwd: "proof" } }] },
          { text: "Created and verified output 42.", toolCalls: [] },
        ]
        return scripts[this.recoveryStep++]
      }
    }
    const provider: Provider = {
      id: "test",
      name: "Test",
      type: "openai",
      baseURL: "http://unused",
      apiKey: "test",
      models: [{ id: "test-model", name: "Test model" }],
    }
    const harness = new AgentHarness({
      panelId: "panel-recovery",
      workspace,
      provider,
      model: provider.models[0],
      systemPrompt: "Use tools.",
      client: new RecoveringProvider(),
      session: new MemorySession(),
      onEvent: (event) => events.push(event),
    })

    const result = await harness.run("Create a file, run it, and verify its output")
    expect(result.completed).toBe(true)
    expect(events.some((event) => event.type === "controller_retry")).toBe(true)
    expect(await readFile(join(workspace, "proof/a.js"), "utf8")).toContain("42")
  })

  test("repairs run_command path aliases and treats nonzero exits as failures", async () => {
    const workspace = await mkdtemp(join(tmpdir(), "rsi-harness-"))
    temporaryDirectories.push(workspace)
    const provider: Provider = {
      id: "test", name: "Test", type: "openai", baseURL: "http://unused", apiKey: "test",
      models: [{ id: "test-model", name: "Test model" }],
    }
    class BadExitProvider extends ScriptedProvider {
      override async stream() {
        return { text: "", toolCalls: [{ id: "bad", name: "run_command", args: { command: "exit 7", path: "." } }] }
      }
    }
    const events: HarnessEvent[] = []
    const harness = new AgentHarness({
      panelId: "panel-exit", workspace, provider, model: provider.models[0], systemPrompt: "test",
      client: new BadExitProvider(), session: new MemorySession(), maxSteps: 1,
      onEvent: (event) => events.push(event),
    })
    await harness.run("Run a test")
    expect(events.some((event) => event.type === "tool_result" && event.data.isError === true)).toBe(true)
  })
})
