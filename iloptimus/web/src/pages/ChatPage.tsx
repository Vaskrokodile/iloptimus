import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowUp, Bot, ChevronDown, Copy, Paperclip, RotateCcw, SlidersHorizontal, User } from "lucide-react";
import { createEnvironmentFromChat, getModels, sendChat, type ModelInfo } from "../api/client";

type Message = { role: "user" | "assistant"; text: string };

const starters = [
  "Design an IL pipeline for mathematical reasoning",
  "Which local model fits my hardware?",
  "Explain the difference between IL and RL",
];

const previousChats: Record<string, Message[]> = {
  "0": [
    { role: "user", text: "How should I design a reward for reasoning quality?" },
    { role: "assistant", text: "Use correctness as a hard gate, then reward the qualities you want the model to internalize: verification, efficient decomposition, and calibrated uncertainty. A practical IL reward is correctness × (0.6 + 0.4 × reasoning quality), so wrong answers never receive signal while good reasoning separates correct traces." },
  ],
  "1": [
    { role: "user", text: "Compare Qwen and Llama for a small local training run." },
    { role: "assistant", text: "Qwen is often the stronger starting point for compact reasoning and coding experiments, while Llama offers a broad ecosystem and familiar tooling. In Model Library, choose the best quantization for your detected memory, then send it into Optimus Lab." },
  ],
};

const slashCommands = [
  { command: "/il", title: "Create an IL environment", description: "Describe a capability to teach through ideal demonstrations", kind: "prompt" },
  { command: "/rl", title: "Create an RL environment", description: "Describe a world, its goal, and the reward signal", kind: "prompt" },
  { command: "/models", title: "Open Model Library", description: "Choose which local model to chat with", kind: "navigate", to: "/models" },
  { command: "/lab", title: "Open Optimus Lab", description: "Configure and launch a training run", kind: "navigate", to: "/studio" },
  { command: "/clear", title: "Clear this chat", description: "Start again without changing the selected model", kind: "clear" },
];

export default function ChatPage() {
  const [params] = useSearchParams();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const savedModel = (() => { try { return JSON.parse(localStorage.getItem("iloptimus-chat-model") || "null"); } catch { return null; } })();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState(savedModel?.id || "qwen2.5-1.5b");
  const [thinking, setThinking] = useState(false);
  const [commandIndex, setCommandIndex] = useState(0);
  const endRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const model = models.find((item) => item.id === modelId)?.name || savedModel?.name || "Qwen2.5-1.5B";
  const commandQuery = input.startsWith("/") && !input.includes(" ") ? input.slice(1).toLowerCase() : null;
  const visibleCommands = commandQuery === null ? [] : slashCommands.filter((item) => item.command.slice(1).startsWith(commandQuery));
  const commandPaletteOpen = visibleCommands.length > 0;

  useEffect(() => {
    const chat = params.get("chat");
    setMessages(chat ? previousChats[chat] || [] : []);
  }, [params]);

  useEffect(() => { getModels().then((items) => {
    setModels(items);
    const installed = items.filter((item) => item.local.status === "downloaded");
    if (!installed.some((item) => item.id === modelId) && installed[0]) setModelId(installed[0].id);
  }).catch(() => setModels([])); }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  const send = async (text = input) => {
    const clean = text.trim();
    if (!clean || thinking) return;
    setMessages((current) => [...current, { role: "user", text: clean }]);
    setInput("");
    setThinking(true);
    try {
      const command = clean.match(/^\/(il|rl)\s+(.+)/i);
      if (command) {
        const environment = await createEnvironmentFromChat(command[1].toUpperCase() as "IL" | "RL", command[2], modelId);
        setMessages((current) => [...current, { role: "assistant", text: `${environment.name} is ready. I created ${environment.tasks.length} gradable tasks, a ${environment.mode} reward design, and a taskset you can use in Optimus Lab. You’ll find it in My Environments.` }]);
        return;
      }
      const response = await sendChat(modelId, clean, messages);
      setMessages((current) => [...current, { role: "assistant", text: response.answer }]);
    } catch (cause) {
      const detail = cause instanceof Error ? cause.message.replace(/^\{"detail":"?|"\}$/g, "") : "The local request failed";
      setMessages((current) => [...current, { role: "assistant", text: clean.startsWith("/") ? `I could not build that environment: ${detail}` : detail }]);
    } finally {
      setThinking(false);
    }
  };

  const runCommand = (item: typeof slashCommands[number]) => {
    if (item.kind === "prompt") { setInput(`${item.command} `); return; }
    if (item.kind === "navigate" && item.to) { navigate(item.to); return; }
    if (item.kind === "clear") { setMessages([]); setInput(""); }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (commandPaletteOpen && event.key === "ArrowDown") { event.preventDefault(); setCommandIndex((index) => (index + 1) % visibleCommands.length); return; }
    if (commandPaletteOpen && event.key === "ArrowUp") { event.preventDefault(); setCommandIndex((index) => (index - 1 + visibleCommands.length) % visibleCommands.length); return; }
    if (commandPaletteOpen && event.key === "Escape") { event.preventDefault(); setInput(""); return; }
    if (commandPaletteOpen && event.key === "Enter") { event.preventDefault(); runCommand(visibleCommands[Math.min(commandIndex, visibleCommands.length - 1)]); setCommandIndex(0); return; }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  };

  return (
    <section className="chat-page">
      <header className="chat-header">
        <div className="model-status"><span className="status-dot" /><select value={modelId} onChange={(event) => { const next = models.find((item) => item.id === event.target.value); setModelId(event.target.value); if (next) localStorage.setItem("iloptimus-chat-model", JSON.stringify({ id: next.id, name: next.name })); }} aria-label="Active model">{models.some((item) => item.local.status === "downloaded") ? models.filter((item) => item.local.status === "downloaded").map((item) => <option key={item.id} value={item.id}>{item.name}</option>) : <option value="">Download a model first</option>}</select><ChevronDown /></div>
        <button className="header-action" onClick={() => navigate("/models")}><SlidersHorizontal /> Model library</button>
      </header>

      <div className={`chat-thread ${messages.length === 0 ? "empty-thread" : ""}`}>
        {messages.length === 0 ? (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="welcome-state">
            <div className="welcome-title-row">
              <h1>What do you want<br />to do today?</h1>
              <div className="mascot-stage" aria-hidden="true"><img src="/wolf-mascot-v2.png" alt="" /></div>
            </div>
            <p className="welcome-copy">Chat with local models, explore ideas, and turn the best conversations into trainable intuition.</p>
            <div className="starter-grid">
              {starters.map((starter) => <button key={starter} onClick={() => send(starter)}>{starter}<ArrowUp /></button>)}
            </div>
          </motion.div>
        ) : (
          <div className="message-list">
            {messages.map((message, index) => (
              <motion.article key={`${message.role}-${index}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={`message ${message.role}`}>
                <div className="message-avatar">{message.role === "assistant" ? <Bot /> : <User />}</div>
                <div><div className="message-meta">{message.role === "assistant" ? model : "You"}</div><p>{message.text}</p>{message.role === "assistant" && <div className="message-tools"><button aria-label="Copy"><Copy /></button><button aria-label="Regenerate"><RotateCcw /></button></div>}</div>
              </motion.article>
            ))}
            {thinking && <div className="thinking"><span /><span /><span /></div>}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <form className="composer" onSubmit={(event: FormEvent) => { event.preventDefault(); send(); }}>
        {commandPaletteOpen && <div className="command-palette" role="listbox" aria-label="Slash commands"><div className="command-palette-label">Commands</div>{visibleCommands.map((item,index)=><button type="button" key={item.command} className={index===commandIndex?"active":""} onMouseDown={(event)=>{event.preventDefault();runCommand(item);setCommandIndex(0)}}><code>{item.command}</code><span><strong>{item.title}</strong><small>{item.description}</small></span><em>↵</em></button>)}</div>}
        <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={onKeyDown} placeholder="Message your model…" rows={1} aria-label="Message" />
        <div className="composer-tools"><button type="button" className="attach" aria-label="Attach file"><Paperclip /></button><span>Local context</span><div className="command-hints"><button type="button" onClick={()=>setInput("/il ")}>/il</button><button type="button" onClick={()=>setInput("/rl ")}>/rl</button></div></div>
        <button className="send-button" disabled={!input.trim() || thinking || !models.some((item) => item.id === modelId && item.local.status === "downloaded")} aria-label="Send message"><ArrowUp /></button>
      </form>
      <p className="chat-footnote">Local models can make mistakes. Verify important outputs.</p>
    </section>
  );
}
