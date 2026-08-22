import {
  Activity,
  ArrowLeft,
  Bot,
  Box,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Cpu,
  Database,
  ExternalLink,
  FileCode2,
  Gauge,
  Globe2,
  HardDrive,
  Layers3,
  Library,
  Maximize2,
  MessageSquare,
  Minus,
  Network,
  Plus,
  Search,
  ShieldCheck,
  Sparkles,
  TestTube2,
  Wrench,
  X,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

type Category = "interface" | "runtime" | "build" | "learn" | "memory" | "proof";

type MapNode = {
  id: string;
  title: string;
  caption: string;
  category: Category;
  x: number;
  y: number;
  parent?: string;
  loop?: boolean;
  metric?: string;
  details: string[];
};

const WORLD = { width: 1940, height: 1260 };

const categoryMeta: Record<Category, { label: string; icon: typeof Bot }> = {
  interface: { label: "Use", icon: MessageSquare },
  runtime: { label: "Run", icon: Cpu },
  build: { label: "Build", icon: Wrench },
  learn: { label: "Improve", icon: BrainCircuit },
  memory: { label: "Remember", icon: Database },
  proof: { label: "Verify", icon: ShieldCheck },
};

const nodes: MapNode[] = [
  {
    id: "optimus", title: "Optimus Studio", caption: "Local model operating system", category: "learn", x: 970, y: 630,
    metric: "One local harness", details: ["Chat, build, train, evaluate and improve from one localhost workspace.", "Models, runs, environments, skills and evidence remain on the user's machine."],
  },
  {
    id: "chat", title: "Chat workspace", caption: "Talk to loaded local models", category: "interface", x: 410, y: 190, parent: "optimus",
    details: ["Streaming conversations with history and model switching.", "Selectable context window, visible consumption and hardware-aware TPS estimate."],
  },
  {
    id: "commands", title: "Slash commands", caption: "/il · /rl · /ttc", category: "interface", x: 125, y: 90, parent: "chat",
    details: ["Command picker appears as soon as '/' is typed.", "Natural-language requests enter environment creation or test-time compute without leaving chat."],
  },
  {
    id: "skills", title: "Skill routing", caption: "Prompt-aware capabilities", category: "memory", x: 415, y: 55, parent: "chat",
    details: ["Installed Markdown skills are matched to the prompt and injected compactly.", "Skills can teach small models stable procedures instead of asking them to invent every workflow."],
  },
  {
    id: "tool-calls", title: "Tools + MCP", caption: "Search, fetch, time, calculate", category: "runtime", x: 705, y: 100, parent: "chat",
    details: ["Structured tool calls are parsed, repaired, executed and returned to the model.", "Built-ins and MCP servers extend a local model with web and utility capabilities."],
  },
  {
    id: "tool-safety", title: "Tool guardrails", caption: "Validated requests", category: "proof", x: 790, y: 255, parent: "tool-calls",
    details: ["URL validation and bounded execution reduce unsafe or malformed tool calls.", "Raw tool JSON is converted into a tool result cycle instead of being shown as the assistant answer."],
  },
  {
    id: "models", title: "Model library", caption: "Find, download and load", category: "runtime", x: 1235, y: 165, parent: "optimus",
    details: ["Detects local hardware and presents compatible model formats and precision.", "Downloads models into local storage, then loads them for chat, environment generation and training."],
  },
  {
    id: "hardware", title: "Hardware profile", caption: "Memory · backend · architecture", category: "runtime", x: 1090, y: 35, parent: "models",
    details: ["Hardware discovery informs model fit, context capacity and training presets.", "The context control estimates throughput from model size, architecture, selected length and machine profile."],
  },
  {
    id: "mlx", title: "Local inference", caption: "MLX / compatible runtimes", category: "runtime", x: 1430, y: 65, parent: "models",
    details: ["Loaded weights generate locally with streaming tokens and measured throughput.", "The runtime is reused by chat, builders, graders and autonomous attempts."],
  },
  {
    id: "context", title: "Context control", caption: "Budget + live fill", category: "interface", x: 1515, y: 225, parent: "models",
    details: ["A clickable ring shows how quickly the active conversation consumes context.", "The slider exposes the speed-versus-memory tradeoff before generation."],
  },
  {
    id: "builder", title: "No-code IL / RL", caption: "Describe a world, get a runnable spec", category: "build", x: 1685, y: 475, parent: "optimus",
    details: ["A guided visual builder and /il or /rl chat commands produce the same typed environment contract.", "Small models start from trusted frameworks and modify constrained fields rather than writing a simulator from scratch."],
  },
  {
    id: "il-env", title: "IL environments", caption: "Tasks, examples, criteria", category: "build", x: 1815, y: 300, parent: "builder",
    details: ["Define task distributions, success criteria, examples and curriculum without code.", "Generated environments are validated before being saved."],
  },
  {
    id: "rl-env", title: "RL state machines", caption: "State · action · reward", category: "build", x: 1850, y: 555, parent: "builder",
    details: ["Define multi-step state, actions, transitions, scenarios, termination and reward rules.", "The deterministic simulator makes the result playable and trainable rather than a prose-only plan."],
  },
  {
    id: "environments", title: "My environments", caption: "Save, inspect, play, reuse", category: "interface", x: 1740, y: 745, parent: "builder",
    details: ["Every generated environment is persisted and available from the sidebar.", "Users can inspect the contract, run scenarios, and select it as a training source in Optimus Lab."],
  },
  {
    id: "lab", title: "Optimus Lab", caption: "End-to-end training runs", category: "learn", x: 1400, y: 1045, parent: "optimus", loop: true,
    details: ["Choose a model, environment and hardware-aware preset, then launch a reproducible run.", "Live events expose benchmarks, training metrics, artifacts and terminal state."],
  },
  {
    id: "baseline", title: "Baseline benchmark", caption: "Measure before changing weights", category: "proof", x: 1725, y: 930, parent: "lab", loop: true,
    details: ["The exact target contract is tested first.", "If the baseline already passes, training is skipped—saving time and avoiding an unnecessary adapter."],
  },
  {
    id: "training", title: "Adapter training", caption: "SFT · QLoRA · GRPO", category: "learn", x: 1720, y: 1115, parent: "lab", loop: true,
    details: ["Supervised and reward-optimized paths run only when evidence justifies adaptation.", "MLX uses its efficient LoRA/quantized path; paged QLoRA is selected only where the backend truly supports it."],
  },
  {
    id: "perf", title: "Speed engine", caption: "More useful steps per second", category: "runtime", x: 1390, y: 1210, parent: "lab", loop: true,
    metric: "0.265 → 0.587 steps/s", details: ["Frozen-prefix generation reuses invariant prompt work and emits only the variable suffix.", "Curated token retention rose from 61.6% to 99.64%; useful throughput rose from 30.4 to 70.3 tok/s in the measured benchmark."],
  },
  {
    id: "runs", title: "Run artifacts", caption: "Adapters, logs, metrics, manifests", category: "memory", x: 1075, y: 1185, parent: "lab",
    details: ["Each run gets a procedural local folder with configuration, stream logs, metrics, checkpoints and outputs.", "Artifacts remain inspectable and reusable across later sessions."],
  },
  {
    id: "ttc", title: "Test-time compute", caption: "Attempt → verify → improve → retry", category: "learn", x: 600, y: 1040, parent: "optimus", loop: true,
    details: ["The harness spends extra inference only where objective checks say it is useful.", "It can research, curate examples, retrieve skills, train an adapter when warranted, and rerun the exact holdout."],
  },
  {
    id: "contract", title: "Frozen contract", caption: "One target, one verifier", category: "proof", x: 215, y: 1165, parent: "ttc", loop: true,
    details: ["The task and hard gates are frozen before attempts begin.", "This prevents the system from declaring victory by silently changing the test."],
  },
  {
    id: "verifier", title: "Objective verifier", caption: "Scores artifacts, not rhetoric", category: "proof", x: 215, y: 975, parent: "ttc", loop: true,
    details: ["Generated artifacts are executed and checked against machine-readable capability gates.", "Failures become structured feedback for the next attempt."],
  },
  {
    id: "curation", title: "Automated curation", caption: "Deduplicate · balance · retain", category: "learn", x: 500, y: 1190, parent: "ttc", loop: true,
    metric: "79 → 111 examples", details: ["Candidates are normalized, quality-filtered, deduplicated and balanced by capability coverage.", "Data quality and retention are measured so more training steps do not mean more low-value tokens."],
  },
  {
    id: "failure", title: "Failure analysis", caption: "Turn errors into reusable lessons", category: "learn", x: 665, y: 840, parent: "ttc", loop: true,
    details: ["Verifier output is classified into missing capability, malformed structure, runtime failure or weak strategy.", "The result drives the smallest useful intervention rather than reflexively retraining."],
  },
  {
    id: "memory", title: "Failure-skill memory", caption: "Retrieve the lesson next time", category: "memory", x: 375, y: 755, parent: "failure", loop: true,
    details: ["Validated Markdown skills are stored in a retrieval bank and matched to similar future tasks.", "A skill is promoted only after the model artifact passes, avoiding permanent memory of unverified advice."],
  },
  {
    id: "harness", title: "Harness adaptation", caption: "Repair interfaces, preserve intent", category: "learn", x: 730, y: 685, parent: "failure", loop: true,
    details: ["Deterministic parsers repair safe structural mismatches and compilers fill only invalid defaults.", "This improves the model-system outcome without pretending the underlying weights changed."],
  },
  {
    id: "retry", title: "Exact retry", caption: "Same gate, improved support", category: "proof", x: 965, y: 905, parent: "ttc", loop: true,
    details: ["The model reruns the unchanged task with retrieved lessons, curated context or a trained adapter.", "Only a measured verifier improvement is accepted."],
  },
  {
    id: "sakura", title: "Sakura Island proof", caption: "Autonomous full-scene run", category: "proof", x: 930, y: 1125, parent: "retry", loop: true,
    metric: "0.9439 · 37.811 s", details: ["Attempts one and two failed; structured feedback guided attempt three to a complete interactive island scene.", "The compiler normalized coordinates and supplied one invalid motion default; Chromium then passed every hard gate.", "No weight training was needed on the successful run: the measured improvement came from test-time iteration and harness adaptation."],
  },
  {
    id: "storage", title: "Local persistence", caption: "~/.iloptimus", category: "memory", x: 365, y: 490, parent: "optimus",
    details: ["Models, chats, environments, runs, learning sessions, tools, skills and performance profiles live locally.", "Manifests and hashes keep autonomous outputs attributable and reproducible."],
  },
  {
    id: "agents", title: "RSI workspace", caption: "Persistent local coding agents", category: "runtime", x: 120, y: 425, parent: "storage",
    details: ["Workspace tabs expose long-running agent panels and tool events.", "Agents can inspect local evidence and continue bounded work while preserving artifacts."],
  },
  {
    id: "evidence", title: "Evidence trail", caption: "Screenshots · reports · hashes", category: "proof", x: 135, y: 640, parent: "storage",
    details: ["Generated outputs retain provenance from prompt through verifier result.", "The research report links directly to run artifacts and the final rendered Sakura scene."],
  },
];

const iconForNode = (node: MapNode) => {
  const byId: Record<string, typeof Bot> = {
    optimus: Sparkles, chat: MessageSquare, commands: FileCode2, skills: Library, "tool-calls": Globe2,
    "tool-safety": ShieldCheck, models: Box, hardware: Cpu, mlx: Zap, context: Gauge, builder: Network,
    "il-env": BrainCircuit, "rl-env": Activity, environments: Layers3, lab: TestTube2, baseline: CircleDot,
    training: BrainCircuit, perf: Zap, runs: HardDrive, ttc: BrainCircuit, contract: ShieldCheck,
    verifier: CheckCircle2, curation: Database, failure: Search, memory: Library, harness: Wrench,
    retry: Activity, sakura: Sparkles, storage: Database, agents: Bot, evidence: ShieldCheck,
  };
  return byId[node.id] ?? categoryMeta[node.category].icon;
};

export default function OptimusMindMapPage() {
  const viewportRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const [zoom, setZoom] = useState(.56);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selectedId, setSelectedId] = useState("optimus");
  const [loopOnly, setLoopOnly] = useState(false);
  const [query, setQuery] = useState("");

  const fit = useCallback(() => {
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect) return;
    const nextZoom = Math.min((rect.width - 64) / WORLD.width, (rect.height - 64) / WORLD.height, .86);
    setZoom(nextZoom);
    setPan({ x: (rect.width - WORLD.width * nextZoom) / 2, y: (rect.height - WORLD.height * nextZoom) / 2 });
  }, []);

  useEffect(() => {
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [fit]);

  const visibleNodes = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return nodes.filter((node) => {
      if (loopOnly && node.id !== "optimus" && !node.loop) return false;
      if (!needle) return true;
      return [node.title, node.caption, ...node.details].join(" ").toLowerCase().includes(needle);
    });
  }, [loopOnly, query]);

  const visibleIds = useMemo(() => new Set(visibleNodes.map((node) => node.id)), [visibleNodes]);
  const selected = nodes.find((node) => node.id === selectedId) ?? nodes[0];
  const SelectedIcon = iconForNode(selected);

  const adjustZoom = (delta: number) => setZoom((value) => Math.min(1.2, Math.max(.28, value + delta)));

  return (
    <div className="optimus-map-page">
      <header className="optimus-map-header">
        <div className="optimus-map-brand">
          <Link to="/" aria-label="Back to Optimus Studio"><ArrowLeft /></Link>
          <span className="optimus-map-mark"><Network /></span>
          <div><strong>Optimus system map</strong><small>Capabilities + autonomous improvement</small></div>
        </div>
        <div className="optimus-map-modes" aria-label="Map mode">
          <button className={!loopOnly ? "active" : ""} onClick={() => setLoopOnly(false)}>Whole platform</button>
          <button className={loopOnly ? "active" : ""} onClick={() => setLoopOnly(true)}>Self-improvement loop</button>
        </div>
        <Link className="optimus-map-paper" to="/research/sakura-island">Read the Sakura report <ExternalLink /></Link>
      </header>

      <main className="optimus-map-main">
        <div className="optimus-map-toolbar">
          <label className="optimus-map-search"><Search /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find a capability" aria-label="Find a capability" />{query && <button onClick={() => setQuery("")} aria-label="Clear search"><X /></button>}</label>
          <div className="optimus-map-legend" aria-label="Categories">
            {(Object.entries(categoryMeta) as [Category, typeof categoryMeta[Category]][]).map(([key, item]) => <span key={key} className={`legend-${key}`}><i />{item.label}</span>)}
          </div>
          <div className="optimus-map-zoom">
            <button onClick={() => adjustZoom(-.08)} aria-label="Zoom out"><Minus /></button>
            <span>{Math.round(zoom * 100)}%</span>
            <button onClick={() => adjustZoom(.08)} aria-label="Zoom in"><Plus /></button>
            <button onClick={fit} aria-label="Fit map"><Maximize2 /></button>
          </div>
        </div>

        <section className="optimus-map-stage">
          <div
            ref={viewportRef}
            className="optimus-map-viewport"
            onPointerDown={(event) => {
              if (event.target !== event.currentTarget) return;
              dragRef.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
            onPointerMove={(event) => {
              if (!dragRef.current) return;
              setPan({ x: dragRef.current.panX + event.clientX - dragRef.current.x, y: dragRef.current.panY + event.clientY - dragRef.current.y });
            }}
            onPointerUp={() => { dragRef.current = null; }}
            onPointerCancel={() => { dragRef.current = null; }}
          >
            <div className="optimus-map-grid" />
            <div className="optimus-map-world" style={{ width: WORLD.width, height: WORLD.height, transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}>
              <svg className="optimus-map-edges" viewBox={`0 0 ${WORLD.width} ${WORLD.height}`} role="img" aria-label="Connections between Optimus capabilities">
                <title>Optimus capability connections</title>
                {visibleNodes.map((node) => {
                  if (!node.parent || !visibleIds.has(node.parent)) return null;
                  const source = nodes.find((item) => item.id === node.parent)!;
                  const bend = Math.abs(node.x - source.x) * .38;
                  const direction = node.x > source.x ? 1 : -1;
                  return <path key={`${source.id}-${node.id}`} className={`edge-${node.category} ${node.loop ? "edge-loop" : ""}`} d={`M ${source.x} ${source.y} C ${source.x + bend * direction} ${source.y}, ${node.x - bend * direction} ${node.y}, ${node.x} ${node.y}`} />;
                })}
              </svg>
              {visibleNodes.map((node) => {
                const Icon = iconForNode(node);
                return (
                  <button
                    key={node.id}
                    className={`optimus-map-node node-${node.category} ${node.id === "optimus" ? "root" : ""} ${selectedId === node.id ? "selected" : ""}`}
                    style={{ left: node.x, top: node.y }}
                    onClick={() => setSelectedId(node.id)}
                    aria-pressed={selectedId === node.id}
                  >
                    <span className="optimus-map-node-icon"><Icon /></span>
                    <span><strong>{node.title}</strong><small>{node.caption}</small>{node.metric && <em>{node.metric}</em>}</span>
                    <ChevronRight className="optimus-map-node-arrow" />
                  </button>
                );
              })}
              {!visibleNodes.length && <div className="optimus-map-empty">No matching capability</div>}
            </div>
          </div>

          <aside className={`optimus-map-detail detail-${selected.category}`} aria-live="polite">
            <span className="optimus-map-detail-icon"><SelectedIcon /></span>
            <small>{categoryMeta[selected.category].label}{selected.loop ? " · self-improvement loop" : ""}</small>
            <h1>{selected.title}</h1>
            <p>{selected.caption}</p>
            {selected.metric && <div className="optimus-map-detail-metric">{selected.metric}</div>}
            <ul>{selected.details.map((detail) => <li key={detail}><CheckCircle2 /> <span>{detail}</span></li>)}</ul>
            {selected.id === "sakura" && <Link to="/research/sakura-island">Open the full evidence report <ExternalLink /></Link>}
          </aside>
        </section>
      </main>
    </div>
  );
}
