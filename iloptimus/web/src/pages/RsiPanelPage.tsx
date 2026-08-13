import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Bot, CheckCircle2, CircleStop, Clock3, Code2, FolderOpen, Play, TerminalSquare, Wrench } from "lucide-react";
import { useParams } from "react-router-dom";
import { getRsiPanel, promptRsiPanel, stopRsiPanel, streamRsiEvents, type RsiEvent, type RsiPanel } from "../api/client";
import { refreshWorkspaceTabs } from "../components/WorkspaceTabs";

function eventText(event: RsiEvent): string {
  const data = event.data || {};
  if (event.type === "assistant_delta") return String(data.delta || data.text || "");
  if (event.type === "assistant_message") return String(data.text || data.content || "");
  if (event.type === "tool_call") return `${String(data.name || "tool")} ${JSON.stringify(data.arguments || {})}`;
  if (event.type === "tool_result") return String(data.content || data.result || "Completed");
  if (event.type === "started") return String(data.prompt || "Agent started");
  if (event.type === "completed") return String(data.text || "Task completed");
  if (event.type === "failed") return event.error || String(data.error || "Agent failed");
  return "";
}

export default function RsiPanelPage() {
  const { panelId = "" } = useParams();
  const [panel, setPanel] = useState<RsiPanel | null>(null);
  const [events, setEvents] = useState<RsiEvent[]>([]);
  const [prompt, setPrompt] = useState("");
  const [error, setError] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const sequence = useMemo(() => events.reduce((largest, event) => Math.max(largest, event.sequence || 0), 0), [events]);

  useEffect(() => {
    let source: EventSource | null = null;
    let cancelled = false;
    getRsiPanel(panelId).then((next) => {
      if (cancelled) return;
      setPanel(next);
      setEvents(next.events || []);
      const after = (next.events || []).reduce((largest, event) => Math.max(largest, event.sequence || 0), 0);
      source = streamRsiEvents(panelId, after, (event) => {
        if (event.type === "heartbeat") return;
        setEvents((current) => current.some((item) => item.sequence === event.sequence) ? current : [...current, event]);
        if (["started", "completed", "failed", "ready", "stopped"].includes(event.type)) {
          getRsiPanel(panelId).then(setPanel).catch(() => undefined);
          refreshWorkspaceTabs();
        }
      });
    }).catch((cause) => setError(cause instanceof Error ? cause.message : "Panel not found"));
    return () => { cancelled = true; source?.close(); };
  }, [panelId]);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [events]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const clean = prompt.trim();
    if (!clean || panel?.status === "running") return;
    setError("");
    try {
      setPanel(await promptRsiPanel(panelId, clean));
      setPrompt("");
      refreshWorkspaceTabs();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start the agent");
    }
  };

  const stop = async () => {
    setPanel(await stopRsiPanel(panelId));
    refreshWorkspaceTabs();
  };

  if (!panel) return <section className="rsi-panel-page"><div className="rsi-empty">{error || "Opening RSI agent…"}</div></section>;

  return (
    <section className="rsi-panel-page">
      <header className="rsi-panel-header">
        <div><span className={`panel-live-dot ${panel.status}`} /><div><small>Persistent RSI workspace</small><h1>{panel.title}</h1></div></div>
        <div className="rsi-panel-meta"><span><Bot />{panel.model_id}</span><span title={panel.workspace}><FolderOpen />{panel.workspace.split("/").pop()}</span><span><Clock3 />#{sequence}</span></div>
        <button className="rsi-stop" onClick={stop} disabled={panel.status === "stopped"}><CircleStop /> Stop</button>
      </header>

      <div className="rsi-timeline">
        {events.length === 0 && <div className="rsi-empty"><TerminalSquare /><strong>This panel is ready.</strong><span>Give the local model a concrete task. It can inspect, create, edit, and execute files inside this workspace.</span></div>}
        {events.map((event) => {
          const text = eventText(event);
          if (!text) return null;
          const Icon = event.type === "tool_call" || event.type === "tool_result" ? Wrench : event.type === "completed" ? CheckCircle2 : event.type === "started" ? Play : event.type === "assistant_message" || event.type === "assistant_delta" ? Code2 : Bot;
          return <article className={`rsi-event ${event.type}`} key={`${event.sequence}-${event.type}`}><div className="rsi-event-icon"><Icon /></div><div><div className="rsi-event-label">{event.type.replaceAll("_", " ")}</div><pre>{text}</pre></div></article>;
        })}
        <div ref={endRef} />
      </div>

      <form className="rsi-prompt" onSubmit={submit}>
        {error && <div className="rsi-error">{error}</div>}
        <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Give this RSI agent a task in its admitted workspace…" rows={2} />
        <button disabled={!prompt.trim() || panel.status === "running" || panel.status === "stopped"}><Play />{panel.status === "running" ? "Working…" : "Run task"}</button>
      </form>
    </section>
  );
}
