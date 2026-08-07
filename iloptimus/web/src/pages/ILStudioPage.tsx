import { useEffect, useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FlaskConical,
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  TrendingUp,
  Settings,
  Terminal,
  Activity,
  ChevronDown,
  ChevronRight,
  Sparkles,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  getModels,
  getTasksets,
  getHardware,
  getRuns,
  getRun,
  createRun,
  streamRunEvents,
  type ModelInfo,
  type TasksetInfo,
  type HardwareInfo,
  type RunState,
  type LogEvent,
} from "../api/client";

const stageLabels: Record<string, string> = {
  "initializing": "Init",
  "loading-model": "Load",
  "benchmarking-baseline": "Baseline",
  "sft-training": "SFT",
  "benchmarking-post-sft": "Post-SFT",
  "grpo-training": "GRPO",
  "benchmarking-post-grpo": "Post-GRPO",
  "done": "Done",
};

const stageOrder = [
  "initializing",
  "loading-model",
  "benchmarking-baseline",
  "sft-training",
  "benchmarking-post-sft",
  "grpo-training",
  "benchmarking-post-grpo",
  "done",
];

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.4, 0, 0.2, 1] as any } },
};

export default function ILStudioPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [tasksets, setTasksets] = useState<TasksetInfo[]>([]);
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [runs, setRuns] = useState<RunState[]>([]);
  const [loading, setLoading] = useState(true);

  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedTaskset, setSelectedTaskset] = useState<string>("");
  const [sftIters, setSftIters] = useState(100);
  const [grpoIters, setGrpoIters] = useState(50);
  const [grpoGroupSize, setGrpoGroupSize] = useState(4);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<RunState | null>(null);
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [starting, setStarting] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const loadInitial = useCallback(async () => {
    const [m, t, h, r] = await Promise.all([
      getModels(),
      getTasksets(),
      getHardware(),
      getRuns(),
    ]);
    setModels(m);
    setTasksets(t);
    setHw(h);
    setRuns(r);
    const recommended = m.find((x) => x.compatibility.status === "recommended");
    setSelectedModel(recommended?.id || m[0]?.id || "");
    setSelectedTaskset(t[0]?.id || "");
    const running = r.find((x) => x.status === "running");
    if (running) setActiveRunId(running.id);
  }, []);

  useEffect(() => {
    loadInitial().finally(() => setLoading(false));
  }, [loadInitial]);

  useEffect(() => {
    if (!activeRunId) return;
    if (eventSourceRef.current) eventSourceRef.current.close();
    setEvents([]);

    const es = streamRunEvents(
      activeRunId,
      (event) => {
        setEvents((prev) => [...prev, event]);
        if (event.stage === "done" || event.level === "error") {
          getRun(activeRunId).then(setActiveRun).catch(() => {});
        }
      },
      () => { getRun(activeRunId).then(setActiveRun).catch(() => {}); }
    );
    eventSourceRef.current = es;

    const pollInterval = setInterval(() => {
      getRun(activeRunId).then((r) => {
        setActiveRun(r);
        if (r.status === "completed" || r.status === "failed" || r.status === "cancelled") {
          clearInterval(pollInterval);
        }
      }).catch(() => {});
    }, 2000);

    getRun(activeRunId).then(setActiveRun).catch(() => {});

    return () => { es.close(); clearInterval(pollInterval); };
  }, [activeRunId]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  const handleStartRun = async () => {
    if (!selectedModel || !selectedTaskset) return;
    setStarting(true);
    try {
      const result = await createRun({
        model_id: selectedModel,
        taskset_id: selectedTaskset,
        sft_iters: sftIters,
        grpo_iters: grpoIters,
        grpo_group_size: grpoGroupSize,
      });
      setActiveRunId(result.id);
      setEvents([]);
      getRuns().then(setRuns);
    } catch (err) {
      console.error("Failed to start run:", err);
    } finally {
      setStarting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="w-12 h-12 rounded-2xl shimmer" />
      </div>
    );
  }

  const selectedModelInfo = models.find((m) => m.id === selectedModel);
  const selectedTasksetInfo = tasksets.find((t) => t.id === selectedTaskset);

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/20 mb-3">
          <Sparkles className="w-3.5 h-3.5 text-accent" />
          <span className="text-xs font-medium text-accent">SFT + GRPO Pipeline</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-fg-primary mb-2">IL Studio</h1>
        <p className="text-fg-secondary text-sm">
          Configure and run Intuition Learning pipelines. Track progress in real time.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration panel */}
        <motion.div variants={fadeUp} initial="hidden" animate="show" className="lg:col-span-1 space-y-4">
          <div className="glass rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <Settings className="w-5 h-5 text-accent" />
              <h2 className="text-base font-semibold text-fg-primary">Pipeline Config</h2>
            </div>

            <div className="space-y-4">
              {/* Model selector */}
              <div>
                <label className="text-xs font-medium text-fg-muted mb-1.5 block uppercase tracking-wide">Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="input-base w-full"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.params_b}B, {m.compatibility.status})
                    </option>
                  ))}
                </select>
                {selectedModelInfo && (
                  <div className="mt-2 text-xs">
                    <span
                      className={`badge ${
                        selectedModelInfo.compatibility.status === "recommended" ? "badge-green"
                        : selectedModelInfo.compatibility.status === "feasible" ? "badge-blue"
                        : selectedModelInfo.compatibility.status === "tight" ? "badge-yellow"
                        : "badge-red"
                      }`}
                    >
                      {selectedModelInfo.compatibility.status} · {selectedModelInfo.compatibility.best_precision}
                    </span>
                    <p className="text-fg-muted mt-1.5 leading-relaxed">
                      {selectedModelInfo.compatibility.reason}
                    </p>
                  </div>
                )}
              </div>

              {/* Taskset selector */}
              <div>
                <label className="text-xs font-medium text-fg-muted mb-1.5 block uppercase tracking-wide">Taskset</label>
                <select
                  value={selectedTaskset}
                  onChange={(e) => setSelectedTaskset(e.target.value)}
                  className="input-base w-full"
                >
                  {tasksets.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name} ({t.num_tasks} tasks)
                    </option>
                  ))}
                </select>
                {selectedTasksetInfo && (
                  <p className="text-xs text-fg-muted mt-1.5">
                    {selectedTasksetInfo.domain} · {selectedTasksetInfo.needs_sandbox ? "sandboxed" : "no sandbox"}
                  </p>
                )}
              </div>

              {/* Basic params */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-medium text-fg-muted mb-1.5 block uppercase tracking-wide">SFT Iters</label>
                  <input
                    type="number"
                    value={sftIters}
                    onChange={(e) => setSftIters(Number(e.target.value))}
                    className="input-base w-full"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-fg-muted mb-1.5 block uppercase tracking-wide">GRPO Iters</label>
                  <input
                    type="number"
                    value={grpoIters}
                    onChange={(e) => setGrpoIters(Number(e.target.value))}
                    className="input-base w-full"
                  />
                </div>
              </div>

              {/* Advanced settings */}
              <div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-1 text-sm text-fg-secondary hover:text-fg-primary transition-colors"
                >
                  {showAdvanced ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                  Advanced
                </button>
                <AnimatePresence>
                  {showAdvanced && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="mt-3">
                        <label className="text-xs font-medium text-fg-muted mb-1.5 block uppercase tracking-wide">GRPO Group Size</label>
                        <input
                          type="number"
                          value={grpoGroupSize}
                          onChange={(e) => setGrpoGroupSize(Number(e.target.value))}
                          className="input-base w-full"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Start button */}
              <button
                onClick={handleStartRun}
                disabled={starting || !selectedModel || !selectedTaskset}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                {starting ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Starting...</>
                ) : (
                  <><Play className="w-4 h-4" /> Start IL Pipeline</>
                )}
              </button>
            </div>
          </div>

          {/* Previous runs */}
          {runs.length > 0 && (
            <div className="glass rounded-2xl p-5">
              <h3 className="text-sm font-semibold text-fg-primary mb-3">Recent Runs</h3>
              <div className="space-y-2">
                {runs.slice(-5).reverse().map((r) => (
                  <button
                    key={r.id}
                    onClick={() => { setActiveRunId(r.id); setEvents([]); }}
                    className={`w-full text-left p-2.5 rounded-xl transition-all ${
                      activeRunId === r.id
                        ? "bg-accent/15 border border-accent/30"
                        : "bg-bg-glass/30 hover:bg-bg-glass/50 border border-transparent"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono text-fg-muted">{r.id}</span>
                      <RunStatusBadge status={r.status} />
                    </div>
                    <div className="text-xs text-fg-muted mt-1">
                      {r.config.model_id} · {r.config.taskset_id}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </motion.div>

        {/* Right: Live tracking panel */}
        <motion.div variants={fadeUp} initial="hidden" animate="show" className="lg:col-span-2 space-y-4">
          <AnimatePresence mode="wait">
            {activeRun ? (
              <motion.div
                key="active"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-4"
              >
                {/* Progress header */}
                <div className="glass rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-accent" />
                      <h2 className="text-base font-semibold text-fg-primary">Run {activeRun.id}</h2>
                    </div>
                    <RunStatusBadge status={activeRun.status} />
                  </div>

                  {/* Progress bar */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between text-xs text-fg-secondary mb-1.5">
                      <span className="font-medium">{stageLabels[activeRun.stage] || activeRun.stage}</span>
                      <span>{(activeRun.progress * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-2 progress-track">
                      <motion.div
                        className="h-full progress-fill"
                        animate={{ width: `${activeRun.progress * 100}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  </div>

                  {/* Stage pipeline */}
                  <div className="flex items-center gap-1 overflow-x-auto pb-2">
                    {stageOrder.map((stage, i) => {
                      const currentIdx = stageOrder.indexOf(activeRun.stage);
                      const isDone = i < currentIdx;
                      const isCurrent = i === currentIdx;
                      return (
                        <div key={stage} className="flex items-center gap-1 flex-shrink-0">
                          <div
                            className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                              isDone
                                ? "bg-success/15 text-success"
                                : isCurrent
                                ? "bg-accent/20 text-accent pulse-glow"
                                : "bg-bg-glass/30 text-fg-muted"
                            }`}
                          >
                            {isDone && "✓ "}{stageLabels[stage]}
                          </div>
                          {i < stageOrder.length - 1 && <div className="w-2 h-px bg-fg-muted/20" />}
                        </div>
                      );
                    })}
                  </div>

                  <div className="flex items-center gap-2 mt-3 text-xs text-fg-muted">
                    <Clock className="w-3.5 h-3.5" />
                    Elapsed: {formatTime(activeRun.elapsed_seconds)}
                  </div>
                </div>

                {/* Accuracy comparison */}
                {(activeRun.baseline_accuracy > 0 || activeRun.post_sft_accuracy > 0 || activeRun.post_grpo_accuracy > 0) && (
                  <div className="glass rounded-2xl p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <TrendingUp className="w-5 h-5 text-accent" />
                      <h3 className="text-base font-semibold text-fg-primary">Accuracy Progression</h3>
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                      <AccuracyCard label="Baseline" value={activeRun.baseline_accuracy} color="text-fg-secondary" />
                      <AccuracyCard
                        label="Post-SFT"
                        value={activeRun.post_sft_accuracy}
                        color="text-info"
                        delta={activeRun.post_sft_accuracy - activeRun.baseline_accuracy}
                      />
                      <AccuracyCard
                        label="Post-GRPO"
                        value={activeRun.post_grpo_accuracy}
                        color="text-success"
                        delta={activeRun.post_grpo_accuracy - activeRun.baseline_accuracy}
                      />
                    </div>
                  </div>
                )}

                {/* Charts */}
                {(activeRun.sft_loss_history.length > 0 || activeRun.grpo_reward_history.length > 0) && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {activeRun.sft_loss_history.length > 0 && (
                      <div className="glass rounded-2xl p-5">
                        <h3 className="text-sm font-semibold text-fg-primary mb-3">SFT Loss</h3>
                        <ResponsiveContainer width="100%" height={180}>
                          <LineChart data={activeRun.sft_loss_history.map((loss, i) => ({ iter: i + 1, loss }))}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                            <XAxis dataKey="iter" stroke="rgba(148,163,184,0.5)" fontSize={10} />
                            <YAxis stroke="rgba(148,163,184,0.5)" fontSize={10} />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "rgba(20,25,42,0.95)",
                                border: "1px solid rgba(255,255,255,0.08)",
                                borderRadius: "12px",
                                backdropFilter: "blur(20px)",
                              }}
                            />
                            <Line type="monotone" dataKey="loss" stroke="rgb(129,140,248)" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                    {activeRun.grpo_reward_history.length > 0 && (
                      <div className="glass rounded-2xl p-5">
                        <h3 className="text-sm font-semibold text-fg-primary mb-3">GRPO Reward</h3>
                        <ResponsiveContainer width="100%" height={180}>
                          <LineChart data={activeRun.grpo_reward_history.map((reward, i) => ({ iter: i + 1, reward }))}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                            <XAxis dataKey="iter" stroke="rgba(148,163,184,0.5)" fontSize={10} />
                            <YAxis stroke="rgba(148,163,184,0.5)" fontSize={10} />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "rgba(20,25,42,0.95)",
                                border: "1px solid rgba(255,255,255,0.08)",
                                borderRadius: "12px",
                                backdropFilter: "blur(20px)",
                              }}
                            />
                            <Line type="monotone" dataKey="reward" stroke="rgb(52,211,153)" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </div>
                )}

                {/* Live log */}
                <div className="glass rounded-2xl p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Terminal className="w-5 h-5 text-accent" />
                    <h3 className="text-sm font-semibold text-fg-primary">Live Log</h3>
                    <span className="text-xs text-fg-muted ml-auto">{events.length} events</span>
                  </div>
                  <div className="bg-black/30 dark:bg-black/40 rounded-xl p-3 max-h-96 overflow-y-auto font-mono text-xs space-y-1 backdrop-blur-sm">
                    {events.length === 0 && <div className="text-fg-muted">Waiting for events...</div>}
                    {events.map((event, i) => <LogLine key={i} event={event} />)}
                    <div ref={logEndRef} />
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="glass rounded-2xl flex flex-col items-center justify-center h-96 text-center"
              >
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-accent/15 to-accent/5 flex items-center justify-center mb-4">
                  <FlaskConical className="w-7 h-7 text-accent" strokeWidth={1.5} />
                </div>
                <h3 className="text-lg font-medium text-fg-primary mb-2">No active run</h3>
                <p className="text-sm text-fg-muted max-w-sm">
                  Configure your pipeline on the left and click "Start IL Pipeline" to begin.
                  You'll see live logs, training curves, and accuracy progression here.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}

function RunStatusBadge({ status }: { status: string }) {
  const config: Record<string, { icon: any; badge: string }> = {
    pending: { icon: Clock, badge: "badge-gray" },
    running: { icon: Loader2, badge: "badge-blue" },
    completed: { icon: CheckCircle2, badge: "badge-green" },
    failed: { icon: XCircle, badge: "badge-red" },
    cancelled: { icon: XCircle, badge: "badge-gray" },
  };
  const cfg = config[status] || config["pending"];
  const Icon = cfg.icon;
  return (
    <span className={cfg.badge}>
      <Icon className={`w-3 h-3 ${status === "running" ? "animate-spin" : ""}`} />
      {status}
    </span>
  );
}

function AccuracyCard({
  label, value, color, delta,
}: { label: string; value: number; color: string; delta?: number }) {
  return (
    <div className="rounded-xl p-4 text-center bg-bg-glass/30 border border-white/5">
      <div className="text-xs text-fg-muted mb-1 uppercase tracking-wide font-medium">{label}</div>
      <div className={`text-2xl font-bold ${color} tracking-tight`}>
        {value > 0 ? `${(value * 100).toFixed(1)}%` : "—"}
      </div>
      {delta !== undefined && delta !== 0 && value > 0 && (
        <div className={`text-xs mt-1 font-medium ${delta > 0 ? "text-success" : "text-danger"}`}>
          {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}

function LogLine({ event }: { event: LogEvent }) {
  const levelColors: Record<string, string> = {
    info: "text-fg-secondary",
    warn: "text-warning",
    error: "text-danger",
    success: "text-success",
    metric: "text-info",
  };
  const time = new Date(event.timestamp * 1000).toLocaleTimeString();
  const color = levelColors[event.level] || "text-fg-secondary";
  return (
    <div className="flex gap-2">
      <span className="text-fg-muted flex-shrink-0">{time}</span>
      <span className="text-fg-muted flex-shrink-0 opacity-60">[{event.stage}]</span>
      <span className={color}>{event.message}</span>
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}
