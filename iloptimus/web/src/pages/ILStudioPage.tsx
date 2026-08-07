import { useEffect, useState, useRef, useCallback } from "react";
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
  createRun,
  streamRunEvents,
  type ModelInfo,
  type TasksetInfo,
  type HardwareInfo,
  type RunState,
  type LogEvent,
} from "../api/client";

const stageLabels: Record<string, string> = {
  "initializing": "Initializing",
  "loading-model": "Loading Model",
  "benchmarking-baseline": "Baseline Benchmark",
  "sft-training": "SFT Training",
  "benchmarking-post-sft": "Post-SFT Benchmark",
  "grpo-training": "GRPO RL Training",
  "benchmarking-post-grpo": "Post-GRPO Benchmark",
  "done": "Complete",
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

export default function ILStudioPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [tasksets, setTasksets] = useState<TasksetInfo[]>([]);
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [runs, setRuns] = useState<RunState[]>([]);
  const [loading, setLoading] = useState(true);

  // Form state
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedTaskset, setSelectedTaskset] = useState<string>("");
  const [sftIters, setSftIters] = useState(100);
  const [grpoIters, setGrpoIters] = useState(50);
  const [grpoGroupSize, setGrpoGroupSize] = useState(4);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Active run
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
    // Auto-select first recommended model and first taskset
    const recommended = m.find((x) => x.compatibility.status === "recommended");
    setSelectedModel(recommended?.id || m[0]?.id || "");
    setSelectedTaskset(t[0]?.id || "");
    // If there's a running run, select it
    const running = r.find((x) => x.status === "running");
    if (running) {
      setActiveRunId(running.id);
    }
  }, []);

  useEffect(() => {
    loadInitial().finally(() => setLoading(false));
  }, [loadInitial]);

  // Stream events for active run
  useEffect(() => {
    if (!activeRunId) return;

    // Close previous stream
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setEvents([]);

    const es = streamRunEvents(
      activeRunId,
      (event) => {
        setEvents((prev) => [...prev, event]);
        // Refresh run state periodically
        if (event.stage === "done" || event.level === "error") {
          getRun(activeRunId).then(setActiveRun).catch(() => {});
        }
      },
      () => {
        // On error, try to refresh state
        getRun(activeRunId).then(setActiveRun).catch(() => {});
      }
    );
    eventSourceRef.current = es;

    // Also poll run state
    const pollInterval = setInterval(() => {
      getRun(activeRunId).then((r) => {
        setActiveRun(r);
        if (r.status === "completed" || r.status === "failed" || r.status === "cancelled") {
          clearInterval(pollInterval);
        }
      }).catch(() => {});
    }, 2000);

    // Load initial run state
    getRun(activeRunId).then(setActiveRun).catch(() => {});

    return () => {
      es.close();
      clearInterval(pollInterval);
    };
  }, [activeRunId]);

  // Auto-scroll log
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
      // Refresh runs list
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
        <div className="text-gray-500 animate-pulse">Loading IL Studio...</div>
      </div>
    );
  }

  const selectedModelInfo = models.find((m) => m.id === selectedModel);
  const selectedTasksetInfo = tasksets.find((t) => t.id === selectedTaskset);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">IL Studio</h1>
        <p className="text-gray-400">
          Configure and run Intuition Learning pipelines. Track progress in real time.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration panel */}
        <div className="lg:col-span-1 space-y-4">
          <div className="card">
            <div className="flex items-center gap-2 mb-4">
              <Settings className="w-5 h-5 text-brand-400" />
              <h2 className="text-lg font-semibold text-white">Pipeline Config</h2>
            </div>

            {/* Model selector */}
            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-400 mb-1.5 block">Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-600"
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
                        selectedModelInfo.compatibility.status === "recommended"
                          ? "badge-green"
                          : selectedModelInfo.compatibility.status === "feasible"
                          ? "badge-blue"
                          : selectedModelInfo.compatibility.status === "tight"
                          ? "badge-yellow"
                          : "badge-red"
                      }`}
                    >
                      {selectedModelInfo.compatibility.status} · {selectedModelInfo.compatibility.best_precision}
                    </span>
                    <p className="text-gray-500 mt-1.5">
                      {selectedModelInfo.compatibility.reason}
                    </p>
                  </div>
                )}
              </div>

              {/* Taskset selector */}
              <div>
                <label className="text-sm text-gray-400 mb-1.5 block">Taskset</label>
                <select
                  value={selectedTaskset}
                  onChange={(e) => setSelectedTaskset(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-600"
                >
                  {tasksets.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name} ({t.num_tasks} tasks)
                    </option>
                  ))}
                </select>
                {selectedTasksetInfo && (
                  <p className="text-xs text-gray-500 mt-1.5">
                    {selectedTasksetInfo.domain} · {selectedTasksetInfo.needs_sandbox ? "sandboxed" : "no sandbox"}
                  </p>
                )}
              </div>

              {/* Basic params */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-sm text-gray-400 mb-1.5 block">SFT Iters</label>
                  <input
                    type="number"
                    value={sftIters}
                    onChange={(e) => setSftIters(Number(e.target.value))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-600"
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-400 mb-1.5 block">GRPO Iters</label>
                  <input
                    type="number"
                    value={grpoIters}
                    onChange={(e) => setGrpoIters(Number(e.target.value))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-600"
                  />
                </div>
              </div>

              {/* Advanced settings */}
              <div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-1 text-sm text-gray-400 hover:text-gray-200"
                >
                  {showAdvanced ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                  Advanced
                </button>
                {showAdvanced && (
                  <div className="mt-3 space-y-3">
                    <div>
                      <label className="text-sm text-gray-400 mb-1.5 block">GRPO Group Size</label>
                      <input
                        type="number"
                        value={grpoGroupSize}
                        onChange={(e) => setGrpoGroupSize(Number(e.target.value))}
                        className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-600"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Start button */}
              <button
                onClick={handleStartRun}
                disabled={starting || !selectedModel || !selectedTaskset}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                {starting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Start IL Pipeline
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Previous runs */}
          {runs.length > 0 && (
            <div className="card">
              <h3 className="text-sm font-semibold text-white mb-3">Recent Runs</h3>
              <div className="space-y-2">
                {runs.slice(-5).reverse().map((r) => (
                  <button
                    key={r.id}
                    onClick={() => {
                      setActiveRunId(r.id);
                      setEvents([]);
                    }}
                    className={`w-full text-left p-2 rounded-lg transition-colors ${
                      activeRunId === r.id
                        ? "bg-brand-600/20 border border-brand-600/40"
                        : "bg-gray-800/50 hover:bg-gray-800"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono text-gray-400">{r.id}</span>
                      <RunStatusBadge status={r.status} />
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {r.config.model_id} · {r.config.taskset_id}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Live tracking panel */}
        <div className="lg:col-span-2 space-y-4">
          {activeRun ? (
            <>
              {/* Progress header */}
              <div className="card">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-brand-400" />
                    <h2 className="text-lg font-semibold text-white">
                      Run {activeRun.id}
                    </h2>
                  </div>
                  <RunStatusBadge status={activeRun.status} />
                </div>

                {/* Progress bar */}
                <div className="mb-4">
                  <div className="flex items-center justify-between text-xs text-gray-400 mb-1.5">
                    <span>{stageLabels[activeRun.stage] || activeRun.stage}</span>
                    <span>{(activeRun.progress * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-brand-600 rounded-full transition-all duration-500"
                      style={{ width: `${activeRun.progress * 100}%` }}
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
                          className={`px-2 py-1 rounded text-xs ${
                            isDone
                              ? "bg-green-500/15 text-green-400"
                              : isCurrent
                              ? "bg-brand-600/20 text-brand-400"
                              : "bg-gray-800 text-gray-600"
                          }`}
                        >
                          {isDone && "✓ "}
                          {stageLabels[stage]}
                        </div>
                        {i < stageOrder.length - 1 && (
                          <div className="w-2 h-px bg-gray-700" />
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Timer */}
                <div className="flex items-center gap-2 mt-3 text-xs text-gray-500">
                  <Clock className="w-3.5 h-3.5" />
                  Elapsed: {formatTime(activeRun.elapsed_seconds)}
                </div>
              </div>

              {/* Accuracy comparison */}
              {(activeRun.baseline_accuracy > 0 ||
                activeRun.post_sft_accuracy > 0 ||
                activeRun.post_grpo_accuracy > 0) && (
                <div className="card">
                  <div className="flex items-center gap-2 mb-4">
                    <TrendingUp className="w-5 h-5 text-brand-400" />
                    <h3 className="text-lg font-semibold text-white">Accuracy Progression</h3>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <AccuracyCard
                      label="Baseline"
                      value={activeRun.baseline_accuracy}
                      color="text-gray-400"
                    />
                    <AccuracyCard
                      label="Post-SFT"
                      value={activeRun.post_sft_accuracy}
                      color="text-blue-400"
                      delta={activeRun.post_sft_accuracy - activeRun.baseline_accuracy}
                    />
                    <AccuracyCard
                      label="Post-GRPO"
                      value={activeRun.post_grpo_accuracy}
                      color="text-green-400"
                      delta={activeRun.post_grpo_accuracy - activeRun.baseline_accuracy}
                    />
                  </div>
                </div>
              )}

              {/* Charts */}
              {(activeRun.sft_loss_history.length > 0 ||
                activeRun.grpo_reward_history.length > 0) && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {activeRun.sft_loss_history.length > 0 && (
                    <div className="card">
                      <h3 className="text-sm font-semibold text-white mb-3">SFT Loss</h3>
                      <ResponsiveContainer width="100%" height={180}>
                        <LineChart
                          data={activeRun.sft_loss_history.map((loss, i) => ({ iter: i + 1, loss }))}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="iter" stroke="#6b7280" fontSize={10} />
                          <YAxis stroke="#6b7280" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#1f2937",
                              border: "1px solid #374151",
                              borderRadius: "8px",
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="loss"
                            stroke="#818cf8"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                  {activeRun.grpo_reward_history.length > 0 && (
                    <div className="card">
                      <h3 className="text-sm font-semibold text-white mb-3">GRPO Reward</h3>
                      <ResponsiveContainer width="100%" height={180}>
                        <LineChart
                          data={activeRun.grpo_reward_history.map((reward, i) => ({ iter: i + 1, reward }))}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="iter" stroke="#6b7280" fontSize={10} />
                          <YAxis stroke="#6b7280" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#1f2937",
                              border: "1px solid #374151",
                              borderRadius: "8px",
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="reward"
                            stroke="#34d399"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              )}

              {/* Live log */}
              <div className="card">
                <div className="flex items-center gap-2 mb-3">
                  <Terminal className="w-5 h-5 text-brand-400" />
                  <h3 className="text-sm font-semibold text-white">Live Log</h3>
                  <span className="text-xs text-gray-500 ml-auto">{events.length} events</span>
                </div>
                <div className="bg-gray-950 rounded-lg p-3 max-h-96 overflow-y-auto font-mono text-xs space-y-1">
                  {events.length === 0 && (
                    <div className="text-gray-600">Waiting for events...</div>
                  )}
                  {events.map((event, i) => (
                    <LogLine key={i} event={event} />
                  ))}
                  <div ref={logEndRef} />
                </div>
              </div>
            </>
          ) : (
            <div className="card flex flex-col items-center justify-center h-96 text-center">
              <FlaskConical className="w-12 h-12 text-gray-700 mb-4" />
              <h3 className="text-lg font-medium text-gray-400 mb-2">No active run</h3>
              <p className="text-sm text-gray-600 max-w-sm">
                Configure your pipeline on the left and click "Start IL Pipeline" to begin.
                You'll see live logs, training curves, and accuracy progression here.
              </p>
            </div>
          )}
        </div>
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
  label,
  value,
  color,
  delta,
}: {
  label: string;
  value: number;
  color: string;
  delta?: number;
}) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-4 text-center">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>
        {value > 0 ? `${(value * 100).toFixed(1)}%` : "—"}
      </div>
      {delta !== undefined && delta !== 0 && value > 0 && (
        <div className={`text-xs mt-1 ${delta > 0 ? "text-green-400" : "text-red-400"}`}>
          {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}

function LogLine({ event }: { event: LogEvent }) {
  const levelColors: Record<string, string> = {
    info: "text-gray-400",
    warn: "text-yellow-400",
    error: "text-red-400",
    success: "text-green-400",
    metric: "text-blue-400",
  };
  const time = new Date(event.timestamp * 1000).toLocaleTimeString();
  const color = levelColors[event.level] || "text-gray-400";
  return (
    <div className="flex gap-2">
      <span className="text-gray-600 flex-shrink-0">{time}</span>
      <span className="text-gray-600 flex-shrink-0">[{event.stage}]</span>
      <span className={color}>{event.message}</span>
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

// Need to import getRun for polling
import { getRun } from "../api/client";
