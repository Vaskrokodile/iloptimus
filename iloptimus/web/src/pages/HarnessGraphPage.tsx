import { useEffect, useState, useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import {
  Network,
  TrendingUp,
  Activity,
  Target,
  Zap,
  RefreshCw,
  Trash2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Wrench,
  Sparkles,
  Brain,
  Layers,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";
import {
  getHarnessGraph,
  getHarnessGraphEfficiency,
  getHarnessGraphTopActions,
  ingestToolLogs,
  resetHarnessGraph,
  type HarnessGraph,
  type EfficiencySnapshot,
  type ActionNode,
} from "../api/client";

const ACTION_ICONS: Record<string, typeof Wrench> = {
  tool_call: Wrench,
  tool_success: CheckCircle2,
  tool_failure: XCircle,
  skill_created: Sparkles,
  skill_deleted: Trash2,
  skill_used: Brain,
  skill_retrieved: Brain,
  duplicate_call: AlertTriangle,
  good_action: CheckCircle2,
  mistake: AlertTriangle,
  uncertainty_detected: AlertTriangle,
  learning_triggered: Brain,
  learning_succeeded: CheckCircle2,
  learning_failed: XCircle,
  artifact_generated: Layers,
  artifact_verified: CheckCircle2,
  artifact_rejected: XCircle,
};

const CATEGORY_COLORS: Record<string, string> = {
  action: "rgb(129,140,248)",
  mistake: "rgb(248,113,113)",
};

const tooltipStyle = {
  backgroundColor: "rgba(20,25,42,0.95)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "12px",
  backdropFilter: "blur(20px)",
};

export default function HarnessGraphPage() {
  const [graph, setGraph] = useState<HarnessGraph | null>(null);
  const [efficiency, setEfficiency] = useState<EfficiencySnapshot[]>([]);
  const [topActions, setTopActions] = useState<ActionNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [g, eff, top] = await Promise.all([
      getHarnessGraph(),
      getHarnessGraphEfficiency(500),
      getHarnessGraphTopActions(30),
    ]);
    setGraph(g);
    setEfficiency(eff);
    setTopActions(top);
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh().catch(() => setLoading(false));
  }, [refresh]);

  const handleIngest = async () => {
    setIngesting(true);
    try {
      await ingestToolLogs();
      await refresh();
    } finally {
      setIngesting(false);
    }
  };

  const handleReset = async () => {
    if (!window.confirm("Reset the entire harness graph? All tracked actions and weights will be cleared.")) return;
    await resetHarnessGraph();
    await refresh();
  };

  const efficiencyData = useMemo(
    () => efficiency.map((e, i) => ({
      idx: i + 1,
      efficiency: e.efficiency * 100,
      successRate: e.success_rate * 100,
      totalActions: e.total_actions,
      totalTasks: e.total_tasks,
      timestamp: e.timestamp,
    })),
    [efficiency]
  );

  const actionBreakdown = useMemo(() => {
    if (!graph) return [];
    const byType: Record<string, { type: string; count: number; weight: number; successes: number; failures: number }> = {};
    for (const node of graph.nodes) {
      const existing = byType[node.action_type] || { type: node.action_type, count: 0, weight: 0, successes: 0, failures: 0 };
      existing.count += 1;
      existing.weight += node.weight;
      existing.successes += node.successes;
      existing.failures += node.failures;
      byType[node.action_type] = existing;
    }
    return Object.values(byType).sort((a, b) => b.count - a.count);
  }, [graph]);

  const scatterData = useMemo(() => {
    if (!graph) return [];
    return graph.nodes
      .filter((n) => n.observations > 0)
      .map((n) => ({
        x: n.observations,
        y: n.weight * 100,
        z: n.successes + n.failures,
        label: n.label,
        key: n.key,
        category: n.category,
      }));
  }, [graph]);

  const currentEfficiency = efficiency.length > 0 ? efficiency[efficiency.length - 1].efficiency : 0;
  const currentSuccessRate = efficiency.length > 0 ? efficiency[efficiency.length - 1].success_rate : 0;

  if (loading) {
    return (
      <section className="product-page">
        <div className="glass rounded-2xl p-12 flex items-center justify-center">
          <RefreshCw className="w-6 h-6 text-accent animate-spin" />
          <span className="ml-3 text-fg-muted">Loading harness graph…</span>
        </div>
      </section>
    );
  }

  return (
    <section className="product-page">
      <header className="product-hero compact-hero">
        <div>
          <span className="product-kicker"><Network /> Harness graph</span>
          <h1>Algorithmic harness<br />self-improvement.</h1>
          <p>
            Every tool call, skill, mistake, and good action the model takes is tracked as a graph node
            with an adaptive weight linked to task outcomes. Weights converge toward each action's true
            success contribution — never fixed, always adapting.
          </p>
        </div>
        <div className="flex flex-col gap-2">
          <button className="hero-button" onClick={refresh}>
            <RefreshCw /> Refresh
          </button>
          <button className="hero-button" onClick={handleIngest} disabled={ingesting}>
            {ingesting ? <RefreshCw className="animate-spin" /> : <Zap />} Ingest tool logs
          </button>
          <button className="hero-button danger" onClick={handleReset}>
            <Trash2 /> Reset graph
          </button>
        </div>
      </header>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard
          icon={TrendingUp}
          label="Efficiency"
          value={`${(currentEfficiency * 100).toFixed(1)}%`}
          subtitle="Weighted success"
          color="rgb(52,211,153)"
        />
        <StatCard
          icon={Target}
          label="Success rate"
          value={`${(currentSuccessRate * 100).toFixed(1)}%`}
          subtitle="Raw task outcomes"
          color="rgb(129,140,248)"
        />
        <StatCard
          icon={Activity}
          label="Actions tracked"
          value={graph?.total_actions?.toString() ?? "0"}
          subtitle="Distinct action nodes"
          color="rgb(251,191,36)"
        />
        <StatCard
          icon={Layers}
          label="Tasks resolved"
          value={graph?.total_tasks?.toString() ?? "0"}
          subtitle={`${graph?.pending_tasks ?? 0} pending`}
          color="rgb(244,114,182)"
        />
      </div>

      {/* Efficiency time series */}
      {efficiencyData.length > 0 && (
        <div className="glass rounded-2xl p-5 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-accent" />
            <h3 className="text-sm font-semibold text-fg-primary">Model success efficiency over time</h3>
            <span className="text-xs text-fg-muted ml-auto">{efficiencyData.length} snapshots</span>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={efficiencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="idx" stroke="rgba(148,163,184,0.5)" fontSize={10} />
              <YAxis domain={[0, 100]} stroke="rgba(148,163,184,0.5)" fontSize={10} unit="%" />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="efficiency" stroke="rgb(52,211,153)" strokeWidth={2} dot={false} name="Efficiency" />
              <Line type="monotone" dataKey="successRate" stroke="rgb(129,140,248)" strokeWidth={2} dot={false} name="Success rate" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Two-column: action breakdown + weight scatter */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        {actionBreakdown.length > 0 && (
          <div className="glass rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="w-5 h-5 text-accent" />
              <h3 className="text-sm font-semibold text-fg-primary">Action type breakdown</h3>
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={actionBreakdown} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                <XAxis type="number" stroke="rgba(148,163,184,0.5)" fontSize={10} />
                <YAxis type="category" dataKey="type" stroke="rgba(148,163,184,0.5)" fontSize={9} width={120} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {actionBreakdown.map((entry, i) => {
                    const node = graph?.nodes.find((n) => n.action_type === entry.type);
                    const color = node ? CATEGORY_COLORS[node.category] : "rgb(129,140,248)";
                    return <Cell key={i} fill={color} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {scatterData.length > 0 && (
          <div className="glass rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Network className="w-5 h-5 text-accent" />
              <h3 className="text-sm font-semibold text-fg-primary">
                Adaptive weight landscape
              </h3>
              <span className="text-xs text-fg-muted ml-auto">observations vs weight</span>
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                <XAxis type="number" dataKey="x" name="observations" stroke="rgba(148,163,184,0.5)" fontSize={10} />
                <YAxis type="number" dataKey="y" name="weight" domain={[0, 100]} stroke="rgba(148,163,184,0.5)" fontSize={10} unit="%" />
                <ZAxis type="number" dataKey="z" range={[30, 400]} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  cursor={{ strokeDasharray: "3 3" }}
                  formatter={(value: number, name: string) => [value, name]}
                  labelFormatter={(_, payload) => {
                    const item = payload?.[0]?.payload;
                    return item ? item.label : "";
                  }}
                />
                <Scatter data={scatterData}>
                  {scatterData.map((entry, i) => (
                    <Cell key={i} fill={CATEGORY_COLORS[entry.category] || "rgb(129,140,248)"} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Top actions table */}
      {topActions.length > 0 && (
        <div className="glass rounded-2xl p-5 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-5 h-5 text-accent" />
            <h3 className="text-sm font-semibold text-fg-primary">Top actions by observation count</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-fg-muted text-xs border-b border-white/5">
                  <th className="text-left py-2 px-3">Action</th>
                  <th className="text-left py-2 px-3">Type</th>
                  <th className="text-right py-2 px-3">Weight</th>
                  <th className="text-right py-2 px-3">Obs</th>
                  <th className="text-right py-2 px-3">Success</th>
                  <th className="text-right py-2 px-3">Fail</th>
                  <th className="text-left py-2 px-3 w-32">Success bar</th>
                </tr>
              </thead>
              <tbody>
                {topActions.map((node) => {
                  const Icon = ACTION_ICONS[node.action_type] || Activity;
                  const successPct = node.observations > 0 ? (node.successes / node.observations) * 100 : 0;
                  return (
                    <tr
                      key={node.key}
                      className={`border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors ${selectedNode === node.key ? "bg-white/10" : ""}`}
                      onClick={() => setSelectedNode(selectedNode === node.key ? null : node.key)}
                    >
                      <td className="py-2 px-3">
                        <div className="flex items-center gap-2">
                          <Icon className="w-4 h-4 text-fg-muted" />
                          <span className="text-fg-primary">{node.label}</span>
                        </div>
                      </td>
                      <td className="py-2 px-3">
                        <span className={`text-xs px-2 py-0.5 rounded-full ${node.category === "mistake" ? "bg-red-500/20 text-red-300" : "bg-indigo-500/20 text-indigo-300"}`}>
                          {node.category}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-fg-primary">{(node.weight * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3 text-right font-mono text-fg-muted">{node.observations}</td>
                      <td className="py-2 px-3 text-right font-mono text-green-400">{node.successes}</td>
                      <td className="py-2 px-3 text-right font-mono text-red-400">{node.failures}</td>
                      <td className="py-2 px-3">
                        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${successPct}%`,
                              backgroundColor: node.category === "mistake" ? "rgb(248,113,113)" : "rgb(52,211,153)",
                            }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Co-occurrence edges */}
      {graph && graph.edges.length > 0 && (
        <div className="glass rounded-2xl p-5 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Network className="w-5 h-5 text-accent" />
            <h3 className="text-sm font-semibold text-fg-primary">Action co-occurrence edges</h3>
            <span className="text-xs text-fg-muted ml-auto">{graph.edges.length} edges</span>
          </div>
          <div className="overflow-x-auto max-h-80 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-[rgba(20,25,42,0.95)] backdrop-blur-sm">
                <tr className="text-fg-muted text-xs border-b border-white/5">
                  <th className="text-left py-2 px-3">Source</th>
                  <th className="text-left py-2 px-3">Target</th>
                  <th className="text-right py-2 px-3">Co-occur</th>
                  <th className="text-right py-2 px-3">Joint success</th>
                  <th className="text-right py-2 px-3">Weight</th>
                </tr>
              </thead>
              <tbody>
                {graph.edges.slice(0, 100).map((edge, i) => (
                  <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                    <td className="py-2 px-3 text-fg-primary">{edge.source}</td>
                    <td className="py-2 px-3 text-fg-primary">{edge.target}</td>
                    <td className="py-2 px-3 text-right font-mono text-fg-muted">{edge.co_occurrences}</td>
                    <td className="py-2 px-3 text-right font-mono text-green-400">{edge.joint_successes}</td>
                    <td className="py-2 px-3 text-right font-mono text-fg-primary">{(edge.weight * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent tasks */}
      {graph && graph.tasks.length > 0 && (
        <div className="glass rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <Target className="w-5 h-5 text-accent" />
            <h3 className="text-sm font-semibold text-fg-primary">Recent task outcomes</h3>
            <span className="text-xs text-fg-muted ml-auto">{graph.total_tasks} total</span>
          </div>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {graph.tasks.slice(0, 50).map((task) => (
              <motion.div
                key={task.id}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-3 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
              >
                {task.success ? (
                  <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-fg-muted font-mono">{task.id}</span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-white/10 text-fg-muted">{task.kind}</span>
                  </div>
                  <div className="text-xs text-fg-muted mt-0.5">
                    {task.action_keys.length} action{task.action_keys.length === 1 ? "" : "s"} · score {(task.score * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="flex gap-1 flex-wrap max-w-xs">
                  {task.action_keys.slice(0, 4).map((key) => (
                    <span key={key} className="text-xs px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 truncate max-w-32">
                      {key}
                    </span>
                  ))}
                  {task.action_keys.length > 4 && (
                    <span className="text-xs text-fg-muted">+{task.action_keys.length - 4}</span>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {graph && graph.total_actions === 0 && graph.total_tasks === 0 && (
        <div className="glass rounded-2xl p-12 text-center">
          <Network className="w-12 h-12 text-fg-muted mx-auto mb-4" />
          <h2 className="text-lg font-semibold text-fg-primary mb-2">No harness data yet.</h2>
          <p className="text-fg-muted mb-6 max-w-md mx-auto">
            The graph populates as the model interacts with the harness — every chat, tool call, skill
            creation, and learning session feeds adaptive weights. Click "Ingest tool logs" to bootstrap
            from existing audit data.
          </p>
          <button className="hero-button inline-flex" onClick={handleIngest} disabled={ingesting}>
            {ingesting ? <RefreshCw className="animate-spin" /> : <Zap />} Ingest existing tool logs
          </button>
        </div>
      )}
    </section>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  subtitle,
  color,
}: {
  icon: typeof TrendingUp;
  label: string;
  value: string;
  subtitle: string;
  color: string;
}) {
  return (
    <div className="glass rounded-2xl p-4">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4" style={{ color }} />
        <span className="text-xs text-fg-muted">{label}</span>
      </div>
      <div className="text-2xl font-bold text-fg-primary">{value}</div>
      <div className="text-xs text-fg-muted mt-1">{subtitle}</div>
    </div>
  );
}
