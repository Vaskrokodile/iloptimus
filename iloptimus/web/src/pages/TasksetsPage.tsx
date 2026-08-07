import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Boxes,
  Code,
  Brain,
  Workflow,
  Terminal,
  Search,
  ArrowRight,
  FlaskConical,
} from "lucide-react";
import { getTasksets, type TasksetInfo } from "../api/client";

const domainConfig: Record<string, { icon: any; color: string; bg: string }> = {
  coding: { icon: Code, color: "text-green-400", bg: "bg-green-500/10" },
  reasoning: { icon: Brain, color: "text-purple-400", bg: "bg-purple-500/10" },
  "agentic-reasoning": { icon: Workflow, color: "text-blue-400", bg: "bg-blue-500/10" },
  "agentic-coding": { icon: Terminal, color: "text-orange-400", bg: "bg-orange-500/10" },
};

export default function TasksetsPage() {
  const [tasksets, setTasksets] = useState<TasksetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");

  useEffect(() => {
    getTasksets()
      .then(setTasksets)
      .finally(() => setLoading(false));
  }, []);

  const filtered = tasksets.filter(
    (t) =>
      t.name.toLowerCase().includes(search.toLowerCase()) ||
      t.domain.toLowerCase().includes(search.toLowerCase()) ||
      t.description.toLowerCase().includes(search.toLowerCase())
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500 animate-pulse">Loading tasksets...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">IL Tasksets & Environments</h1>
        <p className="text-gray-400">
          {tasksets.length} tasksets · {tasksets.reduce((a, t) => a + t.num_tasks, 0)} handcrafted tasks total
        </p>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
        <input
          type="text"
          placeholder="Search tasksets..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded-lg pl-10 pr-4 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-brand-600"
        />
      </div>

      {/* Taskset cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filtered.map((t) => {
          const cfg = domainConfig[t.domain] || domainConfig["coding"];
          const Icon = cfg.icon;
          return (
            <div
              key={t.id}
              className="card hover:border-brand-600/50 transition-colors"
            >
              <div className="flex items-start gap-4">
                {/* Domain icon */}
                <div className={`w-12 h-12 rounded-xl ${cfg.bg} flex items-center justify-center flex-shrink-0`}>
                  <Icon className={`w-6 h-6 ${cfg.color}`} />
                </div>

                <div className="flex-1 min-w-0">
                  {/* Header */}
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-semibold text-white">{t.name}</h3>
                    <span className="text-xs text-gray-500 font-mono">{t.id}</span>
                  </div>

                  {/* Domain + sandbox */}
                  <div className="flex items-center gap-2 mb-3">
                    <span className="badge badge-gray">{t.domain}</span>
                    {t.needs_sandbox ? (
                      <span className="badge badge-yellow">sandbox</span>
                    ) : (
                      <span className="badge badge-green">no sandbox</span>
                    )}
                    <span className="badge badge-blue">{t.num_tasks} tasks</span>
                  </div>

                  {/* Description */}
                  <p className="text-sm text-gray-400 mb-4">{t.description}</p>

                  {/* Eval config */}
                  <div className="flex items-center gap-4 text-xs text-gray-500 mb-4">
                    <span>
                      Default: {t.eval_config.num_examples} tasks × {t.eval_config.rollouts_per_example} rollouts
                    </span>
                  </div>

                  {/* Tags + CTA */}
                  <div className="flex items-center justify-between">
                    <div className="flex flex-wrap gap-1.5">
                      {t.tags.map((tag) => (
                        <span key={tag} className="badge badge-gray">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <Link
                      to="/studio"
                      className="flex items-center gap-1 text-brand-400 hover:text-brand-300 text-sm font-medium"
                    >
                      <FlaskConical className="w-4 h-4" />
                      Run
                      <ArrowRight className="w-3 h-3" />
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          No tasksets match your search.
        </div>
      )}

      {/* Info banner */}
      <div className="card bg-brand-600/5 border-brand-600/20">
        <div className="flex items-start gap-3">
          <Boxes className="w-5 h-5 text-brand-400 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-white mb-1">About IL Tasksets</h4>
            <p className="text-xs text-gray-400">
              All tasksets use efficiency-aware reward shaping:{" "}
              <code className="text-brand-300">final = correctness × (0.6 + 0.4 × reasoning_quality)</code>.
              Wrong answers always get 0. Right answers with lazy reasoning get 0.6. Right answers with
              thorough, verified reasoning get up to 1.0. The 0.4 spread is the RL signal that shapes
              reasoning behavior.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
