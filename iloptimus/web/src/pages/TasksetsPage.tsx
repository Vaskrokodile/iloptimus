import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
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
  coding: { icon: Code, color: "text-success", bg: "from-success/20 to-success/5" },
  reasoning: { icon: Brain, color: "text-accent", bg: "from-accent/20 to-accent/5" },
  "agentic-reasoning": { icon: Workflow, color: "text-info", bg: "from-info/20 to-info/5" },
  "agentic-coding": { icon: Terminal, color: "text-warning", bg: "from-warning/20 to-warning/5" },
};

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
};

const cardAnim = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.4, 0, 0.2, 1] as any } },
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
        <div className="w-12 h-12 rounded-2xl shimmer" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold tracking-tight text-fg-primary mb-2">
          IL Tasksets & Environments
        </h1>
        <p className="text-fg-secondary text-sm">
          {tasksets.length} tasksets · {tasksets.reduce((a, t) => a + t.num_tasks, 0)} handcrafted tasks total
        </p>
      </motion.div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-fg-muted" />
        <input
          type="text"
          placeholder="Search tasksets..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="input-base w-full pl-10"
        />
      </div>

      {/* Taskset cards */}
      <motion.div
        variants={stagger}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 lg:grid-cols-2 gap-4"
      >
        {filtered.map((t) => {
          const cfg = domainConfig[t.domain] || domainConfig["coding"];
          const Icon = cfg.icon;
          return (
            <motion.div
              key={t.id}
              variants={cardAnim}
              whileHover={{ y: -3 }}
              className="glass glass-hover rounded-2xl p-6"
            >
              <div className="flex items-start gap-4">
                {/* Domain icon */}
                <div
                  className={`w-12 h-12 rounded-xl bg-gradient-to-br ${cfg.bg} flex items-center justify-center flex-shrink-0`}
                >
                  <Icon className={`w-6 h-6 ${cfg.color}`} strokeWidth={2} />
                </div>

                <div className="flex-1 min-w-0">
                  {/* Header */}
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-semibold text-fg-primary">{t.name}</h3>
                    <span className="text-xs text-fg-muted font-mono">{t.id}</span>
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
                  <p className="text-sm text-fg-secondary mb-4 leading-relaxed">{t.description}</p>

                  {/* Eval config */}
                  <div className="flex items-center gap-4 text-xs text-fg-muted mb-4">
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
                      className="flex items-center gap-1.5 text-accent hover:text-accent-hover text-sm font-medium transition-colors"
                    >
                      <FlaskConical className="w-4 h-4" />
                      Run
                      <ArrowRight className="w-3 h-3" />
                    </Link>
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-fg-muted">No tasksets match your search.</div>
      )}

      {/* Info banner */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="glass rounded-2xl p-5 border-accent/20"
      >
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-xl bg-accent/10 flex items-center justify-center flex-shrink-0">
            <Boxes className="w-4.5 h-4.5 text-accent" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-fg-primary mb-1">About IL Tasksets</h4>
            <p className="text-xs text-fg-secondary leading-relaxed">
              All tasksets use efficiency-aware reward shaping:{" "}
              <code className="text-accent font-mono bg-accent/10 px-1.5 py-0.5 rounded">
                final = correctness × (0.6 + 0.4 × reasoning_quality)
              </code>
              . Wrong answers always get 0. Right answers with lazy reasoning get 0.6. Right answers
              with thorough, verified reasoning get up to 1.0. The 0.4 spread is the RL signal that
              shapes reasoning behavior.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
