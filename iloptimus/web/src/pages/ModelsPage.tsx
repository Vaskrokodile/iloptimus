import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Search,
  Filter,
  Zap,
  Cpu,
} from "lucide-react";
import {
  getModels,
  getHardware,
  type ModelInfo,
  type HardwareInfo,
} from "../api/client";

const compatConfig = {
  recommended: { icon: CheckCircle2, badge: "badge-green", label: "Recommended" },
  feasible: { icon: CheckCircle2, badge: "badge-blue", label: "Feasible" },
  tight: { icon: AlertTriangle, badge: "badge-yellow", label: "Tight" },
  "not-recommended": { icon: XCircle, badge: "badge-red", label: "Not Recommended" },
};

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.04 } },
};

const cardAnim = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.4, 0, 0.2, 1] as any } },
};

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    Promise.all([getModels(), getHardware()])
      .then(([m, h]) => {
        setModels(m);
        setHw(h);
      })
      .finally(() => setLoading(false));
  }, []);

  const filtered = models.filter((m) => {
    const matchesSearch =
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.family.toLowerCase().includes(search.toLowerCase());
    const matchesFilter = filter === "all" || m.compatibility.status === filter;
    return matchesSearch && matchesFilter;
  });

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
        <h1 className="text-3xl font-bold tracking-tight text-fg-primary mb-2">Models</h1>
        <p className="text-fg-secondary text-sm">
          {hw && (
            <>
              Your hardware: <span className="font-medium text-fg-primary">{hw.gpu.name}</span> ·{" "}
              {hw.total_memory_gb.toFixed(1)} GB available ·{" "}
              <span className="text-accent font-medium">{hw.recommended_backend}</span> backend
            </>
          )}
        </p>
      </motion.div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-fg-muted" />
          <input
            type="text"
            placeholder="Search models..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="input-base w-full pl-10"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-fg-muted" />
          {["all", "recommended", "feasible", "tight", "not-recommended"].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                filter === f
                  ? "bg-accent text-white shadow-md shadow-accent/20"
                  : "glass text-fg-secondary hover:text-fg-primary"
              }`}
            >
              {f === "all" ? "All" : compatConfig[f as keyof typeof compatConfig]?.label || f}
            </button>
          ))}
        </div>
      </div>

      {/* Model grid */}
      <motion.div
        variants={stagger}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
      >
        {filtered.map((m) => {
          const cfg = compatConfig[m.compatibility.status];
          const Icon = cfg.icon;
          return (
            <motion.div
              key={m.id}
              variants={cardAnim}
              whileHover={{ y: -3 }}
              className={`glass glass-hover rounded-2xl p-5 ${
                m.compatibility.status === "not-recommended" ? "opacity-60" : ""
              }`}
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent/15 to-accent/5 flex items-center justify-center flex-shrink-0">
                    <Cpu className="w-5 h-5 text-accent" strokeWidth={2} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-fg-primary leading-tight">{m.name}</h3>
                    <p className="text-xs text-fg-muted mt-0.5">{m.huggingface_id}</p>
                  </div>
                </div>
                <span className={cfg.badge}>
                  <Icon className="w-3 h-3" />
                  {cfg.label}
                </span>
              </div>

              {/* Description */}
              <p className="text-sm text-fg-secondary mb-4 min-h-[40px] leading-relaxed">
                {m.description}
              </p>

              {/* Specs grid */}
              <div className="grid grid-cols-4 gap-2 mb-4">
                <Spec label="Params" value={`${m.params_b}B`} />
                <Spec
                  label="FP16"
                  value={`${m.fp16_gb}GB`}
                  highlight={m.compatibility.best_precision === "fp16"}
                />
                <Spec
                  label="INT8"
                  value={`${m.int8_gb}GB`}
                  highlight={m.compatibility.best_precision === "int8"}
                />
                <Spec
                  label="INT4"
                  value={`${m.int4_gb}GB`}
                  highlight={m.compatibility.best_precision === "int4"}
                />
              </div>

              {/* Compatibility reason */}
              <div className="flex items-start gap-2 text-xs text-fg-secondary">
                <Zap className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-accent" />
                <span>{m.compatibility.reason}</span>
              </div>

              {/* Tags */}
              <div className="flex flex-wrap gap-1.5 mt-4">
                {m.tags.map((tag) => (
                  <span key={tag} className="badge badge-gray">
                    {tag}
                  </span>
                ))}
                <span className="badge badge-gray">
                  {m.context_length > 99999 ? "128K ctx" : `${m.context_length / 1000}K ctx`}
                </span>
                {m.backends.map((b) => (
                  <span key={b} className="badge badge-blue">
                    {b}
                  </span>
                ))}
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-fg-muted">No models match your filters.</div>
      )}
    </div>
  );
}

function Spec({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`rounded-lg p-2 text-center transition-all ${
        highlight
          ? "bg-accent/15 border border-accent/30"
          : "bg-bg-glass/30 border border-white/5"
      }`}
    >
      <div className="text-[10px] text-fg-muted font-medium uppercase tracking-wide">{label}</div>
      <div className={`text-sm font-semibold mt-0.5 ${highlight ? "text-accent" : "text-fg-primary"}`}>
        {value}
      </div>
    </div>
  );
}
