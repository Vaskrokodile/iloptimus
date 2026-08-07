import { useEffect, useState } from "react";
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Search,
  Filter,
  Zap,
} from "lucide-react";
import {
  getModels,
  getHardware,
  type ModelInfo,
  type HardwareInfo,
} from "../api/client";

const compatConfig = {
  recommended: {
    icon: CheckCircle2,
    badge: "badge-green",
    label: "Recommended",
    color: "text-green-400",
  },
  feasible: {
    icon: CheckCircle2,
    badge: "badge-blue",
    label: "Feasible",
    color: "text-blue-400",
  },
  tight: {
    icon: AlertTriangle,
    badge: "badge-yellow",
    label: "Tight",
    color: "text-yellow-400",
  },
  "not-recommended": {
    icon: XCircle,
    badge: "badge-red",
    label: "Not Recommended",
    color: "text-red-400",
  },
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
        <div className="text-gray-500 animate-pulse">Loading models...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Models</h1>
        <p className="text-gray-400">
          {hw && (
            <>
              Your hardware: {hw.gpu.name} · {hw.total_memory_gb.toFixed(1)} GB available ·{" "}
              <span className="text-brand-400">{hw.recommended_backend}</span> backend
            </>
          )}
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
          <input
            type="text"
            placeholder="Search models..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg pl-10 pr-4 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-brand-600"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-500" />
          {["all", "recommended", "feasible", "tight", "not-recommended"].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                filter === f
                  ? "bg-brand-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              {f === "all" ? "All" : compatConfig[f as keyof typeof compatConfig]?.label || f}
            </button>
          ))}
        </div>
      </div>

      {/* Model grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((m) => {
          const cfg = compatConfig[m.compatibility.status];
          const Icon = cfg.icon;
          return (
            <div
              key={m.id}
              className={`card hover:border-brand-600/50 transition-colors ${
                m.compatibility.status === "not-recommended" ? "opacity-60" : ""
              }`}
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="font-semibold text-white">{m.name}</h3>
                  <p className="text-xs text-gray-500 mt-0.5">{m.huggingface_id}</p>
                </div>
                <span className={cfg.badge}>
                  <Icon className="w-3 h-3" />
                  {cfg.label}
                </span>
              </div>

              {/* Description */}
              <p className="text-sm text-gray-400 mb-4 min-h-[40px]">{m.description}</p>

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
              <div className={`text-xs ${cfg.color} flex items-start gap-2`}>
                <Zap className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
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
                  {m.context_length > 99999
                    ? "128K ctx"
                    : `${m.context_length / 1000}K ctx`}
                </span>
                {m.backends.map((b) => (
                  <span key={b} className="badge badge-blue">
                    {b}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          No models match your filters.
        </div>
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
      className={`rounded-lg p-2 text-center ${
        highlight ? "bg-brand-600/20 border border-brand-600/40" : "bg-gray-800/50"
      }`}
    >
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-sm font-medium ${highlight ? "text-brand-300" : "text-gray-200"}`}>
        {value}
      </div>
    </div>
  );
}
