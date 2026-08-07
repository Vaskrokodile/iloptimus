import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Cpu,
  MemoryStick,
  HardDrive,
  Zap,
  FlaskConical,
  Boxes,
  ArrowRight,
  Activity,
} from "lucide-react";
import {
  getHardware,
  getModels,
  getTasksets,
  type HardwareInfo,
  type ModelInfo,
  type TasksetInfo,
} from "../api/client";

export default function DashboardPage() {
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [tasksets, setTasksets] = useState<TasksetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getHardware(), getModels(), getTasksets()])
      .then(([h, m, t]) => {
        setHw(h);
        setModels(m);
        setTasksets(t);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500 animate-pulse">Detecting hardware...</div>
      </div>
    );
  }

  const recommendedModels = models.filter(
    (m) => m.compatibility.status === "recommended"
  );
  const feasibleModels = models.filter(
    (m) => m.compatibility.status === "feasible"
  );

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="text-center py-8">
        <h1 className="text-4xl font-bold text-white mb-3">
          Intuition Learning Pipeline Studio
        </h1>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Run SFT + GRPO RL pipelines on your local hardware. Detect your GPU,
          pick a compatible model, select a taskset, and train.
        </p>
      </div>

      {/* Hardware summary */}
      {hw && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-5 h-5 text-brand-400" />
            <h2 className="text-xl font-semibold text-white">Hardware Detected</h2>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <HwCard
              icon={Cpu}
              label="CPU"
              value={hw.cpu_name}
              subvalue={`${hw.cpu_cores} cores`}
            />
            <HwCard
              icon={MemoryStick}
              label="Memory"
              value={`${hw.ram_gb} GB`}
              subvalue={hw.gpu.type === "apple-silicon" ? "Unified" : "RAM"}
            />
            <HwCard
              icon={Zap}
              label="GPU"
              value={hw.gpu.name}
              subvalue={
                hw.gpu.type === "apple-silicon"
                  ? `${hw.gpu.vram_gb} GB unified`
                  : hw.gpu.type === "cuda"
                  ? `${hw.gpu.vram_gb} GB VRAM`
                  : "None"
              }
            />
            <HwCard
              icon={HardDrive}
              label="Backend"
              value={hw.recommended_backend.toUpperCase()}
              subvalue={`${hw.total_memory_gb.toFixed(1)} GB for models`}
            />
          </div>
          <div className="flex flex-wrap gap-2 mt-4">
            {hw.labels.map((label) => (
              <span key={label} className="badge badge-blue">
                {label}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Quick stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          icon={Cpu}
          label="Compatible Models"
          value={`${recommendedModels.length + feasibleModels.length}`}
          subvalue={`${recommendedModels.length} recommended, ${feasibleModels.length} feasible`}
          to="/models"
        />
        <StatCard
          icon={Boxes}
          label="Tasksets Available"
          value={`${tasksets.length}`}
          subvalue={`${tasksets.reduce((a, t) => a + t.num_tasks, 0)} total tasks`}
          to="/tasksets"
        />
        <StatCard
          icon={FlaskConical}
          label="IL Studio"
          value="Ready"
          subvalue="Start a training run"
          to="/studio"
        />
      </div>

      {/* Recommended models preview */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-white">
            Recommended for Your Hardware
          </h2>
          <Link
            to="/models"
            className="text-brand-400 hover:text-brand-300 text-sm flex items-center gap-1"
          >
            View all <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {recommendedModels.slice(0, 6).map((m) => (
            <div
              key={m.id}
              className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-4 hover:border-brand-600/50 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-white text-sm">{m.name}</span>
                <span className="badge badge-green">{m.compatibility.best_precision}</span>
              </div>
              <p className="text-xs text-gray-400">{m.params_b}B params · {m.compatibility.best_precision_gb}GB</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function HwCard({
  icon: Icon,
  label,
  value,
  subvalue,
}: {
  icon: any;
  label: string;
  value: string;
  subvalue: string;
}) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-4">
      <div className="flex items-center gap-2 text-gray-400 text-xs mb-2">
        <Icon className="w-4 h-4" />
        {label}
      </div>
      <div className="text-white font-medium text-sm truncate">{value}</div>
      <div className="text-gray-500 text-xs mt-1">{subvalue}</div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  subvalue,
  to,
}: {
  icon: any;
  label: string;
  value: string;
  subvalue: string;
  to: string;
}) {
  return (
    <Link
      to={to}
      className="card hover:border-brand-600/50 transition-colors group"
    >
      <div className="flex items-center gap-2 text-gray-400 text-sm mb-3">
        <Icon className="w-5 h-5" />
        {label}
      </div>
      <div className="text-3xl font-bold text-white mb-1">{value}</div>
      <div className="text-gray-500 text-sm">{subvalue}</div>
      <div className="mt-3 text-brand-400 text-xs flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        Open <ArrowRight className="w-3 h-3" />
      </div>
    </Link>
  );
}
