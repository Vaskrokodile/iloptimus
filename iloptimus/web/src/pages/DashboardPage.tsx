import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Cpu,
  MemoryStick,
  HardDrive,
  Zap,
  FlaskConical,
  Boxes,
  ArrowRight,
  Activity,
  Sparkles,
} from "lucide-react";
import {
  getHardware,
  getModels,
  getTasksets,
  type HardwareInfo,
  type ModelInfo,
  type TasksetInfo,
} from "../api/client";

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.4, 0, 0.2, 1] as any } },
};

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
        <div className="space-y-3">
          <div className="w-12 h-12 rounded-2xl shimmer mx-auto" />
          <div className="text-fg-muted text-sm animate-pulse">Detecting hardware...</div>
        </div>
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
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-8">
      {/* Hero */}
      <motion.div variants={fadeUp} className="text-center py-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/20 mb-4">
          <Sparkles className="w-3.5 h-3.5 text-accent" />
          <span className="text-xs font-medium text-accent">Local Harness for Open-Source Models</span>
        </div>
        <h1 className="text-4xl font-bold tracking-tight text-fg-primary mb-3">
          Train smarter, locally.
        </h1>
        <p className="text-fg-secondary text-lg max-w-2xl mx-auto">
          A full local harness for open-source models. Run IL (SFT + GRPO RL)
          pipelines and PQLoRA (parameter-targeted QLoRA) on your hardware.
          Detect your GPU, pick a compatible model, select a taskset, and train.
        </p>
      </motion.div>

      {/* Hardware summary */}
      {hw && (
        <motion.div variants={fadeUp} className="glass glass-hover rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-5">
            <Activity className="w-5 h-5 text-accent" />
            <h2 className="text-lg font-semibold text-fg-primary">Hardware Detected</h2>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <HwCard icon={Cpu} label="CPU" value={hw.cpu_name} subvalue={`${hw.cpu_cores} cores`} />
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
          <div className="flex flex-wrap gap-2 mt-5">
            {hw.labels.map((label) => (
              <span key={label} className="badge badge-accent">
                {label}
              </span>
            ))}
          </div>
        </motion.div>
      )}

      {/* Quick stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          icon={Cpu}
          label="Compatible Models"
          value={`${recommendedModels.length + feasibleModels.length}`}
          subvalue={`${recommendedModels.length} recommended · ${feasibleModels.length} feasible`}
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
      <motion.div variants={fadeUp} className="glass rounded-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-fg-primary">
            Recommended for Your Hardware
          </h2>
          <Link
            to="/models"
            className="text-accent hover:text-accent-hover text-sm flex items-center gap-1 font-medium transition-colors"
          >
            View all <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {recommendedModels.slice(0, 6).map((m, i) => (
            <motion.div
              key={m.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.05 }}
              className="glass glass-hover rounded-xl p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-fg-primary text-sm">{m.name}</span>
                <span className="badge badge-green">{m.compatibility.best_precision}</span>
              </div>
              <p className="text-xs text-fg-muted">
                {m.params_b}B params · {m.compatibility.best_precision_gb}GB
              </p>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </motion.div>
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
    <div className="rounded-xl p-4 bg-bg-glass/40 border border-white/5">
      <div className="flex items-center gap-2 text-fg-muted text-xs mb-2">
        <Icon className="w-4 h-4" strokeWidth={2} />
        {label}
      </div>
      <div className="text-fg-primary font-medium text-sm truncate">{value}</div>
      <div className="text-fg-muted text-xs mt-1">{subvalue}</div>
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
    <Link to={to} className="glass glass-hover rounded-2xl p-6 group block">
      <div className="flex items-center gap-2 text-fg-muted text-sm mb-3">
        <Icon className="w-5 h-5" strokeWidth={2} />
        {label}
      </div>
      <div className="text-3xl font-bold text-fg-primary mb-1 tracking-tight">{value}</div>
      <div className="text-fg-muted text-sm">{subvalue}</div>
      <div className="mt-3 text-accent text-xs flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        Open <ArrowRight className="w-3 h-3" />
      </div>
    </Link>
  );
}
