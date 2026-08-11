import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Check, Cpu, Database, Download, Gauge, LoaderCircle, Search, SlidersHorizontal, Sparkles } from "lucide-react";
import { downloadModel, getHardware, getModels, getModelStatus, type HardwareInfo, type ModelInfo } from "../api/client";

export default function ModelLibraryPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => { Promise.all([getModels(), getHardware()]).then(([m, h]) => { setModels(m); setHardware(h); }); }, []);
  const visible = useMemo(() => models.filter((model) => (filter === "all" || model.compatibility.status === filter) && `${model.name} ${model.family}`.toLowerCase().includes(query.toLowerCase())), [models, query, filter]);
  const select = (model: ModelInfo, destination = "/") => { localStorage.setItem("iloptimus-chat-model", JSON.stringify({ id: model.id, name: model.name })); navigate(destination); };
  const updateLocal = (modelId: string, local: ModelInfo["local"]) => setModels((current) => current.map((item) => item.id === modelId ? { ...item, local } : item));
  const download = async (model: ModelInfo) => {
    setError("");
    try {
      updateLocal(model.id, await downloadModel(model.id, model.local.precision));
      const timer = window.setInterval(async () => {
        const local = await getModelStatus(model.id);
        updateLocal(model.id, local);
        if (["downloaded", "failed"].includes(local.status)) window.clearInterval(timer);
      }, 1200);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Download failed"); }
  };

  return <section className="product-page">
    <header className="product-hero compact-hero">
      <div><span className="product-kicker"><Database /> Model library</span><h1>Choose the right mind<br />for the work.</h1><p>Every model is checked against this machine before you load or train it.</p></div>
      {hardware && <div className="hardware-pill"><span className="live-dot" /><div><small>Running on</small><strong>{hardware.gpu.name}</strong></div><div><small>Available</small><strong>{hardware.total_memory_gb.toFixed(1)} GB</strong></div><div><small>Backend</small><strong>{hardware.recommended_backend.toUpperCase()}</strong></div></div>}
    </header>
    <div className="library-toolbar"><div className="library-search"><Search /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search model families or checkpoints" /></div><div className="filter-chips"><SlidersHorizontal />{["all","recommended","feasible","tight"].map((item) => <button key={item} className={filter === item ? "selected" : ""} onClick={() => setFilter(item)}>{item}</button>)}</div></div>
    {error && <div className="form-error">{error}</div>}
    <div className="model-list">
      {visible.map((model, index) => <motion.article key={model.id} initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:index*.025}} className="model-row">
        <div className="model-monogram">{model.family.slice(0,2).toUpperCase()}</div>
        <div className="model-identity"><div><h3>{model.name}</h3><span className={`compat ${model.compatibility.status}`}><Check /> {model.compatibility.status}</span></div><p>{model.description}</p><div className="model-tags">{model.tags.slice(0,3).map((tag) => <span key={tag}>{tag}</span>)}</div></div>
        <div className="model-specs"><div><Cpu /><span><small>Parameters</small><strong>{model.params_b}B</strong></span></div><div><Gauge /><span><small>Best precision</small><strong>{model.compatibility.best_precision}</strong></span></div><div><Database /><span><small>Memory</small><strong>{model.compatibility.best_precision_gb} GB</strong></span></div></div>
        <div className="model-actions">{model.local.status === "downloaded" ? <><button className="quiet-action" onClick={() => select(model, "/studio")}>Train</button><button className="row-primary" onClick={() => select(model)}><Sparkles /> Use for chat <ArrowRight /></button></> : model.compatibility.status === "not-recommended" ? <button className="row-primary download-model" disabled>Doesn’t fit this machine</button> : <button className="row-primary download-model" disabled={["queued","downloading"].includes(model.local.status)} onClick={() => download(model)}>{["queued","downloading"].includes(model.local.status) ? <LoaderCircle className="spin" /> : <Download />} {model.local.status === "failed" ? "Retry download" : ["queued","downloading"].includes(model.local.status) ? `Downloading${model.local.size_gb ? ` · ${model.local.size_gb} GB` : "…"}` : `Download ${model.local.precision.toUpperCase()} · ${model.int4_gb} GB`}</button>}</div>
      </motion.article>)}
    </div>
  </section>;
}
