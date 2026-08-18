import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { useSearchParams } from "react-router-dom";
import { Activity, AlertTriangle, ArrowRight, Check, Cpu, FlaskConical, FolderOpen, Play, RotateCcw, Sparkles, Terminal, X } from "lucide-react";
import { createRun, getHardware, getModels, getRun, getRuns, getTasksets, preflightRun, type HardwareInfo, type ModelInfo, type RunPreflight, type RunState, type TasksetInfo } from "../api/client";

const stages = ["Prepare", "Load model", "Baseline", "Intuition", "Reinforcement", "Evaluate"];
const presets = {
  quick: { sft: 2, grpo: 1, group: 2, benchmark: 1, batch: 4, reasoning: 64, answer: 64, label: "Quick test", time: "~2–5 min" },
  balanced: { sft: 8, grpo: 4, group: 2, benchmark: 4, batch: 8, reasoning: 192, answer: 96, label: "Balanced", time: "~15–30 min" },
  deep: { sft: 20, grpo: 8, group: 4, benchmark: 8, batch: 16, reasoning: 256, answer: 128, label: "Deep train", time: "~45–90 min" },
};

export default function OptimusLabPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [tasksets, setTasksets] = useState<TasksetInfo[]>([]);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [modelId, setModelId] = useState("");
  const [tasksetId, setTasksetId] = useState("");
  const [preset, setPreset] = useState<keyof typeof presets>("balanced");
  const [run, setRun] = useState<RunState | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [preflight, setPreflight] = useState<RunPreflight | null>(null);
  const [params] = useSearchParams();

  useEffect(() => {
    Promise.all([getModels(), getTasksets(), getHardware(), getRuns()]).then(([modelList, tasksetList, machine, runs]) => {
      setModels(modelList);
      setTasksets(tasksetList);
      setHardware(machine);
      let saved: { id?: string } | null = null;
      try { saved = JSON.parse(localStorage.getItem("iloptimus-chat-model") || "null"); } catch { saved = null; }
      const installed = modelList.filter((item) => item.local.status === "downloaded");
      setModelId(installed.some((item) => item.id === saved?.id) ? saved!.id! : installed[0]?.id || "");
      const requested = params.get("environment");
      setTasksetId(requested && tasksetList.some((item) => item.id === requested) ? requested : tasksetList[0]?.id || "");
      setRun(runs.find((item) => item.status === "running") || null);
    });
  }, []);

  useEffect(() => {
    if (!run || !["pending", "running"].includes(run.status)) return;
    const timer = window.setInterval(() => getRun(run.id).then(setRun), 1500);
    return () => window.clearInterval(timer);
  }, [run?.id, run?.status]);

  const model = useMemo(() => models.find((item) => item.id === modelId), [models, modelId]);
  const taskset = useMemo(() => tasksets.find((item) => item.id === tasksetId), [tasksets, tasksetId]);
  const currentStage = run ? Math.min(stages.length - 1, Math.floor(run.progress * stages.length)) : 0;
  const config = presets[preset];

  const start = async () => {
    if (!modelId || !tasksetId) return;
    setStarting(true); setError(""); setPreflight(null);
    const cfg = { model_id: modelId, taskset_id: tasksetId, sft_iters: config.sft, grpo_iters: config.grpo, grpo_group_size: config.group, benchmark_tasks: config.benchmark, benchmark_batch_size: config.batch, max_reasoning_tokens: config.reasoning, max_answer_tokens: config.answer };
    try {
      const checked = await preflightRun(cfg);
      setPreflight(checked);
      if (!checked.ready) { setStarting(false); return; }
      const created = await createRun(cfg);
      setRun(await getRun(created.id));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not start training"); } finally { setStarting(false); }
  };

  return <section className="product-page lab-page lab-v2">
    <header className="product-hero lab-hero"><div><span className="product-kicker"><FlaskConical /> Optimus Lab</span><h1>Turn practice<br />into instinct.</h1><p>Pick a model, choose its training world, and launch a complete local learning cycle.</p></div>{hardware && <div className="lab-machine"><span className="live-dot" /><div><small>Compute ready</small><strong>{hardware.gpu.name}</strong><p>{hardware.total_memory_gb.toFixed(1)} GB · {hardware.recommended_backend.toUpperCase()}</p></div></div>}</header>

    {!run && <div className="lab-setup">
      <section className="lab-choice-section"><header><span>01</span><div><h2>Choose the model</h2><p>Downloaded models ready for local training are shown here.</p></div></header><div className="choice-card-grid model-choice-grid">{models.filter((item) => item.local.status === "downloaded").slice(0, 6).map((item) => <button key={item.id} className={item.id === modelId ? "selected" : ""} onClick={() => setModelId(item.id)}><span className="choice-check">{item.id === modelId && <Check />}</span><div className="choice-icon"><Cpu /></div><strong>{item.name}</strong><small>{item.params_b}B parameters</small><div><em>{item.local.precision}</em><em>{item.local.size_gb || item.int4_gb} GB local</em></div></button>)}</div>{!models.some((item) => item.local.status === "downloaded") && <div className="lab-empty-choice">Download a compatible model in Model Library before launching training.</div>}</section>
      <section className="lab-choice-section"><header><span>02</span><div><h2>Choose the environment</h2><p>Built-in tasksets and your no-code environments train the same way.</p></div></header><div className="environment-choice-list">{tasksets.map((item) => <button key={item.id} className={item.id === tasksetId ? "selected" : ""} onClick={() => setTasksetId(item.id)}><span className="choice-check">{item.id === tasksetId && <Check />}</span><Sparkles /><div><strong>{item.name}</strong><small>{item.description}</small></div><em>{item.num_tasks} tasks</em></button>)}</div></section>
      <section className="lab-choice-section"><header><span>03</span><div><h2>Choose training depth</h2><p>Start quick, or give the model more time to internalize the behavior.</p></div></header><div className="preset-grid">{Object.entries(presets).map(([key, item]) => <button key={key} className={preset === key ? "selected" : ""} onClick={() => setPreset(key as keyof typeof presets)}><span className="choice-check">{preset === key && <Check />}</span><strong>{item.label}</strong><small>{item.time}</small><div><span>{item.sft} IL steps</span><span>{item.grpo} RL steps</span><span>{item.benchmark} evals</span></div></button>)}</div></section>
      {error && <div className="form-error">{error}</div>}
      {preflight && <div className={`preflight-panel ${preflight.ready ? "preflight-pass" : "preflight-block"}`} role="status"><div className="preflight-head"><strong>{preflight.ready ? "Run preflight passed" : "Run blocked before model load"}</strong><span>{preflight.backend.toUpperCase()} · {preflight.precision}</span></div><div className="preflight-checks">{preflight.checks.map((check) => <div key={check.id} className={`preflight-check preflight-${check.status}`}><span className="preflight-icon">{check.status === "pass" ? <Check /> : check.status === "warn" ? <AlertTriangle /> : <X />}</span><span><strong>{check.label}:</strong> {check.detail}</span></div>)}</div></div>}
      <div className="launch-summary"><div><span className="live-dot" /><div><small>Ready to launch</small><strong>{model?.name || "Download a model"} <ArrowRight /> {taskset?.name || "Choose an environment"}</strong></div></div><button disabled={starting || !model || !taskset} onClick={start}><Play /> {starting ? "Validating run…" : `Start ${config.label.toLowerCase()}`} <ArrowRight /></button></div>
    </div>}

    {run && <main className="lab-console lab-console-active"><header><div><span className={run.status === "running" ? "run-live" : ""}><Activity /> {run.status}</span><h2>{taskset?.name || "Training"} run</h2></div><div className="run-header-actions"><strong>{Math.round(run.progress * 100)}%</strong>{["completed","failed","cancelled"].includes(run.status) && <button onClick={() => setRun(null)}><RotateCcw /> New run</button>}</div></header><div className="stage-track">{stages.map((stage, index) => <div className={`${index < currentStage ? "complete" : ""} ${index === currentStage ? "current" : ""}`} key={stage}><span>{index < currentStage ? <Check /> : index + 1}</span><strong>{stage}</strong></div>)}</div><div className="run-dashboard"><div className="progress-card"><div className="progress-ring" style={{ "--progress": `${run.progress * 360}deg` } as CSSProperties}><span>{Math.round(run.progress * 100)}%</span></div><div><small>Current stage</small><h3>{run.stage.replaceAll("-", " ")}</h3><p>{run.status === "completed" ? "Training complete. Your adapters and metrics are ready." : "Optimus is running the selected environment locally."}</p></div></div><div className="metrics-strip"><div><small>Baseline</small><strong>{(run.baseline_accuracy * 100).toFixed(0)}%</strong></div><div><small>Post-IL</small><strong>{(run.post_sft_accuracy * 100).toFixed(0)}%</strong></div><div><small>Post-RL</small><strong>{(run.post_grpo_accuracy * 100).toFixed(0)}%</strong></div><div><small>Elapsed</small><strong>{Math.round(run.elapsed_seconds)}s</strong></div></div><div className="terminal-preview"><div><Terminal /> Live run state</div><p><span>stage</span> {run.stage}</p><p><span>model</span> {model?.name}</p><p><span>environment</span> {taskset?.name}</p><p><span>backend</span> {hardware?.recommended_backend}</p>{run.artifact_dir && <p><FolderOpen /><span>saved</span> {run.artifact_dir}</p>}{run.manifest?.git?.revision && <p><span>source</span> {run.manifest.git.revision.slice(0, 12)} · {run.manifest.git.dirty ? "working tree dirty" : "clean source"}</p>}</div></div></main>}
  </section>;
}
