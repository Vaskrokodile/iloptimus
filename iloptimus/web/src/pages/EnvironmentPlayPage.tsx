import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Check, Play, RotateCcw, Terminal, X } from "lucide-react";
import { getEnvironment, resetSimulation, stepSimulation, type EnvironmentSpec, type SimulationStep } from "../api/client";

export default function EnvironmentPlayPage() {
  const { environmentId = "" } = useParams();
  const [environment, setEnvironment] = useState<EnvironmentSpec | null>(null);
  const [scenario, setScenario] = useState(0);
  const [state, setState] = useState<SimulationStep | null>(null);
  const [history, setHistory] = useState<Array<{action:string;reward:number;observation:string}>>([]);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => { getEnvironment(environmentId).then(setEnvironment).catch((cause) => setError(cause instanceof Error ? cause.message : "Could not load environment")); }, [environmentId]);

  const reset = async (nextScenario = scenario) => {
    setError("");
    try {
      const result = await resetSimulation(environmentId, nextScenario);
      setScenario(nextScenario);
      setState(result);
      setHistory([]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not reset simulation");
    }
  };

  const act = async (action: string) => {
    if (!state?.session_id || state.terminated) return;
    try {
      const result = await stepSimulation(environmentId, state.session_id, action);
      setHistory((current) => [...current, {action,reward:result.reward,observation:result.observation}]);
      setState({...result, session_id: state.session_id});
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The action failed");
    }
  };

  if (!environment) return <section className="product-page simulator-page"><p>{error || "Loading simulator…"}</p></section>;
  return <section className="product-page simulator-page">
    <button className="sim-back" onClick={()=>navigate("/environments")}><ArrowLeft/> My environments</button>
    <header className="sim-header"><div><span className="product-kicker"><Terminal/> Live state machine</span><h1>{environment.name}</h1><p>{environment.goal}</p></div><button className="hero-button" onClick={()=>navigate(`/studio?environment=${environment.taskset_id}`)}>Train this environment</button></header>
    <div className="sim-shell">
      <aside className="sim-scenarios"><strong>Episodes</strong>{environment.simulator?.scenarios.map((item,index)=><button key={item.name} className={scenario===index?"selected":""} onClick={()=>reset(index)}><span>{index+1}</span><div>{item.name}<small>{item.solution.length} known optimal steps</small></div></button>)}</aside>
      <main className="sim-console">
        {!state?<div className="sim-start"><Play/><h2>Run the environment.</h2><p>Reset creates a real episode with isolated state. Every action executes validated transition rules.</p><button onClick={()=>reset()}>Reset episode</button></div>:<><div className="sim-status"><span className={state.terminated?(state.success?"success":"failure"):"running"}>{state.terminated?(state.success?<><Check/> Success</>:<><X/> {state.outcome}</>):<>Step {state.step} / {environment.simulator?.max_steps}</>}</span><button onClick={()=>reset()}><RotateCcw/> Reset</button></div><div className="observation-card"><small>Observation</small><strong>{state.observation}</strong></div><div className="state-grid">{Object.entries(state.state).map(([key,value])=><div key={key}><small>{key}</small><strong>{String(value)}</strong></div>)}</div><div className="action-panel"><small>Available actions</small><div>{state.actions.map(action=><button key={action} disabled={state.terminated} onClick={()=>act(action)}>{action.replaceAll("_"," ")}</button>)}</div></div><div className="trajectory-log"><small>Trajectory</small>{history.length===0?<p>No actions yet.</p>:history.map((item,index)=><div key={`${item.action}-${index}`}><span>{index+1}</span><strong>{item.action}</strong><em>{item.reward>0?"+":""}{item.reward.toFixed(2)}</em><small>{item.observation}</small></div>)}</div></>}
        {error&&<div className="form-error">{error}</div>}
      </main>
    </div>
  </section>;
}
