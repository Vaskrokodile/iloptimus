import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Boxes, FlaskConical, Plus, Sparkles, Trash2 } from "lucide-react";
import { deleteEnvironment, getEnvironments, type EnvironmentSpec } from "../api/client";

export default function MyEnvironmentsPage() {
  const [environments, setEnvironments] = useState<EnvironmentSpec[]>([]);
  const navigate = useNavigate();
  const refresh = () => getEnvironments().then(setEnvironments);
  useEffect(() => { refresh(); }, []);
  const remove = async (environment: EnvironmentSpec) => { if (!window.confirm(`Delete “${environment.name}”?`)) return; await deleteEnvironment(environment.id); refresh(); };

  return <section className="product-page">
    <header className="product-hero compact-hero"><div><span className="product-kicker"><Boxes/> My environments</span><h1>Your model’s<br/>training worlds.</h1><p>Everything built in the no-code designer or created through /il and /rl lives here.</p></div><button className="hero-button" onClick={()=>navigate("/pipelines")}><Plus/> New environment</button></header>
    {environments.length===0?<div className="empty-environments"><span><Sparkles/></span><h2>No environments yet.</h2><p>Build one visually, or load a chat model and type <code>/il</code> or <code>/rl</code> followed by your goal.</p><button onClick={()=>navigate("/pipelines")}>Open designer <ArrowRight/></button></div>:<div className="environment-grid">{environments.map(environment=><article className="environment-card" key={environment.id}><header><span className={`env-mode ${environment.mode.toLowerCase()}`}>{environment.mode}</span><button onClick={()=>remove(environment)} aria-label={`Delete ${environment.name}`}><Trash2/></button></header><h3>{environment.name}</h3><p>{environment.description}</p><div className="environment-meta"><span>{environment.domain}</span><span>{environment.tasks.length} tasks</span><span>{environment.interaction.max_steps} step{environment.interaction.max_steps===1?"":"s"}</span></div><div className="reward-mini"><span style={{width:`${environment.reward.correctness*100}%`}}/><span style={{width:`${environment.reward.reasoning*100}%`}}/><span style={{width:`${environment.reward.efficiency*100}%`}}/></div><footer><small>Ready taskset · {environment.taskset_id}</small><button onClick={()=>navigate(`/studio?environment=${environment.taskset_id}`)}><FlaskConical/> Train <ArrowRight/></button></footer></article>)}</div>}
  </section>;
}
