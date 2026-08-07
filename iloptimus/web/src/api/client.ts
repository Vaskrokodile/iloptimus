// API client for IL Optimus backend

export interface HardwareInfo {
  cpu_name: string;
  cpu_cores: number;
  ram_gb: number;
  os: string;
  arch: string;
  gpu: {
    name: string;
    vram_gb: number;
    type: string;
  };
  python_version: string;
  mlx_available: boolean;
  vllm_available: boolean;
  torch_available: boolean;
  recommended_backend: string;
  total_memory_gb: number;
  labels: string[];
}

export interface CompatibilityInfo {
  status: "recommended" | "feasible" | "tight" | "not-recommended";
  best_precision: string;
  best_precision_gb: number;
  reason: string;
  score: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  huggingface_id: string;
  params_b: number;
  fp16_gb: number;
  int8_gb: number;
  int4_gb: number;
  family: string;
  context_length: number;
  backends: string[];
  description: string;
  tags: string[];
  compatibility: CompatibilityInfo;
}

export interface TasksetInfo {
  id: string;
  name: string;
  package_name: string;
  domain: string;
  description: string;
  num_tasks: number;
  needs_sandbox: boolean;
  tags: string[];
  eval_config: Record<string, number>;
}

export interface RunConfig {
  model_id: string;
  taskset_id: string;
  backend: string;
  precision: string;
  sft_iters: number;
  sft_lr: number;
  grpo_iters: number;
  grpo_group_size: number;
  grpo_lr: number;
  grpo_temperature: number;
  max_seq_length: number;
  benchmark_tasks: number;
  rollouts_per_example: number;
}

export interface RunState {
  id: string;
  status: string;
  stage: string;
  progress: number;
  started_at: number;
  elapsed_seconds: number;
  metrics: Record<string, number>;
  baseline_accuracy: number;
  post_sft_accuracy: number;
  post_grpo_accuracy: number;
  sft_loss_history: number[];
  grpo_reward_history: number[];
  config: RunConfig;
}

export interface LogEvent {
  timestamp: number;
  stage: string;
  level: string;
  message: string;
  data: Record<string, any>;
}

const API_BASE = "";

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`);
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function getHardware(): Promise<HardwareInfo> {
  return fetchJSON("/api/hardware");
}

export async function getModels(): Promise<ModelInfo[]> {
  return fetchJSON("/api/models");
}

export async function getTasksets(): Promise<TasksetInfo[]> {
  return fetchJSON("/api/tasksets");
}

export async function getRuns(): Promise<RunState[]> {
  return fetchJSON("/api/runs");
}

export async function getRun(id: string): Promise<RunState> {
  return fetchJSON(`/api/runs/${id}`);
}

export async function createRun(config: Partial<RunConfig>): Promise<{ id: string; status: string }> {
  const res = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Failed to create run: ${await res.text()}`);
  return res.json();
}

export function streamRunEvents(
  runId: string,
  onEvent: (event: LogEvent) => void,
  onError?: (err: Event) => void
): EventSource {
  const es = new EventSource(`/api/runs/${runId}/events`);
  es.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data) as LogEvent;
      onEvent(event);
    } catch (err) {
      console.error("Failed to parse event:", err);
    }
  };
  if (onError) es.onerror = onError;
  return es;
}
