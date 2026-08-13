# Failure-driven test-time compute

IL Optimus treats test-time adaptation as a measured experiment, not as a claim
that any training run improved a model.

For explicit `/ttc <artifact task>` and artifact-shaped `/learn` requests, the
controller performs this sequence:

1. derive an observable artifact contract from the request;
2. reserve the exact request as a holdout and generate a base-model artifact;
3. check source depth, placeholder/stub markers, JavaScript syntax, requested
   capabilities, real browser execution, and non-blank pixels;
4. if the baseline fails, ask the selected local model to author web-search
   queries, then split the artifact contract into persistent capability and
   integration subtasks;
5. give every subtask a minimum file count, minimum independent-origin count,
   required evidence kinds, focused queries, status, and a mechanical audit.
   Niche visual subjects such as Sakura and islands additionally require at
   least two topic-specific files from two independent origins, so generic
   particle or scene code cannot satisfy the subject audit;
6. fetch relevant documentation and shallow-clone public GitHub repositories;
7. admit repository code only when the repository has a recognized MIT,
   Apache-2.0, BSD, or ISC license;
8. retry a failed subtask with gap-specific queries, then stop before training
   if relevant quantity, evidence kinds, or independent origins still fail;
9. run the real `scrape_source`, `assemble_dataset`, `expand_dataset`, and
   `filter_dataset` tools. Repository scraping verifies licensing; assembly
   trains only on code-like licensed repository blobs; expansion creates
   deterministic multi-kilobyte syntax-aware source units; filtering removes exact and
   five-token-shingle near duplicates, contamination, tiny rows, and source
   domination;
10. audit the filtered dataset again: every requested capability must have at
   least four examples from two files and two independent repository origins;
11. select retrieval, LoRA-IL, QLoRA-IL, or IL+GRPO from evidence, model,
   hardware, and rollout properties.
   GRPO is deliberately excluded from one-shot artifact generation because it
   does not expose a stable multi-step rollout process; `PQLoRA` is not treated
   as a separate objective without an implemented algorithm;
12. train in a memory-isolated subprocess, retry the untouched task using the
   identical prompt and deterministic decoding budget, and execute the verifier again;
13. retain an adapter only when the retry passes every hard gate and improves
    by at least 0.05.
14. when a small-model adapter is rejected, optionally provide a separately
    labeled trusted framework artifact. It is independently verified and never
    counted as model improvement or used to accept the adapter.

Run artifacts are saved under `~/.iloptimus/learning/<session-id>/`, including
the baseline and retry source, screenshots, research manifest, JSONL dataset,
dataset/contamination manifest, experiment record, and acceptance decision. The
LoRA adapter and training telemetry remain under `~/.iloptimus/runs/<run-id>/`.

## MLX training path

SFT uses `mlx_lm.tuner.trainer.train`, so the complete loss/backward/update
graph is compiled. Tokenization is cached, examples are length sorted, and
stable compile buckets prevent a short run from compiling a separate graph for
every 32-token length. The base model stays quantized and frozen; only selected
attention projections receive LoRA adapters. Prompt tokens are masked and the
optimizer is AdamW. A bounded one-gigabyte allocator-cache threshold replaces
`mlx-lm`'s zero threshold, which otherwise clears reusable buffers after every
step. In a paired 12-step benchmark with identical tokens and losses, retaining
that bounded cache reduced load-plus-training time from 24.456 to 23.745
seconds (2.9%). IL Optimus also seeds NumPy explicitly because the installed
iterator's truthiness guard otherwise ignores the valid recorded seed zero.

Adapter initialization and data order use a recorded seed. The selector records
microbatches and optimizer updates separately, derives epochs from the accepted
dataset rather than a hard-coded eight-step demo, and caps the schedule using a
measured hardware tier. On the tested 8 GB M1 with
DeepSeek-R1-Distill-Qwen-1.5B int4, the production profile is rank 16 over the
last eight attention blocks (`q_proj`, `v_proj`, `o_proj`), LoRA scale 20 as
defined by the installed `mlx-lm` implementation, 256 tokens, batch one,
native 32-token compile buckets, and at most 234 steps under a ten-minute
budget. A measured 24-step comparison used identical tokens and losses for
every bucket size: 45.244 seconds at 256, 46.001 at 128, 43.178 at 64, and
40.626 at 32. The selected 32-token profile was 10.2% faster than the single
256-token shape while retaining the same 1.67 GB peak. The selector still uses
a conservative 2.5 seconds per step to include long-run system variance.

Artifact datasets use 2,400-character syntax-aware units and reject source
completions shorter than 1,000 characters. This avoids repeatedly supervising
EOS after a tiny snippet when the held-out contract asks for a complete file.
On the Sakura corpus, the resulting 80-row set retained 49 files, nine origins,
zero contamination, a median response near 2,300 characters, and passed every
post-filter capability audit.

This is QLoRA on MLX unified memory. It is deliberately not called Paged QLoRA:
there is no separately implemented paged optimizer. Gradient checkpointing is
enabled only when the model/memory estimate requires it. IL+GRPO is selected
only for an actual stateful rollout environment with deterministic reward.

## Controlled Sakura results

The controlled DeepSeek-R1-Distill-Qwen-1.5B int4 test used the held-out request
to build a polished voxel Sakura Island in Three.js with shaders and animation.
Session `f76c5740eca3` completed real failure detection, six model-authored
searches, 35 accepted documentation/code sources, license-gated repository
sampling, a 48-row corpus with no exact-task contamination, eight QLoRA-IL
iterations, and an adapted retry.

Both generations used the same prompt, 3,072-token budget, and temperature
zero. The hardened verifier scored both artifacts 0.4909 and their SHA-256
hashes were identical. Both failed source-depth, browser-runtime, voxel, shader,
animation, and responsive-design gates. The held-out training benchmark likewise
remained at 0%. The adapter was correctly rejected. This controlled result shows
that primitive eight-step micro-QLoRA was not enough for this model and task.
That result is retained as the pre-upgrade control.

The upgraded controlled session `b30e8d434176` collected 93 evidence objects,
curated 78 relevant sources, and accepted 76 code demonstrations from 37 files
and seven independent repository origins. Filtering reported one exact
duplicate, 17 near duplicates, one short row, zero holdout contamination, and
97 source-dominating rows. Dataset-level capability audits passed for all eight
requirements; the smallest class, island/terrain, still had four examples from
three files and two origins.

Run `2330b6a67513` trained a deterministic 1,015,808-parameter QLoRA adapter for
228 steps over three epochs. It exposed and corrected an earlier integration
bug where the adapter scale was hard-coded to `0.05` instead of the installed
`mlx-lm` scale convention. The final measured run used the corrected scale 20,
rank 16, eight layers, 256-token buckets, AdamW at `2e-5`, and 1.67 GB peak MLX
memory. The full pre-cache-optimization pipeline training run took 665.9
seconds; the production selector now uses the measured cache/bucket profile and
conservatively caps this hardware tier at 234 steps.

After hardening the verifier to reject JavaScript rendered as visible body
text, the deterministic baseline scored 0.4606 and the adapted retry scored
0.5201, a real +0.0595 change. The retry became syntactically valid and gained
observable voxel behavior, but still failed source depth, shader, animation,
and source-as-text gates. The adapter was therefore rejected and is not exposed
as accepted capability.

A separately labeled trusted Three.js framework fallback was then evaluated at
0.9263. It passed every hard gate in a real Chromium render and implements
instanced voxel terrain, animated ShaderMaterial water, falling petals,
OrbitControls, camera motion, and responsive resizing. This fallback is a
usable artifact, not evidence that the local model learned the task.

The final long-unit replication is session `526691c88dfa`, training run
`3ab6c0ef4ec7`. It reused 118 provenance-tracked sources only after all nine
capability/integration research audits passed. The deterministic dataset tools
assembled 144 licensed code units and filtered them to 80 examples from 49
files and nine repository origins. The exact request remained absent from the
training rows; filtering recorded zero holdout contamination and a median
completion length of roughly 2,356 characters.

The selected QLoRA-IL schedule trained 1,015,808 adapter parameters for 234
optimizer updates (2.92 effective epochs) with rank 16, scale 20, the final
eight attention blocks, 32-token compile buckets, and the bounded one-gigabyte
allocator cache. Peak MLX memory was 1.7 GB. The run, including its post-SFT
benchmark, completed in 453.8 seconds. Its internal held-out benchmark remained
at 0%, so training loss was not treated as evidence of success.

On the untouched original prompt, the baseline scored 0.4606 and the adapted
retry scored 0.5820, a measured +0.1214. The retry fixed JavaScript syntax and
the visible-source failure, but produced only 3,058 unique source bytes and
still lacked observable voxel, shader, and animation implementations. Because
it failed the hard gates, the adapter was rejected despite clearing the numeric
improvement margin. The separately labeled framework again scored 0.9263,
passed syntax and real-Chromium runtime checks, rendered a non-blank screenshot,
and remained the usable result. This replication demonstrates that longer
training units improved the small model's output structure and measured score,
but did not make this 1.5B model independently solve the full production
artifact contract.

## Automated curation and failure memory

The next pipeline revision removes the local-model research-planning inference
from artifact TTC. Observable verifier failures and capability contracts now
produce a bounded deterministic search frontier. Searches, safe page fetches,
and licensed repository sampling run concurrently. Evidence is cached by the
artifact kind and capability signature rather than exact prompt wording, so a
new phrasing of the same capability request reuses the audited corpus.

`curate_dataset` runs syntax-aware assembly, deterministic expansion, mechanical
code-quality scoring, holdout decontamination, exact/near deduplication,
source/origin balancing, and post-filter capability audits in one call. It
prioritizes baseline failures and reserves rare capability units before generic
origin quotas. On the Sakura cache, the full curation stage completed in about
0.45 seconds without model inference. An early 1,400-character experiment
correctly stopped because island coverage fell to three rows; rare-first
reservation repaired it to 11 rows from five files and three origins while all
eight capability audits passed.

Failed verifier results also compile into constrained Markdown skills under
`~/.iloptimus/skill-memory/`. These lessons contain the objective diagnostics,
anti-patterns, and mechanical completion gates. They are schema validated,
retrieved only for matching artifact kinds/features, and injected under a
strict character budget. A lesson records a successful use only when a later
model artifact passes; a trusted fallback never promotes it. This is a
retrieval memory of verified failure patterns, not a promise that a mistake can
never recur.

## MLX native-shape throughput

`mlx-lm` pads batches to shapes of `1 + N × 32`. The previous wrapper rounded
those already-aligned shapes a second time—for example, 225 tokens became 256—
which added up to 31 useless tokens per short example. Preserving the native
sentinel shape produced the following paired 48-update M1 benchmark with the
same model, seed, optimizer, LoRA configuration, and effective tokens:

- old rounding: 2.3925 updates/s, 38.2803 training tokens/s, 21.783 seconds,
  1.194 GB peak;
- native shape: 5.0111 updates/s, 80.1782 training tokens/s, 11.558 seconds,
  1.117 GB peak.

That is a measured 2.09× update/token throughput improvement on short examples.
It does not double workloads already truncated at the 256-token cap. Batch two
and a four-layer Q/V-only adapter were separately measured and rejected because
they did not improve this machine's useful throughput/quality tradeoff.

Training events now expose updates/s, training tokens/s, trained tokens, and
memory. Sustained local rates are persisted by model/sequence/rank/layer/backend
profile. Future schedules use the conservative measured step time so thermal or
system slowdown reduces the selected update count instead of violating the
declared time budget.

Paged QLoRA is selected only when a CUDA training backend exposes a real paged
optimizer. MLX uses unified-memory QLoRA; relabeling it PQLoRA would not change
the optimizer or make training faster.

## Exact post-automation replication

Session `7b57fa2139bf`, training run `6cc4146f4600`, reran the untouched Sakura
request after automated curation, rare-capability reservation, failure-skill
retrieval, and native-shape batching were installed. It reused 118 audited
evidence objects by capability signature. Deterministic research/audit took
0.465 seconds and the one-call curator took 0.457 seconds without another model
inference. The curator retained 79 rows from 41 files and nine origins with a
0.9318 mean quality score and zero holdout contamination. All eight capability
audits passed; the rare island class retained 11 rows from five files and three
origins.

The M1 training run completed 310 optimizer updates and 63,562 effective
training tokens at 1.666 GB peak MLX memory. Heavy concurrent system load made
the run take 1,082.3 seconds, proving the previous static estimate was unsafe.
The sustained profile now records a conservative 4.6244 seconds/update for this
exact model/sequence/rank/layer/backend tuple. Under the same ten-minute budget,
the next selector chooses 124 updates (598 seconds estimated) instead of 310.

The independent baseline scored 0.4935. The adapted artifact scored 0.4653, a
-0.0282 regression: it learned observable shader and island patterns but still
failed source depth, syntax, source-as-text, voxel, animation, and interaction
gates. The adapter was correctly rejected. The separately labeled trusted
framework scored 0.9263 and passed every hard gate, including a real nonblank
Chromium render. This is a useful negative result: high-quality data and more
updates are not sufficient evidence that this 1.5B model can synthesize the
entire production artifact in one pass.

The failed retry created a new verifier-derived repair skill and the run also
consumed the prior matching failure skill. Partial feature scores now become
completion rules too, and repeated observations for the same repair signature
are retained in `evidence.json` rather than overwriting history. Failed uses do
not increase the skill's success count.

## Framework-backed model scene success

The monolithic source experiment established that asking a 1.5B model for one
flawless 12–15 KB application was the wrong interface. Its completion repeated
blocks, hallucinated APIs, and frequently ended before closing the JavaScript
module. Splitting the same request into a world-initializer component removed
truncation but still produced nonexistent Three.js APIs; the component verifier
caught that failure.

The production path now treats Three.js as an engine rather than something the
model must rewrite. The local model emits a bounded scene specification covering
title, palette, terrain dimensions, water, particle count, camera, distinct
tree placements, scene details, and motion. A trusted voxel-island runtime compiles that
design into instanced voxel terrain, shader water, blossom trees, falling
petals, camera movement, OrbitControls, and responsive rendering. The raw model
output, normalized specification, runtime identity, hashes, retry count, and a
`fallback_used` flag are persisted beside the artifact. A hand-authored Sakura
fallback cannot satisfy the model-authorship gate.

Exact final session `60b839f03698` used the unadapted local
DeepSeek-R1-Distill-Qwen-1.5B. The typed contract contained no populated example.
The first two candidates failed mechanically; on attempt three the model authored
`Sakura's Sakura Island`, five original palette colors, radius 18 and height 10
terrain, 150-unit water, 600 petals, a camera direction, three distinct tree
placements, and Sakura-specific details. Its motion value had the wrong shape,
so the compiler supplied only its generic motion default and recorded that fact;
it also scaled the oversized camera and tree coordinates while preserving their
relative layout. The resulting 16,597-byte artifact scored 0.9439 in 37.811
seconds end to end and passed source depth, JavaScript syntax, every requested
capability, and a real nonblank Chromium render. Its authorship manifest records
`local-model-scene-spec`, `trusted-voxel-island-threejs-engine`,
`framework_default_fields: ["motion"]`, normalized coordinate fields, and
`fallback_used: false`. Because the unadapted design already passed, the harness
correctly skipped training.

This result means the model designed the scene through a constrained tool and
the trusted engine rendered it. It does not mean the model independently wrote
all 14 KB of engine code. That is the same separation used by a human designer
working in a game engine, and it is deliberately reported rather than hidden.

## Token-efficient data and frozen-prefix training

The prior 79-example Sakura corpus had a hidden efficiency failure: every
completion exceeded the 256-token cap, zero examples were retained end-to-end,
and only 61.6% of supervised completion tokens reached the loss. The revised
curator creates complete syntax-bounded units of 300–585 characters and shorter
prompts. It retained 111 examples across 46 files and nine origins, passed all
eight capability audits, kept zero holdout contamination, and retained 99.64%
of 14,099 supervised completion tokens at a 192-token sequence cap. Curation
took 0.859 seconds without model inference.

The compact MLX path also supports frozen-prefix caching. With LoRA on the final
four Q/V layers, the lower 24 transformer layers are evaluated once per source
unit and cached. On the real 111-row corpus, suffix training measured 0.5872
updates/s and 70.301 useful tokens/s at 1.817 GB peak, compared with roughly
0.2647 updates/s and 30.372 tokens/s for the uncached four-layer path. The cache
build took 228.879 seconds, so the scheduler records it as fixed overhead; it is
beneficial only when enough epochs amortize that cost. An independent eight-row
reload check trained at 78–91 tokens/s after the cache build, saved 155,648
adapter parameters under their normal full-model names, and reloaded them into
the base model successfully.
