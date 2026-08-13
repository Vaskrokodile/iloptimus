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
