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
   queries and execute those calls through the audited tool boundary;
5. fetch relevant documentation and shallow-clone public GitHub repositories;
6. admit repository code only when the repository has a recognized MIT,
   Apache-2.0, BSD, or ISC license;
7. build a provenance manifest with source URLs, SHA-256 hashes, duplicate
   checks, and an exact-task contamination check;
8. select retrieval or QLoRA-IL from evidence volume and hardware support.
   GRPO is deliberately excluded from one-shot artifact generation because it
   does not expose a stable multi-step rollout process; `PQLoRA` is not treated
   as a separate objective without an implemented algorithm;
9. train in a memory-isolated subprocess, retry the untouched task using the
   identical prompt and deterministic decoding budget, and execute the verifier again;
10. retain an adapter only when the retry passes every hard gate and improves
    by at least 0.05.

Run artifacts are saved under `~/.iloptimus/learning/<session-id>/`, including
the baseline and retry source, screenshots, research manifest, JSONL dataset,
dataset/contamination manifest, experiment record, and acceptance decision. The
LoRA adapter and training telemetry remain under `~/.iloptimus/runs/<run-id>/`.

## Current adversarial result

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
that primitive eight-step micro-QLoRA was not enough for this model and task;
future work needs verifier-guided multi-candidate search and materially more
task-shaped optimization, not a larger unverified claim.
