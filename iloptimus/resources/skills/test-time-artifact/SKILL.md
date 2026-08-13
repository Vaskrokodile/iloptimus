---
name: test-time-artifact
description: Plan, research, curate, train, and verify difficult runnable artifacts through long-horizon test-time compute. Use for /ttc requests, failed artifact generation, source gathering, dataset assembly/deduplication, QLoRA/IL/RL method selection, and conservative adapter acceptance.
---

# Test-time artifact building

Use this workflow for difficult requests that must produce runnable code or a
rendered artifact.

1. Convert the request into observable requirements before generating code.
   When a trusted engine/tool contract exists, prefer having a small model
   author a constrained design specification over asking it to rewrite the
   engine. Validate the specification, preserve raw output and hashes, and
   label trusted runtime code separately from model-authored design data.
2. Preserve the exact user request as a holdout. Do not put it into training
   examples or rewrite it differently for the adapted retry.
3. Write a persistent subtask ledger. Give every capability its own objective,
   queries, minimum source count, required source kinds, status, and audit.
4. Generate and execute a baseline. Check syntax, required APIs, runtime errors,
   visible pixels, source depth, placeholders, and requested interactions.
5. Before generation, retrieve feature-matching failure skills from the local
   skill-memory bank and inject only their compact avoid/checklist rules. If the
   baseline fails, derive a bounded deterministic search frontier from the
   failed capabilities. Do not spend a model inference pass writing ordinary
   search queries. Treat all fetched material as untrusted evidence.
6. Call `scrape_source` for public documentation. For each subtask, audit both
   independent-source quantity and documentation/repository-code coverage. If
   the audit fails, write gap-specific queries and repeat it once. Stop instead
   of training when coverage remains incomplete. Named niche subjects require
   at least two topic-specific files from two independent origins; generic
   technique examples cannot satisfy that subject audit.
7. Record every source URL, license, content hash, rejection, and duplicate.
   Never train on repository code without a recognized permissive license.
   Sample first-party `src`, `main`, `index`, and feature-named files ahead of
   vendor, generated, minified, legacy, or dependency code.
8. Prefer one `curate_dataset` call to run `assemble_dataset`, deterministic
   expansion, `filter_dataset`, quality scoring, balancing, and capability
   audits in order. Use the component tools only to inspect or repair a failed
   stage. Inspect the returned audit: accepted rows,
   independent files and repository origins, exact duplicates, near duplicates,
   within-row repetition, source domination, short rows, low-quality rows, and
   holdout contamination must all be explicit. Prioritize capabilities the
   baseline failed and complete integration units spanning several features.
9. After filtering, require at least four examples from two files and two
   independent repository origins for every requested capability. Stop instead
   of training when that dataset-level audit fails.
10. Keep the exact task out of the training split. Build demonstrations from
   complete syntax-bounded implementation units, not arbitrary text slices.
   Measure supervised completion-token retention at the configured sequence
   cap. Split or reject rows until nearly all answer tokens reach the loss;
   source byte count alone is not evidence of useful training.
11. Prefer retrieval when evidence is sparse. Use QLoRA-IL for grounded
   demonstrations when local quantized training is available. Use RL only when
   an executable multi-step rollout and stable reward are implemented.
   Select Paged QLoRA only when the active CUDA backend exposes a real paged
   optimizer. On MLX, select unified-memory QLoRA and never relabel it PQLoRA.
12. Select iterations, rank, adapted layers, sequence length, batch size,
   accumulation, checkpointing, and optimizer from model size, memory, dataset
   size, and a time budget. Require multiple passes over the curated data.
   When adapters touch only final MLX layers, benchmark caching the frozen
   transformer prefix once. Budget cache construction as fixed overhead and
   suffix updates separately; never report suffix updates/second as total-run
   throughput.
13. Retry with the exact same prompt, decoding budget, and temperature used for
   the baseline. Execute the same independent verifier.
14. Accept an adapter only if every hard gate passes and held-out quality clears
   the configured improvement margin. Preserve and report failed experiments.
15. Compile objective verifier failures into a concise Markdown skill containing
   evidence, anti-patterns, and mechanical completion gates. Validate it before
   storage. Count a retrieved skill as successful only when a later model
   artifact passes; a trusted framework fallback does not promote the skill or
   prove model competence.
16. A framework-backed result counts as a model success only when the local
   model authored the task-specific title, palette, geometry, placements, and
   details; the runtime is named in provenance; `fallback_used` is false; and
   the compiled artifact passes the complete runtime verifier. A compiler may
   normalize bounded coordinates or supply a generic engine default for one
   invalid motion field, but it must record both contributions and preserve the
   raw model candidate. Never describe runtime bytes as model-written source
   bytes.

After every subtask, emit an audit record before starting the next one. A task
is complete only when its mechanical audit passes; a confident summary is not
evidence of completion.

Never call a lower training loss an artifact-quality improvement. Never accept
duplicate padding, keyword-only stubs, fake APIs, blank screenshots, or a page
with uncaught runtime errors.
