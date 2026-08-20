import {
  ArrowDown,
  ArrowUpRight,
  Check,
  CircleDot,
  Code2,
  Cpu,
  Download,
  FileJson,
  Gauge,
  GitBranch,
  Layers3,
  ShieldCheck,
  Sparkles,
  Workflow,
  X,
} from "lucide-react";

const SESSION_ID = "8dac8bd6296d";
const artifactUrl = `/research/sakura-island/artifact.html`;
const screenshotUrl = `/research/sakura-island/screenshot.png`;
const experimentUrl = `/research/sakura-island/experiment.json`;
const manifestUrl = `/research/sakura-island/authorship.json`;

const capabilityRows = [
  ["Three.js runtime", "1.00", "Pass"],
  ["Voxel geometry", "1.00", "Pass"],
  ["Custom shader", "1.00", "Pass"],
  ["Island semantics", "1.00", "Pass"],
  ["Sakura semantics", "1.00", "Pass"],
  ["Responsive layout", "1.00", "Pass"],
  ["Animation depth", "1.00", "Pass"],
];

const runStages = [
  {
    time: "00.000s",
    label: "Contract",
    title: "The harness froze the request",
    body: "Six observable capabilities became a holdout contract before generation: Three.js, voxels, shaders, island structure, Sakura semantics, and responsive layout.",
    icon: GitBranch,
  },
  {
    time: "04.200s",
    label: "Attempt 01",
    title: "Malformed candidate rejected",
    body: "The local model produced an incomplete scene specification with missing typed fields. The parser and schema verifier rejected it mechanically.",
    icon: X,
    failed: true,
  },
  {
    time: "12.800s",
    label: "Attempt 02",
    title: "A complete Sakura design emerged",
    body: "The model authored the title, palette, terrain radius and height, water size, petal density, camera, trees, and Sakura-specific scene details. All contract fields were present and valid.",
    icon: Sparkles,
  },
  {
    time: "28.500s",
    label: "Compilation",
    title: "Trusted engine assembled the world",
    body: "The voxel-island runtime compiled the design into executable Three.js. It supplied rendering primitives and bounded oversized coordinates; those interventions were recorded in provenance.",
    icon: Code2,
  },
  {
    time: "187.701s",
    label: "Verification",
    title: "Chromium accepted the artifact",
    body: "Syntax, source depth, placeholders, requested capabilities, runtime errors, and visible pixels were evaluated. Every hard gate passed with a perfect score, ending the loop without adapter training.",
    icon: ShieldCheck,
  },
];

export default function ResearchPaperPage() {
  return (
    <article className="paper-page">
      <header className="paper-topbar">
        <a className="paper-wordmark" href="/" aria-label="Back to IL Optimus">
          <span>IL</span><strong>Optimus Research</strong>
        </a>
        <nav aria-label="Paper sections">
          <a href="#method">Method</a>
          <a href="#trajectory">Trajectory</a>
          <a href="#results">Results</a>
          <a href="#limits">Limits</a>
        </nav>
        <a className="paper-source-button" href={experimentUrl} target="_blank" rel="noreferrer">
          <FileJson /> Run data
        </a>
      </header>

      <main id="top">
        <section className="paper-hero paper-section">
          <div className="paper-kicker"><span /> Autonomous systems report · Experiment 8dac8bd6296d</div>
          <h1>Can a 1.5B local model<br /> build a <em>Sakura Island</em><br /> without a human in the loop?</h1>
          <p className="paper-deck">
            A verifier-driven harness converted one natural-language request into a typed scene contract, rejected one
            failure, compiled the second model-authored design, rendered it in Chromium, and stopped only when every hard
            gate passed with a perfect score.
          </p>
          <div className="paper-authors">
            <div><strong>IL Optimus Autonomous Harness</strong><span>Execution, compilation, verification</span></div>
            <div><strong>Boosted-v1-small · int4</strong><span>Local scene design model</span></div>
            <div><strong>August 2026</strong><span>Local NVIDIA GPU · 100% offline</span></div>
          </div>
          <a className="paper-scroll" href="#abstract"><ArrowDown /> Read the report</a>
        </section>

        <section className="paper-proof paper-section" id="evidence" aria-label="Key result">
          <div className="paper-proof-copy">
            <span className="paper-number">Figure 01</span>
            <h2>The accepted world, rendered by the verifier.</h2>
            <p>
              This is the exact nonblank Chromium screenshot stored by the final run. The model selected the scene identity,
              palette, terrain scale, water scale, petal density, camera intent, tree layout, and semantic details. The
              trusted engine provided rendering primitives—not the design decision.
            </p>
            <div className="paper-proof-actions">
              <a href={artifactUrl} target="_blank" rel="noreferrer">Explore live scene <ArrowUpRight /></a>
              <a href={screenshotUrl} download>Download evidence <Download /></a>
            </div>
          </div>
          <figure className="paper-figure">
            <div className="paper-figure-frame">
              <img src={screenshotUrl} alt="Verified model-designed voxel Sakura Island at purple dusk" />
              <div className="paper-figure-status"><span /> Runtime verified</div>
            </div>
            <figcaption>
              <span>Final verifier capture · 1280 × 800</span>
              <span>Artifact · 36,269 bytes / 778 lines</span>
            </figcaption>
          </figure>
        </section>

        <section className="paper-abstract paper-section" id="abstract">
          <div className="paper-section-label">Abstract</div>
          <div>
            <p className="paper-lede">
              We report an autonomous, framework-mediated artifact generation run in which a quantized 1.5B-parameter local
              model produced a complete interactive voxel Sakura Island after one verifier-rejected attempt.
            </p>
            <div className="paper-columns">
              <p>
                The central result is not that a small model independently wrote a production Three.js engine. It did not.
                The result is that a small model successfully operated a constrained world-building interface: it authored
                15 semantic scene fields, while a trusted runtime handled rendering, normalization, and execution.
              </p>
              <p>
                There was no human intervention between submission and terminal verdict in the reported run. Parsing,
                diagnostics, retries, compilation, browser execution, screenshot capture, scoring, and the decision to skip
                training were made by the harness. The complete run ended in 187.701 seconds with a perfect score of 1.000.
              </p>
            </div>
          </div>
        </section>

        <section className="paper-metrics paper-section" aria-label="Experiment metrics">
          <div><span>187.701s</span><small>End-to-end latency</small></div>
          <div><span>1.000</span><small>Perfect score</small></div>
          <div><span>2</span><small>Generation attempts</small></div>
          <div><span>15</span><small>Model-authored fields</small></div>
          <div><span>0</span><small>Human interventions</small></div>
        </section>

        <section className="paper-method paper-section" id="method">
          <div className="paper-section-label">01 · Method</div>
          <div className="paper-method-grid">
            <div>
              <h2>Intelligence in the model.<br />Reliability in the harness.</h2>
              <p>
                Monolithic source generation was replaced by a narrow design API. The model was asked to author the world;
                the engine was responsible for making that world executable. This separation reduced the completion from
                thousands of fragile source tokens to a bounded scene object that could be audited before execution.
              </p>
            </div>
            <div className="paper-flow" aria-label="Autonomous harness flow">
              <div><span>01</span><GitBranch /><strong>Freeze contract</strong><small>Exact task stays outside adaptation data</small></div>
              <div><span>02</span><Cpu /><strong>Generate locally</strong><small>Structured completion from the 1.5B model</small></div>
              <div><span>03</span><ShieldCheck /><strong>Verify mechanically</strong><small>Schema, syntax, capability, and browser gates</small></div>
              <div><span>04</span><Workflow /><strong>Retry or terminate</strong><small>Diagnostic feedback or measured acceptance</small></div>
            </div>
          </div>
          <div className="paper-boundary">
            <div>
              <span className="paper-number">Model-authored</span>
              <h3>What the 1.5B model decided</h3>
              <ul>
                <li>Scene title and Sakura identity</li>
                <li>Five original palette colors</li>
                <li>Terrain radius and height</li>
                <li>Water size and 600-petal density</li>
                <li>Camera intent and three tree placements</li>
                <li>“Sakura”, “Cherry Blossom”, and “Sunset” details</li>
              </ul>
            </div>
            <div>
              <span className="paper-number">Framework-authored</span>
              <h3>What the trusted harness supplied</h3>
              <ul>
                <li>Voxel and shader rendering primitives</li>
                <li>OrbitControls and responsive canvas</li>
                <li>One generic motion default</li>
                <li>Bounded camera and tree-coordinate normalization</li>
                <li>Syntax and browser execution</li>
                <li>Provenance hashes and terminal scoring</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="paper-trajectory paper-section" id="trajectory">
          <div className="paper-section-label">02 · Autonomous trajectory</div>
          <div className="paper-trajectory-head">
            <h2>The harness failed forward.</h2>
            <p>Every transition below came from stored run events or raw model candidates. No candidate was manually edited.</p>
          </div>
          <div className="paper-timeline">
            {runStages.map(({ time, label, title, body, icon: Icon, failed }) => (
              <div className={`paper-timeline-row ${failed ? "failed" : ""}`} key={label}>
                <time>{time}</time>
                <div className="paper-timeline-node"><Icon /></div>
                <div><span>{label}</span><h3>{title}</h3><p>{body}</p></div>
              </div>
            ))}
          </div>
        </section>

        <section className="paper-results paper-section" id="results">
          <div className="paper-section-label">03 · Results</div>
          <div className="paper-results-grid">
            <div>
              <h2>Seven full capability passes.<br />Zero partials.</h2>
              <p>
                Acceptance required every hard gate—not merely an attractive screenshot. All six capability scores reached 1.0, and the verifier’s browser render gate confirmed visible pixels in Chromium.
              </p>
              <div className="paper-score"><Gauge /><span><strong>100</strong><small>out of 100</small></span></div>
            </div>
            <div className="paper-capability-table">
              {capabilityRows.map(([name, score, verdict]) => (
                <div key={name}><span>{name}</span><strong>{score}</strong><em className={verdict === "Partial" ? "partial" : ""}>{verdict === "Pass" ? <Check /> : <CircleDot />}{verdict}</em></div>
              ))}
            </div>
          </div>
          <div className="paper-hard-gates">
            <span><Check /> File exists</span>
            <span><Check /> Substantial source</span>
            <span><Check /> No placeholders</span>
            <span><Check /> Valid entrypoint</span>
            <span><Check /> JavaScript syntax</span>
            <span><Check /> Chromium render</span>
          </div>
        </section>

        <section className="paper-data paper-section">
          <div className="paper-section-label">04 · Speed substrate</div>
          <div className="paper-data-head">
            <h2>The larger pipeline got faster—even though this run did not need training.</h2>
            <p>The harness first tests the unadapted model. Training is invoked only after failure, so a passing baseline saves the entire adaptation budget.</p>
          </div>
          <div className="paper-data-cards">
            <div><Layers3 /><span>61.6% <ArrowUpRight /> 99.64%</span><strong>Completion-token retention</strong><p>Syntax-bounded examples replaced truncated source chunks.</p></div>
            <div><Gauge /><span>30.4 <ArrowUpRight /> 70.3</span><strong>Useful training tokens / sec</strong><p>Frozen-prefix caching more than doubled measured suffix throughput.</p></div>
            <div><Cpu /><span>79 <ArrowUpRight /> 111</span><strong>Accepted curated examples</strong><p>More independently sourced units passed capability audits.</p></div>
          </div>
        </section>

        <section className="paper-limits paper-section" id="limits">
          <div className="paper-section-label">05 · Interpretation & limits</div>
          <div className="paper-limits-grid">
            <div>
              <h2>What “no human in the loop” means here.</h2>
              <p>
                During final session <code>{SESSION_ID}</code>, no person selected candidates, repaired JSON, tuned a value,
                approved a screenshot, or chose when to stop. The same submitted request moved through generation,
                diagnostics, compilation, browser execution, and acceptance automatically.
              </p>
            </div>
            <div className="paper-caveat">
              <ShieldCheck />
              <h3>What it does not mean</h3>
              <p>
                Humans built the harness, its typed contract, verifier, and voxel-island runtime before the reported run.
                The model did not author all 36.3 KB of executable source. This is autonomous tool use inside a human-built
                system, not an unassisted model inventing its own renderer.
              </p>
            </div>
          </div>
          <blockquote>
            “The breakthrough was not asking a small model to become an engine. It was giving the model an engine-shaped
            language in which its decisions could be tested.”
          </blockquote>
        </section>

        <section className="paper-reproduce paper-section">
          <div>
            <span className="paper-number">Reproducibility record</span>
            <h2>Inspect the evidence, not the claim.</h2>
            <p>The run record, authorship boundary, hashes, raw attempts, verifier score, and exact executable artifact remain stored locally.</p>
          </div>
          <div className="paper-reproduce-links">
            <a href={artifactUrl} target="_blank" rel="noreferrer"><ArrowUpRight /><span><strong>Live artifact</strong><small>Execute the accepted scene</small></span></a>
            <a href={experimentUrl} target="_blank" rel="noreferrer"><FileJson /><span><strong>Experiment record</strong><small>Timing, gates, acceptance</small></span></a>
            <a href={manifestUrl} target="_blank" rel="noreferrer"><ShieldCheck /><span><strong>Authorship manifest</strong><small>Model vs. framework fields</small></span></a>
          </div>
        </section>
      </main>

      <footer className="paper-footer">
        <span>IL Optimus Research · Autonomous local intelligence</span>
        <span>Session {SESSION_ID} · Score 1.000 · 187.701 seconds</span>
      </footer>
    </article>
  );
}
