from pathlib import Path

from iloptimus.core.artifact_composer import (
    ArtifactComponent,
    assemble_threejs_artifact,
    audit_component,
    audit_model_authorship,
    authorship_manifest,
    clean_component_source,
    component_prompt,
    threejs_component_plan,
)


def _valid_source(component: ArtifactComponent) -> str:
    patterns = {
        "world": """function initializeWorld(world) {
world.scene = new THREE.Scene(); world.camera = new THREE.PerspectiveCamera();
world.renderer = new THREE.WebGLRenderer(); world.controls = new OrbitControls(world.camera, world.renderer.domElement);
window.addEventListener('resize', () => { world.renderer.setPixelRatio(devicePixelRatio); console.log(innerWidth); });
document.querySelector('#app').appendChild(world.renderer.domElement);
}""",
        "terrain": """function createVoxelIsland(world) {
const terrain = new THREE.InstancedMesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial(), 64);
for (let i=0;i<64;i++) { terrain.setMatrixAt(i, new THREE.Matrix4()); }
world.island = terrain; world.scene.add(terrain);
}""",
        "water": """function createAnimatedWater(world) {
const uniforms={uTime:{value:0}}; const water=new THREE.Mesh(new THREE.PlaneGeometry(), new THREE.ShaderMaterial({uniforms,vertexShader:'void main(){gl_Position=vec4(position,1.);}',fragmentShader:'void main(){gl_FragColor=vec4(0.,.3,.7,1.);}'}));
world.water=water; world.waterUniforms=uniforms; world.scene.add(water);
}""",
        "sakura": """function createSakuraSystem(world) {
const sakura=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial());
const blossom=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial());
world.petalSystem=blossom; world.scene.add(sakura,blossom);
}""",
        "details": """function createWorldDetails(world) {
const rock=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial());
const bridge=new THREE.Group(); bridge.add(rock); world.scene.add(bridge); world.details=bridge;
}""",
        "animation": """function startExperience(world) {
function frame(){ requestAnimationFrame(frame); world.waterUniforms.uTime.value += .01; world.petalSystem.position.y -= .01; world.controls.update(); world.renderer.render(world.scene,world.camera); }
frame();
}""",
    }
    source = patterns[component.id]
    # Unit fixtures remain readable while satisfying production byte floors.
    return source.replace("\n}", "\n" + "\n".join(f"const detail{i} = {i};" for i in range(70)) + "\n}")


def test_component_cleaning_extracts_first_balanced_function():
    component = threejs_component_plan()[1]
    raw = "```javascript\nfunction createVoxelIsland(world) { const label = `}`; world.x = label; }\n```\nprose"
    assert clean_component_source(raw, component) == (
        "function createVoxelIsland(world) { const label = `}`; world.x = label; }"
    )


def test_component_prompt_has_strict_interface():
    component = threejs_component_plan()[0]
    prompt = component_prompt("Build an island", component, verifier_feedback=["missing resize"])
    assert "initializeWorld(world)" in prompt
    assert "one complete function" in prompt
    assert "missing resize" in prompt


def test_model_components_assemble_with_auditable_authorship(tmp_path: Path):
    components = {component.id: _valid_source(component) for component in threejs_component_plan()}
    audits = {
        component.id: audit_component(components[component.id], component)
        for component in threejs_component_plan()
    }
    assert all(audit.passed for audit in audits.values())
    artifact = tmp_path / "index.html"
    artifact.write_text(assemble_threejs_artifact(components), encoding="utf-8")
    manifest = authorship_manifest(
        artifact,
        components,
        audits,
        model_id="local-test-model",
        adapter_path="adapter",
    )
    assert manifest["fallback_used"] is False
    assert manifest["model_authored_ratio"] >= 0.65
    assert audit_model_authorship(manifest) == []


def test_authorship_rejects_missing_or_failed_components():
    assert audit_model_authorship({"authorship": "framework", "fallback_used": True})
