# Prompt2Sim-pipeline

A local Scenethesis-inspired text-to-3D scene generation pipeline.

Reference paper: [Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation](https://arxiv.org/pdf/2505.02836)

## Current Direction

The repo now has two paths:

- `scripts/run_demo.py`: the original lightweight MVP path.
- `scripts/run_faithful.py`: the strict paper-faithful path under active development.

Use the faithful runner for the warehouse recreation work. It fails when a required model, checkpoint, mesh, CUDA capability, OpenAI call, or Blender export is missing. It should not be treated as successful when it cannot run the intended component.

## Faithful Pipeline

The strict path implements the current local approximation of the paper architecture:

1. OpenAI planner produces a strict `SceneSpec`.
2. OpenAI image generation creates a guidance image.
3. GroundingDINO plus SAM segments planned objects into boxes, masks, and crops.
4. Depth Pro estimates dense depth.
5. Mask/depth projection builds per-object point clouds and a 3D scene graph.
6. Local CLIP retrieval selects mesh-backed assets from the warehouse registry.
7. Depth-derived 3D boxes apply conservative metric scale/yaw pose refinement before SDF optimization.
8. SDF/PyTorch3D optimization adjusts 5-DoF poses and checks support/collision.
9. Blender renders `render.png`, alternate scene views, per-object alignment views, `scene.glb`, and `scene.usd` when Blender supports USD.
10. RoMa matches guidance crops against rendered object views and writes per-object correspondence files plus pose history.
11. The joint pose optimizer consumes Depth Pro graph boxes plus RoMa correspondences, applies bounded 5-DoF updates, and writes loss history before the second SDF/render pass.
12. OpenAI vision judge scores the rendered scene with guidance plus multi-view render inputs and returns strict JSON repair actions.
13. The repair loop applies only actionable scene changes, reruns optimization/render/judge, and writes final artifacts plus `qualification.json`.

The paper describes coarse LLM planning, image-guided scene graph extraction, asset retrieval, pose optimization with semantic correspondence and SDF-based physical constraints, and GPT-4o scene judgment. This repo now follows that structure, but remains smaller than the paper system: the asset database is a curated local warehouse pack, not a large Objaverse-scale subset.

## Hard Requirements

- Windows with NVIDIA GPU and working CUDA.
- Dedicated conda env, currently `scenethesis-faithful`.
- OpenAI API key in local `.env`.
- Blender 5.x installed or `BLENDER_PATH` set.
- Downloaded checkpoints for GroundingDINO, SAM, Depth Pro, CLIP/RoMa dependencies, and PyTorch3D support.
- Local mesh-backed warehouse asset registry at `configs/warehouse_asset_registry.yaml`.

Do not put a real API key in `.env.example`. Use `.env` for secrets.

## Setup

Lightweight MVP dependencies:

```bash
pip install -r requirements.txt
```

Faithful CUDA environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_faithful_env.ps1
conda run -n scenethesis-faithful python scripts/download_faithful_checkpoints.py
conda run -n scenethesis-faithful python scripts/import_polyhaven_assets.py --resolution 1k
conda run -n scenethesis-faithful python scripts/import_hf_simready_assets.py
conda run -n scenethesis-faithful python scripts/build_clip_asset_index.py --registry configs/warehouse_asset_registry.yaml --out assets/indexes/warehouse_clip_index.npz --device cuda
```

`scripts/import_hf_simready_assets.py` uses `configs/hf_simready_warehouse_manifest.yaml`; it downloads only declared Hugging Face SimReady asset directories and declared USD payload dependencies, converts those USD files to local GLB through Blender, and records license/source metadata in `assets/manifests/hf_simready_warehouse_assets.json`. It is not a bulk dataset clone.

Runtime check:

```powershell
conda run -n scenethesis-faithful python scripts/run_faithful.py --prompt "warehouse storage area" --out runs/runtime_check --check-runtime-only
```

The runtime check writes `faithful_runtime_report.json` and exits non-zero if anything required is missing.

## Faithful Warehouse Run

```powershell
conda run -n scenethesis-faithful python scripts/run_faithful.py --prompt "a busy warehouse storage and packing area with two steel storage shelves, many cardboard boxes, wooden crates, plastic crates, barrels, a hand truck, a metal trash can, a tool chest, a packing table, tools, a ladder, jerry cans, and a chair" --out runs/warehouse_faithful_rich_007 --repair-rounds 2
```

To rerun from already generated guidance/segmentation/depth artifacts without pretending missing artifacts are valid:

```powershell
conda run -n scenethesis-faithful python scripts/run_faithful.py --prompt "a busy warehouse storage and packing area with two steel storage shelves, many cardboard boxes, wooden crates, plastic crates, barrels, a hand truck, a metal trash can, a tool chest, a packing table, tools, a ladder, jerry cans, and a chair" --out runs/warehouse_faithful_rich_007 --repair-rounds 2 --resume-from-existing
```

Resume mode validates every required artifact before continuing.

Expected final artifacts:

- `coarse_scene_spec.json`
- `guidance.png`
- `segmentation.json`
- `depth.json`
- `scene_graph_3d.json`
- `clip_retrieval.json`
- `depth_pose_refinement.json`
- `sdf_optimizer.json`
- `scene_spec.json`
- `scene.glb`
- `scene.usd` when supported
- `render.png`
- `views/render_front.png`, `views/render_left.png`, `views/render_right.png`, `views/render_top_oblique.png`
- `alignment_views/<object_id>.png`
- `render_views.json`
- `correspondences/<object_id>.npz`
- `pose_alignment_history.json`
- `correspondence_diagnostics.json`
- `joint_pose_optimizer.json`
- `pose_loss_history.json`
- `metrics.json`
- `judge.json`
- `repair_history.json`
- `qualification.json`
- `report.md`

## Verified Local Result

The latest successful warehouse run is:

```text
runs/warehouse_faithful_rich_007
```

Validation summary:

- Objects: 19
- Collision count: 0
- Floating count: 0
- Unsupported count: 0
- SDF optimizer status: `ok`
- Judge `needs_repair`: `false`
- Render: `runs/warehouse_faithful_rich_007/render.png`

The render is materially better than the early barrel/shelf-only output, but it is not yet at paper-demo quality. The largest remaining visual gaps are scene composition quality and the limited curated asset subset, not the absence of the core segmentation/depth/SDF/RoMa stages.

## Strict Success Rules

Current strict behavior:

- Missing OpenAI key fails.
- Missing Blender fails.
- Missing CUDA fails.
- Missing GroundingDINO/SAM/Depth Pro/RoMa/PyTorch3D/OpenCLIP dependencies fail.
- Missing model checkpoints fail.
- Missing mesh-backed assets fail after retrieval/runtime validation.
- Missing segmentation masks, crops, depth arrays, or guidance images fail resume mode.
- Invalid judge JSON fails.
- No-op judge repair actions fail validation when `needs_repair=true`.
- Missing `qualification.json` prevents a run from being accepted by `validate_run`.
- Failed depth-pose refinement, joint pose optimization, visual support, judge repair, or RoMa correspondence diagnostics mark the run unqualified even if render artifacts exist.
- PyBullet is not used as a support-contact substitute. The PyBullet hook raises until a real simulation integration is implemented.

The renderer may assign a default material only when an imported mesh has no material slots. That is not a geometry or asset substitute.

## Tests

```powershell
conda run -n scenethesis-faithful python -m pytest
```

Tests do not call OpenAI. They cover schema validation, registry loading, prompt requirements, subtype contracts, strict resume validation, collision/layout rules, mesh/SDF-adjacent checks, and judge validation.

## Limits

Still not reproduced from the paper:

- No large Objaverse-scale asset database.
- No exact paper asset subset.
- Depth pose refinement uses Depth Pro point-cloud boxes for conservative metric scale/yaw updates before the first SDF pass.
- The joint pose optimizer combines Depth Pro graph targets and RoMa correspondence yaw targets with bounded SGD-style updates, but it is still simpler than the paper's full differentiable 2D/3D/SDF objective.
- The SDF optimizer is local and practical for the laptop, but the paper ran experiments on A100-class hardware.
- Warehouse asset quality depends on the current curated Poly Haven plus HF SimReady subset.
- Multi-view judge/regeneration is implemented, but still smaller than the paper evaluation loop.

The next high-impact work is broader licensed asset acquisition, better object-aware composition policies, and moving the SDF signed-distance terms directly into the joint pose optimizer instead of running SDF as the hard validation/refinement stage after bounded pose updates.
