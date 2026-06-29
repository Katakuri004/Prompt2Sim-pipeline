# Prompt2Sim Pipeline

Prompt2Sim Pipeline is a local research prototype for generating warehouse-style
3D scenes from language and turning selected scenes into robot simulation tasks.
The project combines prompt planning, image-guided scene generation, asset
retrieval, geometry validation, MuJoCo compilation, deterministic Panda teacher
rollouts, accepted demonstration export, and a first LeRobot ACT training path.

Reference papers:

- [Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation](https://arxiv.org/pdf/2505.02836)
- [GRS: Generating Robotic Simulation Tasks from Real-World Images](https://arxiv.org/abs/2410.15536)

## Current Status

The repository has two main execution paths:

- `scripts/run_faithful.py`: the strict scene-generation pipeline inspired by
  Scenethesis and GRS.
- `scripts/evaluate_mujoco_scene.py`: the MuJoCo evaluation path for compiled
  scenes and robot policies.

The current MuJoCo bridge has been validated on a warehouse barcode-scanner
pick-and-place task with the real Franka Panda MuJoCo model. The deterministic
teacher can build a strict joint waypoint plan, run grasp and completion probes,
execute the full rollout, record RGB observations, save MP4 debug videos, and
export accepted episodes for LeRobot-style imitation learning.

This is still a research prototype. The code is most useful for developers who
want to inspect or extend the scene-to-simulation-to-demonstration workflow. It
is not yet a general, production-ready robot data engine.

## Architecture

At a high level:

```text
language prompt / assets / image guidance
-> faithful scene generation
-> SceneIR
-> MJCF/XML/MJB compilation
-> MuJoCo runtime
-> Panda teacher planning
-> strict rollout evaluation
-> accepted RGB demos
-> canonical LeRobot export
-> ACT policy training
```

The main project pieces are:

- `src/scenethesis_mvp/pipeline/`: faithful scene-generation orchestration.
- `src/scenethesis_mvp/assets/`: asset registries, CLIP shortlisting, and
  visual profile helpers.
- `src/scenethesis_mvp/vision/`: image guidance, segmentation, depth, and
  validation helpers.
- `src/scenethesis_mvp/optimization/`: pose, SDF, and support/collision checks.
- `src/scenethesis_mvp/mujoco_bridge/`: SceneIR to MuJoCo, Panda runtime,
  teacher planning, evaluation, rendering, and replay.
- `src/scenethesis_mvp/lerobot_bridge/`: accepted demo filtering and dataset
  export.
- `scripts/`: command-line entry points.
- `configs/`: runtime, prompt, asset, MuJoCo, and artifact-sync configs.
- `docs/`: design notes, debugging reports, architecture notes, and checkpoint
  procedures.

See [docs/project_architecture_flowchart.md](docs/project_architecture_flowchart.md)
for a Mermaid architecture flowchart.

## What Gets Generated

The pipeline can produce:

- planned scene specs
- guidance images and validation reports
- segmentation, depth, and scene-graph artifacts
- asset correspondence reports
- Blender renders and visual-twin artifacts
- MuJoCo `scene.xml` and compiled `scene.mjb`
- strict teacher plan diagnostics
- grasp and completion probe reports
- rollout traces and dense state traces
- RGB frame streams
- MP4 debug videos
- accepted demo manifests
- canonical LeRobot-style datasets
- ACT policy checkpoints, when LeRobot is installed

Generated artifacts are intentionally ignored by Git. Keep large outputs under
`runs/`, `outputs/`, `data/`, or `models/`.

## Requirements

Core Python package:

- Python 3.10+
- `numpy`
- `pydantic`
- `PyYAML`
- `python-dotenv`
- `trimesh`
- `rtree`
- `openai`

Faithful scene-generation path:

- Windows or Linux with a CUDA-capable NVIDIA GPU
- Blender 5.x, or `BLENDER_PATH` pointing to Blender
- OpenAI API key in `.env`
- GroundingDINO, SAM, Depth Pro, RoMa, OpenCLIP, and PyTorch3D dependencies
- local warehouse asset registry and downloaded model checkpoints

MuJoCo / robot path:

- `mujoco`
- `coacd`
- `imageio`
- real Panda assets vendored or available through the configured model path

LeRobot training path:

- PyTorch with CUDA
- Hugging Face LeRobot installed on the training environment

## Installation

For the base package and tests:

```powershell
python -m pip install -e ".[dev]"
```

For MuJoCo evaluation:

```powershell
python -m pip install -e ".[dev,mujoco]"
```

For the local faithful environment, use the setup script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_faithful_env.ps1
```

Then download/import the external runtime assets that are not committed to Git:

```powershell
conda run -n scenethesis-faithful python scripts/download_faithful_checkpoints.py
conda run -n scenethesis-faithful python scripts/import_polyhaven_assets.py --resolution 1k
conda run -n scenethesis-faithful python scripts/import_hf_simready_assets.py
conda run -n scenethesis-faithful python scripts/derive_open_worktable.py
conda run -n scenethesis-faithful python scripts/build_clip_asset_index.py `
  --registry configs/warehouse_asset_registry.yaml `
  --out assets/indexes/warehouse_clip_index.npz `
  --device cuda
```

Never commit `.env` or real API keys.

## Runtime Check

Run a dependency/runtime check before starting an expensive generation job:

```powershell
conda run -n scenethesis-faithful python scripts/run_faithful.py `
  --prompt "warehouse storage area" `
  --out runs/runtime_check `
  --check-runtime-only
```

The check writes `faithful_runtime_report.json` and exits non-zero if a required
model, checkpoint, mesh, CUDA capability, OpenAI key, or Blender export is
missing.

## Faithful Scene Generation

Example warehouse run:

```powershell
conda run -n scenethesis-faithful python scripts/run_faithful.py `
  --prompt "a coherent indoor warehouse aisle with a pallet rack, cardboard boxes, wooden crates, a full walk-behind pallet stacker, one pallet, a packing table holding a clearly visible metal toolbox and hand drill, an orange plastic safety barrier, industrial lighting, and concrete floor markings" `
  --out runs/warehouse_grs_strict_next `
  --repair-rounds 1
```

Resume mode validates existing artifacts before continuing:

```powershell
conda run -n scenethesis-faithful python scripts/run_faithful.py `
  --prompt "a coherent indoor warehouse aisle with a pallet rack, cardboard boxes, wooden crates, a full walk-behind pallet stacker, one pallet, a packing table holding a clearly visible metal toolbox and hand drill, an orange plastic safety barrier, industrial lighting, and concrete floor markings" `
  --out runs/warehouse_grs_strict_next `
  --repair-rounds 1 `
  --resume-from-existing
```

Expected scene-generation outputs include `scene_spec.json`, `guidance.png`,
`guidance_validation.json`, `segmentation.json`, `depth.json`,
`scene_graph_3d.json`, `clip_shortlist.json`, `asset_correspondence.json`,
`scene.glb`, `scene.usd` when supported, `render.png`, `judge.json`,
`repair_history.json`, and `qualification.json`.

## MuJoCo Teacher Evaluation

The evaluator rebuilds the scene representation, compiles MuJoCo XML/MJB,
validates task feasibility, creates the MuJoCo environment, runs a policy, and
writes an episode report.

Strict single-episode teacher smoke:

```powershell
python scripts/evaluate_mujoco_scene.py `
  --run-dir runs/warehouse_gpt55_full_001 `
  --out outputs/lerobot_phase1/teacher_fix_probe_smoke `
  --target-object barcode_scanner_01 `
  --episodes 1 `
  --policy teacher_pick_place `
  --visual-renderer none `
  --run-grasp-probe
```

Useful output files:

- `mujoco_scene_ir.json`
- `scene.xml`
- `scene.mjb`
- `teacher_plan.json`
- `teacher_plan_search.json`
- `teacher_waypoint_diagnostics.json`
- `grasp_probe.json`
- `grasp_probe_search.json`
- `evaluation_report.json`
- `episodes/episode_000_trace.json`
- `episodes/episode_000_rollout_state_trace.json`

The strict teacher path tests:

- `teacher_plan.ok`
- selected base and grasp candidates
- IK and joint-limit feasibility
- static clearance
- grasp probe feasibility
- micro-lift
- completion preview
- collision count
- target lift
- release after place descent
- stable final placement

## Recording Demonstrations

Accepted demos are generated by running strict evaluation attempts and copying
only episodes that pass the success gates.

```powershell
python scripts/record_mujoco_demonstrations.py `
  --run-dir runs/warehouse_gpt55_full_001 `
  --target-object barcode_scanner_01 `
  --eval-out outputs/lerobot_phase1/demo_rollouts/warehouse_scanner_v001 `
  --demo-root data/lerobot_cache/raw_demos/warehouse_scanner_v001 `
  --attempts 50 `
  --min-accepted 5 `
  --policy teacher_pick_place `
  --visual-renderer none `
  --render-rgb
```

An episode is accepted only if it succeeds, attempts and verifies a grasp, lifts
the target, releases after grasp, places the target, avoids forbidden collisions,
does not drop the object, and does not violate the workspace.

## Exporting LeRobot Data

Validate the export path without requiring LeRobot:

```powershell
python scripts/export_lerobot_dataset_from_mujoco.py `
  --raw-demo-root data/lerobot_cache/raw_demos/warehouse_scanner_v001 `
  --out data/lerobot_cache/datasets/warehouse_scanner_v001 `
  --repo-id local/warehouse_scanner_v001 `
  --fps 20 `
  --canonical-only `
  --overwrite
```

After installing LeRobot, write the real LeRobot dataset:

```powershell
python scripts/export_lerobot_dataset_from_mujoco.py `
  --raw-demo-root data/lerobot_cache/raw_demos/warehouse_scanner_v001 `
  --out data/lerobot_cache/datasets/warehouse_scanner_v001 `
  --repo-id local/warehouse_scanner_v001 `
  --fps 20 `
  --overwrite
```

The export contains:

- `observation.state`
- `observation.images.overhead_rgb`
- `observation.images.wrist_rgb`
- `action`
- `task`

## ACT Training

Once a real LeRobot dataset exists locally:

```powershell
python scripts/train_lerobot_policy.py `
  --dataset-repo-id local/warehouse_scanner_v001 `
  --local-dataset-path data/lerobot_cache/datasets/warehouse_scanner_v001 `
  --output-dir outputs/lerobot_phase1/checkpoints/act_warehouse_scanner_v001 `
  --job-name act_warehouse_scanner_v001 `
  --device cuda
```

Dry-run the command construction first:

```powershell
python scripts/train_lerobot_policy.py `
  --local-dataset-path data/lerobot_cache/datasets/warehouse_scanner_v001 `
  --dry-run `
  --extra-arg=--steps=10
```

Evaluate a trained policy back in MuJoCo:

```powershell
python scripts/evaluate_mujoco_scene.py `
  --run-dir runs/warehouse_gpt55_full_001 `
  --out outputs/lerobot_phase1/evals/act_warehouse_scanner_v001 `
  --target-object barcode_scanner_01 `
  --episodes 5 `
  --policy lerobot `
  --policy-path outputs/lerobot_phase1/checkpoints/act_warehouse_scanner_v001 `
  --render-rgb `
  --visual-renderer both
```

The first ACT run should be treated as a pipeline proof. A useful single-task
policy will require more accepted demos and more object/pose variation than the
small smoke dataset.

## Rendering And Videos

RGB training frames are produced by MuJoCo cameras configured in the SceneIR.
MP4 debug videos are produced by MuJoCo offscreen rendering plus `imageio`.

```powershell
python scripts/evaluate_mujoco_scene.py `
  --run-dir runs/warehouse_gpt55_full_001 `
  --out outputs/lerobot_phase1/video_smoke `
  --target-object barcode_scanner_01 `
  --episodes 1 `
  --policy teacher_pick_place `
  --visual-renderer mujoco_debug `
  --save-video
```

Use these videos for visual QA before scaling data collection.

## Artifact Sync

Large datasets and checkpoints should stay out of Git. The sync helper can push
or pull artifacts through the configured backend:

```powershell
python scripts/sync_training_artifacts.py push-dataset --artifact-id warehouse_scanner_v001
python scripts/sync_training_artifacts.py pull-checkpoint --artifact-id act_warehouse_scanner_v001
```

See [docs/lerobot_phase1_checkpoints.md](docs/lerobot_phase1_checkpoints.md)
for the checkpoint workflow.

## Tests

Base tests:

```powershell
python -m pytest -q
```

Focused MuJoCo/LeRobot checks:

```powershell
python -m pytest -q tests/test_mujoco_bridge.py tests/test_lerobot_phase1.py tests/test_artifact_sync.py
```

Tests should not call OpenAI. Runtime-heavy generation and training are kept as
explicit command-line workflows.

## Repository Hygiene

Committed files should be source code, configs, tests, small authored manifests,
small documentation, and small representative thumbnails.

Do not commit:

- `.env`
- model checkpoints
- generated `runs/`
- generated `outputs/`
- local `data/` caches
- compiled datasets
- training checkpoints
- Blender caches
- raw frame dumps

Keep reproducibility in scripts, configs, manifests, and docs rather than by
checking in large generated artifacts.

## Limitations

Current limitations:

- The asset database is curated and small compared with Objaverse-scale systems.
- Scene generation quality depends strongly on the prompt, asset coverage, and
  visual validation.
- Gaussian splats or visual reconstructions still need a physics-ready geometry
  conversion stage before robot use.
- MuJoCo contact behavior depends on simplified collision meshes, masses,
  friction, and solver parameters.
- The deterministic Panda teacher is still candidate-search based and can be
  expensive when many base/grasp options must be probed.
- The current validated robot task is a controlled barcode-scanner pick-place
  scenario, not a broad benchmark.
- A small accepted demo set proves the data path, not a strong learned policy.

The next high-impact work is to add a cheap strict regression gate, scale to 50+
accepted RGB demos, install/export a real LeRobot dataset, run an ACT pilot,
evaluate the learned policy separately from the teacher, and then introduce
controlled variation in object pose, target placement, camera view, and layout.
