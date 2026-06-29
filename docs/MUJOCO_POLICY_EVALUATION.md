# MuJoCo Policy Evaluation

This bridge evaluates policies inside an accepted Prompt2Sim/Scenethesis run directory.

The source of truth is:

- `scene_spec.json` for object identity, placement, and semantic roles
- the configured asset registry for mesh and dimensions
- `qualification.json` with `accepted=true`

`scene.glb` and `scene.usd` are preserved as visual/canonical artifacts, but they are not treated as physics semantics.

## Strict Pick-Place Evaluation

Use this for a real policy/task attempt:

```powershell
python scripts/evaluate_mujoco_scene.py `
  --run-dir runs/warehouse_asset_matched_final_001 `
  --out runs/warehouse_asset_matched_final_001/mujoco_eval_policy `
  --target-object OBJECT_ID `
  --episodes 1 `
  --policy scripted_pick_place `
  --save-video
```

Strict mode rejects physically invalid Panda tasks before rollout. For example, a warehouse crate wider than the Panda gripper is rejected instead of being teleported, welded, or counted as a false success. Inspect:

- `task_feasibility_report.json`
- `mujoco_runtime_report.json`
- `evaluation_report.json` when rollout runs
- `episodes/episode_000.mp4`
- `episodes/episode_000_trace.json`

## Render/Robot Debug Mode

Use this only to confirm the generated scene, Panda model, MuJoCo compilation, rendering, and trace/video export when the target is not graspable:

```powershell
python scripts/evaluate_mujoco_scene.py `
  --run-dir runs/warehouse_asset_matched_final_001 `
  --out runs/warehouse_asset_matched_final_001/mujoco_eval_render_debug `
  --target-object wooden_crate_01 `
  --episodes 1 `
  --policy noop `
  --save-video `
  --skip-task-feasibility
```

This mode is not a task success test. It is a visualization/debug check.

## Current Warehouse Run

`runs/warehouse_asset_matched_final_001` compiles and renders in MuJoCo, but its crate and boxes are too wide for the configured Franka/Panda gripper. Generate or select a run containing a small graspable object, such as a barcode scanner, small tool, or compact container, to run a strict pick-and-place task.
