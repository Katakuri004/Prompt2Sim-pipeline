# LeRobot Phase 1 Checkpoints

This phase keeps training and evaluation on local storage and uses Google Drive only as an explicit archive/retrieval layer.

## Storage

- Local minimum for the ACT pilot: 40-80 GB free.
- Recommended Drive allocation: 100 GB minimum, 200 GB+ if keeping multiple dataset and policy versions.
- Keep `scene.mjb` once per compiled scene. Do not duplicate it per episode.
- Evaluation sync excludes `visual_twin_frames/**` and `episodes/*_rgb/**` by default.

## Checkpoint 0: Sync Foundation

Local backend smoke test:

```powershell
python scripts\sync_training_artifacts.py push-dataset --artifact-id tiny --local-path data\lerobot_cache\datasets\tiny --backend local --remote-root outputs\sync_test_remote
python scripts\sync_training_artifacts.py pull-dataset --artifact-id tiny --local-path data\lerobot_cache\datasets\tiny_pulled --backend local --remote-root outputs\sync_test_remote
python scripts\sync_training_artifacts.py verify --artifact-id tiny --artifact-type dataset --local-path data\lerobot_cache\datasets\tiny_pulled
```

Google Drive backend:

```powershell
rclone config
rclone lsd gdrive:
python scripts\sync_training_artifacts.py push-dataset --artifact-id warehouse_scanner_v001
```

The configured remote is `gdrive:Scenethesis` in `configs/artifact_sync.yaml`.

## Checkpoint 1: Successful Demos

Record strict successful demos:

```powershell
python scripts\record_mujoco_demonstrations.py `
  --run-dir runs\warehouse_gpt55_full_001 `
  --target-object barcode_scanner_01 `
  --attempts 50 `
  --min-accepted 5 `
  --render-rgb
```

Only episodes with verified grasp, lift, place, release, stable placement, no drop, no workspace violation, and no bad collision are copied into `data/lerobot_cache/raw_demos/warehouse_scanner_v001`.

## Checkpoint 2: Dataset Export

Validate canonical export without requiring LeRobot:

```powershell
python scripts\export_lerobot_dataset_from_mujoco.py --canonical-only --overwrite
```

Write a real LeRobot dataset after installing LeRobot:

```powershell
python scripts\export_lerobot_dataset_from_mujoco.py --overwrite --repo-id local/warehouse_scanner_v001
python scripts\sync_training_artifacts.py push-dataset --artifact-id warehouse_scanner_v001
```

The exporter requires `observation.images.overhead_rgb`, `observation.images.wrist_rgb`, `observation.state`, 9D `action`, and `task`.

## Checkpoint 3: ACT Training

Dry-run:

```powershell
python scripts\train_lerobot_policy.py --local-dataset-path data\lerobot_cache\datasets\warehouse_scanner_v001 --dry-run --extra-arg=--steps=10
```

Train and push checkpoint:

```powershell
python scripts\train_lerobot_policy.py `
  --dataset-repo-id local/warehouse_scanner_v001 `
  --local-dataset-path data\lerobot_cache\datasets\warehouse_scanner_v001 `
  --output-dir outputs\lerobot_phase1\checkpoints\act_warehouse_scanner_v001 `
  --push-checkpoint
```

## Checkpoint 4: Policy Evaluation

Retrieve checkpoint if needed:

```powershell
python scripts\sync_training_artifacts.py pull-checkpoint --artifact-id act_warehouse_scanner_v001
```

Evaluate in MuJoCo with the real policy:

```powershell
python scripts\evaluate_mujoco_scene.py `
  --run-dir runs\warehouse_gpt55_full_001 `
  --out outputs\lerobot_phase1\evals\act_warehouse_scanner_v001 `
  --target-object barcode_scanner_01 `
  --episodes 5 `
  --policy lerobot `
  --policy-path outputs\lerobot_phase1\checkpoints\act_warehouse_scanner_v001 `
  --render-rgb `
  --visual-renderer both
```

## Checkpoint 5: Evaluation Archive

```powershell
python scripts\sync_training_artifacts.py push-eval `
  --artifact-id act_warehouse_scanner_v001_eval `
  --local-path outputs\lerobot_phase1\evals\act_warehouse_scanner_v001
```

Archive manifests are checksum verified and exclude temporary frame directories by default.
