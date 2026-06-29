from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.lerobot_bridge.demo_acceptance import collect_successful_demos
from scenethesis_mvp.mujoco_bridge.evaluator import evaluate_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Record strict successful MuJoCo demos for LeRobot Phase 1.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-object", default="barcode_scanner_01")
    parser.add_argument("--dataset-id", default="warehouse_scanner_v001")
    parser.add_argument("--eval-out", default="outputs/lerobot_phase1/demo_rollouts/warehouse_scanner_v001")
    parser.add_argument("--demo-root", default="data/lerobot_cache/raw_demos/warehouse_scanner_v001")
    parser.add_argument("--config", default="configs/mujoco_eval.yaml")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--min-accepted", type=int, default=5)
    parser.add_argument("--policy", default="teacher_pick_place", choices=["teacher_pick_place", "scripted_pick_place", "noop", "random", "lerobot"])
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--render-rgb", action="store_true")
    parser.add_argument("--visual-renderer", default="none", choices=["blender", "mujoco_debug", "both", "none"])
    args = parser.parse_args()

    evaluate_scene(
        run_dir=args.run_dir,
        out_dir=args.eval_out,
        config_path=args.config,
        target_object=args.target_object,
        episodes=args.attempts,
        policy_id=args.policy,
        policy_path=args.policy_path,
        render_rgb=args.render_rgb,
        save_video=False,
        visual_renderer=args.visual_renderer,
    )
    manifest = collect_successful_demos(
        evaluation_dir=args.eval_out,
        demo_root=args.demo_root,
        dataset_id=args.dataset_id,
        min_accepted=args.min_accepted,
    )
    print(f"dataset_id: {manifest['dataset_id']}")
    print(f"accepted_count: {manifest['accepted_count']}")
    print(f"rejected_count: {manifest['rejected_count']}")
    print(f"demo_manifest: {Path(args.demo_root) / 'demo_manifest.json'}")


if __name__ == "__main__":
    main()
