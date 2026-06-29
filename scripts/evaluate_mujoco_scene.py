from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.mujoco_bridge.evaluator import evaluate_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile an accepted Scenethesis run into MuJoCo and evaluate policy rollouts.")
    parser.add_argument("--run-dir", required=True, help="Accepted faithful run directory.")
    parser.add_argument("--out", required=True, help="MuJoCo evaluation output directory.")
    parser.add_argument("--config", default="configs/mujoco_eval.yaml")
    parser.add_argument("--target-object", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--policy", default="scripted_pick_place", choices=["teacher_pick_place", "teacher_delta_debug", "scripted_pick_place", "noop", "random", "lerobot"])
    parser.add_argument("--policy-path", default=None, help="Local LeRobot checkpoint path when --policy lerobot.")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--render-rgb", action="store_true", help="Render configured RGB observations during rollout.")
    parser.add_argument("--save-video", action="store_true", help="Save strict MuJoCo MP4 replay and first-frame PNG per episode.")
    parser.add_argument("--video-camera", default=None, help="Camera name for replay video; defaults to config visualization.camera.")
    parser.add_argument(
        "--visual-renderer",
        default="both",
        choices=["blender", "mujoco_debug", "both", "none"],
        help="Benchmark visual renderer. Blender is the visual-twin renderer; MuJoCo video is debug only.",
    )
    parser.add_argument("--blender-path", default=None, help="Optional Blender executable path for visual-twin rendering.")
    parser.add_argument("--debug-teacher-plan", action="store_true", help="Write strict teacher waypoint planning diagnostics.")
    parser.add_argument("--run-grasp-probe", action="store_true", help="Run the pre-rollout scanner grasp feasibility probe.")
    parser.add_argument(
        "--skip-task-feasibility",
        action="store_true",
        help="Skip strict graspability/reachability validation. This should only be used for render-only debugging.",
    )
    args = parser.parse_args()

    report = evaluate_scene(
        run_dir=args.run_dir,
        out_dir=args.out,
        config_path=args.config,
        target_object=args.target_object,
        episodes=args.episodes,
        policy_id=args.policy,
        policy_path=args.policy_path,
        compile_only=args.compile_only,
        render_rgb=args.render_rgb,
        save_video=args.save_video,
        video_camera=args.video_camera,
        strict_task=not args.skip_task_feasibility,
        visual_renderer=args.visual_renderer,
        blender_path=args.blender_path,
        debug_teacher_plan=args.debug_teacher_plan,
        run_grasp_probe=args.run_grasp_probe,
    )
    print(f"scene_id: {report.scene_id}")
    print(f"policy: {report.policy_id}")
    print(f"episodes: {len(report.episodes)}")
    print(f"success_rate: {report.success_rate:.3f}")
    print(f"xml: {report.compile_artifacts.get('xml')}")
    if report.compile_artifacts.get("mjb"):
        print(f"mjb: {report.compile_artifacts['mjb']}")
    for artifact in report.visualization_artifacts:
        print(f"artifact: {artifact}")
    if report.visual_twin:
        print(f"visual_twin_status: {report.visual_twin.get('status')}")
        if report.visual_twin.get("reason"):
            print(f"visual_twin_reason: {report.visual_twin['reason']}")


if __name__ == "__main__":
    main()
