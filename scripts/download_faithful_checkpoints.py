from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = {
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "groundingdino_swint_ogc.pth": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "roma_outdoor.pth": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
    "dinov2_vitl14_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}

DEPTH_PRO_CHECKPOINT = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"

REPOS = {
    "GroundingDINO": "https://github.com/IDEA-Research/GroundingDINO.git",
    "ml-depth-pro": "https://github.com/apple/ml-depth-pro.git",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real checkpoints required by faithful Scenethesis mode.")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    checkpoints_dir = root / "models" / "checkpoints"
    repos_dir = root / "models" / "repos"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    for name, url in CHECKPOINTS.items():
        download_file(url, checkpoints_dir / name)
    for name, url in REPOS.items():
        clone_repo(url, repos_dir / name)
    download_depth_pro_weights(repos_dir / "ml-depth-pro")
    print("Faithful checkpoints/repositories are present.")


def download_file(url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size > 0:
        print(f"exists: {target}")
        return
    partial = target.with_suffix(target.suffix + ".part")
    print(f"downloading: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "scenethesis-mvp/0.1"})
    with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if partial.stat().st_size == 0:
        raise RuntimeError(f"download produced an empty file: {target}")
    partial.replace(target)
    print(f"wrote: {target}")


def clone_repo(url: str, target: Path) -> None:
    if target.exists():
        print(f"exists: {target}")
        return
    if not shutil.which("git"):
        raise RuntimeError("git is required to clone faithful model repositories.")
    subprocess.run(["git", "clone", url, str(target)], check=True)


def download_depth_pro_weights(depth_pro_dir: Path) -> None:
    if not (depth_pro_dir / "pyproject.toml").is_file():
        raise RuntimeError(f"Depth Pro repo is missing or incomplete: {depth_pro_dir}")
    checkpoints_dir = depth_pro_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    download_file(DEPTH_PRO_CHECKPOINT, checkpoints_dir / "depth_pro.pt")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        sys.exit(1)
