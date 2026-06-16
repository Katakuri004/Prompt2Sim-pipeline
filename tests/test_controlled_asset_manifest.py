from pathlib import Path

from scenethesis_mvp.schemas.asset_manifest import ControlledAssetManifest, manifest_source_dir
from scenethesis_mvp.utils.io import read_yaml


def test_hf_simready_manifest_is_controlled_and_licensed() -> None:
    manifest = ControlledAssetManifest.model_validate(read_yaml(Path("configs/hf_simready_warehouse_manifest.yaml")))

    assert manifest.bulk_download_allowed is False
    assert manifest.entries
    assert {entry.category for entry in manifest.entries} >= {"forklift", "pallet", "pallet_load", "cart", "scanner", "barrier"}
    assert {target.category for target in manifest.unresolved_targets} >= {"conveyor", "robot_arm"}
    for entry in manifest.entries:
        assert entry.license
        assert entry.attribution
        assert entry.source_usd.endswith(".usd")
        assert manifest_source_dir(entry)
