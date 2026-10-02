import json
from pathlib import Path

import tomllib

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src" / "ascend_distributed_metadata"
BUNDLE_ID = "org.vllm-hust.ascend-distributed-metadata"


def _manifest():
    return json.loads(
        (PACKAGE / "vllm-hust-extension-v0.3.json").read_text(encoding="utf-8")
    )


def test_ecpa_registration_and_activation_share_the_distribution():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry_points = project["project"]["entry-points"]

    assert entry_points["vllm_hust.extension_bundles"] == {
        BUNDLE_ID: "ascend_distributed_metadata"
    }
    assert entry_points["vllm.general_plugins"] == {
        "adm_packed_sync": "ascend_distributed_metadata.packed_sync:register"
    }


def test_ecpa_manifest_keeps_install_separate_from_enablement():
    manifest = _manifest()

    assert manifest["schema_version"] == "0.3-experimental"
    assert manifest["extension_id"] == BUNDLE_ID
    assert manifest["activation"]["entry_points"] == [
        {"group": "vllm.general_plugins", "name": "adm_packed_sync"}
    ]
    assert manifest["activation"]["environment"] == {
        "ADM_PACKED_SYNC_ENABLE": "1"
    }
    assert manifest["resource_claims"] == [
        {
            "resource": "vllm-ascend.dp-metadata-sync",
            "scope": "vllm-workers",
            "mode": "exclusive",
        }
    ]


def test_ecpa_manifest_pins_the_reviewed_runtime_qualification():
    qualification = _manifest()["activation"]["additional_config"][
        "_manager_runtime_qualification"
    ]

    assert qualification["vllm_commit"] == (
        "e1248fa2655fdb8d72f80a5d6c28fa9c75b660c0"
    )
    assert qualification["vllm_ascend_commit"] == (
        "367b8e62da799870a7476ce34f5f7658589a8aad"
    )
