import json
import os
from pathlib import Path
import tomllib

import pytest

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
    assert manifest["extension_version"] == tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    # In the stable 0.3 schema the optional api_range, if present, must be
    # a string. Omission preserves the legacy parser's None semantics.
    assert "api_range" not in manifest["host"]
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


def test_diagnostic_environment_is_available_but_not_enabled_by_profile():
    profile = json.loads((ROOT / ".vllm-hust/optimization.json").read_text())
    activation = profile["activation"]
    for name in ("ADM_PACKED_SYNC_COUNT_PATHS", "ADM_PACKED_SYNC_CAPTURE_DIR", "ADM_DP4_REPORT_ACTIVE"):
        assert name in activation["extra_env_keys"]
        assert name not in activation["environment"]


def test_ecpa_current_schema_shape_when_reference_is_supplied():
    schema_path = os.getenv("ADM_ECPA_SCHEMA_FILE")
    if not schema_path:
        pytest.skip("set ADM_ECPA_SCHEMA_FILE to the reviewed Manager 0.3 schema")
    jsonschema = pytest.importorskip("jsonschema")
    manifest = _manifest()
    # Existing 0.3-experimental is a documented read-only Manager alias. The
    # normative schema itself accepts only 0.3; validate the shared field shape.
    assert manifest["schema_version"] == "0.3-experimental"
    manifest["schema_version"] = "0.3"
    jsonschema.Draft202012Validator(
        json.loads(Path(schema_path).read_text(encoding="utf-8"))
    ).validate(manifest)
