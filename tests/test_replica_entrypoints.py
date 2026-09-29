"""The coordinator and API consumer must load the same opt-in replica."""

from pathlib import Path
import tomllib


def test_replica_entrypoint_is_available_to_both_runtime_processes():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    groups = project["project"]["entry-points"]
    expected = "ascend_distributed_metadata.replica_plugin:register"
    assert groups["vllm.general_plugins"]["adm_epoch_replica"] == expected
    assert groups["vllm.dp_metadata_plugins"]["adm_epoch_replica"] == expected
