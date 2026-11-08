"""Opt-in clean-wheel regression against an independently checked-out Manager."""

import json
import os
from pathlib import Path
import subprocess

import pytest


def test_clean_wheel_manager_lifecycle_fails_closed_without_a_host(tmp_path):
    python = os.environ.get("ADM_ECPA_MANAGER_PYTHON")
    manager = os.environ.get("ADM_ECPA_MANAGER_SRC")
    deps = os.environ.get("ADM_ECPA_MANAGER_DEPS")
    if not all((python, manager, deps)):
        pytest.skip("clean wheel and pinned Manager inputs not supplied")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((manager, deps))
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    env.pop("ADM_PACKED_SYNC_ENABLE", None)
    bundle = "org.vllm-hust.ascend-distributed-metadata"
    launch = [python, "-c", "from vllm_hust_ext.cli import main; raise SystemExit(main())"]

    def cli(*args):
        return subprocess.run(launch + list(args), cwd=tmp_path, env=env,
                              text=True, capture_output=True, timeout=30)

    listing = cli("extension", "list", "--json")
    assert listing.returncode == 0, listing.stderr
    assert any(row["bundle_id"] == bundle for row in json.loads(listing.stdout))
    for action in ("inspect", "check", "plan", "render"):
        result = cli("extension", action, bundle)
        assert result.returncode == 0, result.stderr
    assert cli("extension", "enable", bundle).returncode == 0
    status = cli("extension", "status", bundle)
    assert status.returncode == 0
    states = json.loads(status.stdout)["states"]
    assert "enabled" in states and {"degraded", "incompatible"} & set(states)
    assert "runtime_effective" not in states
    marker = tmp_path / "should-not-start"
    run = cli("run", "--", python, "-c", f"open({str(marker)!r},'w').write('bad')")
    assert run.returncode != 0 and not marker.exists()
    assert "refusing to launch" in run.stderr
    assert cli("extension", "disable", bundle).returncode == 0
    assert cli("extension", "forget", bundle).returncode == 0
