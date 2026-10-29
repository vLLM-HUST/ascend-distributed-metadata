"""Source identity must be stable across Python versions and fail closed."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ascend_distributed_metadata import packed_sync


FIXTURE = Path(__file__).parent / "fixtures" / "reviewed_sync.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewed_source_matches_exact_pin():
    fn = load(FIXTURE, "reviewed_source")._sync_metadata_across_dp
    assert packed_sync._fingerprint(fn) == packed_sync.TARGET_FINGERPRINT


def test_python_311_and_current_runtime_produce_same_fingerprint():
    python311 = shutil.which("python3.11")
    if python311 is None:
        pytest.skip("Python 3.11 interpreter not available")
    script = (
        "import importlib.util; from ascend_distributed_metadata.packed_sync import _fingerprint; "
        f"s=importlib.util.spec_from_file_location('reviewed', {str(FIXTURE.resolve())!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(_fingerprint(m._sync_metadata_across_dp))"
    )
    env = os.environ.copy()
    src = str(FIXTURE.resolve().parents[2] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    hashes = [subprocess.check_output([python, "-c", script], text=True, env=env).strip()
              for python in (python311, sys.executable)]
    assert hashes == [packed_sync.TARGET_FINGERPRINT] * 2


def test_changed_host_source_and_old_signature_are_rejected(tmp_path):
    modified = tmp_path / "changed.py"
    source = FIXTURE.read_text().replace("if self.dp_size == 1:", "if self.dp_size == 2:")
    assert source != FIXTURE.read_text()
    modified.write_text(source)
    assert packed_sync._fingerprint(load(modified, "changed_source")._sync_metadata_across_dp) != (
        packed_sync.TARGET_FINGERPRINT
    )

    class Old018:
        def _sync_metadata_across_dp(self, num_tokens, with_prefill=False, is_draft_model=False):
            return num_tokens, None, with_prefill

    with pytest.raises(packed_sync.PackedSyncViolation, match="reviewed pin"):
        packed_sync.install(Old018, object())


@pytest.mark.skipif(sys.version_info < (3, 12), reason="generic syntax requires Python 3.12+")
def test_generic_source_does_not_normalize_away_type_parameters(tmp_path):
    source = tmp_path / "generic.py"
    source.write_text("def generic[T](x: T) -> T:\n    return x\n")
    assert packed_sync._fingerprint(load(source, "generic_source").generic) is None
