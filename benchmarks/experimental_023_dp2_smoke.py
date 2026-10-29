"""Bounded, synthetic four-card diagnostic. Not a dataset score or benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

MODEL = "/data/shared_upload/models/vspec/Qwen3.5-35B-A3B"
PROMPTS = [
    "Explain what a key-value cache does in a transformer.",
    "Give three reasons to validate a numerical experiment.",
    "Explain the difference between latency and throughput.",
    "Describe how a distributed process group can synchronize metadata.",
]
PINS = {"/vllm-workspace/vllm": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "/vllm-workspace/vllm-ascend": "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd"}


def request(port, endpoint, payload=None, timeout=3):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{endpoint}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["off", "on"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18761)
    parser.add_argument("--capture-window", action="store_true",
                        help="ON only: diagnostic worker-boundary capture with extra requests")
    args = parser.parse_args()
    if args.capture_window and args.arm != "on":
        parser.error("worker-path capture requires the plugin ON arm")
    for root, expected in PINS.items():
        actual = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        if actual != expected or subprocess.run(["git", "-C", root, "diff", "--quiet"]).returncode or subprocess.run(["git", "-C", root, "diff", "--cached", "--quiet"]).returncode:
            raise RuntimeError(f"unreviewed or dirty runtime: {root}")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
        "--host", "127.0.0.1", "--port", str(args.port), "--served-model-name", "adm-probe",
        "--tensor-parallel-size", "2", "--data-parallel-size", "2",
        "--distributed-executor-backend", "mp", "--dtype", "bfloat16",
        "--max-model-len", "4096", "--max-num-seqs", "4", "--max-num-batched-tokens", "4096",
        "--gpu-memory-utilization", "0.9", "--kv-cache-memory-bytes", "2147483648",
        "--seed", "17", "--enforce-eager", "--enable-chunked-prefill",
        "--limit-mm-per-prompt", '{"image":0,"video":0}', "--generation-config", "vllm"]
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("ADM_"):
            env.pop(key)
    env.update({"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_PLUGINS": "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,adm_packed_sync",
        "ADM_PACKED_SYNC_ENABLE": "1" if args.arm == "on" else "0",
        "ADM_EXPERIMENTAL_023_ENABLE": "1", "ADM_DP2_REPORT_ACTIVE": "1",
        "NO_PROXY": "localhost,127.0.0.1,::1", "no_proxy": "localhost,127.0.0.1,::1"})
    capture_root = args.output / f"capture-{args.output.name}"
    if args.capture_window:
        capture_root.mkdir()
        env.update({"ADM_PACKED_SYNC_COUNT_PATHS": "1", "ADM_PACKED_SYNC_CAPTURE_DIR": str(capture_root.resolve())})
    receipt = {"arm": args.arm, "command": cmd, "seed": 17, "devices": [0, 1, 2, 3], "source_pins": PINS,
               "prompts_sha256": hashlib.sha256(json.dumps(PROMPTS, separators=(",", ":")).encode()).hexdigest(),
               "classification": "synthetic eager diagnostic, not official score or performance claim",
               "status": "starting", "results": []}
    path = args.output / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2))
    server = None
    try:
        with (args.output / "server.log").open("w") as log:
            server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}")
                try:
                    request(args.port, "/v1/models")
                    break
                except Exception:
                    time.sleep(3)
            else:
                raise TimeoutError("server did not become ready in 900 seconds")
            def invoke(prompt):
                start = time.monotonic()
                result = request(args.port, "/v1/completions", {
                    "model": "adm-probe", "prompt": prompt, "temperature": 0,
                    "seed": 17, "max_tokens": 32, "ignore_eos": True}, timeout=240)
                choice = result["choices"][0]
                return {"elapsed_s": time.monotonic() - start,
                        "text_sha256": hashlib.sha256(choice["text"].encode()).hexdigest(),
                        "finish_reason": choice["finish_reason"], "usage": result.get("usage")}
            with ThreadPoolExecutor(max_workers=4) as pool:
                if args.capture_window:
                    (capture_root / "begin").touch(exist_ok=False)
                    list(pool.map(invoke, PROMPTS))  # triggers per-worker begin boundaries
                    if len(list(capture_root.glob("begin-*.json"))) != 4:
                        raise RuntimeError("missing per-worker begin captures")
                receipt["results"] = list(pool.map(invoke, PROMPTS))
                if args.capture_window:
                    (capture_root / "end").touch(exist_ok=False)
                    list(pool.map(invoke, PROMPTS))  # triggers end boundaries
                    begins = sorted(capture_root.glob("begin-*.json"))
                    ends = sorted(capture_root.glob("end-*.json"))
                    if len(begins) != 4 or len(ends) != 4:
                        raise RuntimeError("missing per-worker end captures")
                    from ascend_distributed_metadata.evidence import captured_window
                    windows = []
                    for begin in begins:
                        end = capture_root / begin.name.replace("begin-", "end-", 1)
                        windows.append(captured_window(json.loads(begin.read_text()), json.loads(end.read_text())))
                    if sorted(w["dp_rank"] for w in windows) != [0, 0, 1, 1]:
                        raise RuntimeError("unexpected per-worker DP rank coverage")
                    receipt["worker_windows"] = windows
                    receipt["classification"] += "; capture enabled, timing is observer-perturbed"
            receipt["status"] = "complete"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
        path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
