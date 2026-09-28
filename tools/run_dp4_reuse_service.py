"""Owner-run DP4 MOD comparison on frozen BF16 or W8A8 MoE traces."""

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/workspace/kmx')
PYTHON = ROOT / 'envs/adm-runtimequal-d7f1c842/bin/python'
VLLM = ROOT / 'envs/adm-runtimequal-d7f1c842/bin/vllm'
BENCH = ROOT / 'adm/bench-qwen35-adm-trace-20260927.py'
CORE = ROOT / 'worktrees/adm-hostqual-d7f1c842/third_party/vllm-hust'
ASCEND = ROOT / 'worktrees/adm-hostqual-d7f1c842/third_party/vllm-ascend-hust'
BENCH_REPO = ROOT / 'vllm-hust-benchmark'
GRAPH = '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'
PORT = 18088
VARIANTS = {
    'base': (ROOT / 'adm-mod-delivery-20260924',
             ROOT / 'artifacts/adm-dp4-base-20260928-Ea194E/runtime-plugin',
             '16362b2d6c229ec1c900b87cdb2c974039db95a6',
             'daa17b4d246653e6f98a99c3141fd84df556c60af4ca38fea5588085be33a7c0'),
    'candidate': (ROOT / 'worktrees/adm-dp4-reuse-bf16-20260928',
                  ROOT / 'artifacts/adm-dp4-base-20260928-Ea194E/runtime-plugin',
                  None,
                  '1ae1e4387bbdbd78f5da4fc9637abfb301e508656ed6db48d15cacebbb5b7cbd'),
}
MODELS = {
    'bf16': {
        'path': ROOT / 'models/Qwen3.5-35B-A3B-modelscope',
        'index': 'model.safetensors.index.json',
        'workload': ROOT / 'results/adm/qwen35-adm-trace-workload-20260927-8J25GT/workload.json',
        'workload_sha256': 'c9a800375f8810f17206f5930a2f189fb52e8a49890d27208bbee67092be2656',
        'served_name': 'Qwen3.5-35B-A3B',
        'tp': 2,
        'devices': '0,1,2,3,4,5,6,7',
    },
    'w8a8': {
        'path': ROOT / 'models/Qwen3-30B-A3B-w8a8',
        'index': 'quant_model_weight_w8a8_dynamic.safetensors.index.json',
        'workload': ROOT / 'results/adm/qwen3moe-coder-trace-workload-20260927-QotVCq/workload.json',
        'workload_sha256': '3010faf313424f6158a1e9acefdc92a2a9fdf1b296d33cffa1ef2dead3b87c70',
        'served_name': 'Qwen3-30B-A3B-W8A8',
        'tp': 1,
        'devices': '0,1,2,3',
    },
}


def git(repo, *args):
    return subprocess.check_output(['/usr/bin/git', '-C', str(repo), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=VARIANTS)
    parser.add_argument('--model', choices=MODELS, required=True)
    args = parser.parse_args()
    repo, plugin, revision, source_pin = VARIANTS[args.variant]
    config = MODELS[args.model]
    model = config['path']
    workload = config['workload']
    devices = config['devices']
    run_root = Path(subprocess.check_output([
        '/usr/bin/mktemp', '-d',
        str(ROOT / f'results/adm/dp4-reuse-{args.model}-{args.variant}-20260928-XXXXXX'),
    ], text=True).strip())
    print(f'run_root={run_root}', flush=True)
    expected = {
        repo: revision,
        CORE: 'e1248fa2655fdb8d72f80a5d6c28fa9c75b660c0',
        ASCEND: '367b8e62da799870a7476ce34f5f7658589a8aad',
        BENCH_REPO: 'e8a68923768adffb85c2d0fa255b2cec79af8ab5',
    }
    admission = {'variant': args.variant, 'repo': str(repo), 'plugin': str(plugin),
                 'model': str(model), 'graph': GRAPH, 'devices': devices,
                 'tp': config['tp'], 'dp': 4, 'port': PORT,
                 'workload_sha256': config['workload_sha256'],
                 'prefix_caching': False, 'revisions': {}}
    for path, pin in expected.items():
        head = git(path, 'rev-parse', 'HEAD')
        dirty = git(path, 'status', '--porcelain')
        admission['revisions'][str(path)] = {'head': head, 'dirty': dirty}
        assert pin is None or head == pin, f'revision changed: {path}: {head}'
        assert not dirty or path == BENCH_REPO and dirty == '?? results/', f'dirty: {path}: {dirty}'
        if path == repo and args.variant == 'candidate':
            assert git(path, 'branch', '--show-current') == 'perf/dp4-reuse-word-bf16-20260928'
    assert PYTHON.is_file() and VLLM.is_file() and plugin.is_dir()
    assert (model / config['index']).is_file()
    assert workload.is_file() and BENCH.is_file()
    assert hashlib.sha256(workload.read_bytes()).hexdigest() == config['workload_sha256']
    source = repo / 'src/ascend_distributed_metadata/packed_sync.py'
    admission['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert admission['source_sha256'] == source_pin
    admission['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    admission['bench_sha256'] = hashlib.sha256(BENCH.read_bytes()).hexdigest()
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', PORT))
    npu = subprocess.check_output(['npu-smi', 'info'], text=True)
    (run_root / 'npu-before.log').write_text(npu)
    for device in map(int, devices.split(',')):
        assert f'No running processes found in NPU {device}' in npu, f'NPU {device} busy'
    (run_root / 'admission.json').write_text(json.dumps(admission, indent=2) + '\n')
    print(f"source_sha256={admission['source_sha256']}", flush=True)

    env = os.environ.copy()
    env.update({
        'PYTHONPATH': f"{repo / 'src'}:{plugin}:{env.get('PYTHONPATH', '')}",
        'VLLM_VERSION': '0.20.2',
        'TORCH_DEVICE_BACKEND_AUTOLOAD': '0',
        'VLLM_ASCEND_TORCH_PREFLIGHT': '0',
        'VLLM_PLUGINS': 'ascend,ascend_kv_connector,ascend_model,ascend_model_loader,ascend_service_profiling,adm_packed_sync',
        'ADM_PACKED_SYNC_ENABLE': '1',
        'ADM_DP4_REPORT_ACTIVE': '1' if args.variant == 'candidate' else '0',
        'ASCEND_RT_VISIBLE_DEVICES': devices,
    })
    origins = subprocess.run(
        [str(PYTHON), '-c',
         'import importlib.util,sys; '
         'print("python_origin=",sys.executable); '
         '[(print(n+"_origin=",importlib.util.find_spec(n).origin)) '
         'for n in ("vllm","vllm_ascend","ascend_distributed_metadata")]; '
         'assert importlib.util.find_spec("ascend_distributed_metadata").origin == sys.argv[1]',
         str(repo / 'src/ascend_distributed_metadata/__init__.py')],
        env=env, text=True, capture_output=True,
    )
    (run_root / 'import-origins.log').write_text(origins.stdout + origins.stderr)
    assert origins.returncode == 0, f'import origin gate failed: {run_root / "import-origins.log"}'
    command = [str(VLLM), 'serve', str(model), '--host', '127.0.0.1',
               '--port', str(PORT), '--api-server-count', '1',
               '--served-model-name', config['served_name'],
               '--tensor-parallel-size', str(config['tp']), '--data-parallel-size', '4',
               '--max-model-len', '2048', '--max-num-seqs', '32',
               '--gpu-memory-utilization', '0.8', '--no-enable-prefix-caching',
               '--cudagraph-metrics', '--compilation-config', GRAPH]
    (run_root / 'serve-command.json').write_text(json.dumps(command, indent=2) + '\n')
    log = (run_root / 'server.log').open('w')
    server = subprocess.Popen(command, env=env, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    (run_root / 'server.pid').write_text(f'{server.pid}\n')
    try:
        deadline = time.monotonic() + 480
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'server exited {server.returncode}; see {run_root / "server.log"}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=3) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                time.sleep(5)
        else:
            raise TimeoutError('server did not become healthy within 480s')
        print('server_state=ready', flush=True)
        smoke = urllib.request.Request(
            f'http://127.0.0.1:{PORT}/v1/completions',
            data=json.dumps({'model': config['served_name'], 'prompt': 'Hello',
                             'max_tokens': 1, 'temperature': 0}).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(smoke, timeout=120) as response:
            assert response.status == 200
            (run_root / 'smoke.json').write_bytes(response.read())
        print('smoke=passed', flush=True)
        bench_env = env.copy()
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                    'http_proxy', 'https_proxy', 'all_proxy'):
            bench_env.pop(key, None)
        for round_no in (1, 2):
            bench_dir = run_root / f'bench-{round_no}'
            bench_dir.mkdir()
            bench = [str(PYTHON), str(BENCH), str(workload),
                     str(bench_dir / 'result.json'), '--port', str(PORT)]
            with (bench_dir / 'bench.log').open('w') as output:
                finished = subprocess.run(bench, env=bench_env, stdout=output,
                                          stderr=subprocess.STDOUT, timeout=300)
            print(f'round={round_no} bench_exit={finished.returncode}', flush=True)
            if finished.returncode:
                raise RuntimeError(f'benchmark failed: {bench_dir / "bench.log"}')
            result = json.loads((bench_dir / 'result.json').read_text())
            assert result['completed'] == 64 and result['failed'] == 0
            print(f"round={round_no} output_tok_s={result['output_throughput_tokens_per_s']:.3f} "
                  f"mean_latency_s={result['mean_latency_s']:.3f} "
                  f"p99_latency_s={result['p99_latency_s']:.3f}", flush=True)
        if args.variant == 'candidate':
            ranks = sorted(set(int(rank) for rank in re.findall(
                r'adm_dp4_word_active=rank:(\d+),dtype:int64',
                (run_root / 'server.log').read_text())))
            assert ranks == [0, 1, 2, 3], f'DP4 path did not activate on all ranks: {ranks}'
            print(f'dp4_active_ranks={ranks}', flush=True)
        print('qualification=passed', flush=True)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        log.close()
        (run_root / 'server.exit').write_text(f'{server.returncode}\n')
        print(f'server_exit={server.returncode} run_root={run_root}', flush=True)


if __name__ == '__main__':
    main()
