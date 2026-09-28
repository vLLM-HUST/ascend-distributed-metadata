from types import SimpleNamespace

import pytest

from ascend_distributed_metadata import workload_routing


class Client:
    def __init__(self):
        self.lb_engines = [[0, 0], [0, 0]]
        self.core_engines = [b"rank-0", b"rank-1"]
        self.eng_start_index = 0
        self.client_count = 1
        self.reqs_in_flight = {}
        self.native_calls = 0

    def native_route(self, request):
        self.native_calls += 1
        rank = request.data_parallel_rank or 0
        identity = self.core_engines[rank]
        self.reqs_in_flight[request.request_id] = identity
        return identity


def request(name, input_len=128, output_len=128, **kwargs):
    return SimpleNamespace(
        request_id=name,
        prompt_token_ids=[1] * input_len,
        sampling_params=SimpleNamespace(max_tokens=output_len),
        pooling_params=None,
        data_parallel_rank=None,
        **kwargs,
    )


def test_long_request_metadata_changes_next_tied_choice():
    client = Client()
    route = workload_routing._wrap(Client.native_route)

    assert route(client, request("long", 3072, 256)) == b"rank-0"
    assert route(client, request("short", 128, 128)) == b"rank-1"
    # Counts now tie. The extra work assigned to rank 0 breaks the tie.
    assert route(client, request("third", 128, 128)) == b"rank-1"
    assert client.lb_engines == [[1, 0], [2, 0]]
    assert client.native_calls == 0


def test_long_decode_budget_keeps_short_requests_on_other_rank():
    client = Client()
    route = workload_routing._wrap(Client.native_route)

    assert route(client, request("long", 128, 512)) == b"rank-0"
    for index in range(5):
        assert route(client, request(f"short-{index}", 128, 32)) == b"rank-1"
    assert route(client, request("next", 128, 32)) == b"rank-0"


def test_completed_request_is_pruned_before_the_next_choice():
    client = Client()
    route = workload_routing._wrap(Client.native_route)
    route(client, request("long", 3072, 256))
    del client.reqs_in_flight["long"]
    client.lb_engines = [[0, 0], [0, 0]]

    assert route(client, request("next")) == b"rank-0"
    assert set(client._adm_pending_work) == {"next"}


def test_explicit_rank_and_unsupported_request_use_native_route():
    client = Client()
    route = workload_routing._wrap(Client.native_route)
    explicit = request("explicit")
    explicit.data_parallel_rank = 1
    unsupported = request("unsupported")
    unsupported.sampling_params.max_tokens = None

    assert route(client, explicit) == b"rank-1"
    assert route(client, unsupported) == b"rank-0"
    assert client.native_calls == 2
    assert not hasattr(client, "_adm_pending_work")


def test_install_requires_reviewed_method_and_is_idempotent(monkeypatch):
    class Target(Client):
        get_core_engine_for_request = Client.native_route

    with pytest.raises(workload_routing.RoutingViolation, match="reviewed pin"):
        workload_routing.install(Target)
    monkeypatch.setattr(
        workload_routing, "_fingerprint", lambda _fn: workload_routing.TARGET_FINGERPRINT
    )
    assert workload_routing.install(Target)
    assert not workload_routing.install(Target)


def test_disabled_plugin_does_not_import_vllm(monkeypatch):
    monkeypatch.setenv(workload_routing.ENABLE_ENV, "0")
    workload_routing.register()
