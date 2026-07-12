"""GB-4 (EXP-ASYNC-BATCH): bounded parallel fan-out for per-batch LLM calls.

Contract: LLM_BATCH_CONCURRENCY unset/<=1 -> plain sequential loop (control
cannot move); >1 -> thread-pool dispatch with results in SUBMISSION order
regardless of completion order; worker exceptions propagate like the
sequential form unless the worker catches them itself (GB-1 pattern).
"""
import threading
import time

import pytest

from multi_agent_kg.core.parallel import batch_concurrency, map_batches


def test_default_is_sequential_on_caller_thread(monkeypatch):
    monkeypatch.delenv("LLM_BATCH_CONCURRENCY", raising=False)
    assert batch_concurrency() == 1
    seen_threads = set()
    out = map_batches(lambda i, b: seen_threads.add(threading.get_ident()) or (i, b),
                      ["a", "b", "c"])
    assert out == [(0, "a"), (1, "b"), (2, "c")]
    assert seen_threads == {threading.get_ident()}


def test_unparseable_env_falls_back_to_sequential(monkeypatch):
    monkeypatch.setenv("LLM_BATCH_CONCURRENCY", "lots")
    assert batch_concurrency() == 1


def test_parallel_preserves_submission_order(monkeypatch):
    """Batch 0 finishes LAST — results must still come back in index order."""
    monkeypatch.setenv("LLM_BATCH_CONCURRENCY", "4")

    def worker(i, batch):
        time.sleep(0.15 if i == 0 else 0.01)
        return f"r{i}-{batch}"

    assert map_batches(worker, ["a", "b", "c", "d"]) == [
        "r0-a", "r1-b", "r2-c", "r3-d",
    ]


def test_parallel_actually_fans_out(monkeypatch):
    monkeypatch.setenv("LLM_BATCH_CONCURRENCY", "4")
    threads = set()
    barrier = threading.Barrier(3, timeout=5)

    def worker(i, batch):
        threads.add(threading.get_ident())
        barrier.wait()  # deadlocks (-> Barrier timeout) unless truly parallel
        return i

    assert map_batches(worker, [1, 2, 3]) == [0, 1, 2]
    assert len(threads) == 3


def test_worker_exception_propagates(monkeypatch):
    monkeypatch.setenv("LLM_BATCH_CONCURRENCY", "2")

    def worker(i, batch):
        if i == 1:
            raise ValueError("batch 1 failed")
        return i

    with pytest.raises(ValueError, match="batch 1 failed"):
        map_batches(worker, [1, 2, 3])


def test_verification_agent_parallel_matches_sequential(monkeypatch):
    """The real verification fan-out: same aggregation either way."""
    from multi_agent_kg.agents.extraction_verification_agent import (
        ExtractionVerificationAgent,
    )
    from multi_agent_kg.core import LLMConfig

    agent = ExtractionVerificationAgent.__new__(ExtractionVerificationAgent)

    def fake_call_llm(**kwargs):
        # One verified triple per call — three batches must aggregate to 3.
        return {
            "verified_triples": [{"subject": "s", "relation": "r", "object": "o"}],
            "verification_summary": {"total": 1, "verified": 1, "partial": 0,
                                     "rejected": 0, "hallucinated": 0},
        }

    agent.call_llm = fake_call_llm
    triples = [{"subject": f"s{i}", "relation": "r", "object": "o",
                "confidence": 0.9} for i in range(65)]  # 3 batches of 30

    monkeypatch.delenv("LLM_BATCH_CONCURRENCY", raising=False)
    seq = agent._verify_against_source("text", triples)
    monkeypatch.setenv("LLM_BATCH_CONCURRENCY", "3")
    par = agent._verify_against_source("text", triples)

    assert seq == par
    assert par["verification_summary"]["verified"] == 3  # one per batch
