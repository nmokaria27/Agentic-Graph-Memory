"""Bounded parallel fan-out for per-batch LLM calls (GB-4).

Sequential by default: with ``LLM_BATCH_CONCURRENCY`` unset or <= 1 the
dispatch is a plain ordered loop, byte-identical to the pre-GB-4 behavior
(controls must not move). With concurrency > 1, independent batches are
dispatched through a bounded thread pool; results are always collected in
SUBMISSION ORDER regardless of completion order, so aggregation downstream
is deterministic either way. Workers must confine themselves to the LLM
call and batch-local state — all cross-batch aggregation stays on the
caller's thread.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Sequence, TypeVar

B = TypeVar("B")
R = TypeVar("R")


def batch_concurrency() -> int:
    """Read LLM_BATCH_CONCURRENCY at call time (>=1; unparseable -> 1)."""
    try:
        return max(1, int(os.environ.get("LLM_BATCH_CONCURRENCY", "1")))
    except ValueError:
        return 1


def map_batches(worker: Callable[[int, B], R], batches: Sequence[B]) -> List[R]:
    """Run ``worker(index, batch)`` over every batch, results in index order.

    An exception raised by any worker propagates to the caller (for parallel
    dispatch, on the ``.result()`` of the earliest-submitted failing batch) —
    the same stage-level failure semantics as the sequential loop. Workers
    that must survive their own failures (GB-1 degradation) catch inside and
    return a sentinel, exactly as they would in the sequential form.
    """
    workers = min(batch_concurrency(), len(batches))
    if workers <= 1:
        return [worker(i, b) for i, b in enumerate(batches)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, i, b) for i, b in enumerate(batches)]
        return [f.result() for f in futures]
