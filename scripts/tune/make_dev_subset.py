"""Build a deterministic paper-style dev subset (24 train / 12 test) for tuning.

Mirrors Cognee paper §5: a small, fixed, seeded sample so trials are cheap and the
train/held-out split is frozen before any tuning (no cherry-picking). Currently
implements LoComo (the first tune target); MAB uses a simple max_samples cap and does
not need a question-id allowlist.

Writes:
  evaluation/results/tune/locomo_train.json  -> {"by_sample": {sample_id: [qa_idx, ...]}}
  evaluation/results/tune/locomo_test.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scripts.tune.objective import SUBSET_DIR


def build_locomo(data_file: str, n_train: int, n_test: int, seed: int) -> None:
    samples = json.loads(Path(data_file).read_text())
    # Flatten to (sample_id, qa_index) pairs, skipping adversarial cat-5 (abstention)
    # which is noisy for F1 tuning.
    pairs = []
    for s in samples:
        sid = str(s.get("sample_id", ""))
        for i, qa in enumerate(s.get("qa", [])):
            if int(qa.get("category", 4)) == 5:
                continue
            pairs.append((sid, i))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    chosen = pairs[: n_train + n_test]
    train, test = chosen[:n_train], chosen[n_train:]

    def to_by_sample(subset):
        by_sample: dict[str, list[int]] = {}
        for sid, idx in subset:
            by_sample.setdefault(sid, []).append(idx)
        return {"by_sample": by_sample, "n": len(subset)}

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    (SUBSET_DIR / "locomo_train.json").write_text(json.dumps(to_by_sample(train), indent=2))
    (SUBSET_DIR / "locomo_test.json").write_text(json.dumps(to_by_sample(test), indent=2))
    print(f"Wrote {len(train)} train / {len(test)} test question ids to {SUBSET_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a frozen dev subset for tuning")
    ap.add_argument("--benchmark", choices=["locomo"], default="locomo")
    ap.add_argument("--data-file", required=True, help="Path to locomo10.json")
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    build_locomo(args.data_file, args.n_train, args.n_test, args.seed)


if __name__ == "__main__":
    main()
