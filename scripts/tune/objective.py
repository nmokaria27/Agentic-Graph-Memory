"""Shared objective: turn a sampled config into a scalar benchmark score.

Both Stage A (retrieval-only, cached KGs) and Stage B (chunk_size grid) call
``evaluate``. LoComo is optimized first; MAB shares the same entry via ``--benchmark``.

Scalars:
  LoComo -> aggregate_scores()['overall_f1']
  MAB    -> aggregate_metrics()['accuracy']  (substring-EM primary)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from multi_agent_kg.core.config import AnswerFormatConfig, LLMConfig, RetrievalConfig

REPO = Path(__file__).resolve().parents[2]
SUBSET_DIR = REPO / "evaluation" / "results" / "tune"


def _load_mab_module():
    """Import the MAB run_eval module despite its hyphenated package directory.

    run_eval.py uses a sibling import (`from agent_graph_memory_adapter import ...`),
    so its own directory must be on sys.path before exec.
    """
    mab_dir = REPO / "evaluation" / "Memory-Agent-Bench"
    if str(mab_dir) not in sys.path:
        sys.path.insert(0, str(mab_dir))
    path = mab_dir / "run_eval.py"
    spec = importlib.util.spec_from_file_location("mab_run_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_subset(benchmark: str, split: str) -> Optional[Dict[str, Any]]:
    """Return the persisted dev-subset selection for a benchmark/split, or None."""
    path = SUBSET_DIR / f"{benchmark}_{split}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _filter_locomo_samples(samples: List[Dict], subset: Optional[Dict[str, Any]]) -> List[Dict]:
    """Restrict LoComo samples to the (sample_id -> [qa indices]) allowlist."""
    if not subset:
        return samples
    keep = subset.get("by_sample", {})
    out = []
    for s in samples:
        sid = str(s.get("sample_id", ""))
        if sid not in keep:
            continue
        idxs = set(keep[sid])
        s2 = dict(s)
        s2["qa"] = [qa for i, qa in enumerate(s.get("qa", [])) if i in idxs]
        if s2["qa"]:
            out.append(s2)
    return out


def evaluate(
    benchmark: str,
    retrieval_kwargs: Dict[str, Any],
    answer_format_kwargs: Dict[str, Any],
    *,
    split: str = "train",
    load_kg_dir: Optional[str] = None,
    save_kg_dir: Optional[str] = None,
    data_file: Optional[str] = None,
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    chunk_size: Optional[int] = None,
    dataset: str = "eventqa",
    max_samples: int = 24,
) -> float:
    """Run one full eval for a sampled config and return its scalar objective."""
    model = model or os.environ.get("LLM_DEFAULT_MODEL", "gemma4:31b")
    rc = RetrievalConfig(embedding_model=embedding_model, **retrieval_kwargs)
    af = AnswerFormatConfig(**answer_format_kwargs)
    llm = LLMConfig(model=model)

    if benchmark == "locomo":
        from evaluation.LoComo import run_eval as L

        if not data_file:
            raise ValueError("locomo objective requires --data-file")
        samples = json.loads(Path(data_file).read_text())
        samples = _filter_locomo_samples(samples, load_subset("locomo", split))
        result = L.run_locomo_eval(
            samples=samples,
            llm_config=llm,
            retrieval_config=rc,
            load_kg_dir=Path(load_kg_dir) if load_kg_dir else None,
            save_kg_dir=Path(save_kg_dir) if save_kg_dir else None,
            answer_format=af,
            chunk_size=chunk_size,
        )
        return float(result["aggregate"]["overall_f1"])

    if benchmark == "mab":
        mab = _load_mab_module()
        agent = mab.AgentGraphMemoryWrapper(
            model=model,
            embedding_model=embedding_model,
            retrieval_config=rc,
            answer_format=af,
            verbose=False,
        )
        result = mab.run_dataset_eval(
            dataset_name=dataset,
            agent=agent,
            max_samples=max_samples,
            load_kg_dir=Path(load_kg_dir) / dataset if load_kg_dir else None,
        )
        return float(result["aggregate"]["accuracy"])

    raise ValueError(f"Unknown benchmark {benchmark!r}; expected 'locomo' or 'mab'")


__all__ = ["evaluate", "load_subset", "SUBSET_DIR"]
