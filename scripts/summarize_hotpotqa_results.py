"""Print a compact HotpotQA pilot table from result JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_system_rows(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Tuple[str, Dict[str, Any]]] = []
    if "aggregate_metrics" in data:
        rows.append((str(data.get("system", path.stem)), data["aggregate_metrics"]))
    for system, run in data.get("runs", {}).items():
        rows.append((system, run.get("aggregate_metrics", {})))
    return rows


def _format_table(rows: Iterable[Tuple[str, Dict[str, Any]]]) -> str:
    header = "| System | EM | Token F1 | Answer rate | N |"
    sep = "|---|---:|---:|---:|---:|"
    lines = [header, sep]
    for system, metrics in rows:
        lines.append(
            "| {system} | {em:.3f} | {f1:.3f} | {rate:.3f} | {n} |".format(
                system=system,
                em=float(metrics.get("exact_match", 0.0)),
                f1=float(metrics.get("token_f1", 0.0)),
                rate=float(metrics.get("answer_rate", 0.0)),
                n=int(metrics.get("num_questions", 0)),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="HotpotQA result JSON files")
    args = parser.parse_args()

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"warning: missing {path}")
            continue
        rows.extend(_load_system_rows(path))
    print(_format_table(rows))


if __name__ == "__main__":
    main()
