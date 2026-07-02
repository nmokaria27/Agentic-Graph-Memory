"""Compute orphan/degeneracy stats for cached governed KGs.

Usage:
    python evals/orphan_stats.py evals/kg_cache/eventqa          # all contexts
    python evals/orphan_stats.py evals/kg_cache/eventqa/context_0/governed_kg.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.evaluate_kg import compute_rogue_entity_stats, compute_degeneracy_rate


def _load_kg(path: Path):
    data = json.loads(path.read_text())
    kg = data.get("knowledge_graph", data)
    return kg.get("entities", []), kg.get("triples", [])


def _report(name: str, entities, triples):
    rogue = compute_rogue_entity_stats(entities, triples)
    degen = compute_degeneracy_rate(triples)
    print(f"\n=== {name} ===")
    print(f"  entities={len(entities)}  triples={len(triples)}")
    print(f"  rogue_entity_rate = {rogue['rogue_entity_rate']:.3f} "
          f"({rogue['rogue_entity_count']}/{len(entities)})  by_type={rogue['by_type']}")
    print(f"  degeneracy_rate   = {degen['degeneracy_rate']:.3f} "
          f"({degen['degenerate_count']}/{len(triples)})  by_type={degen['by_type']}")
    return rogue, degen, len(entities), len(triples)


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/kg_cache/eventqa")
    if target.is_file():
        e, t = _load_kg(target)
        _report(target.parent.name, e, t)
        return
    kg_files = sorted(target.glob("context_*/governed_kg.json"))
    if not kg_files:
        print(f"No governed_kg.json under {target}")
        return
    tot_e = tot_t = tot_rogue = tot_orphan = tot_degen = 0
    for f in kg_files:
        e, t = _load_kg(f)
        rogue, degen, ne, nt = _report(f.parent.name, e, t)
        tot_e += ne; tot_t += nt
        tot_rogue += rogue["rogue_entity_count"]
        tot_orphan += rogue["by_type"].get("ORPHAN_ENTITY", 0)
        tot_degen += degen["degenerate_count"]
    print(f"\n=== AGGREGATE ({len(kg_files)} contexts) ===")
    print(f"  entities={tot_e}  triples={tot_t}")
    print(f"  rogue_rate={tot_rogue/tot_e if tot_e else 0:.3f}  "
          f"orphan_rate={tot_orphan/tot_e if tot_e else 0:.3f}  "
          f"degeneracy_rate={tot_degen/tot_t if tot_t else 0:.3f}")


if __name__ == "__main__":
    main()
