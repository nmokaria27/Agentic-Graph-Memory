"""
Smoke test for the Pydantic LLM-boundary validation (entity + relation extractors).

Two independent parts:

  PART A — OFFLINE (no LLM/GPU needed, runs in <1s):
    Feeds deliberately-malformed LLM-shaped payloads straight through the
    schemas + the extractor's _coerce_llm_items path to prove:
      - malformed items are coerced/dropped, never crash (the old line-611 /
        line-364 crashes)
      - good items + extra keys are preserved (no yield loss)

  PART B — LIVE (needs your configured LLM backend reachable):
    Runs real entity + relation extraction on a paragraph through the
    validated path and prints counts + samples.

Usage:
    python scripts/smoke_test_pydantic_validation.py            # A then B
    python scripts/smoke_test_pydantic_validation.py --offline  # A only
"""

import argparse
import os
import sys

# Make the repo importable whether run as `python scripts/...` or from a notebook.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def part_a_offline() -> None:
    print("\n=== PART A: offline validation (no LLM) ===")
    from multi_agent_kg.schemas_llm import (
        CoreferenceOut,
        EntityExtractionOut,
        PredictionOut,
        TripleOut,
    )
    from multi_agent_kg.agents.relation_extractor import _coerce_llm_items

    # 1. Entity: confidence as string, one empty item (dropped), extra key.
    ents = EntityExtractionOut.model_validate(
        {
            "entities": [
                {"text": "Metformin", "type": "DRUG", "confidence": "0.9", "junk": 1},
                {"text": "   ", "type": "DRUG"},          # empty -> dropped
                {"text": "HbA1c", "confidence": 1.7},     # clamped to 1.0
            ]
        }
    )
    kept = [e.text for e in ents.entities]
    assert kept == ["Metformin", "HbA1c"], kept
    assert ents.entities[0].confidence == 0.9
    assert ents.entities[1].confidence == 1.0
    print(f"  entities: kept {kept} (1 empty dropped, confidence coerced/clamped)  OK")

    # 2. Coreference: the old line-611 crash input — a bare string. Must not raise.
    groups = CoreferenceOut.model_validate("raw CoT prose, not JSON").entity_groups
    assert groups == []
    print("  coref: bare-string input -> [] (no 'str has no attribute get' crash)  OK")

    # 3. Relations: bare list (old line-364 crash) + NONE kept + extras preserved.
    preds = _coerce_llm_items(
        [
            {"pair_index": "0", "relation": "USED-FOR", "confidence": "0.8", "x": 1},
            {"relation": "NONE", "sentence_index": 9},
        ],
        ("predictions",),
        PredictionOut,
    )
    assert len(preds) == 2, preds
    assert preds[0]["pair_index"] == 0 and preds[0]["confidence"] == 0.8
    assert preds[0]["x"] == 1                       # extra preserved
    assert preds[1]["relation"] == "NONE"           # filter is downstream, not here
    assert preds[1]["sentence_index"] == 9
    print(f"  predictions: bare list -> {len(preds)} kept, extras + NONE preserved  OK")

    # 4. Triples: alias key survives for the downstream relation fallback.
    trips = _coerce_llm_items(
        {"triples": [{"subject": "a", "object": "b", "relation_type": "treats"}]},
        ("triples",),
        TripleOut,
    )
    assert trips[0]["relation_type"] == "treats"
    print("  triples: relation_type alias preserved (no yield loss)  OK")

    print("PART A PASSED — validation is lossless + crash-safe.")


def part_b_live() -> None:
    print("\n=== PART B: live extraction through your backend ===")
    backend = os.getenv("LLM_BACKEND", "ollama")
    print(f"  LLM_BACKEND={backend}  LLM_DEFAULT_MODEL={os.getenv('LLM_DEFAULT_MODEL', '(default)')}")

    from multi_agent_kg.agents.base import AgentContext
    from multi_agent_kg.agents.entity_extractor import EntityExtractor
    from multi_agent_kg.agents.relation_extractor import RelationExtractor
    from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
    from multi_agent_kg.core.config import LLMConfig

    text = (
        "Metformin reduces HbA1c levels in patients with type 2 diabetes mellitus. "
        "The World Health Organization (WHO) recommends it as a first-line therapy. "
        "Empagliflozin, developed by Boehringer Ingelheim, lowers cardiovascular risk."
    )
    ctx = AgentContext(document_id="smoke_doc", text=text)
    llm = LLMConfig()  # uses your configured default model

    kg = KnowledgeGraph()
    ee = EntityExtractor(knowledge_graph=kg, llm_config=llm, use_self_consistency=False)
    ent_result = ee.run(ctx)
    entities = ent_result.items
    print(f"\n  ENTITIES extracted: {len(entities)}")
    for e in entities[:8]:
        print(f"    - {e.get('text')!r:35} type={e.get('type')!r:18} conf={e.get('confidence')}")
        # validated path guarantees confidence is a real float in [0, 1]
        assert isinstance(e.get("confidence"), float), e
        assert 0.0 <= e["confidence"] <= 1.0, e

    re_ = RelationExtractor(
        knowledge_graph=kg,
        llm_config=llm,
        use_self_consistency=False,
        enable_relation_gleaning=False,
    )
    rel_result = re_.run(ctx, entities=entities)
    triples = rel_result.items
    print(f"\n  TRIPLES extracted: {len(triples)}")
    for t in triples[:8]:
        print(f"    - ({t.get('subject')}) -[{t.get('relation')}]-> ({t.get('object')})  conf={t.get('confidence')}")

    assert entities, "no entities extracted — backend may be unreachable or model weak"
    print("\nPART B PASSED — real extraction flows through the validated path.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="run Part A only (no LLM)")
    args = ap.parse_args()

    part_a_offline()
    if not args.offline:
        part_b_live()
    print("\nDONE.")
