"""Tests for the orphan relink pass and value-orphan attribute folding."""

from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.orphan_relink import fold_value_orphans, relink_orphans


def _governed_kg_with_orphans() -> GovernedKnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_entity(
        "apple_inc", labels=["Apple Inc."], entity_type="ORG",
        metadata={"source_segment": "seg_0", "source_text": "Apple Inc. was founded by Steve Jobs in 1976."},
    )
    kg.add_entity(
        "steve_jobs", labels=["Steve Jobs"], entity_type="PERSON",
        metadata={"source_segment": "seg_0", "source_text": "Apple Inc. was founded by Steve Jobs in 1976."},
    )
    # Orphan with usable source text
    kg.add_entity(
        "macintosh", labels=["Macintosh"], entity_type="PRODUCT",
        metadata={"source_segment": "seg_0", "source_text": "Apple released the Macintosh in 1984."},
    )
    # Value-like orphan
    kg.add_entity(
        "1_trillion_usd", labels=["$1 trillion"], entity_type="FINANCIAL_METRIC",
        metadata={"source_segment": "seg_0"},
    )
    gkg = GovernedKnowledgeGraph(kg=kg, governance_mode="audit_only")
    gkg.propose_triple("steve_jobs", "FOUNDED", "apple_inc", confidence=0.9)
    return gkg


def test_relink_orphans_commits_valid_triples(monkeypatch):
    gkg = _governed_kg_with_orphans()

    def fake_chat(messages, model=None, temperature=None, **kwargs):
        return {
            "triples": [
                {
                    "subject_id": "apple_inc",
                    "relation": "RELEASED",
                    "object_id": "macintosh",
                    "confidence": 0.85,
                    "evidence": "Apple released the Macintosh in 1984.",
                },
                {  # hallucinated id → must be dropped
                    "subject_id": "macintosh",
                    "relation": "MADE_OF",
                    "object_id": "unobtainium",
                    "confidence": 0.9,
                },
                {  # no orphan involved → must be dropped
                    "subject_id": "steve_jobs",
                    "relation": "FOUNDED",
                    "object_id": "apple_inc",
                    "confidence": 0.9,
                },
            ]
        }

    monkeypatch.setattr(
        "multi_agent_kg.core.orphan_relink.chat_completion_json", fake_chat
    )
    stats = relink_orphans(gkg, llm_config=LLMConfig(model="test-model"))

    assert stats["orphans_before"] == 2
    assert stats["triples_committed"] == 1
    assert stats["orphans_after"] < stats["orphans_before"]
    committed = [t for t in gkg.triples if t.relation == "RELEASED"]
    assert committed and committed[0].source == "orphan_relink"
    assert committed[0].metadata.get("orphan_relink") is True


def test_relink_skips_orphans_without_source_text(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity("lonely", labels=["Lonely"], entity_type="ORG", metadata={})
    gkg = GovernedKnowledgeGraph(kg=kg, governance_mode="audit_only")

    calls = []
    monkeypatch.setattr(
        "multi_agent_kg.core.orphan_relink.chat_completion_json",
        lambda *a, **k: calls.append(1) or {"triples": []},
    )
    stats = relink_orphans(gkg, llm_config=LLMConfig(model="test-model"))
    assert stats["llm_calls"] == 0
    assert stats["orphans_skipped_no_text"] == 1
    assert not calls


def test_fold_value_orphans_moves_to_host_metadata():
    gkg = _governed_kg_with_orphans()
    stats = fold_value_orphans(gkg)

    assert stats["folded"] == 1
    assert "1_trillion_usd" not in gkg.entities
    # Host shares seg_0 and is connected
    hosts_with_attr = [
        e for e in gkg.entities.values()
        if "1_trillion_usd" in (e.metadata.get("attributes") or {})
    ]
    assert len(hosts_with_attr) == 1
    attr = hosts_with_attr[0].metadata["attributes"]["1_trillion_usd"]
    assert attr["value"] == "$1 trillion"
    assert attr["folded_from_orphan"] is True
    # Non-value orphan (macintosh) untouched
    assert "macintosh" in gkg.entities
