"""Item-level self-consistency consensus (replaces broken whole-response voting).

The old _compute_consensus voted on exact serialized responses; at temperature
> 0 multi-item JSON never matches byte-for-byte, so it returned an arbitrary
sample at 1/n confidence — 3x the cost for no variance reduction. These tests
pin the item-level behavior: union across samples, identity-keyed dedupe, and
support-weighted confidence.
"""
from multi_agent_kg.agents.base import AgentContext, BaseAgent, ExtractionResult


class _Dummy(BaseAgent):
    def run(self, context: AgentContext, **kwargs) -> ExtractionResult:  # pragma: no cover
        raise NotImplementedError


def _agent() -> _Dummy:
    return _Dummy.__new__(_Dummy)  # consensus methods use no instance state


def test_entities_union_with_support_weighted_confidence() -> None:
    # alice appears in all 3 samples, bob in only 1: BOTH survive (union keeps
    # recall — the observed RHF variance drops true facts from single samples),
    # but alice ends up more confident than bob.
    samples = [
        {"entities": [{"text": "Alice Chen", "type": "PERSON"}]},
        {"entities": [{"text": "alice chen", "type": "PERSON"},
                      {"text": "Bob Ramirez", "type": "PERSON"}]},
        {"entities": [{"text": "Alice Chen", "type": "EMPLOYEE"}]},
    ]
    out, confidence = _agent()._compute_consensus(samples)
    names = [e["text"].lower() for e in out["entities"]]
    assert names == ["alice chen", "bob ramirez"]
    alice, bob = out["entities"]
    assert alice["confidence"] > bob["confidence"]
    assert 0.0 < confidence <= 1.0


def test_triples_merge_ignores_volatile_fields() -> None:
    # Same (s, r, o) with different rationale/confidence is ONE item, 2/2 votes.
    samples = [
        {"triples": [{"subject": "alice", "relation": "WORKS_AT", "object": "northwind",
                      "confidence": 0.9, "rationale": "stated directly"}]},
        {"triples": [{"subject": "Alice", "relation": "works_at", "object": "Northwind",
                      "confidence": 0.7, "rationale": "para 1"}]},
    ]
    out, _ = _agent()._compute_consensus(samples)
    assert len(out["triples"]) == 1
    # unanimous support: blended confidence ends up high
    assert out["triples"][0]["confidence"] >= 0.8


def test_sample_local_ids_do_not_split_votes() -> None:
    # Each SC sample invents its own entity ids; the same text with different
    # ids must merge (this duplicate flood collapsed coref in the live run).
    samples = [
        {"entities": [{"id": "e1", "text": "Marcus Webb", "type": "PERSON"}]},
        {"entities": [{"id": "ent_7", "text": "marcus webb", "type": "PERSON"}]},
        {"entities": [{"id": "m-webb", "text": "Marcus Webb", "type": "FOUNDER"}]},
    ]
    out, _ = _agent()._compute_consensus(samples)
    assert len(out["entities"]) == 1
    assert out["entities"][0]["confidence"] > 0.9  # unanimous support


def test_entity_type_disagreement_does_not_duplicate() -> None:
    # PERSON vs EMPLOYEE for the same text must merge, not split votes.
    samples = [
        {"entities": [{"text": "Nina Patel", "type": "PERSON"}]},
        {"entities": [{"text": "Nina Patel", "type": "EMPLOYEE"}]},
    ]
    out, _ = _agent()._compute_consensus(samples)
    assert len(out["entities"]) == 1


def test_scalar_fields_majority_voted() -> None:
    samples = [
        {"primary_domain": "law", "entity_types": [{"type": "STATUTE"}]},
        {"primary_domain": "law", "entity_types": [{"type": "STATUTE"}, {"type": "COURT"}]},
        {"primary_domain": "medicine", "entity_types": [{"type": "STATUTE"}]},
    ]
    out, confidence = _agent()._compute_consensus(samples)
    assert out["primary_domain"] == "law"
    types = [t["type"] for t in out["entity_types"]]
    assert types == ["STATUTE", "COURT"]
    assert 0.0 < confidence <= 1.0


def test_bare_list_responses_merge() -> None:
    samples = [
        [{"text": "Portland"}],
        [{"text": "portland"}, {"text": "Seattle"}],
    ]
    out, _ = _agent()._compute_consensus(samples)
    assert [e["text"].lower() for e in out] == ["portland", "seattle"]


def test_scalar_payloads_keep_exact_voting() -> None:
    samples = [{"answer": "yes"}, {"answer": "yes"}, {"answer": "no"}]
    out, confidence = _agent()._compute_consensus(samples)
    assert out == {"answer": "yes"}
    assert abs(confidence - 2 / 3) < 1e-9


def test_empty_responses() -> None:
    out, confidence = _agent()._compute_consensus([])
    assert out == {} and confidence == 0.0
