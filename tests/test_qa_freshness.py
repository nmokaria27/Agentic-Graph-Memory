"""GB-15 (EXP-QA-PATH): the answer layer must see supersession and dates.

DIAG-LME-FAILURES proved every probed knowledge-update miss had the gold fact
in the graph while QA served a stale/placeholder answer. `_format_triple` is
the single evidence-formatting chokepoint: it must surface provenance dates
and mark superseded triples unusable.
"""
from multi_agent_kg.core.knowledge_graph import Triple
from multi_agent_kg.core.qa_orchestrator import _format_triple


def _t(**meta):
    return Triple(subject="yoga classes", relation="HAS_FREQUENCY",
                  object="twice a week", metadata=meta)


def test_plain_triple_unchanged():
    line = _format_triple(_t())
    assert line == "(yoga classes) -[HAS_FREQUENCY]-> (twice a week)"


def test_session_date_kept_and_takes_precedence():
    line = _format_triple(_t(session_date="2023/05/01",
                             provenance={"refs": [{"document_date": "2023/04/01"}]}))
    assert "[on 2023/05/01]" in line
    assert "as of" not in line


def test_provenance_document_date_surfaces_newest():
    line = _format_triple(_t(provenance={"refs": [
        {"document_date": "2023/03/01"}, {"document_date": "2023/06/09"}]}))
    assert "[as of 2023/06/09]" in line


def test_superseded_triple_is_marked_unusable():
    line = _format_triple(_t(superseded_by="t123", superseded_at="2023-06-10"))
    assert "SUPERSEDED" in line and "do not use" in line


def test_freshness_rule_present_in_expert_prompts():
    import multi_agent_kg.core.qa_orchestrator as qa
    src = open(qa.__file__).read()
    assert src.count("FRESHNESS") >= 2  # domain expert + global fallback prompts
