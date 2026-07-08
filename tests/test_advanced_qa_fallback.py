"""Regression: AdvancedQAOrchestrator must survive the borrowed
QAOrchestrator._build_global_fallback_context path.

That method (called with an AdvancedQAOrchestrator as `self`) invokes
self._community_context — which only QAOrchestrator defined until the
EXP-FRESHNESS-E2E local leg crashed 3/5 questions with
"'AdvancedQAOrchestrator' object has no attribute '_community_context'"
(latent since 0932808; multi-document ingestion triggers the fallback path
far more often).
"""
from multi_agent_kg.core import Domain, GovernedKnowledgeGraph, KnowledgeGraph, LLMConfig, OrgChart
from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator
from multi_agent_kg.core.qa_orchestrator import QAOrchestrator


def _adv() -> AdvancedQAOrchestrator:
    kg = KnowledgeGraph()
    kg.add_entity("acme", ["Acme"], "ORG")
    domains = [
        Domain(domain_id="d1", label="D1", description="", entity_ids={"acme"}, relation_schema={}),
        Domain(domain_id="d2", label="D2", description="", entity_ids=set(), relation_schema={}),
    ]
    gkg = GovernedKnowledgeGraph(kg=kg, org_chart=OrgChart(domains=domains),
                                 governance_mode="audit_only")
    return AdvancedQAOrchestrator(governed_kg=gkg, llm_config=LLMConfig(model="test-model"),
                                  enable_debate=False, enable_critic=False)


def test_community_context_empty_without_communities():
    adv = _adv()
    assert adv._community_context("any question") == ""


def test_borrowed_global_fallback_does_not_crash(monkeypatch):
    """The exact crash path: fallback trigger conditions met, reaches the
    self._community_context call — must return a string, not AttributeError."""
    adv = _adv()
    # Force the fallback gate open: unowned query entity + no supported answers.
    monkeypatch.setattr(adv, "_extract_query_entities",
                        lambda q: ["mystery_entity"], raising=False)
    out = QAOrchestrator._build_global_fallback_context(adv, "where is mystery?", [], set())
    assert isinstance(out, str)
    assert "Fallback trigger" in out
