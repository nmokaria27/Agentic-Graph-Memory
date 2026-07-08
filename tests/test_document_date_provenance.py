"""EXP-FRESHNESS-QA (GB-2) — a source document's date must survive from the
ingestion call into AgentContext, so freshness-sensitive conflict resolution
(conflict_resolution.py) has a recency signal to act on. See EXPERIMENT_LOG.md.
"""
from multi_agent_kg.agents.base import AgentContext
from multi_agent_kg.core import DeliberativeOrchestrator, LLMConfig


def test_agent_context_defaults_document_date_to_none():
    context = AgentContext(document_id="doc1", text="hello")
    assert context.document_date is None


def test_agent_context_carries_explicit_document_date():
    context = AgentContext(document_id="doc1", text="hello", document_date="2023-12-25")
    assert context.document_date == "2023-12-25"


def test_process_document_threads_metadata_date_into_context(monkeypatch):
    """process_document must not drop metadata["date"] on the floor — it has to
    reach the AgentContext it builds. The pipeline's downstream stages need a
    live LLM, so this test intercepts AgentContext construction itself rather
    than running the full 9-stage pipeline."""
    orchestrator = DeliberativeOrchestrator(
        llm_config=LLMConfig(model="test-model"),
        enable_deliberation=False,
        enable_self_consistency=False,
        enable_open_world=False,
        enable_cross_document=False,
        enable_governance=False,
    )

    captured = {}
    import multi_agent_kg.core.deliberative_orchestrator as mod
    real_context_cls = mod.AgentContext

    def spy_context(*args, **kwargs):
        captured.update(kwargs)
        return real_context_cls(*args, **kwargs)

    monkeypatch.setattr(mod, "AgentContext", spy_context)
    # Stage 2 (domain classification) is the first real LLM call; failing it
    # fast avoids a slow real network timeout — we only care that AgentContext
    # was already built (stage 2 runs after context creation) by the time it fires.
    monkeypatch.setattr(
        orchestrator.domain_classifier, "run",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no LLM in unit test")),
    )

    try:
        orchestrator.process_document(
            text="Rachel moved to the suburbs.",
            document_id="doc1",
            metadata={"date": "2023-12-25", "source": "test"},
        )
    except Exception:
        pass  # downstream stages need a real LLM; we only care about context construction

    assert captured.get("document_date") == "2023-12-25"


def test_process_document_leaves_document_date_none_without_metadata_date(monkeypatch):
    orchestrator = DeliberativeOrchestrator(
        llm_config=LLMConfig(model="test-model"),
        enable_deliberation=False,
        enable_self_consistency=False,
        enable_open_world=False,
        enable_cross_document=False,
        enable_governance=False,
    )

    captured = {}
    import multi_agent_kg.core.deliberative_orchestrator as mod
    real_context_cls = mod.AgentContext

    def spy_context(*args, **kwargs):
        captured.update(kwargs)
        return real_context_cls(*args, **kwargs)

    monkeypatch.setattr(mod, "AgentContext", spy_context)
    monkeypatch.setattr(
        orchestrator.domain_classifier, "run",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no LLM in unit test")),
    )

    try:
        orchestrator.process_document(text="Some undated text.", document_id="doc1")
    except Exception:
        pass

    assert captured.get("document_date") is None
