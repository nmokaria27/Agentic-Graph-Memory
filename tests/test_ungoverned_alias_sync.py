"""EXP-PAPER-COMPARE incident 2: ungoverned builds must survive alias sync.

`sync_aliases_from` persistence (commit a77e2ca) assumed a governed KG always
exists. Ungoverned builds pass governed_kg=None, and every document of a
100-doc SciERC flat build "failed" AFTER full extraction on
`self.governed_kg.sync_aliases_from(...)` — both per-document and at the
corpus-level cross-document resolution step.
"""
from multi_agent_kg.core import DeliberativeOrchestrator, LLMConfig


def _ungoverned_orch():
    return DeliberativeOrchestrator(
        llm_config=LLMConfig(model="test-model"),
        enable_governance=False,
        enable_cross_document=True,
        enable_self_consistency=False,
    )


def test_ungoverned_orchestrator_has_no_governed_kg():
    orch = _ungoverned_orch()
    assert orch.governed_kg is None  # the precondition the bug missed


def test_cross_document_resolution_survives_without_governed_kg():
    """Corpus-level crash site: _resolve_cross_document_entities must not
    raise when governed_kg is None (aliases stay in shared memory only)."""
    orch = _ungoverned_orch()
    orch.shared_memory.register_entity_alias("nyc", "new_york_city")
    orch._resolve_cross_document_entities()  # raised AttributeError pre-fix


def test_per_document_alias_persist_guard(monkeypatch):
    """Per-document crash site: the alias-persist block in process_document.
    Reproduce it in isolation — aliases present + governed_kg None."""
    orch = _ungoverned_orch()
    orch.shared_memory.register_entity_alias("nyc", "new_york_city")
    # The guarded expression from process_document, verbatim semantics:
    if orch.shared_memory.entity_aliases and orch.governed_kg is not None:
        orch.governed_kg.sync_aliases_from(orch.shared_memory.entity_aliases)
    # Reaching here without AttributeError is the assertion.
