"""Embedding failover (owner requirement, 2026-07-13).

Primary endpoint exhausts its retry ladder -> sticky process-level failover
to the configured fallback (Fireworks by default when FIREWORKS_API_KEY is
set). No fallback configured -> unchanged behavior (raise).
"""
import pytest

from multi_agent_kg.llm import openai_client as oc


class _FailingEmbeddings:
    def __init__(self):
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        raise TimeoutError("primary wedged")


class _WorkingEmbeddings:
    def __init__(self):
        self.calls = 0
        self.models = []

    def create(self, model, input):
        self.calls += 1
        self.models.append(model)

        class _Item:
            def __init__(self, i):
                self.index = i
                self.embedding = [float(i), 1.0]

        class _Resp:
            data = [_Item(i) for i in range(len(input))]

        return _Resp()


class _Client:
    def __init__(self, embeddings):
        self.embeddings = embeddings


@pytest.fixture(autouse=True)
def _fast_and_clean(monkeypatch):
    monkeypatch.setattr(oc, "_MAX_RETRIES", 2)
    monkeypatch.setattr(oc.time, "sleep", lambda *_: None)
    monkeypatch.setitem(oc._EMBED_FAILOVER, "active", False)
    monkeypatch.setitem(oc._EMBED_FAILOVER, "client", None)
    monkeypatch.setitem(oc._EMBED_FAILOVER, "model", None)


def test_no_fallback_configured_raises(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_FALLBACK_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_FALLBACK_API_KEY", raising=False)
    monkeypatch.setattr(oc, "embed_client", _Client(_FailingEmbeddings()))
    with pytest.raises(Exception, match="Embedding API call failed"):
        oc.get_embeddings(["a"])
    assert oc._EMBED_FAILOVER["active"] is False


def test_failover_activates_and_serves_the_failing_batch(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test-key")
    monkeypatch.delenv("EMBEDDING_FALLBACK_BASE_URL", raising=False)
    primary = _FailingEmbeddings()
    fallback = _WorkingEmbeddings()
    monkeypatch.setattr(oc, "embed_client", _Client(primary))
    monkeypatch.setattr(
        oc, "OpenAI", lambda **kwargs: _Client(fallback)
    )
    out = oc.get_embeddings(["a", "b"])
    assert primary.calls == 2            # full (shortened) ladder exhausted
    assert fallback.calls == 1           # same batch served by fallback
    assert out == [[0.0, 1.0], [1.0, 1.0]]
    assert oc._EMBED_FAILOVER["active"] is True
    assert "qwen3-embedding-8b" in fallback.models[0]


def test_failover_is_sticky_for_subsequent_calls(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test-key")
    primary = _FailingEmbeddings()
    fallback = _WorkingEmbeddings()
    monkeypatch.setattr(oc, "embed_client", _Client(primary))
    monkeypatch.setattr(oc, "OpenAI", lambda **kwargs: _Client(fallback))
    oc.get_embeddings(["a"])             # triggers the switch
    oc.get_embeddings(["b"])             # must NOT touch primary again
    assert primary.calls == 2            # only the first call's ladder
    assert fallback.calls == 2
    # Prefix selection follows the failover model once active.
    assert oc._resolved_embedding_model("mxbai-embed-large") == fallback.models[0]


def test_explicit_fallback_env_overrides_default(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_FALLBACK_BASE_URL", "https://alt.example/v1")
    monkeypatch.setenv("EMBEDDING_FALLBACK_API_KEY", "alt-key")
    monkeypatch.setenv("EMBEDDING_FALLBACK_MODEL", "alt-embed")
    cfg = oc._fallback_embedding_config()
    assert cfg == ("https://alt.example/v1", "alt-key", "alt-embed")
