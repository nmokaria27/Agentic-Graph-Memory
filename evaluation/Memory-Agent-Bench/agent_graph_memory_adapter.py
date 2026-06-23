"""
MemoryAgentBench adapter for Agent-Graph-Memory.

Implements the AgentWrapper interface expected by MemoryAgentBench:
  - send_message(message, memorizing, query_id, context_id) -> dict
  - save_agent() / load_agent() for optional state persistence

Usage:
  Register this adapter in MemoryAgentBench's agent.py by adding
  "agent_graph_memory" to the _initialize_agent_by_type dispatch.

  Or run standalone via run_eval.py in this directory.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow imports from project root
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from multi_agent_kg.core.config import LLMConfig, RetrievalConfig
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.domain_experts import DomainBuilder, OrgChart, QAOrchestrator
from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator
from multi_agent_kg.core.kg_operations import save_governed_kg, load_governed_kg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Rough token count (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


def _extract_answer(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("final_answer", "answer", "output"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, dict):
                inner = val.get("answer", "")
                if inner:
                    return str(inner).strip()
    return str(result).strip() if result else ""


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

class AgentGraphMemoryWrapper:
    """
    Wraps the Agent-Graph-Memory pipeline to satisfy MemoryAgentBench's
    AgentWrapper interface.

    Memorization phase  (memorizing=True):
        Text chunks are accumulated and ingested via the 9-agent extraction
        pipeline.  A GovernedKnowledgeGraph + KGVectorStore are built once
        all chunks for a context have been fed in.

    Query phase  (memorizing=False):
        QAOrchestrator (or AdvancedQAOrchestrator) answers over the KG
        with hybrid retrieval (KG traversal + vector semantic search).
    """

    def __init__(
        self,
        model: str = "gemma4:31b",
        embedding_model: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        use_advanced_qa: bool = True,
        governance_mode: str = "permissive",
        save_dir: Optional[str] = None,
        temperature: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model or os.environ.get(
            "EMBEDDING_MODEL", "mxbai-embed-large"
        )
        self.retrieval_mode = retrieval_mode
        self.use_advanced_qa = use_advanced_qa
        self.governance_mode = governance_mode
        self.save_dir = Path(save_dir) if save_dir else None
        self.temperature = temperature
        self.verbose = verbose

        self.llm_config = LLMConfig(model=model, temperature=temperature)
        self.retrieval_config = RetrievalConfig(
            retrieval_mode=retrieval_mode,
            embedding_model=self.embedding_model,
        )

        # State reset per context
        self._chunks: List[str] = []
        self._governed_kg: Optional[GovernedKnowledgeGraph] = None
        self._qa_system: Optional[Any] = None
        self._context_id: Optional[int] = None
        self._build_time: float = 0.0

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def send_message(
        self,
        message: str,
        memorizing: bool = False,
        query_id: int = 0,
        context_id: int = 0,
    ) -> Dict[str, Any]:
        if memorizing:
            return self._memorize(message, context_id)
        else:
            return self._query(message, query_id, context_id)

    # ------------------------------------------------------------------
    # Memorization
    # ------------------------------------------------------------------

    def _memorize(self, chunk: str, context_id: int) -> Dict[str, Any]:
        """Accumulate chunk; pipeline runs lazily on first query."""
        if context_id != self._context_id:
            self._reset_context(context_id)
        self._chunks.append(chunk)
        return {
            "output": "Memorized",
            "input_len": _count_tokens(chunk),
            "output_len": 0,
            "memory_construction_time": 0.0,
            "query_time_len": 0.0,
        }

    def _ensure_kg_built(self, context_id: int) -> None:
        """Build KG from accumulated chunks if not already done."""
        if self._governed_kg is not None:
            return

        full_text = "\n\n".join(self._chunks)
        t0 = time.time()

        try:
            self._governed_kg = self._run_pipeline(full_text)
        except Exception as exc:
            logger.warning("Pipeline failed (%s); falling back to empty KG.", exc)
            kg = KnowledgeGraph()
            self._governed_kg = GovernedKnowledgeGraph(
                kg=kg,
                governance_mode=self.governance_mode,
            )

        self._build_time = time.time() - t0
        self._qa_system = self._build_qa_system(self._governed_kg)

        if self.save_dir:
            self._persist(context_id)

    def _run_pipeline(self, text: str) -> GovernedKnowledgeGraph:
        """Run the extraction pipeline via DeliberativeOrchestrator (same as run_pipeline.py)."""
        from multi_agent_kg.core import DeliberativeOrchestrator
        from multi_agent_kg.agents.base import ModelTier

        document = {"text": text, "metadata": {"source": "memoryagentbench"}}

        # Force all tiers to same model to avoid OOM from multiple models loading simultaneously
        single_model = self.llm_config.model
        model_tiers = {
            ModelTier.SMALL:  os.environ.get("LLM_SMALL_MODEL",  single_model),
            ModelTier.MEDIUM: os.environ.get("LLM_MEDIUM_MODEL", single_model),
            ModelTier.LARGE:  os.environ.get("LLM_LARGE_MODEL",  single_model),
        }

        governed_kg = GovernedKnowledgeGraph(governance_mode=self.governance_mode)
        orchestrator = DeliberativeOrchestrator(
            llm_config=self.llm_config,
            knowledge_graph=KnowledgeGraph(),
            governed_kg=governed_kg,
            quality_threshold=0.35,
            max_refinement_iterations=1,
            enable_self_consistency=False,
            enable_open_world=True,
            enable_cross_document=False,
            model_tiers=model_tiers,
        )
        orchestrator.process_corpus([document])
        governed_kg = orchestrator.governed_kg

        # Orphan relink pass
        try:
            from multi_agent_kg.core.orphan_relink import fold_value_orphans, relink_orphans
            from multi_agent_kg.core.vector_index import KGVectorStore

            vs = None
            if self.retrieval_config.use_vectors:
                vs = KGVectorStore(model=self.retrieval_config.embedding_model)
                vs.build(governed_kg)
                governed_kg.vector_store = vs

            relink_orphans(
                governed_kg,
                llm_config=self.llm_config,
                retrieval_config=self.retrieval_config,
                vector_store=vs,
            )
            fold_value_orphans(governed_kg)
        except Exception as exc:
            logger.debug("Orphan relink skipped: %s", exc)

        return governed_kg

    # ------------------------------------------------------------------
    # QA system construction
    # ------------------------------------------------------------------

    def _build_qa_system(self, governed_kg: GovernedKnowledgeGraph) -> Any:
        from multi_agent_kg.core.vector_index import KGVectorStore

        vector_store = None
        if self.retrieval_config.use_vectors:
            try:
                vs = governed_kg.vector_store
                if vs is None:
                    vs = KGVectorStore(model=self.retrieval_config.embedding_model)
                    vs.build(governed_kg)
                    governed_kg.vector_store = vs
                vector_store = vs
            except Exception as exc:
                logger.warning("Vector store unavailable: %s", exc)

        # Build org chart from the governed KG
        try:
            builder = DomainBuilder(self.llm_config)
            org_chart = builder.build(governed_kg._kg)
        except Exception as exc:
            logger.warning("DomainBuilder failed (%s); single-domain fallback.", exc)
            org_chart = OrgChart.single_domain(governed_kg._kg)

        if self.use_advanced_qa:
            return AdvancedQAOrchestrator(
                org_chart=org_chart,
                full_kg=governed_kg._kg,
                llm_config=self.llm_config,
                vector_store=vector_store,
                retrieval_config=self.retrieval_config,
                enable_debate=True,
                enable_critic=True,
                max_exploration_rounds=3,
            )
        return QAOrchestrator(
            org_chart=org_chart,
            full_kg=governed_kg._kg,
            llm_config=self.llm_config,
            vector_store=vector_store,
            retrieval_config=self.retrieval_config,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _query(self, query: str, query_id: int, context_id: int) -> Dict[str, Any]:
        if context_id != self._context_id:
            # Shouldn't happen in normal bench flow; handle gracefully
            logger.warning("Query on context %d but built for %d", context_id, self._context_id)

        self._ensure_kg_built(context_id)

        t0 = time.time()
        try:
            result = self._qa_system.query(query)
            answer = _extract_answer(result)
        except Exception as exc:
            logger.error("QA failed: %s", exc)
            answer = ""
            result = {}
        query_time = time.time() - t0

        return {
            "output": answer,
            "input_len": _count_tokens(query),
            "output_len": _count_tokens(answer),
            "memory_construction_time": self._build_time,
            "query_time_len": query_time,
        }

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_context(self, context_id: int) -> None:
        self._chunks = []
        self._governed_kg = None
        self._qa_system = None
        self._context_id = context_id
        self._build_time = 0.0

    def save_agent(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.save_dir
        if target is None or self._governed_kg is None:
            return
        target.mkdir(parents=True, exist_ok=True)
        save_governed_kg(self._governed_kg, str(target / "governed_kg.json"))

    def load_agent(self, path: Optional[str] = None) -> bool:
        target = Path(path) if path else self.save_dir
        kg_file = target / "governed_kg.json" if target else None
        if kg_file is None or not kg_file.exists():
            return False
        try:
            self._governed_kg = load_governed_kg(str(kg_file))
            self._qa_system = self._build_qa_system(self._governed_kg)
            return True
        except Exception as exc:
            logger.warning("Load failed: %s", exc)
            return False

    def _persist(self, context_id: int) -> None:
        if self.save_dir:
            ctx_dir = self.save_dir / f"context_{context_id}"
            self.save_agent(str(ctx_dir))

    # ------------------------------------------------------------------
    # Convenience: expose stats for logging
    # ------------------------------------------------------------------

    def get_kg_stats(self) -> Dict[str, Any]:
        if self._governed_kg is None:
            return {}
        return self._governed_kg.get_stats()
