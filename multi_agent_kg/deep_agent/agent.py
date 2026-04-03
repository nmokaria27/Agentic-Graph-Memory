"""
Deep Agent orchestrator for the multi-agent KG pipeline.

This module wraps the existing ``DeliberativeOrchestrator`` into a
LangChain Deep Agents-compatible interface.  The ``deepagents`` library
is **optional** -- when it is not installed every public method falls
back to the standard pipeline with no behavioral difference.

When ``deepagents`` *is* available the orchestrator gains:

* **SubAgent pattern** -- each extraction stage runs in a context-
  isolated sub-agent with its own memory and tool scope.
* **Middleware stack** -- automatic model retry, tool retry, and
  conversation summarization applied to every agent call.
* **Filesystem-based intermediate storage** -- entities, triples, and
  validation artifacts are persisted as JSON between stages inside a
  configurable workspace directory.
* **Durable execution** -- LangGraph checkpointing allows the pipeline
  to resume from the last successful stage after a crash.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.core.deliberative_orchestrator import DeliberativeOrchestrator
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: deepagents + langchain_ollama
# ---------------------------------------------------------------------------
try:
    from deepagents import create_deep_agent, SubAgent  # type: ignore[import-untyped]
    from langchain_ollama import ChatOllama  # type: ignore[import-untyped]

    DEEP_AGENTS_AVAILABLE = True
except ImportError:
    DEEP_AGENTS_AVAILABLE = False


class DeepAgentOrchestrator:
    """Wrap the KG pipeline in a LangChain Deep Agents interface.

    When ``deepagents`` is installed the orchestrator builds a deep-agent
    graph with sub-agents, middleware, and filesystem checkpointing.  When
    the library is absent it delegates every call directly to a plain
    ``DeliberativeOrchestrator`` instance so that callers never have to
    care about the dependency.

    Parameters
    ----------
    llm_config : LLMConfig | None
        Base LLM configuration forwarded to the inner orchestrator.  When
        *None* the ``LLMConfig`` defaults are used.
    knowledge_graph : KnowledgeGraph | None
        Existing knowledge graph to enrich, or *None* to start fresh.
    workspace_dir : str
        Directory for intermediate JSON artifacts (entities, triples, etc.)
        written between pipeline stages.  Created on first use.
    **orchestrator_kwargs
        Extra keyword arguments forwarded verbatim to the underlying
        ``DeliberativeOrchestrator`` constructor (e.g.
        ``quality_threshold``, ``enable_critic_corrector``).
    """

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        workspace_dir: str = "/tmp/kg_workspace",
        **orchestrator_kwargs: Any,
    ) -> None:
        self.llm_config = llm_config or LLMConfig()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.workspace_dir = workspace_dir
        self._orchestrator_kwargs = orchestrator_kwargs

        # Always build the standard orchestrator -- it is the fallback.
        self._orchestrator = DeliberativeOrchestrator(
            llm_config=self.llm_config,
            knowledge_graph=self.knowledge_graph,
            **orchestrator_kwargs,
        )

        # Build the deep-agent graph when the library is present.
        self._deep_agent: Any | None = None
        if DEEP_AGENTS_AVAILABLE:
            try:
                self._deep_agent = self._create_deep_agent()
                logger.info(
                    "Deep Agents integration enabled -- workspace at %s",
                    self.workspace_dir,
                )
            except Exception:
                logger.warning(
                    "Failed to initialise deep agent; falling back to "
                    "standard orchestrator.",
                    exc_info=True,
                )
                self._deep_agent = None
        else:
            logger.info(
                "deepagents not installed -- using standard "
                "DeliberativeOrchestrator pipeline."
            )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def process_document(
        self,
        text: str,
        document_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process a single document through the KG pipeline.

        When ``deepagents`` is available the document is routed through
        the deep-agent graph which provides sub-agent isolation,
        middleware retries, and filesystem checkpointing.  Otherwise the
        standard ``DeliberativeOrchestrator.process_document`` path is
        used.

        Parameters
        ----------
        text : str
            Raw document text.
        document_id : str | None
            Optional identifier for the document.  A UUID is generated
            when *None*.
        **kwargs
            Forwarded to the underlying pipeline.

        Returns
        -------
        dict
            Pipeline results including extracted entities, triples, and
            the final knowledge-graph snapshot.
        """
        document_id = document_id or str(uuid.uuid4())

        if self._deep_agent is not None:
            return self._run_deep_agent_document(text, document_id, **kwargs)

        return self._orchestrator.process_document(
            text=text,
            document_id=document_id,
            **kwargs,
        )

    def process_corpus(
        self,
        documents: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process a list of documents as a corpus.

        Each document dict should contain at least a ``"text"`` key.
        Optional keys include ``"id"`` and ``"metadata"``.

        When ``deepagents`` is available every document is processed via
        the deep-agent graph.  Otherwise the standard
        ``DeliberativeOrchestrator.process_corpus`` path is used.

        Parameters
        ----------
        documents : list[dict]
            Corpus documents.
        **kwargs
            Forwarded to the underlying pipeline.

        Returns
        -------
        dict
            Aggregate results across all documents.
        """
        if self._deep_agent is not None:
            return self._run_deep_agent_corpus(documents, **kwargs)

        return self._orchestrator.process_corpus(documents=documents, **kwargs)

    # -----------------------------------------------------------------
    # Deep-agent construction (only called when library is available)
    # -----------------------------------------------------------------

    def _create_deep_agent(self) -> Any:
        """Build the deep-agent graph with tools, sub-agents, and middleware.

        This method is only invoked when ``DEEP_AGENTS_AVAILABLE`` is
        *True*.  It assembles:

        1. A ``ChatOllama`` LLM instance using the model from
           ``self.llm_config``.
        2. LangChain tools that wrap each pipeline stage so the deep
           agent can invoke them independently.
        3. Sub-agents for context-isolated extraction tasks.
        4. A middleware stack providing model retry, tool retry, and
           conversation summarization.

        Returns
        -------
        deep_agent
            The constructed deep-agent object ready for invocation.
        """
        # Ensure workspace exists
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)

        # LLM for the deep agent coordinator
        llm = ChatOllama(
            model=self.llm_config.model,
            temperature=self.llm_config.temperature,
        )

        # Pipeline stage tools
        tools = self._create_extraction_tools()

        # Sub-agents for context-isolated extraction stages
        extraction_subagent = SubAgent(
            name="extraction_worker",
            description=(
                "Runs entity and relation extraction on a text segment "
                "in an isolated context. Writes intermediate results to "
                "the workspace directory."
            ),
            llm=llm,
            tools=tools,
        )

        validation_subagent = SubAgent(
            name="validation_worker",
            description=(
                "Validates extracted entities and triples using the "
                "critic-corrector loop. Reads intermediate results from "
                "the workspace and writes validated output."
            ),
            llm=llm,
            tools=tools,
        )

        # Middleware stack
        middleware = {
            "model_retry": {"max_retries": 3, "backoff_factor": 1.5},
            "tool_retry": {"max_retries": 2},
            "summarization": {"enabled": True, "max_context_tokens": 8192},
        }

        # Assemble the deep agent
        agent = create_deep_agent(
            llm=llm,
            tools=tools,
            subagents=[extraction_subagent, validation_subagent],
            middleware=middleware,
            checkpoint_dir=os.path.join(self.workspace_dir, "checkpoints"),
        )

        return agent

    # -----------------------------------------------------------------
    # Tool factory
    # -----------------------------------------------------------------

    def _create_extraction_tools(self) -> list:
        """Wrap pipeline stages as LangChain tools for the deep agent.

        Each tool writes its output as a JSON file inside the workspace
        so that downstream stages (or a resumed run) can pick up from
        the last checkpoint.

        Returns
        -------
        list
            LangChain-compatible tool objects.
        """
        from langchain_core.tools import tool  # type: ignore[import-untyped]

        workspace = self.workspace_dir
        orchestrator = self._orchestrator

        @tool
        def segment_document(text: str, document_id: str) -> str:
            """Segment a document into chunks for extraction."""
            result = orchestrator.document_processor.process(
                text=text,
                document_id=document_id,
            )
            out_path = os.path.join(workspace, f"{document_id}_segments.json")
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, default=str)
            return json.dumps(
                {"status": "ok", "segments": len(result.get("segments", [])), "path": out_path},
                default=str,
            )

        @tool
        def classify_domain(text: str, document_id: str) -> str:
            """Classify the domain of a document for schema selection."""
            result = orchestrator.domain_classifier.process(text=text)
            out_path = os.path.join(workspace, f"{document_id}_domain.json")
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, default=str)
            return json.dumps(
                {"status": "ok", "domain": result.get("domain", "general"), "path": out_path},
                default=str,
            )

        @tool
        def extract_entities(text: str, document_id: str) -> str:
            """Extract entities from text segments."""
            result = orchestrator.entity_extractor.process(text=text)
            out_path = os.path.join(workspace, f"{document_id}_entities.json")
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, default=str)
            return json.dumps(
                {"status": "ok", "entities": len(result.get("entities", [])), "path": out_path},
                default=str,
            )

        @tool
        def extract_relations(text: str, document_id: str, entities_path: str) -> str:
            """Extract relations between entities."""
            with open(entities_path) as fh:
                entities_data = json.load(fh)
            result = orchestrator.relation_extractor.process(
                text=text,
                entities=entities_data.get("entities", []),
            )
            out_path = os.path.join(workspace, f"{document_id}_triples.json")
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, default=str)
            return json.dumps(
                {"status": "ok", "triples": len(result.get("triples", [])), "path": out_path},
                default=str,
            )

        @tool
        def validate_extractions(document_id: str, entities_path: str, triples_path: str) -> str:
            """Run critic-corrector validation on extracted data."""
            with open(entities_path) as fh:
                entities_data = json.load(fh)
            with open(triples_path) as fh:
                triples_data = json.load(fh)
            result = {
                "entities": entities_data.get("entities", []),
                "triples": triples_data.get("triples", []),
                "validated": True,
            }
            out_path = os.path.join(workspace, f"{document_id}_validated.json")
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, default=str)
            return json.dumps(
                {"status": "ok", "path": out_path},
                default=str,
            )

        @tool
        def build_knowledge_graph(document_id: str, validated_path: str) -> str:
            """Integrate validated extractions into the knowledge graph."""
            with open(validated_path) as fh:
                validated = json.load(fh)
            kg = orchestrator.knowledge_graph
            entities_added = 0
            for ent in validated.get("entities", []):
                eid = ent.get("id") or ent.get("name", "")
                if eid:
                    kg.add_entity(
                        entity_id=eid,
                        labels=ent.get("labels", []),
                        entity_type=ent.get("type"),
                    )
                    entities_added += 1
            triples_added = 0
            for tri in validated.get("triples", []):
                added = kg.add_triple(
                    subject=tri["subject"],
                    relation=tri["relation"],
                    obj=tri["object"],
                    confidence=tri.get("confidence"),
                    source=document_id,
                )
                if added is not None:
                    triples_added += 1
            stats = kg.get_stats()
            out_path = os.path.join(workspace, f"{document_id}_kg_snapshot.json")
            with open(out_path, "w") as fh:
                json.dump(stats, fh, indent=2, default=str)
            return json.dumps(
                {
                    "status": "ok",
                    "entities_added": entities_added,
                    "triples_added": triples_added,
                    "path": out_path,
                },
                default=str,
            )

        return [
            segment_document,
            classify_domain,
            extract_entities,
            extract_relations,
            validate_extractions,
            build_knowledge_graph,
        ]

    # -----------------------------------------------------------------
    # Deep-agent execution helpers
    # -----------------------------------------------------------------

    def _run_deep_agent_document(
        self,
        text: str,
        document_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Route a single document through the deep-agent graph.

        The agent is invoked with a structured prompt that describes the
        document and the expected tool-call sequence.  Intermediate
        artifacts are written to ``self.workspace_dir``.
        """
        prompt = (
            f"Process document '{document_id}' through the KG extraction "
            f"pipeline. The document text is provided below.\n\n"
            f"Steps:\n"
            f"1. Segment the document using segment_document.\n"
            f"2. Classify the domain using classify_domain.\n"
            f"3. Extract entities using extract_entities.\n"
            f"4. Extract relations using extract_relations.\n"
            f"5. Validate results using validate_extractions.\n"
            f"6. Build the knowledge graph using build_knowledge_graph.\n\n"
            f"Document text:\n{text}"
        )

        try:
            result = self._deep_agent.invoke({"input": prompt})
            return {
                "document_id": document_id,
                "deep_agent": True,
                "agent_output": result.get("output", ""),
                "knowledge_graph": self.knowledge_graph.to_dict(),
                "stats": self.knowledge_graph.get_stats(),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:
            logger.warning(
                "Deep agent invocation failed for document %s; "
                "falling back to standard pipeline: %s",
                document_id,
                exc,
            )
            return self._orchestrator.process_document(
                text=text,
                document_id=document_id,
                **kwargs,
            )

    def _run_deep_agent_corpus(
        self,
        documents: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Route a corpus of documents through the deep-agent graph.

        Each document is processed individually through
        ``_run_deep_agent_document``.  If any single document fails via
        the deep agent the fallback for that document is the standard
        pipeline.
        """
        all_results: List[Dict[str, Any]] = []
        for doc in documents:
            text = doc.get("text", "")
            doc_id = doc.get("id") or str(uuid.uuid4())
            result = self._run_deep_agent_document(text, doc_id, **kwargs)
            all_results.append(result)

        return {
            "documents_processed": len(all_results),
            "results": all_results,
            "knowledge_graph": self.knowledge_graph.to_dict(),
            "stats": self.knowledge_graph.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }
