"""
BEAM benchmark evaluator for Agent-Graph-Memory.

Phase 1: Build KG from each conversation, answer all probing questions,
         save raw responses to disk.
Phase 2: Run score.py to judge the responses with an LLM judge.

Both phases are separated so the expensive KG pipeline and the expensive
judge can be run independently (and re-run with a different judge without
re-building the KG).

Dataset: HuggingFace Mohammadta/BEAM
  Splits: 100K (20 chats), 500K (35 chats), 1M (35 chats)
  Separate dataset for 10M: Mohammadta/BEAM-10M

Usage:
    # Smoke test — 3 chats, 5 Qs each
    python evaluation/BEAM/run_eval.py --tier 100K --max-samples 3 --max-questions 5

    # Full 100K tier (20 chats, ~400 Qs total)
    python evaluation/BEAM/run_eval.py --tier 100K \
        --save-kg-dir evaluation/results/beam_kg_cache/100K \
        --output evaluation/results/beam_100K_responses.json

    # Load cached KGs, only re-run QA
    python evaluation/BEAM/run_eval.py --tier 100K \
        --load-kg-dir evaluation/results/beam_kg_cache/100K \
        --output evaluation/results/beam_100K_responses.json

    # Then score (separate step):
    python evaluation/BEAM/score.py \
        --responses evaluation/results/beam_100K_responses.json \
        --output evaluation/results/beam_100K_scores.json
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("beam_eval")

_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from multi_agent_kg.core import LLMConfig, GovernedKnowledgeGraph
from multi_agent_kg.core.config import RetrievalConfig
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.kg_operations import save_governed_kg, load_governed_kg


# ---------------------------------------------------------------------------
# BEAM dataset loading
# ---------------------------------------------------------------------------

HF_REPO = "Mohammadta/BEAM"
VALID_TIERS = ["100K", "500K", "1M"]


def load_beam_dataset(tier: str, max_samples: int = -1) -> List[Dict]:
    """Download BEAM from HuggingFace and return list of chat dicts."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("pip install datasets  — required to download BEAM from HuggingFace")

    if tier not in VALID_TIERS:
        raise ValueError(f"tier must be one of {VALID_TIERS}, got {tier!r}")

    logger.info("Loading BEAM %s from HuggingFace (%s)…", tier, HF_REPO)
    ds = load_dataset(HF_REPO, split=tier, trust_remote_code=True)

    rows = list(ds)
    if max_samples > 0:
        rows = rows[:max_samples]

    logger.info("Loaded %d conversations (tier=%s)", len(rows), tier)
    return rows


# ---------------------------------------------------------------------------
# Chat formatter
# ---------------------------------------------------------------------------

def format_beam_chat(row: Dict, max_chars: Optional[int] = None) -> str:
    """
    Convert a BEAM row's `chat` field into flat text for KG pipeline input.

    `chat` is a list-of-lists (batches of turns), each turn is a list of
    {"role": "user"/"assistant", "content": "..."} dicts.
    """
    chat = row.get("chat", [])
    conversation_id = row.get("conversation_id", "unknown")
    seed = row.get("conversation_seed", {})
    category = seed.get("category", "General") if isinstance(seed, dict) else "General"

    lines = [f"[BEAM Conversation — ID: {conversation_id} | Domain: {category}]\n"]

    for batch in chat:
        # each batch is a list of turns; each turn is a list of messages
        if not isinstance(batch, list):
            continue
        for turn in batch:
            if not isinstance(turn, list):
                # flat message dict
                if isinstance(turn, dict):
                    role = turn.get("role", "unknown").upper()
                    content = turn.get("content", "").strip()
                    if content:
                        lines.append(f"{role}: {content}")
                continue
            for msg in turn:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "").strip()
                if content:
                    lines.append(f"{role}: {content}")

    full = "\n".join(lines)
    if max_chars and len(full) > max_chars:
        full = full[:max_chars]
    return full


def parse_probing_questions(raw: Any) -> Dict[str, List[Dict]]:
    """
    Parse the `probing_questions` field from a BEAM row.

    It may be:
      - a dict already (when HF auto-casts)
      - a JSON string
      - a Python-repr string using single-quotes
    Returns {question_type: [{question, rubric, ...}]}
    """
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, str):
        raw = raw.strip()
        # try JSON first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # fall back to ast.literal_eval (handles single-quoted Python reprs)
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass

    logger.warning("Could not parse probing_questions (type=%s); skipping", type(raw))
    return {}


# ---------------------------------------------------------------------------
# KG pipeline (mirrors LoComo run_eval.py exactly)
# ---------------------------------------------------------------------------

def build_kg(
    row: Dict,
    llm_config: LLMConfig,
    retrieval_config: RetrievalConfig,
    governance_mode: str = "permissive",
    max_context_chars: Optional[int] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 150,
) -> GovernedKnowledgeGraph:
    from multi_agent_kg.core import DeliberativeOrchestrator
    from multi_agent_kg.agents.base import ModelTier

    text = format_beam_chat(row, max_chars=max_context_chars)
    conv_id = row.get("conversation_id", "beam_unknown")
    document = {"text": text, "metadata": {"source": f"beam_{conv_id}"}}

    single_model = llm_config.model
    model_tiers = {
        ModelTier.SMALL:  os.environ.get("LLM_SMALL_MODEL",  single_model),
        ModelTier.MEDIUM: os.environ.get("LLM_MEDIUM_MODEL", single_model),
        ModelTier.LARGE:  os.environ.get("LLM_LARGE_MODEL",  single_model),
    }

    governed_kg = GovernedKnowledgeGraph(governance_mode=governance_mode)
    orchestrator = DeliberativeOrchestrator(
        llm_config=llm_config,
        knowledge_graph=KnowledgeGraph(),
        governed_kg=governed_kg,
        quality_threshold=0.35,
        max_refinement_iterations=1,
        enable_self_consistency=False,
        enable_open_world=True,
        enable_cross_document=False,
        model_tiers=model_tiers,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    orchestrator.process_corpus([document])
    governed_kg = orchestrator.governed_kg

    # Orphan relink + vector index (same as LoComo)
    try:
        from multi_agent_kg.core.orphan_relink import fold_value_orphans, relink_orphans
        from multi_agent_kg.core.vector_index import KGVectorStore

        vs = None
        if retrieval_config.use_vectors:
            vs = KGVectorStore(model=retrieval_config.embedding_model)
            vs.build(governed_kg)
            governed_kg.vector_store = vs

        relink_orphans(governed_kg, llm_config=llm_config,
                       retrieval_config=retrieval_config, vector_store=vs)
        fold_value_orphans(governed_kg)
    except Exception as exc:
        logger.warning("Orphan relink / vector build failed (KG still usable): %s", exc)

    return governed_kg


def build_qa_system(
    governed_kg: GovernedKnowledgeGraph,
    llm_config: LLMConfig,
    retrieval_config: RetrievalConfig,
    answer_format=None,
):
    from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator
    from multi_agent_kg.core.domain_experts import DomainBuilder, QAOrchestrator
    from multi_agent_kg.core.vector_index import KGVectorStore

    vector_store = governed_kg.vector_store
    if vector_store is None and retrieval_config.use_vectors:
        try:
            vs = KGVectorStore(model=retrieval_config.embedding_model)
            vs.build(governed_kg)
            governed_kg.vector_store = vs
            vector_store = vs
        except Exception as exc:
            logger.warning("Vector store build failed; QA falls back to KG-only: %s", exc)

    try:
        builder = DomainBuilder(llm_config)
        org_chart = builder.build(governed_kg._kg)
    except Exception as exc:
        logger.warning("DomainBuilder failed; single-domain fallback: %s", exc)
        # Non-LLM fallback: one domain owning every entity. (OrgChart has no
        # single_domain() factory, so build it directly — must not itself raise,
        # this is the resilience path for an overnight run.)
        from multi_agent_kg.core.domain_experts import OrgChart
        from multi_agent_kg.core.governance import Domain

        kg = governed_kg._kg
        fallback_domain = Domain(
            domain_id="general",
            label="General",
            description="All entities (single-domain fallback).",
            entity_ids=set(kg.entities.keys()),
            relation_schema={},
        )
        org_chart = OrgChart(domains=[fallback_domain])

    return AdvancedQAOrchestrator(
        org_chart=org_chart,
        full_kg=governed_kg._kg,
        llm_config=llm_config,
        vector_store=vector_store,
        retrieval_config=retrieval_config,
        answer_format=answer_format,
        enable_debate=True,
        enable_critic=True,
        max_exploration_rounds=3,
    )


def answer_question(qa_system, question: str) -> str:
    """Query the QA system and return the best answer string."""
    try:
        result = qa_system.query(question)
        verbose = (
            result.get("final_answer")
            or result.get("answer")
            or result.get("output")
            or ""
        )
        if isinstance(verbose, dict):
            verbose = verbose.get("answer", "")
        return str(verbose).strip()
    except Exception as exc:
        logger.error("QA query failed for %r: %s", question[:80], exc, exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_beam_eval(
    rows: List[Dict],
    llm_config: LLMConfig,
    retrieval_config: RetrievalConfig,
    governance_mode: str = "permissive",
    max_questions: int = -1,
    max_context_chars: Optional[int] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 150,
    output_path: Optional[Path] = None,
    save_kg_dir: Optional[Path] = None,
    load_kg_dir: Optional[Path] = None,
    answer_format=None,
) -> Dict[str, Any]:
    """
    Run the BEAM answer-generation phase.

    Returns a dict:
    {
        "meta": {tier, model, n_chats, ...},
        "chats": [
            {
                "conversation_id": ...,
                "responses": {
                    question_type: [
                        {"question": ..., "rubric": [...], "llm_response": ...},
                        ...
                    ]
                }
            },
            ...
        ]
    }
    """
    all_chats = []

    for chat_idx, row in enumerate(rows):
        conv_id = row.get("conversation_id", f"chat_{chat_idx}")
        logger.info("=== Chat %d/%d — ID: %s ===", chat_idx + 1, len(rows), conv_id)
        try:
            chat_result = _process_chat(
                row, conv_id, llm_config, retrieval_config, governance_mode,
                max_questions, max_context_chars, chunk_size, chunk_overlap,
                save_kg_dir, load_kg_dir, answer_format,
            )
        except Exception as exc:
            # One bad conversation must not abort an overnight run; log, record a
            # stub so the chat is accounted for, and continue. Already-completed
            # chats stay saved via the incremental write below.
            logger.error("Chat %s failed, skipping: %s", conv_id, exc, exc_info=True)
            chat_result = {
                "conversation_id": conv_id,
                "tier": row.get("_tier", ""),
                "domain": (row.get("conversation_seed", {}) or {}).get("category", ""),
                "_total_questions": 0,
                "_error": str(exc),
                "responses": {},
            }

        all_chats.append(chat_result)
        if output_path:
            _save(output_path, all_chats, llm_config.model)
            logger.info("  Saved progress → %s", output_path)

    result = {
        "meta": {
            "model": llm_config.model,
            "retrieval": retrieval_config.retrieval_mode,
            "n_chats": len(all_chats),
            "n_questions": sum(c.get("_total_questions", 0) for c in all_chats),
        },
        "chats": all_chats,
    }

    if output_path:
        _save(output_path, all_chats, llm_config.model)

    return result


def _process_chat(
    row: Dict,
    conv_id: Any,
    llm_config: LLMConfig,
    retrieval_config: RetrievalConfig,
    governance_mode: str,
    max_questions: int,
    max_context_chars: Optional[int],
    chunk_size: Optional[int],
    chunk_overlap: int,
    save_kg_dir: Optional[Path],
    load_kg_dir: Optional[Path],
    answer_format,
) -> Dict[str, Any]:
    """Build (or load) one chat's KG and answer its probing questions."""
    # --- KG build or load ---
    t0 = time.time()
    kg_save_path = (save_kg_dir / str(conv_id)) if save_kg_dir else None
    kg_load_path = (load_kg_dir / str(conv_id)) if load_kg_dir else None

    if kg_load_path and kg_load_path.exists():
        logger.info("  Loading KG from %s", kg_load_path)
        governed_kg = load_governed_kg(str(kg_load_path))
    else:
        logger.info("  Building KG…")
        governed_kg = build_kg(
            row, llm_config, retrieval_config, governance_mode,
            max_context_chars, chunk_size, chunk_overlap,
        )
        stats = governed_kg.get_stats()
        logger.info("  Built: %d entities, %d triples in %.0fs",
                    stats.get("entities", 0), stats.get("triples", 0),
                    time.time() - t0)
        if kg_save_path:
            kg_save_path.parent.mkdir(parents=True, exist_ok=True)
            save_governed_kg(governed_kg, str(kg_save_path))
            logger.info("  KG saved → %s", kg_save_path)

    build_time = time.time() - t0

    # --- QA system ---
    qa_system = build_qa_system(governed_kg, llm_config, retrieval_config, answer_format)

    # --- Probing questions ---
    raw_pq = row.get("probing_questions", {})
    probing_questions = parse_probing_questions(raw_pq)

    chat_responses: Dict[str, List[Dict]] = {}
    total_q = 0

    for q_type, questions in probing_questions.items():
        if not questions:
            continue
        type_responses = []

        if max_questions > 0:
            questions = questions[:max_questions]

        for q_idx, q_item in enumerate(questions):
            question = q_item.get("question", "")
            rubric = q_item.get("rubric", [])

            t_q = time.time()
            response = answer_question(qa_system, question)
            query_time = time.time() - t_q
            total_q += 1

            logger.info("  [%s] Q%d: %r → %r (%.1fs)",
                        q_type, q_idx + 1, question[:60], response[:60], query_time)

            # Preserve all original fields, add llm_response
            entry = dict(q_item)
            entry["llm_response"] = response
            entry["_query_time_s"] = round(query_time, 2)
            type_responses.append(entry)

        chat_responses[q_type] = type_responses

    return {
        "conversation_id": conv_id,
        "tier": row.get("_tier", ""),
        "domain": (row.get("conversation_seed", {}) or {}).get("category", ""),
        "_build_time_s": round(build_time, 1),
        "_total_questions": total_q,
        "responses": chat_responses,
    }


def _save(path: Path, chats: List[Dict], model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "model": model,
            "n_chats": len(chats),
            "n_questions": sum(c.get("_total_questions", 0) for c in chats),
        },
        "chats": chats,
    }
    path.write_text(json.dumps(out, indent=2, default=str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BEAM Phase 1: build KG + answer probing questions"
    )
    parser.add_argument(
        "--tier",
        choices=VALID_TIERS,
        default="100K",
        help="BEAM tier to evaluate (default: 100K)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=-1,
        help="Max conversations to evaluate (default: all). Smoke test: 3"
    )
    parser.add_argument(
        "--max-questions", type=int, default=-1,
        help="Max probing questions per question-type per chat (default: all). Smoke test: 5"
    )
    parser.add_argument(
        "--max-context-chars", type=int, default=None,
        help="Truncate conversation text to N chars before KG build. "
             "None = no truncation (recommended). 500K chars ≈ 125K tokens."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_DEFAULT_MODEL", os.environ.get("LLM_MODEL", "gemma4:31b")),
        help="Backbone LLM for KG extraction + QA (default: LLM_DEFAULT_MODEL env)"
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large"),
        help="Embedding model for vector retrieval"
    )
    parser.add_argument(
        "--retrieval",
        choices=["lexical", "dense", "hybrid", "graph_completion", "graph_summary", "chunk"],
        default="hybrid",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Chunk size in chars for KG extraction (default: auto). "
             "Recommended for 100K chats: 3000"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=150,
    )
    parser.add_argument(
        "--governance-mode",
        choices=["permissive", "strict", "triage", "audit_only"],
        default="permissive",
    )
    parser.add_argument(
        "--answer-format",
        default="beam",
        help="Answer format profile (default: beam)"
    )
    parser.add_argument(
        "--save-kg-dir", default=None,
        help="Directory to save built KGs for reuse. "
             "e.g. evaluation/results/beam_kg_cache/100K"
    )
    parser.add_argument(
        "--load-kg-dir", default=None,
        help="Load pre-built KGs from this dir instead of running pipeline"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path for responses (default: evaluation/results/beam_{tier}_responses.json)"
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else \
        Path(f"evaluation/results/beam_{args.tier}_responses.json")
    save_kg_dir = Path(args.save_kg_dir) if args.save_kg_dir else None
    load_kg_dir = Path(args.load_kg_dir) if args.load_kg_dir else None

    logger.info("BEAM Evaluation — Phase 1: Answer Generation")
    logger.info("Tier:       %s", args.tier)
    logger.info("Backbone:   %s", args.model)
    logger.info("Retrieval:  %s", args.retrieval)
    logger.info("Embedding:  %s", args.embedding_model)
    logger.info("Max chats:  %s", args.max_samples if args.max_samples > 0 else "all")
    logger.info("Max Qs:     %s/type/chat", args.max_questions if args.max_questions > 0 else "all")
    logger.info("Output:     %s", output_path)

    rows = load_beam_dataset(args.tier, max_samples=args.max_samples)
    # Tag each row with tier for output metadata
    for r in rows:
        r["_tier"] = args.tier

    from evaluation.answer_format_profiles import get_answer_format
    llm_config       = LLMConfig(model=args.model)
    retrieval_config = RetrievalConfig(
        retrieval_mode=args.retrieval,
        embedding_model=args.embedding_model,
    )
    answer_format = get_answer_format(args.answer_format)

    result = run_beam_eval(
        rows=rows,
        llm_config=llm_config,
        retrieval_config=retrieval_config,
        governance_mode=args.governance_mode,
        max_questions=args.max_questions,
        max_context_chars=args.max_context_chars,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_path=output_path,
        save_kg_dir=save_kg_dir,
        load_kg_dir=load_kg_dir,
        answer_format=answer_format,
    )

    print(f"\n{'='*60}")
    print("BEAM Phase 1 Complete")
    print(f"{'='*60}")
    print(json.dumps(result["meta"], indent=2))
    print(f"\nResponses saved to: {output_path}")
    print(f"Next step: python evaluation/BEAM/score.py --responses {output_path}")


if __name__ == "__main__":
    main()
