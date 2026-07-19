"""
FastAPI server for the MaKG frontend.

Endpoints:
  GET  /health             — server status + KG stats
  GET  /kg/data            — full KG data (entities, triples, entity types, relations, org_chart)
  GET  /kg/stats           — summary statistics
  GET  /models             — list of available QA models
  POST /qa                 — question answering (proxies to QA orchestrator)
  POST /ingest             — upload a .txt document for pipeline ingestion
  GET  /pipeline/status    — pipeline job status (all jobs or ?job_id=...)

Run:
  python scripts/api_server.py [--port 8000] [--basic] [--no-debate] [--no-critic]
  # Or via uvicorn directly:
  uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="MaKG API", version="0.1.0")

_default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (populated on startup) ───────────────────────────────────────────

KG_FILE = str(PROJECT_ROOT / "governed_kg_export.json")
CACHE_FILE = str(PROJECT_ROOT / "org_chart_cache.json")
INBOX_DIR = PROJECT_ROOT / "inbox"
INBOX_DIR.mkdir(exist_ok=True)
PIPELINE_LOG_DIR = PROJECT_ROOT / "pipeline_logs"
PIPELINE_LOG_DIR.mkdir(exist_ok=True)
JOBS_FILE = PIPELINE_LOG_DIR / "jobs.json"
JOB_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "jobs"

governed_kg = None
qa_system = None
_startup_time = time.time()
_qa_model_lock = threading.Lock()  # protects per-request model swap on qa_system.llm_config
_kg_reload_lock = threading.Lock()  # protects governed_kg/qa_system swap during reload

# How the KG/QA stack was initialized; reused by /kg/reload after ingestion.
_INIT_CONFIG: dict = {"mode": "none"}

# Job table for /ingest + /pipeline/status (persisted to JOBS_FILE).
JOBS: dict[str, dict] = {}
if JOBS_FILE.exists():
    try:
        JOBS.update(json.loads(JOBS_FILE.read_text()))
        # Jobs that were queued/running when the server died are stale.
        for _job in JOBS.values():
            if _job.get("status") in ("queued", "running"):
                _job["status"] = "failed"
                _job["error"] = "server restarted while job was in flight"
    except (json.JSONDecodeError, OSError):
        pass


def _save_jobs() -> None:
    try:
        tmp = JOBS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(JOBS, indent=2, default=str))
        os.replace(tmp, JOBS_FILE)
    except OSError:
        pass


# ── Request / response models ────────────────────────────────────────────────


class QARequest(BaseModel):
    question: str
    model: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _kg_hash(kg_path: str) -> str:
    h = hashlib.md5()
    with open(kg_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _build_entity_types(entities: list[dict]) -> dict:
    """Derive entity type -> color mapping from entity data."""
    # Color palette for entity types
    PALETTE = [
        "#2e7d9e", "#c17f4a", "#7264c8", "#3d8f6e", "#a05070",
        "#4a6b9e", "#d4a843", "#5e8c61", "#9b6b9e", "#c76f6f",
        "#4a9e8e", "#8b7355", "#6a8fc1", "#c4826d", "#7a9e4a",
    ]
    types_seen: dict[str, dict] = {}
    color_idx = 0
    for e in entities:
        etype = e.get("type", "UNKNOWN")
        if etype not in types_seen:
            types_seen[etype] = {
                "color": PALETTE[color_idx % len(PALETTE)],
                "label": etype.replace("_", " ").title(),
            }
            color_idx += 1
    return types_seen


def _format_entities(raw_entities: list[dict]) -> list[dict]:
    """Normalize entity data for frontend consumption."""
    formatted = []
    for e in raw_entities:
        aliases = e.get("aliases", [])
        name = e.get("name", e.get("id", ""))
        labels = [name] + aliases if name else aliases or [e.get("id", "")]
        # Deduplicate while preserving order
        seen = set()
        unique_labels = []
        for label in labels:
            if label and label not in seen:
                seen.add(label)
                unique_labels.append(label)

        formatted.append({
            "id": e.get("id", ""),
            "labels": unique_labels,
            "type": e.get("type", "UNKNOWN"),
            "metadata": {
                "confidence": e.get("confidence", e.get("attributes", {}).get("confidence", 0.0)),
                "source": e.get("source", e.get("attributes", {}).get("source", "")),
                "description": e.get("description", e.get("attributes", {}).get("description", "")),
                **{k: v for k, v in e.get("attributes", {}).items()
                   if k not in ("confidence", "source", "description")},
            },
        })
    return formatted


def _format_domain_responses(domain_responses: list[dict]) -> list[dict]:
    """Normalize domain response data for frontend consumption."""
    formatted = []
    for dr in domain_responses:
        evidence_strings = []
        for ev in dr.get("evidence", []):
            if isinstance(ev, str):
                evidence_strings.append(ev)
            elif isinstance(ev, dict):
                text = ev.get("text", ev.get("description", ""))
                if text:
                    evidence_strings.append(text)
                for t in ev.get("supporting_triples", []):
                    if isinstance(t, dict):
                        s = t.get("subject", "")
                        r = t.get("relation", "")
                        o = t.get("object", "")
                        if s and r and o:
                            evidence_strings.append(f"({s}) -[{r}]-> ({o})")
        formatted.append({
            "domain_id": dr.get("domain_id", "unknown"),
            "confidence": _safe_float(dr.get("confidence", 0.0)),
            "topics_used": dr.get("topics_used", []),
            "evidence": evidence_strings[:10],
        })
    return formatted


def _format_triples(raw_triples: list[dict]) -> list[dict]:
    """Normalize triple data for frontend consumption."""
    formatted = []
    for i, t in enumerate(raw_triples):
        formatted.append({
            "id": t.get("id", f"t{i}"),
            "subject": t.get("subject", ""),
            "relation": t.get("relation", ""),
            "object": t.get("object", ""),
            "confidence": t.get("confidence", 0.0),
            "source": t.get("source", ""),
        })
    return formatted


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    kg_loaded = governed_kg is not None
    kg_stats = {}
    if kg_loaded:
        stats = governed_kg.kg.get_stats()
        kg_stats = {
            "entities": stats.get("num_entities", 0),
            "triples": stats.get("num_triples", 0),
        }
    return {
        "status": "ok",
        "kg_loaded": kg_loaded,
        "qa_ready": qa_system is not None,
        "kg_stats": kg_stats,
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


def _build_org_chart_summary() -> dict:
    """Return a small org-chart summary for frontend domain coloring."""
    if governed_kg is None or governed_kg.org_chart is None:
        return {"domains": [], "assignments": {}}
    org = governed_kg.org_chart
    domains = []
    for d in org.domains:
        domains.append({
            "domain_id": d.domain_id,
            "label": getattr(d, "label", d.domain_id),
            "description": getattr(d, "description", ""),
            "entity_count": len(getattr(d, "entity_ids", [])),
            "owner_label": getattr(d, "owner_label", ""),
            "topics": [
                {"topic_id": t.topic_id, "label": t.label}
                for t in getattr(d, "topics", [])
            ],
            "relation_types": sorted(getattr(d, "relation_schema", {}).keys()),
            "entity_ids": sorted(getattr(d, "entity_ids", [])),
        })
    # entity_id → primary domain (first one wins for coloring)
    assignments = {eid: dids[0] for eid, dids in org.entity_domain_map().items() if dids}
    return {"domains": domains, "assignments": assignments}


@app.get("/kg/data")
def kg_data():
    if governed_kg is None:
        raise HTTPException(status_code=503, detail="KG not loaded")

    export = governed_kg.kg.to_dict()
    raw_entities = export.get("entities", [])
    raw_triples = export.get("triples", [])

    entities = _format_entities(raw_entities)
    triples = _format_triples(raw_triples)
    entity_types = _build_entity_types(entities)
    relation_types = sorted(set(t["relation"] for t in triples))

    return {
        "entities": entities,
        "triples": triples,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "org_chart": _build_org_chart_summary(),
    }


# Hosted models offered in the QA dropdown when FIREWORKS_API_KEY is set.
# Override with FIREWORKS_QA_MODELS (comma-separated). The "fireworks/" prefix
# is expanded to "accounts/fireworks/models/" by the LLM client, which routes
# these to the Fireworks API per-request regardless of LLM_BACKEND.
_FIREWORKS_QA_DEFAULTS = [
    "fireworks/deepseek-v4-flash",
    "fireworks/deepseek-v4-pro",
    "fireworks/glm-5p2",
    "fireworks/gpt-oss-120b",
    "fireworks/kimi-k2p6",
]


def _live_vllm_models() -> list[str]:
    """Ask the running vLLM server what it's actually serving (2s budget)."""
    if os.getenv("LLM_BACKEND", "").lower() != "vllm":
        return []
    base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    req = urllib.request.Request(f"{base}/models")
    key = os.getenv("VLLM_API_KEY", "")
    if key and key != "EMPTY":
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.load(resp)
        return [m["id"] for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


@app.get("/models")
def list_models():
    """QA model catalog: the model(s) the local vLLM server is actually serving,
    plus Fireworks-hosted models (when FIREWORKS_API_KEY is configured).

    /qa and /qa/stream accept any of these via the request's `model` field;
    Fireworks names are routed to the Fireworks API by the LLM client.
    """
    env_default = os.getenv("LLM_DEFAULT_MODEL", "")
    extra = [m.strip() for m in os.getenv("LLM_AVAILABLE_MODELS", "").split(",") if m.strip()]

    local = _live_vllm_models()
    local_label = "vLLM (local)" if local else "Configured"
    if not local:
        local = [m for m in [env_default, *extra] if m]

    fireworks: list[str] = []
    if os.getenv("FIREWORKS_API_KEY"):
        configured = [m.strip() for m in os.getenv("FIREWORKS_QA_MODELS", "").split(",") if m.strip()]
        fireworks = configured or _FIREWORKS_QA_DEFAULTS

    groups = []
    if local:
        groups.append({"provider": "vllm", "label": local_label, "models": local})
    if fireworks:
        groups.append({"provider": "fireworks", "label": "Fireworks (hosted)", "models": fireworks})

    # Prefer the live local model as default so QA stays on gpu02 unless the
    # user explicitly picks a hosted model.
    default = (env_default if env_default in local else None) or (local[0] if local else None) \
        or (fireworks[0] if fireworks else "")
    flat = [m for g in groups for m in g["models"]]
    return {"models": flat, "default": default, "groups": groups}


@app.get("/kg/stats")
def kg_stats():
    if governed_kg is None:
        raise HTTPException(status_code=503, detail="KG not loaded")

    stats = governed_kg.kg.get_stats()
    gov_stats = governed_kg.get_stats()
    return {
        "kg": stats,
        "governed": gov_stats,
    }


def _build_triple_index() -> dict[tuple[str, str, str], str]:
    """Map (subject, relation, object) → triple id from the current KG."""
    if governed_kg is None:
        return {}
    export = governed_kg.kg.to_dict()
    formatted = _format_triples(export.get("triples", []))
    return {(t["subject"], t["relation"], t["object"]): t["id"] for t in formatted}


@app.post("/qa")
def qa_query(req: QARequest):
    if qa_system is None:
        raise HTTPException(status_code=503, detail="QA system not initialized")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    print(f"\nQ: {req.question}")

    llm_cfg = getattr(qa_system, "llm_config", None)
    with _qa_model_lock:
        prev_model = getattr(llm_cfg, "model", None) if llm_cfg else None
        used_model = req.model if (req.model and llm_cfg) else prev_model
        if req.model and llm_cfg and req.model != prev_model:
            print(f"  switching model: {prev_model!r} -> {req.model!r}")
            llm_cfg.model = req.model
        try:
            result = qa_system.query(req.question)
        finally:
            if llm_cfg and prev_model and getattr(llm_cfg, "model", None) != prev_model:
                llm_cfg.model = prev_model  # restore after query

    print(f"A: {result.get('final_answer', '')[:200]}...")
    return _build_qa_payload(result, used_model)


def _build_qa_payload(result: dict, used_model: str | None) -> dict:
    """Convert a raw QA orchestrator result into the frontend response shape."""
    triple_index = _build_triple_index()
    # Map "entity_id" -> set of triple ids it appears in (for fallback edge lookup)
    entity_triples: dict[str, set[str]] = {}
    for (s, _r, o), tid in triple_index.items():
        entity_triples.setdefault(s, set()).add(tid)
        entity_triples.setdefault(o, set()).add(tid)

    # Basic-mode evidence comes back as strings like "(subject) -[relation]-> (object)".
    # Advanced-mode evidence comes back as dicts with "supporting_triples".
    triple_str_re = re.compile(r"\(([^)]+)\)\s*-\[([^\]]+)\]->\s*\(([^)]+)\)")

    def collect_triple_obj(t: dict, nodes: set, edges: set):
        if not isinstance(t, dict):
            return
        s, r, o = t.get("subject", ""), t.get("relation", ""), t.get("object", "")
        if s: nodes.add(s)
        if o: nodes.add(o)
        tid = triple_index.get((s, r, o))
        if tid:
            edges.add(tid)

    def collect_triple_str(text: str, nodes: set, edges: set):
        for m in triple_str_re.finditer(text or ""):
            s, r, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if s: nodes.add(s)
            if o: nodes.add(o)
            tid = triple_index.get((s, r, o))
            if tid:
                edges.add(tid)

    referenced_nodes: set = set()
    referenced_edges: set = set()
    for dr in result.get("domain_responses", []):
        for ev in dr.get("evidence", []):
            if isinstance(ev, dict):
                # Advanced mode: dict with supporting_triples + text
                for t in ev.get("supporting_triples", []):
                    collect_triple_obj(t, referenced_nodes, referenced_edges)
                if ev.get("text"):
                    collect_triple_str(ev["text"], referenced_nodes, referenced_edges)
            elif isinstance(ev, str):
                # Basic mode: plain "(s) -[r]-> (o)" string
                collect_triple_str(ev, referenced_nodes, referenced_edges)

    # Also parse the final answer text — it often inlines "(entity) -[relation]-> (entity)"
    collect_triple_str(result.get("final_answer", ""), referenced_nodes, referenced_edges)

    for pr in result.get("provenance", {}).get("records", []):
        for t in pr.get("supporting_triples", []):
            collect_triple_obj(t, referenced_nodes, referenced_edges)

    referenced_nodes.discard("")

    # Fallback: if no edges resolved but we have nodes, light up every triple
    # touching those nodes so the user still sees something in the graph.
    if referenced_nodes and not referenced_edges:
        for nid in referenced_nodes:
            referenced_edges.update(entity_triples.get(nid, set()))

    active_model = used_model or os.getenv("LLM_DEFAULT_MODEL", "")
    return {
        "answer": result.get("final_answer", ""),
        "model": active_model,
        "referenced_nodes": sorted(referenced_nodes),
        "referenced_edges": sorted(referenced_edges),
        "coverage": result.get("overall_coverage", 0),
        "confidence": result.get("overall_confidence", 0),
        "provenance": result.get("provenance", {}),
        "domain_responses": _format_domain_responses(result.get("domain_responses", [])),
        "debug": {
            "sub_questions": result.get("sub_questions", []),
            "debate_results": result.get("debate_results", []),
            "critic_result": result.get("critic_result", {}),
        },
    }


@app.post("/qa/stream")
def qa_query_stream(req: QARequest):
    """Server-Sent Events version of /qa.

    Emits `progress` events (stage names from the QA orchestrator) while the
    pipeline runs, then a final `result` event with the same payload as /qa.
    """
    if qa_system is None:
        raise HTTPException(status_code=503, detail="QA system not initialized")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    import queue as _queue

    events: _queue.Queue = _queue.Queue()

    def on_progress(stage: str, info: dict):
        events.put({"event": "progress", "stage": stage, "info": info})

    def worker():
        llm_cfg = getattr(qa_system, "llm_config", None)
        with _qa_model_lock:
            prev_model = getattr(llm_cfg, "model", None) if llm_cfg else None
            used_model = req.model if (req.model and llm_cfg) else prev_model
            if req.model and llm_cfg and req.model != prev_model:
                llm_cfg.model = req.model
            had_callback = getattr(qa_system, "progress_callback", None)
            qa_system.progress_callback = on_progress
            try:
                result = qa_system.query(req.question)
                payload = _build_qa_payload(result, used_model)
                events.put({"event": "result", "payload": payload})
            except Exception as exc:  # noqa: BLE001
                events.put({"event": "error", "message": str(exc)})
            finally:
                qa_system.progress_callback = had_callback
                if llm_cfg and prev_model and getattr(llm_cfg, "model", None) != prev_model:
                    llm_cfg.model = prev_model
                events.put(None)  # sentinel: stream done

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            try:
                item = events.get(timeout=15)
            except _queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Governance endpoints ──────────────────────────────────────────────


class GovernanceReviewRequest(BaseModel):
    index: int
    action: str  # "approve" | "reject"
    rationale: str | None = None


@app.get("/governance/audit")
def governance_audit(offset: int = 0, limit: int = 100, action: str | None = None, domain_id: str | None = None):
    """Paginated governance audit log, newest first."""
    if governed_kg is None:
        raise HTTPException(status_code=503, detail="KG not loaded")
    entries = [d.to_dict() for d in reversed(governed_kg.audit_log)]
    if action:
        entries = [e for e in entries if e.get("action") == action]
    if domain_id:
        entries = [e for e in entries if e.get("domain_id") == domain_id]
    total = len(entries)
    page = entries[offset:offset + max(0, min(limit, 500))]
    return {
        "total": total,
        "offset": offset,
        "entries": page,
        "stats": governed_kg.get_stats(),
    }


@app.get("/governance/pending")
def governance_pending():
    """Pending-review queue (strict/triage governance escalations)."""
    if governed_kg is None:
        raise HTTPException(status_code=503, detail="KG not loaded")
    return {
        "pending": [
            {
                "index": i,
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
                "confidence": t.confidence,
                "source": t.source,
            }
            for i, t in enumerate(governed_kg.pending_review)
        ],
    }


@app.post("/governance/review")
def governance_review(req: GovernanceReviewRequest):
    """Approve or reject a pending-review triple, then persist the KG export."""
    if governed_kg is None:
        raise HTTPException(status_code=503, detail="KG not loaded")
    if req.action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    from multi_agent_kg.core.governed_kg import GovernanceDecision

    pending = governed_kg.pending_review
    if req.index < 0 or req.index >= len(pending):
        raise HTTPException(status_code=404, detail=f"No pending entry at index {req.index}")

    triple = pending[req.index]
    assignment = governed_kg.org_chart.route_triple_for_governance(triple)
    decision = GovernanceDecision(
        triple=triple,
        action=req.action,
        domain_id=assignment.primary_domain_id if assignment else None,
        rationale=req.rationale or f"Manual {req.action} via governance review UI.",
        assignment=assignment,
    )
    governed_kg.resolve_pending(req.index, decision)

    # Persist the updated governed KG so the decision survives restarts.
    try:
        from multi_agent_kg.core import save_governed_kg
        save_governed_kg(governed_kg, KG_FILE)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to persist governed KG after review: {exc}")

    return {
        "action": req.action,
        "committed": decision.committed,
        "remaining_pending": len(governed_kg.pending_review),
        "triple": {
            "subject": triple.subject,
            "relation": triple.relation,
            "object": triple.object,
        },
    }


# ── KG reload ─────────────────────────────────────────────────────────────


def _reload_kg() -> dict:
    """Re-initialize the KG (and QA stack if it was enabled) from disk."""
    with _kg_reload_lock:
        mode = _INIT_CONFIG.get("mode")
        if mode == "full":
            init_kg_and_qa(
                basic=_INIT_CONFIG.get("basic", False),
                no_debate=_INIT_CONFIG.get("no_debate", False),
                no_critic=_INIT_CONFIG.get("no_critic", False),
                exploration_rounds=_INIT_CONFIG.get("exploration_rounds", 3),
            )
        else:
            init_kg_only()
    stats = governed_kg.kg.get_stats() if governed_kg else {}
    return {
        "reloaded": governed_kg is not None,
        "entities": stats.get("num_entities", 0),
        "triples": stats.get("num_triples", 0),
    }


@app.post("/kg/reload")
def kg_reload():
    """Reload governed_kg_export.json from disk (e.g. after a pipeline run)."""
    try:
        return _reload_kg()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")


# ── Ingestion + pipeline status ──────────────────────────────────────────────


def _run_pipeline_job(job_id: str, doc_path: Path):
    """Run scripts/run_pipeline.py against a single document and update JOBS."""
    log_path = PIPELINE_LOG_DIR / f"{job_id}.log"
    checkpoint_dir = JOB_CHECKPOINT_ROOT / job_id
    JOBS[job_id].update({
        "status": "running",
        "started_at": time.time(),
        "log_file": str(log_path),
        "checkpoint_dir": str(checkpoint_dir),
    })
    _save_jobs()
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
                    "--input", str(doc_path),
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                cwd=str(PROJECT_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        JOBS[job_id]["return_code"] = proc.returncode
        JOBS[job_id]["finished_at"] = time.time()
        JOBS[job_id]["status"] = "completed" if proc.returncode == 0 else "failed"
        if proc.returncode == 0:
            # The pipeline rewrote governed_kg_export.json — swap the in-memory
            # KG so /kg/data serves fresh data without a server restart.
            try:
                _reload_kg()
                JOBS[job_id]["kg_reloaded"] = True
            except Exception as exc:  # noqa: BLE001
                JOBS[job_id]["kg_reloaded"] = False
                JOBS[job_id]["reload_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)
        JOBS[job_id]["finished_at"] = time.time()
    _save_jobs()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Accept a .txt or .pdf document, save it, and launch the pipeline in the background."""
    filename = file.filename or "upload.txt"
    lower = filename.lower()
    if not (lower.endswith(".txt") or lower.endswith(".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")
    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Empty file")

    job_id = uuid.uuid4().hex[:12]

    if lower.endswith(".pdf"):
        try:
            import io
            from pypdf import PdfReader
        except ImportError:
            raise HTTPException(status_code=400, detail="PDF support requires the pypdf package")
        try:
            reader = PdfReader(io.BytesIO(content))
            text = "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {exc}")
        if not text:
            raise HTTPException(status_code=400, detail="No extractable text found in PDF")
        content = text.encode("utf-8")
        filename = Path(filename).stem + ".txt"

    safe_name = f"{job_id}_{Path(filename).name}"
    doc_path = INBOX_DIR / safe_name
    doc_path.write_bytes(content)

    JOBS[job_id] = {
        "job_id": job_id,
        "filename": filename,
        "doc_path": str(doc_path),
        "status": "queued",
        "queued_at": time.time(),
    }
    _save_jobs()

    # Fire-and-forget the pipeline subprocess on a worker thread.
    asyncio.get_event_loop().run_in_executor(None, _run_pipeline_job, job_id, doc_path)

    return {"job_id": job_id, "status": "queued", "filename": filename}


def _read_checkpoint_progress(job_id: str | None = None) -> dict:
    """Read per-stage progress written by CheckpointManager for a job.

    The pipeline writes checkpoints/jobs/<job_id>/<doc_id>/manifest.json plus a
    governed_kg_latest.json snapshot after each completed stage.
    """
    if job_id is None:
        return {}
    root = JOB_CHECKPOINT_ROOT / job_id
    if not root.exists():
        return {}
    progress: dict = {}
    try:
        for doc_dir in sorted(root.iterdir()):
            manifest_path = doc_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path) as f:
                manifest = json.load(f)
            progress = {
                "document_id": manifest.get("document_id"),
                "current_stage": manifest.get("last_stage"),
                "completed_stages": manifest.get("completed_stages", []),
                "last_update": manifest.get("last_update"),
            }
            snapshot_path = doc_dir / "governed_kg_latest.json"
            if snapshot_path.exists():
                with open(snapshot_path) as f:
                    snapshot = json.load(f)
                kg_blob = snapshot.get("knowledge_graph", snapshot)
                progress["entity_count"] = len(kg_blob.get("entities", []) or [])
                progress["triple_count"] = len(kg_blob.get("triples", []) or [])
    except (json.JSONDecodeError, OSError):
        pass
    return progress


# Canonical checkpoint stage order (ids match CheckpointManager manifests;
# stage 7 was consolidated into verification). Labels from the orchestrator's
# stage headers — used by the frontend status bar for progress fractions.
PIPELINE_STAGES = [
    {"id": "1",  "label": "Document Processing"},
    {"id": "2",  "label": "Domain Classification"},
    {"id": "2b", "label": "Governance Bootstrap"},
    {"id": "3",  "label": "Entity Extraction"},
    {"id": "3b", "label": "Orphan Relink"},
    {"id": "4",  "label": "Relation Extraction"},
    {"id": "4b", "label": "Connectivity Pass"},
    {"id": "5",  "label": "Evidence Linking"},
    {"id": "6",  "label": "Deliberation"},
    {"id": "8",  "label": "Verification"},
    {"id": "9",  "label": "KG Integration"},
]


@app.get("/pipeline/status")
def pipeline_status(job_id: str | None = None):
    if job_id:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        return {**job, "checkpoint": _read_checkpoint_progress(job_id), "stages": PIPELINE_STAGES}
    jobs = []
    for job in sorted(JOBS.values(), key=lambda j: j.get("queued_at", 0), reverse=True):
        if job.get("status") in ("queued", "running"):
            job = {**job, "checkpoint": _read_checkpoint_progress(job.get("job_id"))}
        jobs.append(job)
    return {"jobs": jobs, "checkpoint": {}, "stages": PIPELINE_STAGES}


# ── Eval run viewer (read-only) ──────────────────────────────────────────────
# Browse cached eval-harness outputs (DocRED-style doc caches) straight from
# evaluation/results/ without moving files. Response shape matches /kg/data so
# the frontend graph canvas renders them unchanged.

EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
_DOCRED_FILE_RE = re.compile(r"^doc_(\d+)_(rhf|singlepass|hybrid)\.json$")


def _discover_docred_runs() -> list[dict]:
    """Stat-only scan of evaluation/results for doc caches, newest first."""
    runs: dict[tuple[str, str], dict] = {}
    if not EVAL_RESULTS_DIR.is_dir():
        return []
    for d in EVAL_RESULTS_DIR.iterdir():
        if not d.is_dir():
            continue
        try:
            files = list(d.iterdir())
        except OSError:
            continue
        for f in files:
            m = _DOCRED_FILE_RE.match(f.name)
            if not m:
                continue
            key = (d.name, m.group(2))
            run = runs.setdefault(key, {
                "dir": d.name, "strategy": m.group(2),
                "doc_count": 0, "doc_indices": [], "latest_mtime": 0.0,
            })
            run["doc_count"] += 1
            run["doc_indices"].append(int(m.group(1)))
            run["latest_mtime"] = max(run["latest_mtime"], f.stat().st_mtime)
    out = sorted(runs.values(), key=lambda r: r["latest_mtime"], reverse=True)
    for r in out:
        r["doc_indices"].sort()
        r["label"] = f"{r['dir']} · {r['strategy']} ({r['doc_count']} docs)"
    return out


def _resolve_eval_run_dir(dir_name: str) -> Path:
    """Validate a run dir name against path traversal; return its Path."""
    if not dir_name or "/" in dir_name or "\\" in dir_name or dir_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid run dir")
    path = (EVAL_RESULTS_DIR / dir_name).resolve()
    if path.parent != EVAL_RESULTS_DIR.resolve() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown run dir: {dir_name}")
    return path


def _iter_run_docs(run_dir: Path, strategy: str, doc: str = "all"):
    """Yield (idx, path) for a run's doc caches, filtered to one doc if given."""
    out = []
    for f in sorted(run_dir.iterdir()):
        m = _DOCRED_FILE_RE.match(f.name)
        if m and m.group(2) == strategy and (doc == "all" or doc == m.group(1)):
            out.append((int(m.group(1)), f))
    return out


@app.get("/eval/runs")
def eval_runs():
    runs = _discover_docred_runs()
    return {"runs": runs, "default": runs[0] if runs else None}


@app.get("/eval/metrics")
def eval_metrics(dir: str, strategy: str):
    """Aggregate the counts blocks of one run: pred/gold sizes, calls, wall time."""
    run_dir = _resolve_eval_run_dir(dir)
    docs = []
    for idx, f in _iter_run_docs(run_dir, strategy):
        try:
            blob = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        c = blob.get("counts") or {}
        docs.append({
            "idx": idx, "title": blob.get("title", ""), "model": blob.get("model", ""),
            **{k: c.get(k) for k in (
                "wall_s", "llm_calls", "empty_calls", "pred_entities",
                "pred_triples", "gold_entities", "gold_triples", "error")},
        })
    if not docs:
        raise HTTPException(status_code=404, detail=f"No {strategy} docs in {dir}")

    def _tot(k):
        return sum(d.get(k) or 0 for d in docs)

    summary = {
        "docs": len(docs),
        "model": docs[0].get("model", ""),
        "pred_entities": _tot("pred_entities"), "gold_entities": _tot("gold_entities"),
        "pred_triples": _tot("pred_triples"), "gold_triples": _tot("gold_triples"),
        "llm_calls": _tot("llm_calls"), "empty_calls": _tot("empty_calls"),
        "errors": sum(1 for d in docs if d.get("error")),
        "mean_wall_s": round(_tot("wall_s") / len(docs), 1),
    }
    return {"summary": summary, "docs": docs}


@app.get("/eval/graph")
def eval_graph(dir: str, strategy: str, doc: str = "all"):
    """Reshape one run's doc caches into the /kg/data response contract.

    doc="all" merges every document (ids namespaced "<idx>::", one color per
    source doc); doc="<idx>" shows a single document with real entity types.
    """
    run_dir = _resolve_eval_run_dir(dir)
    files = _iter_run_docs(run_dir, strategy, doc)
    if not files:
        raise HTTPException(status_code=404, detail=f"No {strategy} docs in {dir} (doc={doc})")

    merged = doc == "all" and len(files) > 1
    entities: list[dict] = []
    triples: list[dict] = []
    for idx, f in files:
        try:
            blob = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        prefix = f"{idx}::" if merged else ""
        title = blob.get("title", f.name)
        for e in blob.get("entities", []):
            etype = f"doc_{idx}" if merged else (e.get("type") or "UNKNOWN")
            if etype == "?":
                etype = "UNKNOWN"
            entities.append({
                "id": prefix + e.get("id", ""),
                "labels": e.get("labels") or [e.get("name", e.get("id", ""))],
                "type": etype,
                "metadata": {"confidence": 1.0, "source": title, "description": ""},
            })
        for i, t in enumerate(blob.get("triples", [])):
            triples.append({
                "id": f"{prefix}t{i}",
                "subject": prefix + t.get("subject", ""),
                "relation": t.get("relation", ""),
                "object": prefix + t.get("object", ""),
                "confidence": t.get("confidence", 1.0),
                "source": title,
            })
    return {
        "entities": entities,
        "triples": triples,
        "entity_types": _build_entity_types(entities),
        "relation_types": sorted({t["relation"] for t in triples}),
        "org_chart": {"domains": [], "assignments": {}},
        "meta": {"dir": dir, "strategy": strategy, "doc": doc, "docs": len(files)},
    }


# ── Frontend static serving ──────────────────────────────────────────────────
# Registered after all API routes so they take precedence over the mount.

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "app" / "dist"
if FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    print(f"NOTE: no frontend build at {FRONTEND_DIST} — serving API only.")


# ── Startup logic ────────────────────────────────────────────────────────────


def init_kg_only():
    """Load KG data only — no LLM calls, no org chart, no QA system."""
    global governed_kg

    _INIT_CONFIG.update({"mode": "data_only"})

    from multi_agent_kg.core import load_governed_kg

    if not os.path.exists(KG_FILE):
        print(f"WARNING: {KG_FILE} not found.")
        return

    print(f"Loading KG from {KG_FILE}...")
    governed_kg = load_governed_kg(KG_FILE)
    stats = governed_kg.kg.get_stats()
    print(f"KG: {stats['num_entities']} entities, {stats['num_triples']} triples")
    print("Data-only mode: /health and /kg/data available. /qa returns 503.")


def init_kg_and_qa(basic: bool = False, no_debate: bool = False,
                   no_critic: bool = False, exploration_rounds: int = 3):
    """Load KG and initialize QA system."""
    global governed_kg, qa_system

    _INIT_CONFIG.update({
        "mode": "full",
        "basic": basic,
        "no_debate": no_debate,
        "no_critic": no_critic,
        "exploration_rounds": exploration_rounds,
    })

    from multi_agent_kg.core import (
        DomainBuilder,
        LLMConfig,
        create_qa_system,
        load_governed_kg,
    )
    from multi_agent_kg.core.domain_experts import OrgChart

    if not os.path.exists(KG_FILE):
        print(f"WARNING: {KG_FILE} not found. Server running without KG data.")
        print("Run the pipeline first to generate the KG, or place a governed_kg_export.json in the project root.")
        return

    model = os.getenv("LLM_DEFAULT_MODEL", "gemma4:31b")
    llm_config = LLMConfig(model=model, temperature=0.2, max_tokens=4096)

    print(f"Loading KG from {KG_FILE}...")
    governed_kg = load_governed_kg(KG_FILE)
    kg = governed_kg.kg
    stats = kg.get_stats()
    print(f"KG: {stats['num_entities']} entities, {stats['num_triples']} triples")

    # Org chart (cached)
    cache_valid = False
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cached_hash = json.load(f).get("_kg_hash")
            if cached_hash == _kg_hash(KG_FILE):
                cache_valid = True
            else:
                print("KG changed since cache — rebuilding org chart...")
                os.remove(CACHE_FILE)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"Cache file corrupted ({e}), rebuilding org chart...")
            os.remove(CACHE_FILE)

    if cache_valid:
        try:
            print(f"Loading cached org chart from {CACHE_FILE}...")
            with open(CACHE_FILE) as f:
                data = json.load(f)
            org_chart = OrgChart.from_dict(data, kg)
            governed_kg.set_org_chart(org_chart)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"Cache load failed ({e}), rebuilding org chart...")
            cache_valid = False

    if not cache_valid:
        print("Building org chart (will be cached for next time)...")
        builder = DomainBuilder(llm_config)
        org_chart = builder.build(kg)
        governed_kg.set_org_chart(org_chart)
        data = {"_kg_hash": _kg_hash(KG_FILE), **org_chart.to_dict()}
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Org chart cached to: {CACHE_FILE}")

    if governed_kg.org_chart.domains:
        print(f"{governed_kg.org_chart.domain_summary()}")

    # QA system
    if basic:
        qa_system = create_qa_system(governed_kg=governed_kg, llm_config=llm_config, advanced=False)
        print("Basic QA system ready.")
    else:
        qa_system = create_qa_system(
            governed_kg=governed_kg,
            llm_config=llm_config,
            advanced=True,
            max_exploration_rounds=exploration_rounds,
            enable_debate=not no_debate,
            enable_critic=not no_critic,
        )
        print("Advanced QA system ready.")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="MaKG API Server")
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--data-only", action="store_true", help="Load KG only — no LLM, no QA")
    parser.add_argument("--basic", action="store_true")
    parser.add_argument("--no-debate", action="store_true")
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--exploration-rounds", type=int, default=3)
    args = parser.parse_args()

    if args.data_only:
        init_kg_only()
    else:
        init_kg_and_qa(
            basic=args.basic,
            no_debate=args.no_debate,
            no_critic=args.no_critic,
            exploration_rounds=args.exploration_rounds,
        )

    print(f"\nMaKG API server at http://{args.host}:{args.port}")
    print("Endpoints: /health, /kg/data, /kg/stats, /kg/reload, /models, /qa, /qa/stream, "
          "/governance/audit, /governance/pending, /governance/review, /ingest, /pipeline/status\n")
    uvicorn.run(app, host=args.host, port=args.port)
