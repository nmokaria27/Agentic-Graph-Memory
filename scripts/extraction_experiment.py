"""Extraction-strategy experiment harness.

Compares extraction approaches on the SAME document under identical conditions so we
can pick the best backend for the pipeline BEFORE committing. Domain-adaptive: every
strategy first runs DomainClassifier, so the discovered domain (law / medical / etc.)
drives the schema hints — the graph stays general and self-organizing.

Strategies:
  rhf         : current DeliberativeOrchestrator multi-stage RHF (baseline, reasoning ON)
  singlepass  : GraphRAG-style ONE-CALL extraction of entities+relationships, guided by
                the discovered domain types, reasoning OFF (fast path), + 1 gleaning round

Metrics: wall-time, LLM calls, entities, triples, orphan rate, empty-output count.

Usage:
  NEMOTRON_THINKING=off python scripts/extraction_experiment.py --strategy singlepass
  python scripts/extraction_experiment.py --strategy rhf
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multi_agent_kg.llm.openai_client as oc

# ── instrument every LLM call ────────────────────────────────────────────────
CALLS = []
_orig = oc.chat_completion
def _timed(messages, *a, **k):
    t = time.time(); out = _orig(messages, *a, **k); dt = time.time() - t
    CALLS.append((dt, len(out or "")))
    return out
oc.chat_completion = _timed

from multi_agent_kg.core.config import LLMConfig
from multi_agent_kg.agents.base import ModelTier, AgentContext

SAMPLE = """Alice Chen works at Northwind Robotics as a senior engineer.
Bob Ramirez lives in Portland and drives a blue Toyota.
The Meridian Bridge was completed in 1998 by the city of Ashford.
Dr. Elena Voss discovered the Kessler enzyme in her lab at Trellis University.
The Falcon Q7 drone has a maximum flight time of 40 minutes.
Marcus Webb founded Lumen Analytics in 2011.
Nina Patel manages the customer success team at Vertex Cloud.
The Orion Mark III telescope weighs 12 kilograms.
Later, Bob Ramirez moved to Seattle and now drives a red Honda.
Marcus Webb stepped down as CEO of Lumen Analytics in 2020.
Dr. Elena Voss later joined the National Institute of Enzymology as director.
The Kessler enzyme was subsequently found in deep-sea hydrothermal vents.
"""


def _dump_graph(strategy, entity_lines, triples):
    """Write the extracted graph so richness/correctness can be inspected qualitatively."""
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"..//exp_{strategy}_kg.json")
    try:
        with open(os.path.abspath(out), "w") as f:
            json.dump({"entities": entity_lines,
                       "triples": [f"({s}) -[{r}]-> ({o})" for s, r, o in triples]}, f, indent=2)
        print(f"[EXP] graph dumped -> exp_{strategy}_kg.json ({len(entity_lines)} ents, {len(triples)} triples)")
    except Exception as exc:
        print(f"[EXP] dump failed: {exc}")


def classify_domain(model):
    """Adaptive step: discover the domain + schema hints for this document."""
    from multi_agent_kg.agents.domain_classifier import DomainClassifier
    dc = DomainClassifier(llm_config=LLMConfig(model=model))
    ctx = AgentContext(document_id="exp", text=SAMPLE)
    res = dc.run(ctx, segments=[{"text": SAMPLE, "segment_id": "s0", "id": "s0"}])
    d = getattr(res, "items", None)
    if not isinstance(d, dict):
        d = getattr(res, "metadata", {}) or {}
    # domain fields may be nested one level down (e.g. items['domain_context'])
    if "primary_domain" not in d:
        for v in d.values():
            if isinstance(v, dict) and "primary_domain" in v:
                return v
    return d


SINGLEPASS_PROMPT = """You are extracting a knowledge graph from text in the domain: {domain}.

Suggested entity types (extend freely if the text needs new ones): {etypes}
Suggested relation types (extend freely if the text needs new ones): {rtypes}

Extract EVERY significant entity and EVERY relationship in ONE pass. Use the exact surface
form for entity names. Prefer specific relations; connect every entity to at least one other
when the text supports it.

TEXT:
{text}

Return ONLY JSON:
{{"entities":[{{"name":"...","type":"..."}}],
  "relationships":[{{"source":"...","relation":"...","target":"..."}}]}}"""

GLEAN_PROMPT = """Some entities below have NO relationship yet. Using ONLY the text, add any
supported relationships for them. Do not invent facts.

TEXT:
{text}

UNCONNECTED ENTITIES: {orphans}
EXISTING RELATIONSHIPS: {rels}

Return ONLY JSON: {{"relationships":[{{"source":"...","relation":"...","target":"..."}}]}}"""


def run_singlepass(model, domain_info):
    dom = domain_info.get("primary_domain", "general") if isinstance(domain_info, dict) else "general"
    etypes = ", ".join(e.get("type", e) if isinstance(e, dict) else str(e)
                       for e in (domain_info.get("entity_types", []) if isinstance(domain_info, dict) else []))[:400] or "PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT"
    rtypes = ", ".join(r.get("type", r) if isinstance(r, dict) else str(r)
                       for r in (domain_info.get("relation_types", []) if isinstance(domain_info, dict) else []))[:400] or "WORKS_AT, FOUNDED, LOCATED_IN, DISCOVERED, MOVED_TO"
    p = SINGLEPASS_PROMPT.format(domain=dom, etypes=etypes, rtypes=rtypes, text=SAMPLE)
    res = oc.chat_completion_json(
        messages=[{"role": "system", "content": "You are a precise knowledge-graph extractor. Return ONLY valid JSON."},
                  {"role": "user", "content": p}],
        model=model, temperature=0.1, max_tokens=4096)
    ents = res.get("entities", []) if isinstance(res, dict) else []
    rels = res.get("relationships", []) if isinstance(res, dict) else []
    # gleaning: connect orphans
    names = {e.get("name", "").lower() for e in ents if isinstance(e, dict)}
    connected = set()
    for r in rels:
        if isinstance(r, dict):
            connected.add(str(r.get("source", "")).lower()); connected.add(str(r.get("target", "")).lower())
    orphans = [e.get("name") for e in ents if isinstance(e, dict) and e.get("name", "").lower() not in connected]
    if orphans:
        g = oc.chat_completion_json(
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": GLEAN_PROMPT.format(text=SAMPLE, orphans=", ".join(orphans),
                                                                      rels=json.dumps(rels)[:800])}],
            model=model, temperature=0.1, max_tokens=2048)
        rels += (g.get("relationships", []) if isinstance(g, dict) else [])
    # recompute orphan rate
    connected = set()
    for r in rels:
        if isinstance(r, dict):
            connected.add(str(r.get("source", "")).lower()); connected.add(str(r.get("target", "")).lower())
    orphan_n = sum(1 for e in ents if isinstance(e, dict) and e.get("name", "").lower() not in connected)
    rel_types = {str(r.get("relation", "")).lower() for r in rels if isinstance(r, dict)}
    _dump_graph("singlepass",
                [e.get("name", "") + "::" + str(e.get("type")) for e in ents if isinstance(e, dict)],
                [(str(r.get("source")), str(r.get("relation")), str(r.get("target"))) for r in rels if isinstance(r, dict)])
    return {"entities": len(ents), "triples": len(rels),
            "relation_types": len(rel_types), "avg_degree": round(2 * len(rels) / max(len(ents), 1), 2),
            "orphans": orphan_n, "orphan_rate": round(orphan_n / max(len(ents), 1), 3),
            "sample_entities": [e.get("name") + "::" + str(e.get("type")) for e in ents[:12] if isinstance(e, dict)]}


def run_rhf(model, self_consistency=False):
    from multi_agent_kg.core import DeliberativeOrchestrator
    from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
    from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
    gkg = GovernedKnowledgeGraph(governance_mode="permissive")
    orch = DeliberativeOrchestrator(
        llm_config=LLMConfig(model=model), knowledge_graph=KnowledgeGraph(), governed_kg=gkg,
        quality_threshold=0.35, max_refinement_iterations=1, enable_self_consistency=self_consistency,
        enable_open_world=True, enable_cross_document=False,
        model_tiers={t: model for t in ModelTier})
    orch.process_corpus([{"text": SAMPLE, "metadata": {"source": "exp"}}])
    gkg = orch.governed_kg
    ents = gkg.entities
    # orphan = entity id never appearing as subject/object of an active triple
    conn = set()
    for t in gkg.kg.get_active_triples():
        conn.add(t.subject); conn.add(t.object)
    orphan_n = sum(1 for eid in ents if eid not in conn)
    rel_types = {getattr(t, "relation", "").lower() for t in gkg.triples}
    _dump_graph("rhf",
                [f"{eid}::{getattr(e, 'entity_type', '?')}" for eid, e in ents.items()],
                [(t.subject, t.relation, t.object) for t in gkg.triples])
    return {"entities": len(ents), "triples": len(gkg.triples),
            "relation_types": len(rel_types), "avg_degree": round(2 * len(gkg.triples) / max(len(ents), 1), 2),
            "orphans": orphan_n, "orphan_rate": round(orphan_n / max(len(ents), 1), 3),
            "sample_entities": list(ents.keys())[:12]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["rhf", "singlepass"], required=True)
    ap.add_argument("--sc", action="store_true",
                    help="enable item-level self-consistency (3 samples/stage; ~3x calls, attacks run variance)")
    args = ap.parse_args()
    model = os.environ.get("LLM_DEFAULT_MODEL", "nvidia/nemotron-3-nano")
    print(f"[EXP] strategy={args.strategy} model={model} NEMOTRON_THINKING={oc.NEMOTRON_THINKING}")
    t0 = time.time()
    dom = classify_domain(model)
    dom_name = dom.get("primary_domain", "?") if isinstance(dom, dict) else "?"
    if args.strategy == "singlepass":
        m = run_singlepass(model, dom)
    else:
        m = run_rhf(model, self_consistency=args.sc)
    wall = time.time() - t0
    empties = sum(1 for dt, ln in CALLS if ln == 0)
    print("\n===== RESULT =====")
    print(json.dumps({"strategy": args.strategy, "domain": dom_name, "wall_s": round(wall, 1),
                      "llm_calls": len(CALLS), "empty_calls": empties, **m}, indent=2))


if __name__ == "__main__":
    main()
