"""
QA server for the KG Explorer's "Ask your own question" feature.
Run this alongside the HTML viewer: python3 qa_server.py

Serves at http://localhost:5050/qa  (POST {"question": "..."})

Uses the AdvancedQAOrchestrator by default with:
  - Active graph exploration
  - Multi-expert debate
  - Self-reflection critic
  - Persistent session memory
  - Provenance chain tracking

Pass --basic to use the simpler QAOrchestrator instead.

The org chart is cached to org_chart_cache.json after the first build.
Delete that file to force a rebuild.
"""

import json
import hashlib
import os
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from multi_agent_kg.core import LLMConfig, load_kg, DomainBuilder, QAOrchestrator
from multi_agent_kg.core.domain_experts import OrgChart, Domain, TopicSubAgent

CACHE_FILE = "org_chart_cache.json"
KG_FILE = "kg_export.json"


def _kg_hash(kg_path: str) -> str:
    """Return MD5 hex digest of the KG file for cache invalidation."""
    h = hashlib.md5()
    with open(kg_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_org_chart(org_chart: OrgChart, path: str, kg_path: str = KG_FILE) -> None:
    """Serialize OrgChart to JSON, embedding the KG hash for invalidation."""
    data = {
        "_kg_hash": _kg_hash(kg_path),
        "domains": [
            {
                "domain_id": d.domain_id,
                "label": d.label,
                "description": d.description,
                "entity_ids": sorted(d.entity_ids),
                "relation_schema": d.relation_schema,
                "topics": [
                    {
                        "topic_id": t.topic_id,
                        "label": t.label,
                        "description": t.description,
                        "entity_ids": sorted(t.entity_ids),
                        "relation_types": sorted(t.relation_types),
                        "keywords": t.keywords,
                    }
                    for t in d.topics
                ],
            }
            for d in org_chart.domains
        ],
        "cross_domain_relation_count": len(org_chart.cross_domain_relations),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Org chart cached to: {path}")


def load_org_chart(path: str, kg) -> OrgChart:
    """Deserialize OrgChart from JSON cache."""
    with open(path) as f:
        data = json.load(f)

    domains = []
    for dd in data["domains"]:
        topics = [
            TopicSubAgent(
                topic_id=t["topic_id"],
                label=t["label"],
                description=t["description"],
                entity_ids=set(t.get("entity_ids", [])),
                relation_types=set(t.get("relation_types", [])),
                keywords=t.get("keywords", []),
            )
            for t in dd.get("topics", [])
        ]
        domains.append(Domain(
            domain_id=dd["domain_id"],
            label=dd["label"],
            description=dd["description"],
            entity_ids=set(dd.get("entity_ids", [])),
            relation_schema=dd.get("relation_schema", {}),
            topics=topics,
        ))

    # Rebuild cross-domain relations from KG
    all_domain_ids = set()
    entity_to_domain = {}
    for d in domains:
        all_domain_ids.add(d.domain_id)
        for eid in d.entity_ids:
            entity_to_domain[eid] = d.domain_id

    cross = [
        t for t in kg.triples
        if entity_to_domain.get(t.subject) != entity_to_domain.get(t.object)
        and t.subject in entity_to_domain
        and t.object in entity_to_domain
    ]

    return OrgChart(domains=domains, cross_domain_relations=cross)


# --- Setup (runs once at startup) ---
print("Loading KG and building QA system...")
llm_config = LLMConfig(model="gemma3:27b", temperature=0.2, max_tokens=4096)

kg = load_kg("kg_export.json")
stats = kg.get_stats()
print(f"KG: {stats['num_entities']} entities, {stats['num_triples']} triples")

cache_valid = False
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as _f:
        cached_hash = json.load(_f).get("_kg_hash")
    current_hash = _kg_hash(KG_FILE)
    if cached_hash == current_hash:
        cache_valid = True
    else:
        print(f"KG has changed since cache was built — rebuilding org chart...")
        os.remove(CACHE_FILE)

if cache_valid:
    print(f"Loading cached org chart from {CACHE_FILE}...")
    org_chart = load_org_chart(CACHE_FILE, kg)
    print(f"{org_chart.domain_summary()}")
else:
    print("Building org chart (this takes ~10 min, will be cached for next time)...")
    builder = DomainBuilder(llm_config)
    org_chart = builder.build(kg)
    save_org_chart(org_chart, CACHE_FILE)
    print(f"{org_chart.domain_summary()}")

# Parse command-line args for mode selection
parser = argparse.ArgumentParser(description="KG QA Server")
parser.add_argument("--basic", action="store_true",
                    help="Use basic QAOrchestrator instead of the default AdvancedQAOrchestrator")
parser.add_argument("--no-debate", action="store_true", help="Disable debate arena")
parser.add_argument("--no-critic", action="store_true", help="Disable critic agent")
parser.add_argument("--exploration-rounds", type=int, default=3, help="Max exploration rounds per expert")
args, _ = parser.parse_known_args()

if args.basic:
    qa = QAOrchestrator(org_chart=org_chart, full_kg=kg, llm_config=llm_config)
    print("Basic QA system ready.\n")
else:
    from multi_agent_kg.core.advanced_qa import AdvancedQAOrchestrator
    qa = AdvancedQAOrchestrator(
        org_chart=org_chart,
        full_kg=kg,
        llm_config=llm_config,
        max_exploration_rounds=args.exploration_rounds,
        enable_debate=not args.no_debate,
        enable_critic=not args.no_critic,
    )
    print("Advanced QA system ready (active exploration + debate + critic + memory + provenance)\n")


class QAHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/qa":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        question = body.get("question", "")

        if not question:
            self._json_response({"error": "No question provided"}, 400)
            return

        print(f"\nQ: {question}")
        result = qa.query(question)
        print(f"A: {result['final_answer'][:200]}...")

        self._json_response(result)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, code=200):
        self.send_response(code)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format, *args):
        pass  # suppress default logging


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    port = int(os.getenv("QA_PORT", "5050"))
    server = ReusableHTTPServer(("localhost", port), QAHandler)
    print(f"QA server listening on http://localhost:{port}/qa")
    print("Open kg_explorer.html in your browser and type questions!\n")
    server.serve_forever()
