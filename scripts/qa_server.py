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

from multi_agent_kg.core import (
    DomainBuilder,
    LLMConfig,
    create_qa_system,
    load_governed_kg,
)
from multi_agent_kg.core.domain_experts import OrgChart

CACHE_FILE = "org_chart_cache.json"
KG_FILE = "governed_kg_export.json"


def _kg_hash(kg_path: str) -> str:
    """Return MD5 hex digest of the KG file for cache invalidation."""
    h = hashlib.md5()
    with open(kg_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_org_chart(org_chart: OrgChart, path: str, kg_path: str = KG_FILE) -> None:
    """Serialize OrgChart to JSON, embedding the KG hash for invalidation."""
    data = {"_kg_hash": _kg_hash(kg_path), **org_chart.to_dict()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Org chart cached to: {path}")


def load_org_chart(path: str, kg) -> OrgChart:
    """Deserialize OrgChart from JSON cache."""
    with open(path) as f:
        data = json.load(f)
    return OrgChart.from_dict(data, kg)


# --- Setup (runs once at startup) ---
print("Loading KG and building QA system...")
llm_config = LLMConfig(model="gemma4:31b", temperature=0.2, max_tokens=4096)

governed_kg = load_governed_kg(KG_FILE)
kg = governed_kg.kg
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
    governed_kg.set_org_chart(org_chart)
    print(f"{org_chart.domain_summary()}")
else:
    print("Building org chart (this takes ~10 min, will be cached for next time)...")
    builder = DomainBuilder(llm_config)
    org_chart = builder.build(kg)
    governed_kg.set_org_chart(org_chart)
    save_org_chart(org_chart, CACHE_FILE)
    print(f"{org_chart.domain_summary()}")

if governed_kg.org_chart.domains:
    org_chart = governed_kg.org_chart

# Parse command-line args for mode selection
parser = argparse.ArgumentParser(description="KG QA Server")
parser.add_argument("--basic", action="store_true",
                    help="Use basic QAOrchestrator instead of the default AdvancedQAOrchestrator")
parser.add_argument("--no-debate", action="store_true", help="Disable debate arena")
parser.add_argument("--no-critic", action="store_true", help="Disable critic agent")
parser.add_argument("--exploration-rounds", type=int, default=3, help="Max exploration rounds per expert")
args, _ = parser.parse_known_args()

if args.basic:
    qa = create_qa_system(governed_kg=governed_kg, llm_config=llm_config, advanced=False)
    print("Basic QA system ready.\n")
else:
    qa = create_qa_system(
        governed_kg=governed_kg,
        llm_config=llm_config,
        advanced=True,
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
