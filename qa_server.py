"""
Minimal QA server for the KG Explorer's "Ask your own question" feature.
Run this alongside the HTML viewer: python3 qa_server.py

Serves at http://localhost:5050/qa  (POST {"question": "..."})
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv

load_dotenv()

from multi_agent_kg.core import LLMConfig, load_kg, DomainBuilder, QAOrchestrator

# --- Setup (runs once at startup) ---
print("Loading KG and building QA system...")
llm_config = LLMConfig(model="gemma3:27b", temperature=0.2, max_tokens=4096)

kg = load_kg("kg_export.json")
stats = kg.get_stats()
print(f"KG: {stats['num_entities']} entities, {stats['num_triples']} triples")

builder = DomainBuilder(llm_config)
org_chart = builder.build(kg)
print(f"{org_chart.domain_summary()}")

qa = QAOrchestrator(org_chart=org_chart, full_kg=kg, llm_config=llm_config)
print("QA system ready!\n")


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
        try:
            self.send_response(code)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())
        except BrokenPipeError:
            print("  (client disconnected before response was sent)")

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
