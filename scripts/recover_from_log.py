#!/usr/bin/env python3
"""
Emergency Recovery Script for multi-agent-kg.

This script parses 'pipeline_debug.log' to reconstruct extracted entities 
and triples if the main pipeline process was killed or crashed before 
the final export.
"""

import os
import json
import re
from typing import List, Dict, Any

def recover_from_log(log_path: str = "pipeline_debug.log"):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    print(f"Analyzing {log_path} for extractable data...")
    
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Recover Entities
    # We look for the "Entity Extraction SUMMARY" or the terminal output pattern
    # since memory logging might be disabled in current run.
    # Pattern: "-> X entities extracted"
    entity_summary_pattern = re.compile(r"Segment \d+/63 \(.*?\)\s+— extracting entities.*?\n\s+-> (\d+) entities extracted")
    
    # But better, we look for LLM CALL logs if they exist. 
    # In the user's log, we only see the stage headers.
    
    # Look for any JSON-like structures that look like entity lists
    # [{"id": "...", "type": "..."}]
    recovered_entities = []
    seen_entity_ids = set()
    
    # Try finding LLM responses in the log
    json_blocks = re.finditer(r'\{.*?\n\}', content, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block.group(0))
            if isinstance(data, dict) and "entities" in data:
                for ent in data["entities"]:
                    eid = ent.get("id") or ent.get("text")
                    if eid and eid not in seen_entity_ids:
                        recovered_entities.append(ent)
                        seen_entity_ids.add(eid)
        except:
            continue

    print(f"✓ Recovered {len(recovered_entities)} unique entities from JSON blocks.")

    # 2. Recover Triples
    # Same for triples
    recovered_triples = []
    seen_triple_keys = set()
    
    json_blocks = re.finditer(r'\{.*?\n\}', content, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block.group(0))
            if isinstance(data, dict):
                triples = data.get("triples", data.get("linked_triples", []))
                if isinstance(triples, list):
                    for t in triples:
                        subj = t.get("subject")
                        rel = t.get("relation")
                        obj = t.get("object")
                        if subj and rel and obj:
                            key = f"{subj}|{rel}|{obj}"
                            if key not in seen_triple_keys:
                                recovered_triples.append(t)
                                seen_triple_keys.add(key)
        except:
            continue

    print(f"✓ Recovered {len(recovered_triples)} unique triples.")

    # 3. Create a valid KG export format
    export_data = {
        "knowledge_graph": {
            "entities": recovered_entities,
            "triples": recovered_triples
        },
        "metadata": {
            "recovery_timestamp": "recovered",
            "source": "log_recovery"
        }
    }

    output_path = "kg_recovered.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"\nSUCCESS: Recovery complete. Saved to: {output_path}")
    print("You can use this file with the IncrementalEnricher to continue your work.")

if __name__ == "__main__":
    recover_from_log()
