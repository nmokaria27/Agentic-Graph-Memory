"""Dreamify-style hyperparameter tuning for Agent-Graph-Memory (Cognee paper item ①).

Two-stage search:
  Stage A (tune_retrieval.py): cheap retrieval-only TPE against cached KGs (no rebuild).
  Stage B (tune_chunk.py): coarse chunk_size grid (rebuilds KGs) + mandatory short
                           Stage-A re-tune on the winning topology.

Winning configs are written to evaluation/results/tune/<bench>_best_config.json and are
NEVER baked into core RetrievalConfig defaults (would regress other datasets).
"""
