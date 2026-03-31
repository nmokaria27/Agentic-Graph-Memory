"""
KGAFE — Knowledge-Graph-grounded Atomic Fact Evaluation.

A novel evaluation framework that decomposes QA answers into atomic facts
and verifies each against the knowledge graph using three-tier verification:
  1. Exact triple match
  2. Path-based inference (multi-hop)
  3. LLM semantic entailment

Inspired by FActScore (Min et al., 2023), SAFE (Wei et al., 2024), and
KARMA (NeurIPS 2025), but uniquely grounded in a structured KG rather
than unstructured text or web search.
"""
