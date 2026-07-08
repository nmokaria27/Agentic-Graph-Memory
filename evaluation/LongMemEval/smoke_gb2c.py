"""GB-2c smoke test: reproduce the exact q0 scenario (personal_best_time held
via HAS_TIME=27:12 and, on update, HAS_VALUE=25:50 — two different relation
names for the same fact) and confirm the REAL LLMConflictResolver (not a
stub) supersedes correctly once _relation_aware_conflicts surfaces the pair.

Cheap: 1 resolver call, no re-extraction. Run on whichever lane .env points at.
"""
from multi_agent_kg.core.conflict_resolution import LLMConflictResolver
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph
from multi_agent_kg.core.knowledge_graph import is_superseded
from multi_agent_kg.core.vector_index import KGVectorStore


def main() -> None:
    gkg = GovernedKnowledgeGraph(governance_mode="permissive",
                                 conflict_resolver=LLMConflictResolver())
    gkg.vector_store = KGVectorStore()  # real embeddings (gpu01 Ollama / .env)
    for eid in ("personal_best_time", "twenty_seven_twelve", "twenty_five_fifty",
               "charity_5k_run"):
        gkg.add_entity(eid)

    print("Proposing OLD fact: personal_best_time -[HAS_TIME]-> 27:12 "
         "(evidence: earlier session)")
    gkg.propose_triple(
        "personal_best_time", "HAS_TIME", "twenty_seven_twelve", confidence=0.9,
        metadata={"evidence": "My personal best time in the charity 5K run is 27:12.",
                  "provenance": {"refs": [{"document_date": "2023/05/25 (Thu)"}]}},
    )
    print("Proposing NEW fact: personal_best_time -[HAS_VALUE]-> 25:50 "
         "(evidence: later session, updated PB)\n")
    decision = gkg.propose_triple(
        "personal_best_time", "HAS_VALUE", "twenty_five_fifty", confidence=0.9,
        metadata={"evidence": "I beat my personal best — new time is 25:50!",
                  "provenance": {"refs": [{"document_date": "2023/05/27 (Sat)"}]}},
    )

    old = [t for t in gkg.kg.triples if t.object == "twenty_seven_twelve"][0]
    stats = gkg.get_stats()["conflict_resolution"]
    print(f"conflict_resolution stats: {stats}")
    print(f"decision.action = {decision.action!r}, committed = {decision.committed}")
    print(f"old triple (HAS_TIME=27:12) superseded = {is_superseded(old)}")
    if old.metadata.get("conflict_resolution") or decision.triple.metadata.get("conflict_resolution"):
        cr = decision.triple.metadata.get("conflict_resolution", {})
        print(f"resolver reasoning: {cr.get('reasoning', '(none)')!r}")

    if is_superseded(old):
        print("\nGB2C_SMOKE: PASS — relation-name variance no longer blinds conflict detection")
    else:
        print("\nGB2C_SMOKE: NO SUPERSEDE — either not detected as conflict, or "
             "resolver judged coexist (reasoning above)")


if __name__ == "__main__":
    main()
