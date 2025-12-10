"""
Quick test script to verify the installation and basic functionality.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple, Entity
from multi_agent_kg.core.config import RelationType, RelationSchema


def test_knowledge_graph():
    """Test basic knowledge graph operations."""
    print("Testing Knowledge Graph...")

    kg = KnowledgeGraph()

    # Add entities
    kg.add_entity("Aspirin", labels=["acetylsalicylic acid"], entity_type="Drug")
    kg.add_entity("Headache", entity_type="Symptom")
    kg.add_entity("Fever", entity_type="Symptom")

    # Add triples
    kg.add_triple("Aspirin", "treats", "Headache", confidence=0.9)
    kg.add_triple("Aspirin", "treats", "Fever", confidence=0.85)

    print(f"✓ Added {len(kg.entities)} entities")
    print(f"✓ Added {len(kg.triples)} triples")

    # Test conflict detection
    conflicting = [Triple("Aspirin", "treats", "Diabetes", confidence=0.5)]
    conflicts = kg.find_conflicts(conflicting)

    print(f"✓ Conflict detection works (found {len(conflicts)} conflicts)")

    return True


def test_relation_schema():
    """Test relation schema."""
    print("\nTesting Relation Schema...")

    schema = RelationSchema(
        types=[
            RelationType(
                name="treats",
                description="Medical treatment relationship",
                allowed_subject_types=["Drug"],
                allowed_object_types=["Disease", "Symptom"],
            ),
        ]
    )

    print(f"✓ Created schema with {len(schema.types)} relation types")
    print(f"✓ Schema has 'treats' relation: {schema.has_relation_type('treats')}")

    return True


def test_imports():
    """Test that all modules can be imported."""
    print("\nTesting Module Imports...")

    try:
        from multi_agent_kg.llm.openai_client import chat_completion
        from multi_agent_kg.agents.base_agent import Agent
        from multi_agent_kg.agents.ingestion_agent import IngestionAgent
        from multi_agent_kg.agents.segmenter_agent import SegmenterAgent
        from multi_agent_kg.agents.entity_agent import EntityAgent
        from multi_agent_kg.agents.relation_agent import RelationAgent
        from multi_agent_kg.agents.schema_agent import SchemaAgent
        from multi_agent_kg.agents.conflict_agent import ConflictAgent
        from multi_agent_kg.agents.verifier_agent import VerifierAgent
        from multi_agent_kg.core.orchestrator import Orchestrator

        print("✓ All modules imported successfully")
        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("MULTI-AGENT KG - INSTALLATION TEST")
    print("=" * 70 + "\n")

    tests = [
        ("Module Imports", test_imports),
        ("Knowledge Graph", test_knowledge_graph),
        ("Relation Schema", test_relation_schema),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\nAll tests passed! Installation is successful.")
        print("\nNext steps:")
        print("  1. Set your OPENAI_API_KEY in .env file")
        print("  2. Run: python -m multi_agent_kg.examples.run_pipeline")
    else:
        print("\nSome tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
