"""
Simple usage example for quick start.
"""

from multi_agent_kg.core.orchestrator import Orchestrator
from multi_agent_kg.core.config import RelationType, RelationSchema, LLMConfig


def main():
    """Quick start example."""

    # Define a simple relation schema
    schema = RelationSchema(
        types=[
            RelationType(
                name="treats",
                description="Treatment relationship",
            ),
            RelationType(
                name="causes",
                description="Causal relationship",
            ),
        ]
    )

    # Create orchestrator
    orchestrator = Orchestrator(
        relation_schema=schema,
        llm_config=LLMConfig(model="gpt-4", temperature=0.2),
    )

    # Sample text
    text = """
    Aspirin is commonly used to treat headaches and reduce fever.
    Studies show that smoking causes lung cancer and heart disease.
    """

    # Process the text
    print("Processing text...\n")
    kg = orchestrator.process_document(text=text)

    # Print results
    print("\nKnowledge Graph:")
    kg.print_graph()

    # Export to JSON
    orchestrator.export_to_json("output_kg.json")
    print("\n✓ Exported to output_kg.json")


if __name__ == "__main__":
    main()
