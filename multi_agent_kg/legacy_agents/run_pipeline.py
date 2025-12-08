"""
Example pipeline demonstrating end-to-end knowledge graph enrichment.

This script demonstrates:
1. Creating a relation schema
2. Processing sample text through the pipeline
3. Visualizing the resulting knowledge graph
"""

from multi_agent_kg.core.orchestrator import Orchestrator
from multi_agent_kg.core.config import RelationType, RelationSchema, LLMConfig


def create_medical_schema() -> RelationSchema:
    """Create a sample medical relation schema."""
    return RelationSchema(
        types=[
            RelationType(
                name="treats",
                description="A treatment or therapy for a medical condition",
                allowed_subject_types=["Drug", "Therapy", "Treatment"],
                allowed_object_types=["Disease", "Condition", "Symptom"],
            ),
            RelationType(
                name="causes",
                description="A causal relationship where the subject causes the object",
                allowed_subject_types=None,  # Allow any type
                allowed_object_types=None,
            ),
            RelationType(
                name="associated_with",
                description="An association or correlation between entities",
            ),
            RelationType(
                name="part_of",
                description="A hierarchical relationship where subject is part of object",
            ),
            RelationType(
                name="prevents",
                description="The subject prevents or reduces risk of the object",
                allowed_subject_types=["Drug", "Therapy", "Treatment", "Lifestyle"],
                allowed_object_types=["Disease", "Condition"],
            ),
        ],
        allow_new_types=False,  # Set to True for open-world mode
    )


def create_general_schema() -> RelationSchema:
    """Create a general-purpose relation schema."""
    return RelationSchema(
        types=[
            RelationType(
                name="works_for",
                description="Employment or affiliation relationship",
                allowed_subject_types=["Person"],
                allowed_object_types=["Organization", "Company"],
            ),
            RelationType(
                name="located_in",
                description="Geographic or spatial location",
                allowed_subject_types=["Organization", "Person", "Event"],
                allowed_object_types=["Location", "City", "Country"],
            ),
            RelationType(
                name="founded_by",
                description="Creation or founding relationship",
                allowed_subject_types=["Organization", "Company"],
                allowed_object_types=["Person"],
            ),
            RelationType(
                name="produces",
                description="Production or creation relationship",
                allowed_subject_types=["Organization", "Company"],
                allowed_object_types=["Product", "Service"],
            ),
        ],
        allow_new_types=True,  # Allow discovery of new relations
    )


# Sample documents
MEDICAL_SAMPLE = """
Aspirin is a widely used medication for treating headaches, fever, and inflammation.
Studies have shown that regular aspirin use can prevent heart attacks and strokes in 
high-risk patients. However, aspirin can cause stomach irritation and bleeding in 
some individuals.

Diabetes is a chronic condition that affects how the body processes blood sugar. 
Type 2 diabetes is often associated with obesity and lack of physical activity.
Metformin is the first-line treatment for type 2 diabetes and helps control blood 
sugar levels.

Recent research suggests that smoking causes lung cancer and significantly increases 
the risk of heart disease. Quitting smoking can prevent many of these health problems
and improve overall health outcomes.
"""

GENERAL_SAMPLE = """
Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne
in 1976. The company is headquartered in Cupertino, California, and produces popular 
products including the iPhone, iPad, and Mac computers.

Microsoft Corporation was founded by Bill Gates and Paul Allen in Redmond, Washington.
Microsoft produces software products including the Windows operating system and Office 
productivity suite. The company is also a major player in cloud computing through its 
Azure platform.

Elon Musk is the CEO of Tesla, an electric vehicle manufacturer located in Austin, Texas.
Tesla produces electric cars, battery storage systems, and solar panels. Musk also 
founded SpaceX, a aerospace company that develops rockets and spacecraft.
"""


def run_medical_example() -> None:
    """Run medical knowledge extraction example."""
    print("MEDICAL KNOWLEDGE EXTRACTION EXAMPLE")

    # Create schema and orchestrator
    schema = create_medical_schema()
    llm_config = LLMConfig(
        model="gpt-3.5-turbo",  # Use gpt-3.5-turbo (widely available)
        temperature=0.2,
    )

    orchestrator = Orchestrator(
        relation_schema=schema,
        llm_config=llm_config,
        enable_summarization=True,
        enable_verification=True,
        confidence_threshold=0.6,
    )

    # Process document
    kg = orchestrator.process_document(text=MEDICAL_SAMPLE)

    # Display results
    print("\n📊 FINAL KNOWLEDGE GRAPH:")
    kg.print_graph()

    # Export to JSON
    orchestrator.export_to_json("medical_knowledge_graph.json")


def run_general_example() -> None:
    """Run general knowledge extraction example."""
    print("GENERAL KNOWLEDGE EXTRACTION EXAMPLE")

    # Create schema and orchestrator
    schema = create_general_schema()
    llm_config = LLMConfig(
        model="gpt-3.5-turbo",  # Use gpt-3.5-turbo (widely available)
        temperature=0.2,
    )

    orchestrator = Orchestrator(
        relation_schema=schema,
        llm_config=llm_config,
        enable_summarization=False,  # Skip summarization for short text
        enable_verification=True,
        confidence_threshold=0.7,
    )

    # Process document (with open-world mode enabled via schema)
    kg = orchestrator.process_document(text=GENERAL_SAMPLE, open_world=True)

    # Display results
    print("\n📊 FINAL KNOWLEDGE GRAPH:")
    kg.print_graph()

    # Export to JSON
    orchestrator.export_to_json("general_knowledge_graph.json")


def run_custom_example() -> None:
    """Run a custom example with user-provided text."""
    print("CUSTOM KNOWLEDGE EXTRACTION")

    # Get custom text
    print("Enter your text (press Ctrl+D or Ctrl+Z when done):")
    print("-" * 70)

    try:
        lines = []
        while True:
            try:
                line = input()
                lines.append(line)
            except EOFError:
                break
        custom_text = "\n".join(lines)
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return

    if not custom_text.strip():
        print("No text provided.")
        return

    # Create a flexible schema
    schema = RelationSchema(allow_new_types=True)  # Open-world mode

    orchestrator = Orchestrator(
        relation_schema=schema,
        llm_config=LLMConfig(model="gpt-3.5-turbo", temperature=0.2),
        enable_summarization=len(custom_text) > 1000,
        enable_verification=True,
    )

    # Process
    kg = orchestrator.process_document(text=custom_text, open_world=True)

    # Display
    print("\n📊 FINAL KNOWLEDGE GRAPH:")
    kg.print_graph()

    # Export
    orchestrator.export_to_json("custom_knowledge_graph.json")


def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 70)
    print("MULTI-AGENT KNOWLEDGE GRAPH ENRICHMENT - EXAMPLES")
    print("=" * 70)
    print("\nSelect an example to run:")
    print("  1. Medical Knowledge Extraction")
    print("  2. General Knowledge Extraction (Companies & People)")
    print("  3. Custom Text (enter your own)")
    print("  4. Run All Examples")
    print("  0. Exit")
    print()

    try:
        choice = input("Enter choice (0-4): ").strip()

        if choice == "1":
            run_medical_example()
        elif choice == "2":
            run_general_example()
        elif choice == "3":
            run_custom_example()
        elif choice == "4":
            run_medical_example()
            run_general_example()
        elif choice == "0":
            print("Goodbye!")
            return
        else:
            print("Invalid choice.")
            return

        print("\n✅ Example completed successfully!")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
