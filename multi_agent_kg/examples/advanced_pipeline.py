"""
Advanced Pipeline Example - Open World Multi-Document Processing.

Demonstrates:
1. Open-world relation discovery (no predefined schemas)
2. Multi-document context maintenance
3. Agent collaboration
4. Cross-document entity resolution
"""

from multi_agent_kg.core.advanced_orchestrator import AdvancedOrchestrator
from multi_agent_kg.core.config import LLMConfig


# Sample documents for multi-document processing
DOCUMENTS = [
    {
        "id": "doc_apple_microsoft",
        "text": """
Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne
in 1976. The company is headquartered in Cupertino, California, and produces popular 
products including the iPhone, iPad, and Mac computers. Steve Jobs served as CEO until 
his death in 2011, when Tim Cook took over as the new CEO.

Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975 in Redmond, Washington.
Microsoft produces the Windows operating system, which competes with Apple's macOS. Bill Gates
stepped down as CEO in 2000, and Satya Nadella has been CEO since 2014. Microsoft also owns
LinkedIn and GitHub.
""",
        "metadata": {"domain": "technology", "type": "company_overview"}
    },
    {
        "id": "doc_tesla_spacex",
        "text": """
Elon Musk is the CEO of Tesla, an electric vehicle manufacturer located in Austin, Texas.
Tesla was originally founded by Martin Eberhard and Marc Tarpenning in 2003, with Musk
joining as chairman and lead investor. The company produces electric cars including the 
Model S, Model 3, Model X, and Model Y.

SpaceX, officially known as Space Exploration Technologies Corp., was founded by Elon Musk
in 2002. The company develops rockets including the Falcon 9 and Starship. SpaceX has
contracts with NASA for crew transportation to the International Space Station.
""",
        "metadata": {"domain": "technology", "type": "company_overview"}
    },
    {
        "id": "doc_tech_ceos",
        "text": """
The technology industry has been shaped by visionary leaders. Steve Jobs transformed Apple
into a consumer electronics powerhouse. Bill Gates made Microsoft the dominant software company.
Elon Musk disrupted both automotive and aerospace industries with Tesla and SpaceX.

Tim Cook, who succeeded Jobs at Apple, has focused on services and wearables. Satya Nadella
pivoted Microsoft toward cloud computing with Azure. Jeff Bezos built Amazon into an 
e-commerce and cloud computing giant before stepping down as CEO in 2021.
""",
        "metadata": {"domain": "technology", "type": "industry_analysis"}
    },
]

MEDICAL_DOCUMENTS = [
    {
        "id": "doc_aspirin",
        "text": """
Aspirin, also known as acetylsalicylic acid, was developed by Felix Hoffmann at Bayer
in 1897. It is commonly used to treat headaches, fever, and inflammation. Aspirin works
by inhibiting cyclooxygenase enzymes, which reduces prostaglandin synthesis.

Recent studies show that low-dose aspirin can prevent heart attacks and strokes in 
high-risk patients. However, aspirin may cause stomach irritation and increases the
risk of gastrointestinal bleeding, especially when combined with other blood thinners
like warfarin.
""",
        "metadata": {"domain": "medical", "type": "drug_information"}
    },
    {
        "id": "doc_diabetes",
        "text": """
Type 2 diabetes is a metabolic disorder characterized by high blood sugar levels.
Risk factors include obesity, lack of physical activity, and genetic predisposition.
The condition is associated with increased risk of heart disease, kidney disease,
and nerve damage.

Metformin is the first-line treatment for type 2 diabetes. It works by reducing
glucose production in the liver and improving insulin sensitivity. Other treatments
include sulfonylureas, GLP-1 agonists like Ozempic, and SGLT2 inhibitors.
""",
        "metadata": {"domain": "medical", "type": "disease_information"}
    },
]


def run_tech_corpus_example():
    """Process technology company documents."""
    print("\n" + "🏢" * 35)
    print("TECHNOLOGY CORPUS EXAMPLE")
    print("Open-World Multi-Document Processing")
    print("🏢" * 35)

    # Create orchestrator with open-world settings
    orchestrator = AdvancedOrchestrator(
        llm_config=LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.3,
        ),
        enable_summarization=False,
        enable_verification=True,
        enable_collaboration=True,
        confidence_threshold=0.5,
        refinement_rounds=1,
    )

    # Process entire corpus
    kg = orchestrator.process_corpus(
        documents=DOCUMENTS,
        resolve_cross_document_entities=True,
    )

    # Print final knowledge graph
    print("\n📊 FINAL KNOWLEDGE GRAPH:")
    kg.print_graph()

    # Print discovered ontology
    ontology = orchestrator.get_discovered_ontology()
    print("\n📚 DISCOVERED RELATION ONTOLOGY:")
    print("-" * 50)
    for name, info in sorted(ontology.items(), key=lambda x: x[1]["frequency"], reverse=True):
        print(f"  {name}:")
        print(f"    Definition: {info['definition']}")
        print(f"    Frequency: {info['frequency']}")
        print()

    # Export all data
    orchestrator.export_all("tech_corpus")

    return orchestrator


def run_medical_corpus_example():
    """Process medical documents."""
    print("\n" + "🏥" * 35)
    print("MEDICAL CORPUS EXAMPLE")
    print("Open-World Multi-Document Processing")
    print("🏥" * 35)

    orchestrator = AdvancedOrchestrator(
        llm_config=LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.2,
        ),
        enable_summarization=True,
        enable_verification=True,
        confidence_threshold=0.6,
    )

    kg = orchestrator.process_corpus(
        documents=MEDICAL_DOCUMENTS,
        resolve_cross_document_entities=True,
    )

    print("\n📊 FINAL KNOWLEDGE GRAPH:")
    kg.print_graph()

    ontology = orchestrator.get_discovered_ontology()
    print("\n📚 DISCOVERED RELATION ONTOLOGY:")
    print("-" * 50)
    for name, info in sorted(ontology.items(), key=lambda x: x[1]["frequency"], reverse=True):
        print(f"  {name}: {info['definition']}")
        print(f"    (frequency: {info['frequency']})")

    orchestrator.export_all("medical_corpus")

    return orchestrator


def run_single_document_example():
    """Process a single document in open-world mode."""
    print("\n" + "📄" * 35)
    print("SINGLE DOCUMENT EXAMPLE")
    print("Open-World Relation Discovery")
    print("📄" * 35)

    sample_text = """
    Artificial Intelligence has transformed many industries. OpenAI, founded by Sam Altman
    and others in 2015, created ChatGPT which revolutionized conversational AI. Google
    developed BERT and later Gemini to compete in the AI race. Meta AI released LLaMA
    as an open-source alternative.

    The AI industry faces regulatory challenges. The European Union passed the AI Act
    in 2024, which imposes restrictions on high-risk AI systems. In the US, President
    Biden signed an executive order on AI safety in 2023.
    """

    orchestrator = AdvancedOrchestrator(
        llm_config=LLMConfig(model="gpt-3.5-turbo", temperature=0.3),
        enable_verification=True,
        confidence_threshold=0.5,
    )

    kg = orchestrator.process_document(
        text=sample_text,
        document_id="ai_industry_doc",
        metadata={"domain": "AI", "type": "industry_overview"},
    )

    print("\n📊 KNOWLEDGE GRAPH:")
    kg.print_graph()

    return orchestrator


def run_interactive_example():
    """Interactive mode for processing custom documents."""
    print("\n" + "✨" * 35)
    print("INTERACTIVE MODE")
    print("Enter your own documents")
    print("✨" * 35)

    orchestrator = AdvancedOrchestrator(
        llm_config=LLMConfig(model="gpt-3.5-turbo", temperature=0.3),
        enable_verification=True,
        confidence_threshold=0.5,
    )

    documents = []
    doc_num = 1

    while True:
        print(f"\n--- Document {doc_num} ---")
        print("Enter text (empty line to finish document, 'done' to process all):")

        lines = []
        while True:
            try:
                line = input()
                if line.lower() == 'done':
                    break
                if line == '':
                    break
                lines.append(line)
            except EOFError:
                break

        if lines and lines[0].lower() != 'done':
            text = '\n'.join(lines)
            documents.append({
                "id": f"user_doc_{doc_num}",
                "text": text,
            })
            doc_num += 1
            print(f"✓ Added document ({len(text)} chars)")
            
            cont = input("\nAdd another document? (y/n): ").strip().lower()
            if cont != 'y':
                break
        else:
            break

    if documents:
        print(f"\n Processing {len(documents)} documents...")
        kg = orchestrator.process_corpus(documents)
        
        print("\n📊 KNOWLEDGE GRAPH:")
        kg.print_graph()
        
        orchestrator.export_all("interactive_output")
    else:
        print("No documents provided.")

    return orchestrator


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("🚀 ADVANCED MULTI-AGENT KNOWLEDGE GRAPH")
    print("   Open-World | Multi-Document | Agent Collaboration")
    print("=" * 70)
    print("\nSelect an example:")
    print("  1. Technology Companies Corpus (3 documents)")
    print("  2. Medical Knowledge Corpus (2 documents)")
    print("  3. Single Document (AI Industry)")
    print("  4. Interactive Mode (enter your own text)")
    print("  0. Exit")
    print()

    try:
        choice = input("Enter choice (0-4): ").strip()

        if choice == "1":
            run_tech_corpus_example()
        elif choice == "2":
            run_medical_corpus_example()
        elif choice == "3":
            run_single_document_example()
        elif choice == "4":
            run_interactive_example()
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
