"""
Example: Using Adaptive Configuration Features

Demonstrates:
1. Custom domain schemas
2. Dynamic batch sizing
3. Threshold auto-tuning
"""

from multi_agent_kg.core import (
    DomainTaxonomy,
    DomainSchema,
    AdaptiveBatchCalculator,
    ThresholdAutoTuner,
    DeliberativeOrchestrator,
    LLMConfig,
)


def example_custom_domain():
    """Example: Add a custom domain schema."""
    
    # Initialize taxonomy (loads base domains)
    taxonomy = DomainTaxonomy()
    
    # Define custom domain
    custom_schema = DomainSchema(
        name="Legal Documents",
        description="Legal contracts, court cases, and regulations",
        entity_types=[
            {"type": "PARTY", "description": "Legal entity or individual"},
            {"type": "STATUTE", "description": "Law or regulation"},
            {"type": "CASE", "description": "Court case or precedent"},
            {"type": "CONTRACT", "description": "Legal agreement"},
        ],
        relation_types=[
            {"type": "CITED_BY", "source_types": ["CASE"], "target_types": ["STATUTE"]},
            {"type": "PARTY_TO", "source_types": ["PARTY"], "target_types": ["CONTRACT", "CASE"]},
        ],
        quality_threshold=0.75,  # Higher threshold for legal domain
        keywords=["court", "statute", "contract", "plaintiff", "defendant"],
    )
    
    # Register custom domain
    taxonomy.register_domain(custom_schema)
    
    # Use in orchestrator
    orchestrator = DeliberativeOrchestrator(
        llm_config=LLMConfig(),
        quality_threshold=0.6,  # Default threshold
    )
    
    # Override threshold for specific domain
    domain = taxonomy.get_domain("Legal Documents")
    if domain and domain.quality_threshold:
        orchestrator.quality_threshold = domain.quality_threshold
    
    print(f"Using threshold {orchestrator.quality_threshold} for Legal Documents")


def example_adaptive_batching():
    """Example: Calculate optimal batch sizes for different models."""
    
    calculator = AdaptiveBatchCalculator()
    
    # Entity batching for different models
    entity_avg_size = 150  # chars per entity JSON
    
    gpt35_batch = calculator.calculate_batch_size(
        "gpt-3.5-turbo",
        item_avg_size=entity_avg_size,
        prompt_overhead=500,
    )
    
    gpt4o_batch = calculator.calculate_batch_size(
        "gpt-4o-mini",
        item_avg_size=entity_avg_size,
        prompt_overhead=500,
    )
    
    print(f"Optimal batch sizes:")
    print(f"  gpt-3.5-turbo: {gpt35_batch} entities")
    print(f"  gpt-4o-mini: {gpt4o_batch} entities")
    
    # Relation batching (larger items)
    relation_avg_size = 300
    
    rel_batch = calculator.calculate_batch_size(
        "gpt-4o-mini",
        item_avg_size=relation_avg_size,
        prompt_overhead=800,
    )
    
    print(f"  Relations: {rel_batch} head bindings")


def example_threshold_tuning():
    """Example: Auto-tune thresholds based on validation set."""
    
    # Mock validation data
    predictions = [
        {"text": "COVID-19", "type": "DISEASE", "confidence": 0.95},
        {"text": "vaccine", "type": "INTERVENTION", "confidence": 0.85},
        {"text": "patients", "type": "POPULATION", "confidence": 0.65},
        {"text": "maybe-entity", "type": "UNKNOWN", "confidence": 0.45},
    ]
    
    ground_truth = [
        {"text": "COVID-19", "type": "DISEASE"},
        {"text": "vaccine", "type": "INTERVENTION"},
        {"text": "patients", "type": "POPULATION"},
    ]
    
    tuner = ThresholdAutoTuner()
    
    # Find optimal threshold for F1
    optimal_threshold = tuner.find_optimal_threshold(
        predictions,
        ground_truth,
        objective="f1",
    )
    
    print(f"Optimal threshold for F1: {optimal_threshold:.2f}")
    
    # Evaluate at different thresholds
    for threshold in [0.4, 0.6, 0.8]:
        p, r, f1 = tuner.evaluate_threshold(predictions, ground_truth, threshold)
        print(f"  Threshold {threshold:.1f}: P={p:.2f}, R={r:.2f}, F1={f1:.2f}")


def example_save_load_domains():
    """Example: Save and load custom domains."""
    
    taxonomy = DomainTaxonomy()
    
    # Add custom domain
    taxonomy.register_domain(DomainSchema(
        name="Climate Science",
        description="Climate change research and environmental science",
        entity_types=[
            {"type": "CLIMATE_PHENOMENON", "description": "Weather or climate events"},
            {"type": "GREENHOUSE_GAS", "description": "Gases contributing to warming"},
            {"type": "REGION", "description": "Geographic area"},
        ],
        relation_types=[
            {"type": "CAUSES", "source_types": ["GREENHOUSE_GAS"], "target_types": ["CLIMATE_PHENOMENON"]},
            {"type": "AFFECTS", "source_types": ["CLIMATE_PHENOMENON"], "target_types": ["REGION"]},
        ],
        keywords=["climate", "warming", "emissions", "temperature"],
    ))
    
    # Export to file
    taxonomy.export_domains("custom_domains.json")
    print("Exported domains to custom_domains.json")
    
    # Load from file
    new_taxonomy = DomainTaxonomy(custom_domains_path="custom_domains.json")
    climate_domain = new_taxonomy.get_domain("Climate Science")
    print(f"Loaded domain: {climate_domain.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    print("\n1. Custom Domain Schema")
    print("-" * 60)
    example_custom_domain()
    
    print("\n2. Adaptive Batch Sizing")
    print("-" * 60)
    example_adaptive_batching()
    
    print("\n3. Threshold Auto-Tuning")
    print("-" * 60)
    example_threshold_tuning()
    
    print("\n4. Save/Load Domains")
    print("-" * 60)
    example_save_load_domains()
