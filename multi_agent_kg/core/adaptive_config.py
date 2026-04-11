"""
Adaptive Configuration System.

Provides:
- Dynamic batch size calculation based on model context windows
- Extensible domain taxonomy
- Custom entity schemas per domain
- Auto-tuning of quality thresholds based on validation
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ModelSpec:
    """Specification for an LLM model."""
    name: str
    max_context_tokens: int
    max_output_tokens: int
    avg_chars_per_token: float = 4.0  # GPT models average


# Model specifications
MODEL_SPECS = {
    # Ollama models (default)
    "qwen3:4b": ModelSpec("qwen3:4b", 32768, 8192),
    "qwen3:8b": ModelSpec("qwen3:8b", 32768, 8192),
    "gemma4:31b": ModelSpec("gemma4:31b", 32768, 8192),
    "gemma3:27b": ModelSpec("gemma3:27b", 32768, 8192),
    "mistral:latest": ModelSpec("mistral:latest", 32768, 8192),
    "mistral-small3.1:latest": ModelSpec("mistral-small3.1:latest", 32768, 8192),
    "deepseek-r1:14b": ModelSpec("deepseek-r1:14b", 32768, 8192),
    # OpenAI models (for LLM_BACKEND=openai)
    "gpt-3.5-turbo": ModelSpec("gpt-3.5-turbo", 16385, 4096),
    "gpt-4o-mini": ModelSpec("gpt-4o-mini", 128000, 16384),
    "gpt-4o": ModelSpec("gpt-4o", 128000, 16384),
    "gpt-4-turbo": ModelSpec("gpt-4-turbo", 128000, 4096),
}


@dataclass
class DomainSchema:
    """Custom schema for a domain."""
    name: str
    description: str
    entity_types: List[Dict[str, Any]]
    relation_types: List[Dict[str, Any]]
    quality_threshold: Optional[float] = None
    extraction_examples: List[Dict[str, Any]] = field(default_factory=list)
    parent_domain: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


class DomainTaxonomy:
    """Extensible domain taxonomy with custom schemas."""
    
    def __init__(self, custom_domains_path: Optional[str] = None):
        """Initialize taxonomy with optional custom domains."""
        self.domains: Dict[str, DomainSchema] = {}
        self._load_base_domains()
        
        if custom_domains_path:
            self.load_custom_domains(custom_domains_path)
    
    def _load_base_domains(self):
        """Load base domain schemas."""
        # NO BUILT-IN DOMAINS - fully agent-driven discovery
        # Domains, entity types, and relation types are discovered dynamically
        # from the actual document content by the agents
        pass
    
    def register_domain(self, schema: DomainSchema):
        """Register a custom domain schema."""
        self.domains[schema.name] = schema
    
    def get_domain(self, name: str) -> Optional[DomainSchema]:
        """Get domain schema by name."""
        return self.domains.get(name, self.domains.get("General"))
    
    def classify_domain(self, text: str, existing_classification: Optional[str] = None) -> DomainSchema:
        """Classify text to best matching domain based on keywords."""
        if existing_classification and existing_classification in self.domains:
            return self.domains[existing_classification]
        
        # Simple keyword-based matching
        text_lower = text.lower()
        best_match = "General"
        best_score = 0
        
        for domain_name, schema in self.domains.items():
            if domain_name == "General":
                continue
            score = sum(1 for keyword in schema.keywords if keyword in text_lower)
            if score > best_score:
                best_score = score
                best_match = domain_name
        
        return self.domains[best_match]
    
    def load_custom_domains(self, path: str):
        """Load custom domains from JSON file."""
        with open(path, 'r') as f:
            custom_domains = json.load(f)
        
        for domain_data in custom_domains:
            schema = DomainSchema(**domain_data)
            self.register_domain(schema)
    
    def export_domains(self, path: str):
        """Export all domains to JSON file."""
        domains_list = []
        for schema in self.domains.values():
            domains_list.append({
                "name": schema.name,
                "description": schema.description,
                "entity_types": schema.entity_types,
                "relation_types": schema.relation_types,
                "quality_threshold": schema.quality_threshold,
                "extraction_examples": schema.extraction_examples,
                "parent_domain": schema.parent_domain,
                "keywords": schema.keywords,
            })
        
        with open(path, 'w') as f:
            json.dump(domains_list, f, indent=2)


class AdaptiveBatchCalculator:
    """Calculate optimal batch sizes based on model context windows."""
    
    @staticmethod
    def calculate_batch_size(
        model_name: str,
        item_avg_size: int,
        prompt_overhead: int = 500,
        safety_margin: float = 0.7,
    ) -> int:
        """
        Calculate optimal batch size for a given model and item size.
        
        Args:
            model_name: Name of the LLM model
            item_avg_size: Average size of each item in chars
            prompt_overhead: Fixed prompt text size
            safety_margin: Use only X% of max tokens (default 0.7)
        
        Returns:
            Optimal batch size
        """
        spec = MODEL_SPECS.get(model_name, MODEL_SPECS["mistral:latest"])
        
        # Calculate available tokens for content
        available_output_tokens = int(spec.max_output_tokens * safety_margin)
        
        # Convert to chars
        available_chars = available_output_tokens * spec.avg_chars_per_token
        
        # Subtract prompt overhead
        available_for_items = available_chars - prompt_overhead
        
        # Calculate how many items fit
        batch_size = max(1, int(available_for_items / item_avg_size))
        
        return min(batch_size, 50)  # Cap at 50 for practical reasons
    
    @staticmethod
    def estimate_item_size(items: List[Dict[str, Any]]) -> int:
        """Estimate average item size from sample."""
        if not items:
            return 100  # Default estimate
        
        sample_size = min(10, len(items))
        sample_items = items[:sample_size]
        
        total_size = sum(len(json.dumps(item)) for item in sample_items)
        return total_size // sample_size


class ThresholdAutoTuner:
    """Auto-tune quality thresholds based on validation set performance."""
    
    def __init__(self):
        self.validation_results: List[Tuple[float, float, float]] = []
        # (threshold, precision, recall)
    
    def evaluate_threshold(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        threshold: float,
    ) -> Tuple[float, float, float]:
        """
        Evaluate P/R/F1 at a given threshold.
        
        Args:
            predictions: Extracted items with confidence scores
            ground_truth: Gold standard items
            threshold: Confidence threshold to test
        
        Returns:
            (precision, recall, f1)
        """
        # Filter predictions by threshold
        filtered = [p for p in predictions if p.get("confidence", 0) >= threshold]
        
        # Simple exact match for demo (replace with proper matching)
        pred_set = {self._item_key(p) for p in filtered}
        gold_set = {self._item_key(g) for g in ground_truth}
        
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def find_optimal_threshold(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        objective: str = "f1",  # "precision", "recall", or "f1"
    ) -> float:
        """
        Find threshold that optimizes objective metric.
        
        Args:
            predictions: Extracted items with confidence scores
            ground_truth: Gold standard items
            objective: Metric to optimize
        
        Returns:
            Optimal threshold
        """
        # Test thresholds from 0.3 to 0.95 in steps of 0.05
        thresholds = [i / 20 for i in range(6, 20)]  # 0.3 to 0.95
        
        best_threshold = 0.6
        best_score = 0
        
        for threshold in thresholds:
            precision, recall, f1 = self.evaluate_threshold(
                predictions, ground_truth, threshold
            )
            
            score = {"precision": precision, "recall": recall, "f1": f1}[objective]
            
            self.validation_results.append((threshold, precision, recall))
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    @staticmethod
    def _item_key(item: Dict[str, Any]) -> str:
        """Generate key for item matching."""
        if "subject" in item:  # Triple
            return f"{item.get('subject')}|{item.get('relation')}|{item.get('object')}"
        elif "text" in item:  # Entity
            return f"{item.get('text')}|{item.get('type', '')}"
        else:
            return json.dumps(item, sort_keys=True)
