"""
Atomic Fact Decomposer — breaks QA answers into independently verifiable atomic claims.

Based on the FActScore decomposition principle: each atomic fact should be a single,
self-contained claim that can be verified independently against the knowledge graph.

Key design choices:
- Context-independent: each fact includes enough context to stand alone
- Entity-grounded: every fact mentions at least one KG entity
- Relation-oriented: facts are structured to map onto KG triples
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from multi_agent_kg.llm.openai_client import chat_completion_json


@dataclass
class AtomicFact:
    """A single, independently verifiable claim extracted from a QA answer."""

    fact_id: str
    text: str  # The atomic claim in natural language
    source_sentence: str  # Original sentence this was extracted from
    entities_mentioned: List[str] = field(default_factory=list)
    relation_implied: Optional[str] = None  # e.g., "causes", "is_type_of"
    fact_type: str = "claim"  # claim | definition | comparison | negation | quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "text": self.text,
            "source_sentence": self.source_sentence,
            "entities_mentioned": self.entities_mentioned,
            "relation_implied": self.relation_implied,
            "fact_type": self.fact_type,
        }


DECOMPOSITION_PROMPT = """You are an atomic fact decomposition engine. Given a QA answer about
a knowledge graph, decompose it into the smallest possible independent facts.

RULES:
1. Each atomic fact must be a SINGLE claim that can be verified independently.
2. Each fact must be self-contained — include enough context (entity names, types)
   so the fact makes sense without the original answer.
3. Preserve ALL factual claims, even minor ones. Do not skip details.
4. Do NOT include opinions, hedging language, or meta-commentary
   (e.g., skip "According to the knowledge graph..." or "The expert reports...").
5. If the answer says "X is related to Y through Z", decompose into:
   - "X is related to Z"
   - "Z is related to Y"
6. For each fact, identify:
   - entities_mentioned: entity names that appear in the fact
   - relation_implied: the relationship being stated (e.g., "causes", "treats", "is_a")
   - fact_type: one of [claim, definition, comparison, negation, quantity]

ENTITY NAMES IN THE KNOWLEDGE GRAPH (for grounding):
{entity_names}

ANSWER TO DECOMPOSE:
{answer}

Return JSON:
{{
    "atomic_facts": [
        {{
            "text": "Metformin reduces HbA1c levels",
            "source_sentence": "Metformin is known to reduce HbA1c levels in type 2 diabetes patients.",
            "entities_mentioned": ["metformin", "hba1c"],
            "relation_implied": "reduces",
            "fact_type": "claim"
        }}
    ]
}}

Return ONLY the JSON. Be thorough — extract EVERY factual claim."""


class AtomicDecomposer:
    """Decomposes QA answers into atomic facts for verification."""

    def __init__(self, model: str = "gemma3:27b"):
        self.model = model

    def decompose(
        self,
        answer: str,
        entity_names: List[str],
    ) -> List[AtomicFact]:
        """
        Decompose a QA answer into atomic facts.

        Args:
            answer: The QA answer text to decompose
            entity_names: Known entity names from the KG (for grounding)

        Returns:
            List of AtomicFact objects
        """
        if not answer or not answer.strip():
            return []

        # Truncate entity list to avoid prompt overflow
        entity_sample = entity_names[:200]
        entity_str = ", ".join(entity_sample)
        if len(entity_names) > 200:
            entity_str += f"  ... and {len(entity_names) - 200} more"

        prompt = DECOMPOSITION_PROMPT.format(
            entity_names=entity_str,
            answer=answer,
        )

        result = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise fact decomposition system. "
                        "Extract every atomic claim from the given text. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.1,
        )

        facts = []
        for i, item in enumerate(result.get("atomic_facts", [])):
            fact = AtomicFact(
                fact_id=f"af_{i:03d}",
                text=item.get("text", ""),
                source_sentence=item.get("source_sentence", ""),
                entities_mentioned=item.get("entities_mentioned", []),
                relation_implied=item.get("relation_implied"),
                fact_type=item.get("fact_type", "claim"),
            )
            if fact.text.strip():
                facts.append(fact)

        return facts
