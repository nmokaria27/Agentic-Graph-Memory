"""
Auto-Benchmark Generator — creates QA evaluation benchmarks from the KG itself.

Novel contribution: instead of requiring human annotation, this generates
questions and gold answers directly from KG structure:

1. Single-hop questions: Direct triple-based Q&A
2. Multi-hop questions: Path-based reasoning Q&A
3. Aggregation questions: "How many X relate to Y?"
4. Comparison questions: "What's the difference between X and Y?"
5. Negative questions: "Does X relate to Z?" (when it doesn't)

Gold answers are DERIVED from the graph, so they're provably correct.
This enables fully automatic evaluation without human annotation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.domain_experts import find_paths
from multi_agent_kg.core.kg_operations import normalize_for_matching
from multi_agent_kg.llm.openai_client import chat_completion_json


@dataclass
class BenchmarkQuestion:
    """A single auto-generated benchmark question with gold answer."""

    question_id: str
    question: str
    gold_answer: str
    question_type: str  # single_hop | multi_hop | aggregation | comparison | negative
    difficulty: str  # easy | medium | hard
    supporting_triples: List[Dict[str, str]] = field(default_factory=list)
    supporting_paths: List[List[Dict[str, str]]] = field(default_factory=list)
    entities_involved: List[str] = field(default_factory=list)
    expected_domains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "supporting_triples": self.supporting_triples,
            "supporting_paths": self.supporting_paths,
            "entities_involved": self.entities_involved,
            "expected_domains": self.expected_domains,
        }


class BenchmarkGenerator:
    """
    Generates QA benchmarks automatically from KG structure.

    Uses KG triples, paths, and entity properties to create diverse
    question types with provably correct gold answers.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        model: str = "gemma3:27b",
        seed: int = 42,
    ):
        self.kg = kg
        self.model = model
        self.rng = random.Random(seed)

        # Pre-compute useful indexes
        self._subj_index: Dict[str, List[Triple]] = defaultdict(list)
        self._obj_index: Dict[str, List[Triple]] = defaultdict(list)
        self._rel_index: Dict[str, List[Triple]] = defaultdict(list)
        for t in kg.triples:
            self._subj_index[t.subject].append(t)
            self._obj_index[t.object].append(t)
            self._rel_index[t.relation].append(t)

    def generate(
        self,
        n_questions: int = 50,
        question_types: Optional[List[str]] = None,
    ) -> List[BenchmarkQuestion]:
        """
        Generate a balanced benchmark of n_questions questions.

        Args:
            n_questions: Total number of questions to generate
            question_types: Which types to include (default: all)

        Returns:
            List of BenchmarkQuestion objects
        """
        if question_types is None:
            question_types = [
                "single_hop", "multi_hop", "aggregation",
                "comparison", "negative",
            ]

        # Compute per-type counts
        per_type = max(1, n_questions // len(question_types))
        remainder = n_questions - per_type * len(question_types)

        questions: List[BenchmarkQuestion] = []
        qid_counter = 0

        for qtype in question_types:
            count = per_type + (1 if remainder > 0 else 0)
            remainder -= 1

            if qtype == "single_hop":
                batch = self._generate_single_hop(count, qid_counter)
            elif qtype == "multi_hop":
                batch = self._generate_multi_hop(count, qid_counter)
            elif qtype == "aggregation":
                batch = self._generate_aggregation(count, qid_counter)
            elif qtype == "comparison":
                batch = self._generate_comparison(count, qid_counter)
            elif qtype == "negative":
                batch = self._generate_negative(count, qid_counter)
            else:
                continue

            questions.extend(batch)
            qid_counter += len(batch)

        self.rng.shuffle(questions)
        return questions

    def _generate_single_hop(
        self, count: int, start_id: int
    ) -> List[BenchmarkQuestion]:
        """Generate single-hop questions from individual triples."""
        if not self.kg.triples:
            return []

        # Sample triples with decent confidence
        candidates = [
            t for t in self.kg.triples
            if t.confidence is None or t.confidence >= 0.5
        ]
        if not candidates:
            candidates = self.kg.triples

        sampled = self.rng.sample(candidates, min(count * 3, len(candidates)))
        questions = []

        for triple in sampled:
            if len(questions) >= count:
                break

            # Get human-readable names
            subj_name = self._entity_label(triple.subject)
            obj_name = self._entity_label(triple.object)
            rel_name = triple.relation.replace("_", " ")

            # Generate question using LLM for natural phrasing
            q_text = self._phrase_question(
                f"What {rel_name} {subj_name}?",
                subj_name, obj_name, rel_name,
            )
            gold = f"{subj_name} {rel_name} {obj_name}."

            questions.append(BenchmarkQuestion(
                question_id=f"sh_{start_id + len(questions):04d}",
                question=q_text,
                gold_answer=gold,
                question_type="single_hop",
                difficulty="easy",
                supporting_triples=[{
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                }],
                entities_involved=[triple.subject, triple.object],
            ))

        return questions

    def _generate_multi_hop(
        self, count: int, start_id: int
    ) -> List[BenchmarkQuestion]:
        """Generate multi-hop questions requiring path traversal."""
        questions = []

        # Find entity pairs connected by 2-3 hop paths
        entities = list(self.kg.entities.keys())
        attempts = 0
        max_attempts = count * 20

        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            if len(entities) < 2:
                break

            e1, e2 = self.rng.sample(entities, 2)
            paths = find_paths(self.kg, e1, e2, max_hops=3)

            # Only use paths with 2-3 hops (single hop = easy)
            multi_paths = [p for p in paths if 2 <= len(p) <= 3]
            if not multi_paths:
                continue

            path = multi_paths[0]  # Use shortest multi-hop path
            subj_name = self._entity_label(e1)
            obj_name = self._entity_label(e2)

            # Build the chain description
            chain = []
            for t in path:
                chain.append(
                    f"{self._entity_label(t.subject)} {t.relation.replace('_', ' ')} "
                    f"{self._entity_label(t.object)}"
                )

            # Build gold answer describing the path
            gold = f"{subj_name} is connected to {obj_name} through: {'; '.join(chain)}."

            q_text = self._phrase_question(
                f"How is {subj_name} related to {obj_name}?",
                subj_name, obj_name, "connection",
            )

            questions.append(BenchmarkQuestion(
                question_id=f"mh_{start_id + len(questions):04d}",
                question=q_text,
                gold_answer=gold,
                question_type="multi_hop",
                difficulty="medium" if len(path) == 2 else "hard",
                supporting_triples=[{
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object,
                } for t in path],
                supporting_paths=[[{
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object,
                } for t in path]],
                entities_involved=[e1, e2] + [
                    t.object for t in path[:-1]
                ],
            ))

        return questions

    def _generate_aggregation(
        self, count: int, start_id: int
    ) -> List[BenchmarkQuestion]:
        """Generate aggregation questions (counting, listing)."""
        questions = []

        # Find entities with multiple outgoing/incoming triples
        for entity_id in self.rng.sample(
            list(self.kg.entities.keys()),
            min(count * 3, len(self.kg.entities)),
        ):
            if len(questions) >= count:
                break

            out_triples = self._subj_index.get(entity_id, [])
            in_triples = self._obj_index.get(entity_id, [])

            if len(out_triples) >= 2:
                # Group by relation type
                by_rel: Dict[str, List[Triple]] = defaultdict(list)
                for t in out_triples:
                    by_rel[t.relation].append(t)

                for rel, triples in by_rel.items():
                    if len(triples) >= 2 and len(questions) < count:
                        name = self._entity_label(entity_id)
                        rel_name = rel.replace("_", " ")
                        objects = [self._entity_label(t.object) for t in triples]

                        q_text = f"What are all the things that {name} {rel_name}?"
                        gold = f"{name} {rel_name}: {', '.join(objects)}."

                        questions.append(BenchmarkQuestion(
                            question_id=f"ag_{start_id + len(questions):04d}",
                            question=q_text,
                            gold_answer=gold,
                            question_type="aggregation",
                            difficulty="medium",
                            supporting_triples=[{
                                "subject": t.subject,
                                "relation": t.relation,
                                "object": t.object,
                            } for t in triples],
                            entities_involved=[entity_id] + [t.object for t in triples],
                        ))
                        break

        return questions

    def _generate_comparison(
        self, count: int, start_id: int
    ) -> List[BenchmarkQuestion]:
        """Generate comparison questions between similar entities."""
        questions = []

        # Group entities by type
        by_type: Dict[str, List[str]] = defaultdict(list)
        for eid, entity in self.kg.entities.items():
            if entity.type:
                by_type[entity.type].append(eid)

        for etype, eids in by_type.items():
            if len(eids) < 2 or len(questions) >= count:
                continue

            pairs = []
            for i in range(len(eids)):
                for j in range(i + 1, len(eids)):
                    pairs.append((eids[i], eids[j]))
            if not pairs:
                continue

            pair = self.rng.choice(pairs)
            e1_name = self._entity_label(pair[0])
            e2_name = self._entity_label(pair[1])

            # Get triples for both entities
            e1_triples = self._subj_index.get(pair[0], []) + self._obj_index.get(pair[0], [])
            e2_triples = self._subj_index.get(pair[1], []) + self._obj_index.get(pair[1], [])

            if not e1_triples or not e2_triples:
                continue

            # Build comparison answer
            e1_facts = [
                f"{self._entity_label(t.subject)} {t.relation.replace('_', ' ')} {self._entity_label(t.object)}"
                for t in e1_triples[:3]
            ]
            e2_facts = [
                f"{self._entity_label(t.subject)} {t.relation.replace('_', ' ')} {self._entity_label(t.object)}"
                for t in e2_triples[:3]
            ]

            gold = (
                f"{e1_name}: {'; '.join(e1_facts)}. "
                f"{e2_name}: {'; '.join(e2_facts)}."
            )

            questions.append(BenchmarkQuestion(
                question_id=f"cmp_{start_id + len(questions):04d}",
                question=f"Compare {e1_name} and {e2_name}.",
                gold_answer=gold,
                question_type="comparison",
                difficulty="medium",
                supporting_triples=[{
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object,
                } for t in e1_triples[:3] + e2_triples[:3]],
                entities_involved=[pair[0], pair[1]],
            ))

        return questions

    def _generate_negative(
        self, count: int, start_id: int
    ) -> List[BenchmarkQuestion]:
        """Generate negative questions (answer should be 'no' or 'not found')."""
        questions = []
        entities = list(self.kg.entities.keys())

        attempts = 0
        while len(questions) < count and attempts < count * 20:
            attempts += 1
            if len(entities) < 2:
                break

            e1, e2 = self.rng.sample(entities, 2)

            # Check that there's NO direct triple between them.
            # Use normalised comparison since triples may store display
            # names while entity keys use snake_case IDs.
            e1_n = normalize_for_matching(e1)
            e2_n = normalize_for_matching(e2)
            has_direct = any(
                (normalize_for_matching(t.subject) == e1_n and normalize_for_matching(t.object) == e2_n) or
                (normalize_for_matching(t.subject) == e2_n and normalize_for_matching(t.object) == e1_n)
                for t in self.kg.triples
            )

            if has_direct:
                continue

            # Also check no short path exists
            paths = find_paths(self.kg, e1, e2, max_hops=2)
            if paths:
                continue

            e1_name = self._entity_label(e1)
            e2_name = self._entity_label(e2)

            # Pick a plausible-sounding relation
            if self.kg.triples:
                rel = self.rng.choice(self.kg.triples).relation.replace("_", " ")
            else:
                rel = "is related to"

            questions.append(BenchmarkQuestion(
                question_id=f"neg_{start_id + len(questions):04d}",
                question=f"Does {e1_name} {rel} {e2_name}?",
                gold_answer=f"No, the knowledge graph does not contain a relationship between {e1_name} and {e2_name}.",
                question_type="negative",
                difficulty="easy",
                supporting_triples=[],
                entities_involved=[e1, e2],
            ))

        return questions

    def _entity_label(self, entity_id: str) -> str:
        """Get the best human-readable label for an entity."""
        entity = self.kg.entities.get(entity_id)
        if entity and entity.labels:
            return entity.labels[0]
        return entity_id.replace("_", " ")

    def _phrase_question(
        self, template: str, subj: str, obj: str, rel: str,
    ) -> str:
        """
        Use LLM to rephrase a template question more naturally.
        Falls back to the template on failure.
        """
        prompt = f"""Rephrase this question to sound more natural, like a human would ask it.
Keep it concise (one sentence). Preserve the meaning exactly.

Original: {template}
Context: This is about the relationship between "{subj}" and "{obj}" ({rel}).

Return JSON:
{{"question": "Your rephrased question here"}}

Return ONLY the JSON."""

        try:
            result = chat_completion_json(
                messages=[
                    {"role": "system", "content": "Rephrase questions naturally. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=0.3,
            )
            return result.get("question", template)
        except Exception:
            return template
