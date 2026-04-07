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
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple
from multi_agent_kg.core.domain_experts import OrgChart, find_paths
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
        org_chart: Optional[OrgChart] = None,
    ):
        self.kg = kg
        self.model = model
        self.rng = random.Random(seed)
        self.org_chart = org_chart
        self._entity_to_domain: Dict[str, str] = {}
        if org_chart:
            for domain in org_chart.domains:
                for entity_id in domain.entity_ids:
                    self._entity_to_domain[entity_id] = domain.domain_id

        # Pre-compute useful indexes
        self._subj_index: Dict[str, List[Triple]] = defaultdict(list)
        self._obj_index: Dict[str, List[Triple]] = defaultdict(list)
        self._rel_index: Dict[str, List[Triple]] = defaultdict(list)
        for t in kg.triples:
            self._subj_index[t.subject].append(t)
            self._obj_index[t.object].append(t)
            self._rel_index[t.relation].append(t)

    def _entity(self, entity_id: str):
        return self.kg.entities.get(entity_id)

    def _entity_type(self, entity_id: str) -> str:
        entity = self._entity(entity_id)
        return (entity.type or "") if entity else ""

    def _entity_domain(self, entity_id: str) -> str:
        return self._entity_to_domain.get(entity_id, "")

    def _is_good_entity(self, entity_id: str) -> bool:
        entity = self._entity(entity_id)
        label = self._entity_label(entity_id).strip()
        low = label.lower()
        etype = self._entity_type(entity_id).upper()

        bad_tokens = {
            "quartile", "group", "groups", "patient", "patients", "study",
            "analysis", "findings", "result", "results", "changes",
            "phenogroup",
        }
        bad_types = {
            "STUDY_DESIGN", "ANALYSIS_TECHNIQUE",
            "STATISTICAL_METHOD", "STATISTICAL_MODEL",
        }

        if not label or len(label) < 3:
            return False
        if any(token in low for token in bad_tokens):
            return False
        if re.fullmatch(r"[0-9.%+\- ]+", label):
            return False
        if etype in bad_types:
            return False
        return True

    def _is_good_triple(self, triple: Triple) -> bool:
        return (
            self._is_good_entity(triple.subject)
            and self._is_good_entity(triple.object)
            and len(triple.relation) >= 3
        )

    def _question_template(self, subject: str, relation: str, obj: str) -> str:
        rel = relation.replace("_", " ").lower()

        if rel.startswith("associated with"):
            return f"Is {subject} associated with {obj}?"
        if "mediated by" in rel or rel.startswith("mediates"):
            return f"How is the relationship involving {subject} mediated by {obj}?"
        if rel.startswith("measured by") or "used to measure" in rel:
            return f"How is {subject} measured?"
        if rel.startswith("part of"):
            return f"What system or process is {subject} part of?"
        if rel.startswith("affects"):
            return f"How does {subject} affect {obj}?"
        if rel.startswith("manifests as"):
            return f"How does {subject} manifest clinically?"
        return f"What is the relationship between {subject} and {obj}?"

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
                "comparison", "negative", "cross_domain",
            ]

        if n_questions <= 0 or not question_types:
            return []

        # Distribute the requested budget across types without exceeding n_questions.
        shuffled_types = list(question_types)
        self.rng.shuffle(shuffled_types)
        per_type = n_questions // len(shuffled_types)
        remainder = n_questions % len(shuffled_types)

        questions: List[BenchmarkQuestion] = []
        qid_counter = 0

        for idx, qtype in enumerate(shuffled_types):
            count = per_type + (1 if idx < remainder else 0)
            if count == 0:
                continue

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
            elif qtype == "cross_domain":
                batch = self._generate_cross_domain(count, qid_counter)
            else:
                continue

            questions.extend(batch)
            qid_counter += len(batch)

        self.rng.shuffle(questions)
        return questions[:n_questions]

    def _generate_single_hop(
        self, count: int, start_id: int
    ) -> List[BenchmarkQuestion]:
        """Generate single-hop questions from individual triples."""
        if not self.kg.triples:
            return []

        # Sample triples with decent confidence
        candidates = [
            t for t in self.kg.triples
            if (t.confidence is None or t.confidence >= 0.5)
            and self._is_good_triple(t)
        ]
        if not candidates:
            candidates = [t for t in self.kg.triples if self._is_good_triple(t)]
        if not candidates:
            return []

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
                self._question_template(subj_name, triple.relation, obj_name),
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
        entities = [eid for eid in self.kg.entities if self._is_good_entity(eid)]
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

            if self.org_chart:
                cross_domain_paths = []
                for path in multi_paths:
                    domains = {
                        self._entity_domain(t.subject)
                        for t in path
                        if self._entity_domain(t.subject)
                    } | {
                        self._entity_domain(t.object)
                        for t in path
                        if self._entity_domain(t.object)
                    }
                    if len(domains) >= 2:
                        cross_domain_paths.append(path)
                if cross_domain_paths:
                    multi_paths = cross_domain_paths

            path = multi_paths[0]  # Use shortest multi-hop path
            if not all(self._is_good_triple(t) for t in path):
                continue
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
                expected_domains=sorted({
                    self._entity_domain(eid)
                    for eid in [e1, e2] + [t.object for t in path[:-1]]
                    if self._entity_domain(eid)
                }),
            ))

        return questions

    def _generate_aggregation(
        self, count: int, start_id: int
    ) -> List[BenchmarkQuestion]:
        """Generate aggregation questions (counting, listing)."""
        questions = []

        # Find entities with multiple outgoing/incoming triples
        for entity_id in self.rng.sample(
            [eid for eid in self.kg.entities if self._is_good_entity(eid)],
            min(count * 3, len([eid for eid in self.kg.entities if self._is_good_entity(eid)])),
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
                    good_triples = [t for t in triples if self._is_good_triple(t)]
                    if len(good_triples) >= 2 and len(questions) < count:
                        name = self._entity_label(entity_id)
                        rel_name = rel.replace("_", " ")
                        objects = [self._entity_label(t.object) for t in good_triples]

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
                            } for t in good_triples],
                            entities_involved=[entity_id] + [t.object for t in good_triples],
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
            if entity.type and self._is_good_entity(eid):
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
            e1_triples = [t for t in e1_triples if self._is_good_triple(t)]
            e2_triples = [t for t in e2_triples if self._is_good_triple(t)]

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
        entities = [eid for eid in self.kg.entities if self._is_good_entity(eid)]

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
                good_triples = [t for t in self.kg.triples if self._is_good_triple(t)]
                if not good_triples:
                    break
                rel = self.rng.choice(good_triples).relation.replace("_", " ")
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

    def _generate_cross_domain(
        self, count: int, start_id: int
    ) -> List[BenchmarkQuestion]:
        """Generate harder cross-domain questions that require bridging domains."""
        if not self.org_chart:
            return self._generate_multi_hop(count, start_id)

        questions = []
        multi_hop_candidates: List[List[Triple]] = []
        direct_candidates: List[List[Triple]] = []
        seen_pairs = set()
        cross_relations = list(self.org_chart.cross_domain_relations)
        self.rng.shuffle(cross_relations)
        max_seed_relations = min(max(count * 20, 25), len(cross_relations))

        for triple in cross_relations[:max_seed_relations]:
            subj_domain = self._entity_domain(triple.subject)
            obj_domain = self._entity_domain(triple.object)
            if not subj_domain or not obj_domain or subj_domain == obj_domain:
                continue
            if not self._is_good_triple(triple):
                continue
            pair_key = tuple(sorted((triple.subject, triple.object)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            direct_candidates.append([triple])

            paths = find_paths(self.kg, triple.subject, triple.object, max_hops=3)
            added_for_pair = 0
            for path in paths:
                if not (2 <= len(path) <= 3):
                    continue
                if all(self._is_good_triple(step) for step in path):
                    path_domains = {
                        self._entity_domain(step.subject)
                        for step in path
                        if self._entity_domain(step.subject)
                    } | {
                        self._entity_domain(step.object)
                        for step in path
                        if self._entity_domain(step.object)
                    }
                    if len(path_domains) >= 2:
                        multi_hop_candidates.append(path)
                        added_for_pair += 1
                if added_for_pair >= 2:
                    break

        if len(multi_hop_candidates) < count and self.org_chart:
            domain_entities = [
                (entity_id, self._entity_domain(entity_id))
                for entity_id in self.kg.entities
                if self._is_good_entity(entity_id) and self._entity_domain(entity_id)
            ]
            self.rng.shuffle(domain_entities)
            pair_attempts = 0
            max_pair_attempts = min(len(domain_entities) * 4, count * 50)
            while len(multi_hop_candidates) < count * 4 and pair_attempts < max_pair_attempts:
                pair_attempts += 1
                if len(domain_entities) < 2:
                    break
                (e1, d1), (e2, d2) = self.rng.sample(domain_entities, 2)
                if d1 == d2:
                    continue
                pair_key = tuple(sorted((e1, e2)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                paths = find_paths(self.kg, e1, e2, max_hops=3)
                for path in paths:
                    if not (2 <= len(path) <= 3):
                        continue
                    if not all(self._is_good_triple(step) for step in path):
                        continue
                    path_domains = {
                        self._entity_domain(step.subject)
                        for step in path
                        if self._entity_domain(step.subject)
                    } | {
                        self._entity_domain(step.object)
                        for step in path
                        if self._entity_domain(step.object)
                    }
                    if len(path_domains) >= 2:
                        multi_hop_candidates.append(path)
                        break

        candidate_paths = multi_hop_candidates
        if len(candidate_paths) < count:
            candidate_paths = candidate_paths + direct_candidates

        self.rng.shuffle(candidate_paths)
        for path in candidate_paths:
            if len(questions) >= count:
                break

            e1 = path[0].subject
            e2 = path[-1].object
            subj_name = self._entity_label(e1)
            obj_name = self._entity_label(e2)
            path_domains = sorted({
                self._entity_domain(t.subject)
                for t in path
                if self._entity_domain(t.subject)
            } | {
                self._entity_domain(t.object)
                for t in path
                if self._entity_domain(t.object)
            })
            if len(path_domains) < 2:
                continue

            chain = []
            for t in path:
                chain.append(
                    f"{self._entity_label(t.subject)} {t.relation.replace('_', ' ')} "
                    f"{self._entity_label(t.object)}"
                )

            q_text = self._phrase_question(
                f"Explain how {subj_name} connects to {obj_name} across different research areas.",
                subj_name,
                obj_name,
                "cross-domain connection",
            )
            gold = f"{subj_name} connects to {obj_name} through: {'; '.join(chain)}."

            questions.append(BenchmarkQuestion(
                question_id=f"xd_{start_id + len(questions):04d}",
                question=q_text,
                gold_answer=gold,
                question_type="cross_domain",
                difficulty="hard" if len(path) >= 2 else "medium",
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
                entities_involved=[e1, e2] + [t.object for t in path[:-1]],
                expected_domains=path_domains,
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
