"""
Multi-Agent Judge Panel — independent LLM judges score QA answers on separate dimensions.

Inspired by LLM-as-Judge research (2024-2025):
- Binary scoring > granular scales for reliability
- One criterion per judge reduces conflation
- Chain-of-thought reasoning improves alignment with human judgment
- Ensemble aggregation + meta-judge mitigates position/verbosity bias

Three independent judges:
  1. Correctness Judge — is the answer factually accurate given KG evidence?
  2. Completeness Judge — does the answer cover all relevant aspects from the KG?
  3. Groundedness Judge — is every claim in the answer traceable to KG triples?

Plus a Meta-Judge that aggregates individual verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from multi_agent_kg.llm.openai_client import chat_completion_json


@dataclass
class JudgeVerdict:
    """Verdict from a single judge on a single dimension."""
    dimension: str  # correctness | completeness | groundedness
    score: float  # 0.0-1.0
    binary_verdict: bool  # pass/fail
    reasoning: str = ""
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "binary_verdict": self.binary_verdict,
            "reasoning": self.reasoning,
            "issues": self.issues,
        }


@dataclass
class PanelVerdict:
    """Aggregated verdict from all judges."""
    individual_verdicts: List[JudgeVerdict] = field(default_factory=list)
    overall_score: float = 0.0
    overall_pass: bool = False
    meta_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "individual_verdicts": [v.to_dict() for v in self.individual_verdicts],
            "overall_score": self.overall_score,
            "overall_pass": self.overall_pass,
            "meta_reasoning": self.meta_reasoning,
        }


class JudgePanel:
    """
    Multi-agent judge panel with three independent evaluators and a meta-judge.
    Each judge scores a single dimension using chain-of-thought reasoning.
    """

    def __init__(self, model: str = "gemma3:27b"):
        self.model = model

    def evaluate(
        self,
        question: str,
        answer: str,
        kg_evidence: str,
        atomic_facts: Optional[List[Dict[str, Any]]] = None,
        verification_results: Optional[List[Dict[str, Any]]] = None,
    ) -> PanelVerdict:
        """
        Evaluate a QA answer using the full judge panel.

        Args:
            question: The original question
            answer: The generated answer
            kg_evidence: Relevant KG triples/paths as text
            atomic_facts: Decomposed atomic facts (from AtomicDecomposer)
            verification_results: Triple verification results (from TripleVerifier)

        Returns:
            PanelVerdict with individual and aggregated scores
        """
        verdicts = []

        # Run all three judges
        verdicts.append(self._judge_correctness(question, answer, kg_evidence))
        verdicts.append(self._judge_completeness(question, answer, kg_evidence))
        verdicts.append(
            self._judge_groundedness(
                question, answer, kg_evidence, atomic_facts, verification_results
            )
        )

        # Meta-judge aggregation
        panel = self._meta_judge(question, answer, verdicts)
        return panel

    def _judge_correctness(
        self, question: str, answer: str, kg_evidence: str
    ) -> JudgeVerdict:
        """Judge 1: Is the answer factually correct given the KG evidence?"""
        prompt = f"""You are a CORRECTNESS judge. Your ONLY job is to determine whether
the answer contains factual errors when compared against the knowledge graph evidence.

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

KNOWLEDGE GRAPH EVIDENCE (ground truth):
{kg_evidence}

EVALUATION CRITERIA:
- Does the answer make any claims that CONTRADICT the KG evidence?
- Does the answer state incorrect relationships between entities?
- Does the answer confuse entities or attribute properties to the wrong entity?
- Are numerical values, directions of relationships, or causal chains accurate?

Think step-by-step:
1. List each factual claim in the answer
2. Check each claim against the KG evidence
3. Identify any errors

Return JSON:
{{
    "reasoning": "Step-by-step analysis of each claim...",
    "errors_found": ["list of specific factual errors, if any"],
    "score": 0.0-1.0,
    "binary_verdict": true/false
}}

A score of 1.0 means no errors. 0.0 means completely wrong.
binary_verdict is true if score >= 0.7 (mostly correct).

Return ONLY the JSON."""

        result = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict factual correctness judge. "
                        "Only evaluate correctness, not completeness or style. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.1,
        )

        return JudgeVerdict(
            dimension="correctness",
            score=min(1.0, max(0.0, result.get("score", 0.5))),
            binary_verdict=result.get("binary_verdict", False),
            reasoning=result.get("reasoning", ""),
            issues=result.get("errors_found", []),
        )

    def _judge_completeness(
        self, question: str, answer: str, kg_evidence: str
    ) -> JudgeVerdict:
        """Judge 2: Does the answer cover all relevant aspects available in the KG?"""
        prompt = f"""You are a COMPLETENESS judge. Your ONLY job is to determine whether
the answer addresses all aspects of the question that the knowledge graph can answer.

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

KNOWLEDGE GRAPH EVIDENCE (all relevant facts):
{kg_evidence}

EVALUATION CRITERIA:
- What aspects of the question are answerable from the KG evidence?
- Which of those aspects does the answer address?
- Are there important KG facts that the answer SHOULD have included but didn't?
- Does the answer address the core of the question or only peripherally?

Think step-by-step:
1. List all aspects of the question
2. List all relevant facts from the KG evidence
3. Check which facts are reflected in the answer
4. Identify gaps

Return JSON:
{{
    "reasoning": "Step-by-step analysis of coverage...",
    "question_aspects": ["list each aspect of the question"],
    "covered_aspects": ["aspects that the answer addresses"],
    "missing_aspects": ["important aspects the answer missed"],
    "score": 0.0-1.0,
    "binary_verdict": true/false
}}

Score = covered_aspects / answerable_aspects.
binary_verdict is true if score >= 0.6 (covers most aspects).

Return ONLY the JSON."""

        result = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a completeness judge. Only evaluate whether "
                        "the answer is thorough, not whether it's correct. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.1,
        )

        return JudgeVerdict(
            dimension="completeness",
            score=min(1.0, max(0.0, result.get("score", 0.5))),
            binary_verdict=result.get("binary_verdict", False),
            reasoning=result.get("reasoning", ""),
            issues=result.get("missing_aspects", []),
        )

    def _judge_groundedness(
        self,
        question: str,
        answer: str,
        kg_evidence: str,
        atomic_facts: Optional[List[Dict[str, Any]]] = None,
        verification_results: Optional[List[Dict[str, Any]]] = None,
    ) -> JudgeVerdict:
        """Judge 3: Is every claim in the answer grounded in KG triples?"""

        # If we have atomic fact verification results, use them directly
        verification_context = ""
        if verification_results:
            supported = sum(1 for v in verification_results if v.get("verdict") == "supported")
            contradicted = sum(1 for v in verification_results if v.get("verdict") == "contradicted")
            unverifiable = sum(1 for v in verification_results if v.get("verdict") == "unverifiable")
            total = len(verification_results)

            verification_context = f"""
ATOMIC FACT VERIFICATION RESULTS (from three-tier KG verification):
- Total atomic facts: {total}
- Supported by KG: {supported}
- Contradicted by KG: {contradicted}
- Unverifiable from KG: {unverifiable}

Detailed results:
"""
            for v in verification_results[:20]:
                verification_context += (
                    f"  - [{v.get('verdict', '?')}] (Tier {v.get('tier', '?')}, "
                    f"conf={v.get('confidence', 0):.2f}): {v.get('fact_id', '?')}\n"
                )

        prompt = f"""You are a GROUNDEDNESS judge. Your ONLY job is to determine whether
every claim in the answer is grounded in (traceable to) the knowledge graph.

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

KNOWLEDGE GRAPH EVIDENCE:
{kg_evidence}
{verification_context}

EVALUATION CRITERIA:
- Does the answer ONLY make claims that are supported by the KG evidence?
- Does the answer fabricate facts (hallucinate) that aren't in the KG?
- Are there claims that go beyond what the KG evidence can support?
- Is speculation clearly marked as such?

Think step-by-step:
1. List each claim in the answer
2. For each claim, identify whether it's grounded in a KG triple
3. Flag any ungrounded claims (hallucinations)

Return JSON:
{{
    "reasoning": "Step-by-step groundedness analysis...",
    "grounded_claims": ["claims with KG support"],
    "ungrounded_claims": ["claims without KG support (hallucinations)"],
    "score": 0.0-1.0,
    "binary_verdict": true/false
}}

Score = grounded_claims / total_claims.
binary_verdict is true if score >= 0.8 (minimal hallucination).

Return ONLY the JSON."""

        result = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict groundedness judge. Identify any claims "
                        "not supported by the knowledge graph. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.1,
        )

        return JudgeVerdict(
            dimension="groundedness",
            score=min(1.0, max(0.0, result.get("score", 0.5))),
            binary_verdict=result.get("binary_verdict", False),
            reasoning=result.get("reasoning", ""),
            issues=result.get("ungrounded_claims", []),
        )

    def _meta_judge(
        self,
        question: str,
        answer: str,
        verdicts: List[JudgeVerdict],
    ) -> PanelVerdict:
        """
        Meta-judge: aggregate individual verdicts, resolving disagreements.
        Uses weighted average with groundedness weighted higher (hallucination is worst).
        """
        weights = {
            "correctness": 0.35,
            "completeness": 0.25,
            "groundedness": 0.40,
        }

        weighted_sum = 0.0
        weight_total = 0.0
        for v in verdicts:
            w = weights.get(v.dimension, 0.33)
            weighted_sum += v.score * w
            weight_total += w

        overall_score = weighted_sum / max(weight_total, 0.01)
        overall_pass = all(v.binary_verdict for v in verdicts) and overall_score >= 0.65

        # Build meta-reasoning from individual analyses
        meta_parts = []
        for v in verdicts:
            status = "PASS" if v.binary_verdict else "FAIL"
            meta_parts.append(
                f"{v.dimension.upper()}: {status} ({v.score:.2f})"
            )
            if v.issues:
                meta_parts.append(f"  Issues: {'; '.join(v.issues[:3])}")

        return PanelVerdict(
            individual_verdicts=verdicts,
            overall_score=round(overall_score, 3),
            overall_pass=overall_pass,
            meta_reasoning=" | ".join(meta_parts),
        )
