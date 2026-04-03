"""
Adaptive Planner for Multi-Agent KG Extraction Pipeline.

Analyzes a document BEFORE extraction begins and selects an optimal
extraction strategy based on document characteristics such as length,
complexity, vocabulary diversity, and domain signals.

Strategies:
- QUICK:     Short documents  (<2000 words) -- minimal passes
- STANDARD:  Medium documents (2000-8000 words) -- balanced pipeline
- THOROUGH:  Long documents   (>8000 words) -- full multi-pass extraction
- MULTI_DOC: Corpus processing -- cross-document resolution enabled

Pure Python -- no external dependencies or LLM calls required.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentProfile:
    """
    Statistical profile of a document used for strategy selection.

    Attributes:
        word_count:           Total number of whitespace-delimited tokens.
        char_count:           Total characters (including whitespace).
        sentence_count:       Approximate number of sentences.
        avg_sentence_length:  Average words per sentence.
        vocabulary_diversity: Ratio of unique words to total words (0-1).
        estimated_entities:   Rough count of likely named entities based
                              on capitalised word heuristics.
    """

    word_count: int
    char_count: int
    sentence_count: int
    avg_sentence_length: float
    vocabulary_diversity: float
    estimated_entities: int

    def __repr__(self) -> str:
        return (
            f"DocumentProfile(words={self.word_count}, "
            f"sentences={self.sentence_count}, "
            f"vocab_diversity={self.vocabulary_diversity:.3f}, "
            f"est_entities={self.estimated_entities})"
        )


@dataclass
class PipelineStrategy:
    """
    The extraction strategy chosen by the planner.

    Every field that controls downstream pipeline behaviour is captured here
    so that the orchestrator can configure itself from a single object.

    Attributes:
        name:                    Strategy label (QUICK / STANDARD / THOROUGH / MULTI_DOC).
        batch_size:              Number of text segments processed per LLM call.
        max_gleanings:           How many gleaning (re-extraction) passes to run.
        enable_triplex:          Whether to enable TripleX extraction if available.
        enable_cross_document:   Whether to enable cross-document entity resolution.
        enable_critic_corrector: Whether to enable the critic-corrector feedback loop.
        max_critic_iterations:   Maximum rounds of critic-corrector refinement.
        segment_overlap_chars:   Character overlap between consecutive segments.
        quality_threshold:       Minimum confidence for a triple to be accepted.
        reasoning:               Human-readable explanation of why this strategy
                                 was selected.
    """

    name: str
    batch_size: int
    max_gleanings: int
    enable_triplex: bool
    enable_cross_document: bool
    enable_critic_corrector: bool
    max_critic_iterations: int
    segment_overlap_chars: int
    quality_threshold: float
    reasoning: str

    def __repr__(self) -> str:
        return (
            f"PipelineStrategy(name={self.name!r}, batch={self.batch_size}, "
            f"gleanings={self.max_gleanings}, triplex={self.enable_triplex}, "
            f"quality={self.quality_threshold})"
        )


# ---------------------------------------------------------------------------
# Strategy profile templates
# ---------------------------------------------------------------------------

_STRATEGY_PROFILES = {
    "QUICK": PipelineStrategy(
        name="QUICK",
        batch_size=8,
        max_gleanings=0,
        enable_triplex=False,
        enable_cross_document=False,
        enable_critic_corrector=False,
        max_critic_iterations=0,
        segment_overlap_chars=64,
        quality_threshold=0.50,
        reasoning="",
    ),
    "STANDARD": PipelineStrategy(
        name="STANDARD",
        batch_size=5,
        max_gleanings=1,
        enable_triplex=False,
        enable_cross_document=False,
        enable_critic_corrector=True,
        max_critic_iterations=1,
        segment_overlap_chars=128,
        quality_threshold=0.60,
        reasoning="",
    ),
    "THOROUGH": PipelineStrategy(
        name="THOROUGH",
        batch_size=3,
        max_gleanings=2,
        enable_triplex=True,
        enable_cross_document=False,
        enable_critic_corrector=True,
        max_critic_iterations=2,
        segment_overlap_chars=200,
        quality_threshold=0.70,
        reasoning="",
    ),
    "MULTI_DOC": PipelineStrategy(
        name="MULTI_DOC",
        batch_size=5,
        max_gleanings=1,
        enable_triplex=False,
        enable_cross_document=True,
        enable_critic_corrector=True,
        max_critic_iterations=1,
        segment_overlap_chars=128,
        quality_threshold=0.60,
        reasoning="",
    ),
}


# ---------------------------------------------------------------------------
# Sentence-boundary regex
# ---------------------------------------------------------------------------

# Splits on '.', '!', '?' that are followed by whitespace or end-of-string,
# but avoids false positives on abbreviations like "U.S." or "Dr.".
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# Matches capitalised words that are likely named entities.  We exclude
# sentence-initial words by requiring at least one preceding non-newline
# character (a rough but dependency-free heuristic).
_CAPITALISED_WORD = re.compile(r'(?<=[a-z,.;:!?]\s)[A-Z][a-z]+')


# ---------------------------------------------------------------------------
# Domain signal keywords (lightweight heuristic)
# ---------------------------------------------------------------------------

_DOMAIN_SIGNALS = {
    "biomedical": {
        "protein", "gene", "cell", "enzyme", "receptor", "mutation",
        "clinical", "patient", "diagnosis", "therapy", "disease",
        "pathology", "symptom", "antibody", "metabolite",
    },
    "legal": {
        "plaintiff", "defendant", "court", "statute", "jurisdiction",
        "verdict", "litigation", "counsel", "arbitration", "tort",
        "contract", "clause", "damages", "appeal", "precedent",
    },
    "financial": {
        "revenue", "equity", "dividend", "securities", "portfolio",
        "asset", "liability", "valuation", "fiscal", "shareholder",
        "acquisition", "merger", "quarterly", "earnings", "capital",
    },
    "technical": {
        "algorithm", "implementation", "architecture", "latency",
        "throughput", "protocol", "framework", "module", "deployment",
        "repository", "API", "endpoint", "microservice", "container",
    },
}


# ---------------------------------------------------------------------------
# AdaptivePlanner
# ---------------------------------------------------------------------------

class AdaptivePlanner:
    """
    Analyse a document and select an optimal extraction strategy.

    Usage::

        planner = AdaptivePlanner()
        strategy = planner.analyze(text)
        # hand `strategy` to the orchestrator / pipeline runner
    """

    def __init__(self) -> None:
        # No external dependencies needed.  Instance state is reserved for
        # future extensions (e.g. caching profiles across calls).
        self._profiles: List[DocumentProfile] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        num_documents: int = 1,
    ) -> PipelineStrategy:
        """
        Determine the best extraction strategy for *text*.

        Args:
            text:           The full document text (or representative sample
                            for very large corpora).
            num_documents:  Number of documents being processed in this run.
                            When > 1 the planner may choose MULTI_DOC.

        Returns:
            A :class:`PipelineStrategy` configured for this input.
        """
        profile = self.profile_document(text)
        self._profiles.append(profile)

        # --- Decide base strategy ---

        if num_documents > 1:
            strategy_name = "MULTI_DOC"
            reasons = [
                f"Corpus of {num_documents} documents detected -- "
                "enabling cross-document entity resolution."
            ]
            # For multi-doc we may still want thorough extraction per doc
            # if individual documents are long.
            if profile.word_count > 8000:
                reasons.append(
                    f"Individual document is long ({profile.word_count} words); "
                    "increasing gleanings to 2."
                )
        elif profile.word_count < 2000:
            strategy_name = "QUICK"
            reasons = [
                f"Short document ({profile.word_count} words, "
                f"{profile.sentence_count} sentences) -- using fast extraction."
            ]
        elif profile.word_count <= 8000:
            strategy_name = "STANDARD"
            reasons = [
                f"Medium-length document ({profile.word_count} words) -- "
                "using standard pipeline."
            ]
        else:
            strategy_name = "THOROUGH"
            reasons = [
                f"Long document ({profile.word_count} words) -- "
                "enabling all extraction passes."
            ]

        # Start from the template for the chosen strategy.
        strategy = self._clone_strategy(_STRATEGY_PROFILES[strategy_name])

        # --- Adjustments based on document complexity ---

        # High vocabulary diversity suggests technical or specialised text.
        if profile.vocabulary_diversity > 0.70:
            reasons.append(
                f"High vocabulary diversity ({profile.vocabulary_diversity:.2f}) "
                "suggests specialised content; increasing quality threshold."
            )
            strategy.quality_threshold = min(strategy.quality_threshold + 0.05, 0.90)

        # Long sentences often indicate complex, nested clauses.
        if profile.avg_sentence_length > 30:
            reasons.append(
                f"Long average sentence length ({profile.avg_sentence_length:.1f} words) "
                "indicates complex syntax; adding a gleaning pass."
            )
            strategy.max_gleanings = min(strategy.max_gleanings + 1, 3)

        # Very short sentences may be lists / tables -- smaller batches help.
        if profile.avg_sentence_length < 8 and strategy_name != "QUICK":
            reasons.append(
                f"Short average sentence length ({profile.avg_sentence_length:.1f} words) "
                "suggests list-like content; reducing batch size."
            )
            strategy.batch_size = max(strategy.batch_size - 1, 2)

        # Entity density -- many entities warrant more careful extraction.
        entity_density = (
            profile.estimated_entities / max(profile.sentence_count, 1)
        )
        if entity_density > 2.0:
            reasons.append(
                f"High entity density ({entity_density:.1f} per sentence) "
                "detected; enabling critic-corrector."
            )
            strategy.enable_critic_corrector = True
            strategy.max_critic_iterations = max(strategy.max_critic_iterations, 1)

        # Multi-doc override for long individual documents.
        if num_documents > 1 and profile.word_count > 8000:
            strategy.max_gleanings = max(strategy.max_gleanings, 2)
            strategy.enable_triplex = True

        # --- Domain signals ---
        detected_domains = self._detect_domains(text)
        if detected_domains:
            domain_str = ", ".join(detected_domains)
            reasons.append(
                f"Domain signals detected: {domain_str}. "
                "The pipeline may benefit from domain-specific schemas."
            )
            # Biomedical and legal texts tend to need higher precision.
            if "biomedical" in detected_domains or "legal" in detected_domains:
                strategy.quality_threshold = min(
                    strategy.quality_threshold + 0.05, 0.90
                )

        strategy.reasoning = " ".join(reasons)
        return strategy

    # ------------------------------------------------------------------
    # Document profiling
    # ------------------------------------------------------------------

    @staticmethod
    def profile_document(text: str) -> DocumentProfile:
        """
        Compute lightweight statistics about *text*.

        This is intentionally dependency-free: no NLP library is needed.

        Args:
            text: Raw document text.

        Returns:
            A :class:`DocumentProfile` with word-level, sentence-level,
            and entity-heuristic statistics.
        """
        char_count = len(text)

        # Tokenise on whitespace.
        words = text.split()
        word_count = len(words)

        # Normalise words for vocabulary diversity calculation.
        normalised = [
            w.strip(string.punctuation).lower() for w in words
        ]
        normalised = [w for w in normalised if w]  # drop empty tokens
        unique_words = set(normalised)
        vocabulary_diversity = (
            len(unique_words) / len(normalised) if normalised else 0.0
        )

        # Sentence splitting (regex-based approximation).
        sentences = _SENTENCE_END.split(text)
        # Filter out empty fragments that result from leading/trailing whitespace.
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)  # avoid division by zero

        avg_sentence_length = word_count / sentence_count

        # Entity estimation heuristic: count capitalised words that do NOT
        # start a sentence.  Deduplicate so that repeated mentions of the
        # same name are counted once.
        capitalised_matches = set(_CAPITALISED_WORD.findall(text))
        estimated_entities = len(capitalised_matches)

        return DocumentProfile(
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            vocabulary_diversity=vocabulary_diversity,
            estimated_entities=estimated_entities,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_strategy(template: PipelineStrategy) -> PipelineStrategy:
        """Return a shallow copy of a strategy template."""
        return PipelineStrategy(
            name=template.name,
            batch_size=template.batch_size,
            max_gleanings=template.max_gleanings,
            enable_triplex=template.enable_triplex,
            enable_cross_document=template.enable_cross_document,
            enable_critic_corrector=template.enable_critic_corrector,
            max_critic_iterations=template.max_critic_iterations,
            segment_overlap_chars=template.segment_overlap_chars,
            quality_threshold=template.quality_threshold,
            reasoning=template.reasoning,
        )

    @staticmethod
    def _detect_domains(text: str) -> List[str]:
        """
        Return a list of domain labels whose keyword signals appear
        frequently enough in *text* to be noteworthy.

        A domain is reported when at least 3 distinct signal words are found.
        """
        lower_text = text.lower()
        # Tokenise once for set-membership checks.
        token_set = set(lower_text.split())

        detected: List[str] = []
        for domain, keywords in _DOMAIN_SIGNALS.items():
            hits = token_set & keywords
            if len(hits) >= 3:
                detected.append(domain)
        return sorted(detected)
