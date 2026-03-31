"""
Adapter for converting SciERC dataset to pipeline input format and
extracting gold-standard annotations for evaluation.

SciERC format (one JSON object per line):
{
    "doc_key": "X96-1059",
    "sentences": [["token", "token", ...], ...],
    "ner": [[[start, end, type], ...], ...],       # per-sentence
    "relations": [[[s1, e1, s2, e2, type], ...], ...],  # per-sentence
    "clusters": [[[start, end], ...], ...]          # coreference
}

Entity types: Task, Method, Metric, Material, OtherScientificTerm, Generic
Relation types: USED-FOR, FEATURE-OF, PART-OF, COMPARE, HYPONYM-OF,
                CONJUNCTION, EVALUATE-FOR

Token indices are document-level (not sentence-level), and entity spans
are inclusive on both ends: tokens[start:end+1].
"""

import json
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
GoldEntity = Dict[str, Any]   # {"text": str, "type": str, "span": (int, int)}
GoldTriple = Dict[str, Any]   # {"subject": str, "relation": str, "object": str,
                               #  "subject_type": str, "object_type": str}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
# SciERC relation labels are UPPER-CASE with hyphens; the pipeline may
# produce title-case or lower-case variants.  Keep a canonical mapping so
# that comparisons are consistent.

RELATION_NORMALISE = {
    "USED-FOR": "Used-for",
    "FEATURE-OF": "Feature-of",
    "PART-OF": "Part-of",
    "COMPARE": "Compare",
    "HYPONYM-OF": "Hyponym-of",
    "CONJUNCTION": "Conjunction",
    "EVALUATE-FOR": "Evaluate-for",
}

ENTITY_TYPE_NORMALISE = {
    "TASK": "Task",
    "METHOD": "Method",
    "METRIC": "Metric",
    "MATERIAL": "Material",
    "OTHERSCIENTIFICTERM": "OtherScientificTerm",
    "GENERIC": "Generic",
}


def normalise_relation(rel: str) -> str:
    """Return a canonical relation label."""
    return RELATION_NORMALISE.get(rel.upper(), rel)


def normalise_entity_type(etype: str) -> str:
    """Return a canonical entity type label."""
    return ENTITY_TYPE_NORMALISE.get(etype.upper(), etype)


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------
class SciERCAdapter:
    """Converts SciERC documents into pipeline-ready and evaluation-ready formats."""

    def __init__(self, filepath: str, skip_generic: bool = True) -> None:
        """
        Load a SciERC JSON-lines file.

        Args:
            filepath: Path to a SciERC JSON-lines file (train/dev/test.json).
            skip_generic: If True, drop entities with type "Generic" (pronouns
                          like "it", "this method" that are hard to evaluate).
        """
        self.filepath = filepath
        self.skip_generic = skip_generic
        self.documents: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        with open(self.filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.documents.append(json.loads(line))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.documents[idx]

    # ------------------------------------------------------------------
    def _flatten_tokens(self, doc: Dict[str, Any]) -> List[str]:
        """Flatten the list-of-lists sentences into a single token list."""
        tokens: List[str] = []
        for sent in doc["sentences"]:
            tokens.extend(sent)
        return tokens

    def _span_to_text(self, tokens: List[str], start: int, end: int) -> str:
        """Convert an inclusive (start, end) span to surface text."""
        return " ".join(tokens[start : end + 1])

    # ------------------------------------------------------------------
    # Pipeline input
    # ------------------------------------------------------------------
    def to_pipeline_input(
        self, max_docs: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert SciERC documents to the pipeline's expected input format:

            [{"id": "...", "text": "...", "metadata": {...}}, ...]
        """
        docs = self.documents[:max_docs] if max_docs else self.documents
        result: List[Dict[str, Any]] = []
        for doc in docs:
            tokens = self._flatten_tokens(doc)
            text = " ".join(tokens)
            result.append(
                {
                    "id": doc["doc_key"],
                    "text": text,
                    "metadata": {
                        "source": "scierc",
                        "type": "scientific_abstract",
                        "doc_key": doc["doc_key"],
                    },
                }
            )
        return result

    # ------------------------------------------------------------------
    # Gold-standard extraction
    # ------------------------------------------------------------------
    def get_gold_entities(self, doc: Dict[str, Any]) -> List[GoldEntity]:
        """
        Extract gold-standard entities from a single SciERC document.

        Returns list of dicts:
            {"text": str, "type": str, "span": (start, end)}
        """
        tokens = self._flatten_tokens(doc)
        entities: List[GoldEntity] = []
        for sent_ner in doc["ner"]:
            for ent in sent_ner:
                start, end, etype = ent[0], ent[1], ent[2]
                etype = normalise_entity_type(etype)
                if self.skip_generic and etype == "Generic":
                    continue
                entities.append(
                    {
                        "text": self._span_to_text(tokens, start, end),
                        "type": etype,
                        "span": (start, end),
                    }
                )
        return entities

    def get_gold_triples(self, doc: Dict[str, Any]) -> List[GoldTriple]:
        """
        Extract gold-standard relation triples from a single SciERC document.

        Returns list of dicts:
            {"subject": str, "relation": str, "object": str,
             "subject_type": str, "object_type": str}
        """
        tokens = self._flatten_tokens(doc)

        # Build a span -> type lookup from NER annotations
        span_type: Dict[Tuple[int, int], str] = {}
        for sent_ner in doc["ner"]:
            for ent in sent_ner:
                span_type[(ent[0], ent[1])] = normalise_entity_type(ent[2])

        triples: List[GoldTriple] = []
        for sent_rel in doc["relations"]:
            for rel in sent_rel:
                s1, e1, s2, e2, rtype = rel[0], rel[1], rel[2], rel[3], rel[4]
                subj_text = self._span_to_text(tokens, s1, e1)
                obj_text = self._span_to_text(tokens, s2, e2)
                subj_type = span_type.get((s1, e1), "Unknown")
                obj_type = span_type.get((s2, e2), "Unknown")

                if self.skip_generic and (
                    subj_type == "Generic" or obj_type == "Generic"
                ):
                    continue

                triples.append(
                    {
                        "subject": subj_text,
                        "relation": normalise_relation(rtype),
                        "object": obj_text,
                        "subject_type": subj_type,
                        "object_type": obj_type,
                    }
                )
        return triples

    def get_gold_for_doc(
        self, doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Return a combined gold-standard dict for a single document, in a
        format parallel to the pipeline's kg_export.json:

            {
                "doc_key": str,
                "entities": [...],
                "triples": [...]
            }
        """
        return {
            "doc_key": doc["doc_key"],
            "entities": self.get_gold_entities(doc),
            "triples": self.get_gold_triples(doc),
        }

    def get_all_gold(
        self, max_docs: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return gold-standard data for all (or first N) documents."""
        docs = self.documents[:max_docs] if max_docs else self.documents
        return [self.get_gold_for_doc(doc) for doc in docs]


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Convert SciERC data to pipeline input format."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(
            os.path.dirname(__file__), "..", "datasets", "scierc", "test.json"
        ),
        help="Path to SciERC JSON-lines file",
    )
    parser.add_argument(
        "--output-pipeline",
        default=None,
        help="Write pipeline-input JSON to this path",
    )
    parser.add_argument(
        "--output-gold",
        default=None,
        help="Write gold-standard JSON to this path",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit to first N documents",
    )
    parser.add_argument(
        "--include-generic",
        action="store_true",
        help="Include Generic-type entities (pronouns, etc.)",
    )
    args = parser.parse_args()

    adapter = SciERCAdapter(args.input, skip_generic=not args.include_generic)
    print(f"Loaded {len(adapter)} documents from {args.input}")

    pipeline_docs = adapter.to_pipeline_input(max_docs=args.max_docs)
    gold_data = adapter.get_all_gold(max_docs=args.max_docs)

    # Summary
    total_ents = sum(len(g["entities"]) for g in gold_data)
    total_rels = sum(len(g["triples"]) for g in gold_data)
    print(f"  Documents: {len(pipeline_docs)}")
    print(f"  Total gold entities: {total_ents}")
    print(f"  Total gold triples: {total_rels}")

    if args.output_pipeline:
        with open(args.output_pipeline, "w", encoding="utf-8") as f:
            json.dump(pipeline_docs, f, indent=2)
        print(f"  Pipeline input written to {args.output_pipeline}")

    if args.output_gold:
        with open(args.output_gold, "w", encoding="utf-8") as f:
            json.dump(gold_data, f, indent=2)
        print(f"  Gold standard written to {args.output_gold}")


if __name__ == "__main__":
    main()
