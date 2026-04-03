"""
Relation Library -- persistent, cross-document catalog of relation types.

The library tracks every relation type ever discovered by the pipeline,
accumulating frequency counts, definitions, source/target entity type
constraints, and concrete examples.  It serves two purposes:

1. **Consistency**: When the RelationExtractor encounters a relation that
   is semantically identical to one already in the library (e.g.
   "INTRODUCED_PRODUCT" vs "PRODUCT_LAUNCH"), the library normalises it
   to the canonical form.

2. **Reuse across documents**: When a new document is processed, the
   DomainClassifier and RelationExtractor receive the library's known
   relations as "suggested types", improving extraction quality and
   schema alignment over time.

Persistence:
- The library can be serialised to / loaded from a JSON file so it
  survives across pipeline runs.
- It can also be stored in SharedMemory for intra-run access.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set


@dataclass
class RelationEntry:
    """A single relation type tracked by the library."""

    name: str
    """Canonical UPPER_SNAKE_CASE name (e.g. FOUNDED_BY)."""

    definition: str = ""
    """Human-readable description of the relation."""

    aliases: List[str] = field(default_factory=list)
    """Alternative names that have been normalised to this entry
    (e.g. FOUNDED_BY might have alias CREATED_BY)."""

    source_entity_types: List[str] = field(default_factory=list)
    """Entity types allowed as subjects (e.g. ["COMPANY"])."""

    target_entity_types: List[str] = field(default_factory=list)
    """Entity types allowed as objects (e.g. ["PERSON"])."""

    frequency: int = 0
    """How many times this relation has been extracted (across all docs)."""

    examples: List[Dict[str, str]] = field(default_factory=list)
    """Concrete (subject, object, evidence) examples (kept to max 5)."""

    source_documents: List[str] = field(default_factory=list)
    """Document IDs where this relation was found."""


class RelationLibrary:
    """Cross-document relation type catalog with normalisation.

    Usage::

        lib = RelationLibrary.load("relation_library.json")

        # Register a new relation discovered during extraction
        lib.register(
            name="FOUNDED_BY",
            definition="Company founded by person",
            source_types=["COMPANY"],
            target_types=["PERSON"],
            document_id="doc_1",
            example={"subject": "Apple Inc.", "object": "Steve Jobs",
                     "evidence": "Founded in 1976 by Steve Jobs..."},
        )

        # Normalise a potentially new name against existing entries
        canonical = lib.normalise("CREATED_BY")  # -> "FOUNDED_BY" if similar

        # Get suggested relation types for a new document
        suggestions = lib.get_suggestions(min_frequency=2)

        lib.save("relation_library.json")
    """

    def __init__(self) -> None:
        self._entries: Dict[str, RelationEntry] = {}
        # Fast lookup: lowered alias -> canonical name
        self._alias_index: Dict[str, str] = {}

    # ------------------------------------------------------------------
    #  Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        definition: str = "",
        source_types: Optional[List[str]] = None,
        target_types: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        example: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register or update a relation type.  Returns the canonical name."""
        canonical = self.normalise(name)

        if canonical in self._entries:
            entry = self._entries[canonical]
            entry.frequency += 1
            if definition and not entry.definition:
                entry.definition = definition
        else:
            entry = RelationEntry(
                name=canonical,
                definition=definition,
                source_entity_types=source_types or [],
                target_entity_types=target_types or [],
                frequency=1,
            )
            self._entries[canonical] = entry
            self._alias_index[canonical.lower()] = canonical

        # Merge source/target types
        if source_types:
            for st in source_types:
                if st not in entry.source_entity_types:
                    entry.source_entity_types.append(st)
        if target_types:
            for tt in target_types:
                if tt not in entry.target_entity_types:
                    entry.target_entity_types.append(tt)

        # Add document reference
        if document_id and document_id not in entry.source_documents:
            entry.source_documents.append(document_id)

        # Store example (cap at 5)
        if example and len(entry.examples) < 5:
            entry.examples.append(example)

        return canonical

    def add_alias(self, alias: str, canonical: str) -> None:
        """Map *alias* to *canonical* so future calls to ``normalise(alias)``
        return *canonical*."""
        canonical_upper = canonical.upper().replace(" ", "_")
        if canonical_upper in self._entries:
            entry = self._entries[canonical_upper]
            if alias not in entry.aliases:
                entry.aliases.append(alias)
            self._alias_index[alias.lower()] = canonical_upper

    # ------------------------------------------------------------------
    #  Normalisation
    # ------------------------------------------------------------------

    def normalise(self, name: str) -> str:
        """Return the canonical name for *name*, or *name* itself if unknown.

        Matching is case-insensitive and underscore/space agnostic.
        """
        key = name.strip().lower().replace(" ", "_")

        # Direct hit
        if key in self._alias_index:
            return self._alias_index[key]

        # Check without underscores
        key_no_sep = key.replace("_", "")
        for existing_key, canonical in self._alias_index.items():
            if existing_key.replace("_", "") == key_no_sep:
                # Register the new spelling as an alias
                self._alias_index[key] = canonical
                return canonical

        # Token containment check: if the new name's tokens are a subset of an existing entry
        new_tokens = set(key.replace("_", " ").split())
        if new_tokens:
            for existing_key, canonical in self._alias_index.items():
                existing_tokens = set(existing_key.replace("_", " ").split())
                # If new name is a subset of existing, or existing is subset of new
                if new_tokens <= existing_tokens or existing_tokens <= new_tokens:
                    self._alias_index[key] = canonical
                    return canonical

        # No match -- return as-is (UPPER_SNAKE_CASE normalised)
        return name.upper().replace(" ", "_")

    def merge_similar(self) -> int:
        """Merge semantically similar relation entries.

        Finds entries whose canonical names share significant tokens and
        merges them, keeping the most frequent as canonical.

        Returns the number of merges performed.
        """
        merges = 0
        names = list(self._entries.keys())
        merged_into = {}  # old_name -> new_canonical

        for i, name_a in enumerate(names):
            if name_a in merged_into:
                continue
            tokens_a = set(name_a.lower().replace("_", " ").split())

            for name_b in names[i+1:]:
                if name_b in merged_into:
                    continue
                tokens_b = set(name_b.lower().replace("_", " ").split())

                # Merge if one is a subset of the other
                if tokens_a <= tokens_b or tokens_b <= tokens_a:
                    entry_a = self._entries[name_a]
                    entry_b = self._entries[name_b]

                    # Keep the more frequent one as canonical
                    if entry_a.frequency >= entry_b.frequency:
                        keep, remove = name_a, name_b
                    else:
                        keep, remove = name_b, name_a

                    # Merge
                    self._entries[keep].frequency += self._entries[remove].frequency
                    self._entries[keep].aliases.append(remove)
                    for alias in self._entries[remove].aliases:
                        if alias not in self._entries[keep].aliases:
                            self._entries[keep].aliases.append(alias)
                    for doc in self._entries[remove].source_documents:
                        if doc not in self._entries[keep].source_documents:
                            self._entries[keep].source_documents.append(doc)
                    for ex in self._entries[remove].examples:
                        if len(self._entries[keep].examples) < 5:
                            self._entries[keep].examples.append(ex)

                    # Remove and update index
                    del self._entries[remove]
                    self._alias_index[remove.lower()] = keep
                    merged_into[remove] = keep
                    merges += 1

        return merges

    # ------------------------------------------------------------------
    #  Querying
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[RelationEntry]:
        """Lookup a relation entry by name or alias."""
        canonical = self.normalise(name)
        return self._entries.get(canonical)

    def get_suggestions(self, min_frequency: int = 1) -> List[Dict[str, Any]]:
        """Return relation types suitable for passing to the DomainClassifier
        or RelationExtractor as ``suggested_types``."""
        suggestions = []
        for entry in self._entries.values():
            if entry.frequency >= min_frequency:
                suggestions.append({
                    "type": entry.name,
                    "definition": entry.definition,
                    "source_types": entry.source_entity_types,
                    "target_types": entry.target_entity_types,
                    "frequency": entry.frequency,
                })
        return sorted(suggestions, key=lambda s: s["frequency"], reverse=True)

    def all_relation_names(self) -> List[str]:
        """Return all canonical relation names."""
        return list(self._entries.keys())

    @property
    def size(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise to JSON file."""
        data = {
            name: asdict(entry)
            for name, entry in self._entries.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "RelationLibrary":
        """Load from JSON file.  Returns empty library if file doesn't exist."""
        lib = cls()
        if not os.path.exists(path):
            return lib
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, entry_dict in data.items():
            entry = RelationEntry(**{
                k: v for k, v in entry_dict.items()
                if k in RelationEntry.__dataclass_fields__
            })
            lib._entries[entry.name] = entry
            lib._alias_index[entry.name.lower()] = entry.name
            for alias in entry.aliases:
                lib._alias_index[alias.lower()] = entry.name
        return lib

    def to_dict(self) -> Dict[str, Any]:
        """Return serialisable dict (for storing in SharedMemory)."""
        return {
            name: asdict(entry)
            for name, entry in self._entries.items()
        }

    def __repr__(self) -> str:
        return f"RelationLibrary({self.size} relations)"
