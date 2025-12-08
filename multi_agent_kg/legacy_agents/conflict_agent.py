"""
Conflict detection agent for identifying conflicting triples.
"""

from typing import Any, List, Optional, Tuple
from multi_agent_kg.agents.base_agent import Agent
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph, Triple, Conflict


class ConflictAgent(Agent):
    """
    Agent responsible for detecting conflicts between candidate triples and existing KG.

    A conflict is defined as: same subject and relation but different object.
    """

    def __init__(
        self,
        name: str = "ConflictAgent",
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ):
        """
        Initialize the conflict detection agent.

        Args:
            name: Name of the agent
            knowledge_graph: Reference to the knowledge graph
        """
        super().__init__(name, knowledge_graph)

    def run(
        self,
        candidate_triples: List[Triple],
        auto_resolve: bool = False,
        **kwargs: Any,
    ) -> Tuple[List[Triple], List[Conflict]]:
        """
        Detect conflicts between candidate triples and existing knowledge graph.

        Args:
            candidate_triples: List of triples to check
            auto_resolve: If True, automatically resolve conflicts (keep higher confidence)
            **kwargs: Additional arguments

        Returns:
            Tuple of (non-conflicting triples, list of conflicts)
        """
        if not self.knowledge_graph:
            self.log("No knowledge graph available, returning all triples", level="WARNING")
            return candidate_triples, []

        self.log(f"Checking {len(candidate_triples)} triples for conflicts")

        # Find conflicts
        conflicts = self.knowledge_graph.find_conflicts(candidate_triples)

        if not conflicts:
            self.log("No conflicts detected")
            return candidate_triples, []

        self.log(f"Found {len(conflicts)} conflicts")

        # Log conflicts
        for conflict in conflicts:
            self.log(
                f"Conflict: ({conflict.subject}) -[{conflict.relation}]-> "
                f"{conflict.existing_object} vs {conflict.new_object}",
                level="WARNING",
            )

        # Separate conflicting and non-conflicting triples
        conflicting_triples = {conflict.new_triple for conflict in conflicts}
        non_conflicting = [t for t in candidate_triples if t not in conflicting_triples]

        if auto_resolve:
            resolved = self._auto_resolve_conflicts(conflicts)
            non_conflicting.extend(resolved)
            self.log(f"Auto-resolved {len(resolved)} conflicts")

        self.log(f"Returning {len(non_conflicting)} non-conflicting triples")

        return non_conflicting, conflicts

    def _auto_resolve_conflicts(self, conflicts: List[Conflict]) -> List[Triple]:
        """
        Automatically resolve conflicts by choosing the triple with higher confidence.

        Args:
            conflicts: List of conflicts

        Returns:
            List of resolved triples to keep
        """
        resolved = []

        for conflict in conflicts:
            existing = conflict.existing_triple
            new = conflict.new_triple

            # Choose based on confidence
            existing_conf = existing.confidence or 0.5
            new_conf = new.confidence or 0.5

            if new_conf > existing_conf:
                self.log(
                    f"Resolving conflict in favor of new triple "
                    f"(confidence: {new_conf} > {existing_conf})"
                )
                resolved.append(new)
            else:
                self.log(
                    f"Resolving conflict in favor of existing triple "
                    f"(confidence: {existing_conf} >= {new_conf})"
                )
                # Keep existing (don't add to resolved)

        return resolved
