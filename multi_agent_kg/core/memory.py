"""
Shared Memory System for Multi-Agent Knowledge Graph.

Implements:
- Episodic memory for document context
- Working memory for agent communication
- Long-term memory for accumulated knowledge
- Blackboard pattern for agent collaboration
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import json
import hashlib


class MemoryType(str, Enum):
    """Types of memory in the system."""
    EPISODIC = "episodic"      # Document-level context
    WORKING = "working"         # Current task context
    SEMANTIC = "semantic"       # Extracted knowledge
    PROCEDURAL = "procedural"   # Learned patterns


@dataclass
class MemoryEntry:
    """A single entry in the memory system."""
    id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    source: str                           # Which agent created this
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0          # Decay over time
    references: List[str] = field(default_factory=list)  # Links to other entries
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score,
            "references": self.references,
            "metadata": self.metadata,
        }


@dataclass
class BlackboardEntry:
    """
    Entry on the shared blackboard for agent communication.
    
    Agents post hypotheses, requests, and refinements here.
    Other agents can read, respond, or refine these entries.
    """
    id: str
    author: str                           # Which agent posted this
    entry_type: str                        # "hypothesis", "request", "refinement", "vote"
    content: Dict[str, Any]
    status: str = "pending"               # pending, accepted, rejected, refined
    responses: List[Dict[str, Any]] = field(default_factory=list)
    votes: Dict[str, float] = field(default_factory=dict)  # agent -> confidence
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class SharedMemory:
    """
    Shared memory system accessible by all agents.
    
    Provides:
    - Episodic memory: Context from processed documents
    - Working memory: Current task state
    - Semantic memory: Accumulated knowledge
    - Cross-document entity resolution
    """

    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
        self.blackboard: Dict[str, BlackboardEntry] = {}
        
        # Indices for fast lookup
        self._by_type: Dict[MemoryType, Set[str]] = {t: set() for t in MemoryType}
        self._by_source: Dict[str, Set[str]] = {}
        self._entity_mentions: Dict[str, Set[str]] = {}  # entity -> memory_ids
        
        # Cross-document entity resolution
        self.entity_aliases: Dict[str, str] = {}  # alias -> canonical_id
        self.entity_contexts: Dict[str, List[Dict]] = {}  # entity -> contexts
        
        # Document tracking
        self.processed_documents: List[Dict[str, Any]] = []
        self.document_embeddings: Dict[str, List[float]] = {}

    def store(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        source: str,
        references: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a new memory entry.
        
        Args:
            memory_type: Type of memory
            content: The actual content
            source: Which agent is storing this
            references: Links to other memory entries
            metadata: Additional metadata
            
        Returns:
            Memory entry ID
        """
        # Generate unique ID
        content_hash = hashlib.md5(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        entry_id = f"{memory_type.value}_{source}_{content_hash}"
        
        entry = MemoryEntry(
            id=entry_id,
            memory_type=memory_type,
            content=content,
            source=source,
            references=references or [],
            metadata=metadata or {},
        )
        
        self.memories[entry_id] = entry
        self._by_type[memory_type].add(entry_id)
        
        if source not in self._by_source:
            self._by_source[source] = set()
        self._by_source[source].add(entry_id)
        
        # Index entities mentioned
        if "entities" in content:
            for entity in content["entities"]:
                entity_id = entity if isinstance(entity, str) else entity.get("id", str(entity))
                if entity_id not in self._entity_mentions:
                    self._entity_mentions[entity_id] = set()
                self._entity_mentions[entity_id].add(entry_id)
        
        return entry_id

    def retrieve(
        self,
        memory_type: Optional[MemoryType] = None,
        source: Optional[str] = None,
        entity: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        """
        Retrieve memories matching criteria.
        
        Args:
            memory_type: Filter by memory type
            source: Filter by source agent
            entity: Filter by entity mention
            limit: Maximum entries to return
            
        Returns:
            List of matching memory entries
        """
        candidates = set(self.memories.keys())
        
        if memory_type:
            candidates &= self._by_type.get(memory_type, set())
        
        if source:
            candidates &= self._by_source.get(source, set())
        
        if entity:
            # Also check aliases
            canonical = self.entity_aliases.get(entity, entity)
            entity_mems = self._entity_mentions.get(canonical, set())
            entity_mems |= self._entity_mentions.get(entity, set())
            candidates &= entity_mems
        
        # Sort by relevance and recency
        entries = [self.memories[mid] for mid in candidates]
        entries.sort(key=lambda e: (e.relevance_score, e.timestamp), reverse=True)
        
        return entries[:limit]

    def get_entity_context(self, entity_id: str) -> Dict[str, Any]:
        """
        Get all context about an entity across all documents.
        
        Args:
            entity_id: The entity to look up
            
        Returns:
            Aggregated context about the entity
        """
        canonical = self.entity_aliases.get(entity_id, entity_id)
        
        contexts = self.entity_contexts.get(canonical, [])
        memories = self.retrieve(entity=canonical)
        
        return {
            "canonical_id": canonical,
            "aliases": [k for k, v in self.entity_aliases.items() if v == canonical],
            "contexts": contexts,
            "memory_count": len(memories),
            "sources": list(set(m.source for m in memories)),
            "first_seen": min((m.timestamp for m in memories), default=None),
            "last_seen": max((m.timestamp for m in memories), default=None),
        }

    def register_entity_alias(self, alias: str, canonical_id: str) -> None:
        """Register an alias for entity resolution."""
        self.entity_aliases[alias] = canonical_id
        
        # Merge contexts
        if alias in self.entity_contexts:
            if canonical_id not in self.entity_contexts:
                self.entity_contexts[canonical_id] = []
            self.entity_contexts[canonical_id].extend(self.entity_contexts[alias])

    def add_entity_context(self, entity_id: str, context: Dict[str, Any]) -> None:
        """Add context about an entity from a document."""
        canonical = self.entity_aliases.get(entity_id, entity_id)
        if canonical not in self.entity_contexts:
            self.entity_contexts[canonical] = []
        self.entity_contexts[canonical].append(context)

    # ========== Blackboard Methods ==========

    def post_to_blackboard(
        self,
        author: str,
        entry_type: str,
        content: Dict[str, Any],
    ) -> str:
        """
        Post an entry to the shared blackboard.
        
        Entry types:
        - "hypothesis": A proposed triple or entity
        - "request": Asking other agents for information
        - "refinement": Improving an existing entry
        - "conflict": Flagging a potential conflict
        - "vote": Voting on another entry
        
        Args:
            author: Agent posting the entry
            entry_type: Type of blackboard entry
            content: The content being posted
            
        Returns:
            Entry ID
        """
        entry_id = f"bb_{author}_{datetime.now().timestamp()}"
        
        entry = BlackboardEntry(
            id=entry_id,
            author=author,
            entry_type=entry_type,
            content=content,
        )
        
        self.blackboard[entry_id] = entry
        return entry_id

    def respond_to_blackboard(
        self,
        entry_id: str,
        responder: str,
        response_type: str,
        response_content: Dict[str, Any],
    ) -> None:
        """
        Respond to a blackboard entry.
        
        Args:
            entry_id: ID of entry to respond to
            responder: Agent responding
            response_type: Type of response (agree, disagree, refine, etc.)
            response_content: Response content
        """
        if entry_id not in self.blackboard:
            return
        
        entry = self.blackboard[entry_id]
        entry.responses.append({
            "responder": responder,
            "type": response_type,
            "content": response_content,
            "timestamp": datetime.now().isoformat(),
        })
        entry.updated_at = datetime.now()

    def vote_on_blackboard(
        self,
        entry_id: str,
        voter: str,
        confidence: float,
    ) -> None:
        """
        Vote on a blackboard entry.
        
        Args:
            entry_id: ID of entry to vote on
            voter: Agent voting
            confidence: Confidence score (0-1)
        """
        if entry_id not in self.blackboard:
            return
        
        entry = self.blackboard[entry_id]
        entry.votes[voter] = confidence
        entry.updated_at = datetime.now()

    def get_blackboard_entries(
        self,
        entry_type: Optional[str] = None,
        status: Optional[str] = None,
        min_votes: int = 0,
    ) -> List[BlackboardEntry]:
        """Get blackboard entries matching criteria."""
        entries = list(self.blackboard.values())
        
        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]
        
        if status:
            entries = [e for e in entries if e.status == status]
        
        if min_votes > 0:
            entries = [e for e in entries if len(e.votes) >= min_votes]
        
        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def resolve_blackboard_entry(self, entry_id: str, status: str) -> None:
        """Mark a blackboard entry as resolved."""
        if entry_id in self.blackboard:
            self.blackboard[entry_id].status = status
            self.blackboard[entry_id].updated_at = datetime.now()

    # ========== Document Tracking ==========

    def register_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a processed document."""
        self.processed_documents.append({
            "id": doc_id,
            "content_preview": content[:500],
            "length": len(content),
            "processed_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        })

    def get_document_history(self) -> List[Dict[str, Any]]:
        """Get history of processed documents."""
        return self.processed_documents

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "total_memories": len(self.memories),
            "by_type": {t.value: len(ids) for t, ids in self._by_type.items()},
            "by_source": {s: len(ids) for s, ids in self._by_source.items()},
            "unique_entities": len(self._entity_mentions),
            "entity_aliases": len(self.entity_aliases),
            "blackboard_entries": len(self.blackboard),
            "documents_processed": len(self.processed_documents),
        }

    def export(self) -> Dict[str, Any]:
        """Export entire memory system."""
        return {
            "memories": [m.to_dict() for m in self.memories.values()],
            "blackboard": [
                {
                    "id": e.id,
                    "author": e.author,
                    "entry_type": e.entry_type,
                    "content": e.content,
                    "status": e.status,
                    "responses": e.responses,
                    "votes": e.votes,
                }
                for e in self.blackboard.values()
            ],
            "entity_aliases": self.entity_aliases,
            "documents": self.processed_documents,
            "stats": self.get_stats(),
        }
