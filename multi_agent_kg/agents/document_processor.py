"""
Document Processor Agent.

Responsible for:
- Document ingestion and preprocessing
- Text segmentation using sentence embeddings
- Chunk metadata enrichment
- Document structure analysis

This is the entry point worker agent in the pipeline.
"""

from typing import Any, Dict, List, Optional
import re

from multi_agent_kg.agents.base import (
    BaseAgent,
    AgentRole,
    AgentContext,
    ExtractionResult,
    ModelTier,
    MemoryType,
)
from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.memory import SharedMemory
from multi_agent_kg.core.communication import MessageBus
from multi_agent_kg.core.config import LLMConfig


class DocumentProcessor(BaseAgent):
    """
    Document Processor Agent - Entry point for the extraction pipeline.
    
    Responsibilities:
    1. Load and validate document content
    2. Segment text into coherent chunks
    3. Enrich chunks with positional and structural metadata
    4. Store document in shared memory for cross-reference
    
    Uses SharedMemory to:
    - Register documents for cross-document entity resolution
    - Store segment boundaries for evidence linking
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        shared_memory: Optional[SharedMemory] = None,
        message_bus: Optional[MessageBus] = None,
        llm_config: Optional[LLMConfig] = None,
        min_segment_length: int = 1500,
        max_segment_length: int = 2000,
        overlap: int = 150,
    ):
        super().__init__(
            name="DocumentProcessor",
            role=AgentRole.WORKER,
            knowledge_graph=knowledge_graph,
            shared_memory=shared_memory,
            message_bus=message_bus,
            llm_config=llm_config,
            default_tier=ModelTier.SMALL,  # Simple task, use small model
        )
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.overlap = overlap

    def run(
        self,
        context: AgentContext,
        source_path: Optional[str] = None,
        **kwargs,
    ) -> ExtractionResult:
        """
        Process a document and return segmented chunks.
        
        Args:
            context: Processing context with document text
            source_path: Optional file path for metadata
            
        Returns:
            ExtractionResult with segments and metadata
        """
        self.stats["calls"] += 1
        text = context.text
        
        if not text:
            return ExtractionResult(
                items=[],
                confidence=0.0,
                metadata={"error": "No text provided"},
            )
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Segment text
        segments = self._segment_text(cleaned_text)
        
        # Enrich segments with metadata
        enriched_segments = self._enrich_segments(
            segments, 
            context.document_id,
            source_path,
        )
        
        # Store in shared memory
        if self.shared_memory:
            self._store_document(
                text=cleaned_text,
                segments=enriched_segments,
                document_id=context.document_id,
                source_path=source_path,
            )
        
        # Calculate confidence based on segment quality
        confidence = self._calculate_confidence(enriched_segments)
        
        self.log(f"Processed document into {len(enriched_segments)} segments")
        
        return ExtractionResult(
            items=enriched_segments,
            confidence=confidence,
            metadata={
                "document_id": context.document_id,
                "original_length": len(text),
                "cleaned_length": len(cleaned_text),
                "segment_count": len(enriched_segments),
                "source_path": source_path,
            },
        )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving scientific characters.
        
        Keeps Greek letters (α, β, γ …), math symbols (±, ≥, ≤, μ),
        accented characters, and other Unicode that is common in
        scientific / medical literature.
        """
        import unicodedata

        # Normalize Unicode to NFC so combining chars are composed
        text = unicodedata.normalize("NFC", text)

        # Collapse runs of whitespace (spaces, tabs, newlines) to a single space
        text = re.sub(r'\s+', ' ', text)

        # Remove only control characters (C0/C1) except common whitespace
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        return text.strip()

    def _segment_text(self, text: str) -> List[str]:
        """
        Segment text into chunks using sentence boundaries with overlap.
        
        Uses a simple but effective approach:
        1. Split by sentence-ending punctuation
        2. Combine short sentences until min_segment_length
        3. Split long segments at natural boundaries
        4. Overlap trailing sentences from the previous segment
           into the next to avoid losing entities at boundaries
        """
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        segments: List[str] = []
        current_segment = ""
        # Keep tail sentences of the previous segment for overlap
        overlap_buffer: List[str] = []

        for sentence in sentences:
            prospective = (current_segment + " " + sentence).strip() if current_segment else sentence

            # If adding this sentence would exceed max, flush current segment
            if len(prospective) > self.max_segment_length and current_segment:
                segments.append(current_segment.strip())

                # Build overlap from tail of flushed segment
                overlap_buffer = self._build_overlap_buffer(current_segment)
                overlap_text = " ".join(overlap_buffer)
                current_segment = (overlap_text + " " + sentence).strip() if overlap_text else sentence
            else:
                current_segment = prospective

            # If current segment is at least min length and sentence ends with period
            if len(current_segment) >= self.min_segment_length and sentence.endswith('.'):
                segments.append(current_segment.strip())
                overlap_buffer = self._build_overlap_buffer(current_segment)
                current_segment = " ".join(overlap_buffer)

        # Don't forget the last segment
        if current_segment and len(current_segment.strip()) >= self.min_segment_length // 4:
            segments.append(current_segment.strip())

        # Handle case where text is too short
        if not segments and text:
            segments = [text]

        return segments

    def _build_overlap_buffer(self, segment_text: str) -> List[str]:
        """Return the last N characters' worth of sentences for overlap."""
        if self.overlap <= 0:
            return []
        sentence_pattern = r'(?<=[.!?])\s+'
        sents = re.split(sentence_pattern, segment_text)
        buf: List[str] = []
        total = 0
        for s in reversed(sents):
            s = s.strip()
            if not s:
                continue
            total += len(s)
            buf.insert(0, s)
            if total >= self.overlap:
                break
        return buf

    def _enrich_segments(
        self,
        segments: List[str],
        document_id: str,
        source_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Enrich segments with metadata."""
        enriched = []
        char_offset = 0
        
        for i, segment in enumerate(segments):
            enriched.append({
                "text": segment,
                "segment_id": f"{document_id}_seg_{i}",
                "index": i,
                "char_start": char_offset,
                "char_end": char_offset + len(segment),
                "word_count": len(segment.split()),
                "document_id": document_id,
                "source_path": source_path,
                "is_first": i == 0,
                "is_last": i == len(segments) - 1,
            })
            char_offset += len(segment) + 1  # +1 for space/newline
        
        return enriched

    def _store_document(
        self,
        text: str,
        segments: List[Dict[str, Any]],
        document_id: str,
        source_path: Optional[str],
    ) -> None:
        """Store document and segments in shared memory."""
        # Register document
        self.shared_memory.register_document(
            doc_id=document_id,
            content=text,
            metadata={
                "source_path": source_path,
                "segment_count": len(segments),
            },
        )
        
        # Store episodic memory of this processing
        self.store_in_memory(
            memory_type=MemoryType.EPISODIC,
            content={
                "document_id": document_id,
                "segments": [s["segment_id"] for s in segments],
                "segment_count": len(segments),
            },
            metadata={"source_path": source_path},
        )

    def _calculate_confidence(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate processing confidence based on segment quality."""
        if not segments:
            return 0.0
        
        # Factors that affect confidence
        avg_word_count = sum(s["word_count"] for s in segments) / len(segments)
        
        # Ideal segment has 50-150 words
        if 50 <= avg_word_count <= 150:
            quality = 1.0
        elif 30 <= avg_word_count < 50 or 150 < avg_word_count <= 200:
            quality = 0.8
        else:
            quality = 0.6
        
        # More segments generally means better coverage
        coverage = min(1.0, len(segments) / 3)  # At least 3 segments is ideal
        
        return (quality + coverage) / 2
