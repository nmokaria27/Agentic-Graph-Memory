"""
Per-stage checkpointing for the deliberative pipeline.

The pipeline runs 9 stages per document and can take many hours. Without
checkpointing, a crash or kill mid-run loses all work. CheckpointManager
writes stage results to disk after each stage and lets a re-run skip
already-completed stages.

Layout per document:

    checkpoints/{document_id}/
        manifest.json
        stage_<n>.json
        governed_kg_latest.json
        knowledge_graph_latest.json

Writes are atomic (tmp file + os.replace) so a SIGKILL during a write
cannot leave a half-written checkpoint behind.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from multi_agent_kg.core.knowledge_graph import KnowledgeGraph
from multi_agent_kg.core.governed_kg import GovernedKnowledgeGraph


class CheckpointManager:
    """Per-document, per-stage checkpoint store."""

    MANIFEST_NAME = "manifest.json"
    GOVERNED_KG_NAME = "governed_kg_latest.json"
    KG_NAME = "knowledge_graph_latest.json"

    def __init__(
        self,
        base_dir: str,
        document_id: str,
        enabled: bool = True,
        resume: bool = False,
    ) -> None:
        self.enabled = enabled
        self.resume = resume
        self.document_id = document_id
        self.dir = Path(base_dir) / document_id
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    # ---------- paths ----------

    @property
    def manifest_path(self) -> Path:
        return self.dir / self.MANIFEST_NAME

    @property
    def governed_kg_path(self) -> Path:
        return self.dir / self.GOVERNED_KG_NAME

    @property
    def kg_path(self) -> Path:
        return self.dir / self.KG_NAME

    def _stage_path(self, stage: str) -> Path:
        return self.dir / f"stage_{stage}.json"

    # ---------- writes ----------

    def save(
        self,
        stage: str,
        data: Any,
        governed_kg: Optional[GovernedKnowledgeGraph] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ) -> None:
        """Persist a stage's result + latest KG/governed-KG snapshots."""
        if not self.enabled:
            return
        self._atomic_write(self._stage_path(stage), data)
        if governed_kg is not None:
            payload = governed_kg.to_dict()
            payload["stats"] = governed_kg.get_stats()
            self._atomic_write(self.governed_kg_path, payload)
        if knowledge_graph is not None:
            self._atomic_write(self.kg_path, knowledge_graph.to_dict())
        self._update_manifest(stage)

    def _update_manifest(self, stage: str) -> None:
        manifest = self.load_manifest()
        completed = manifest.setdefault("completed_stages", [])
        if stage not in completed:
            completed.append(stage)
        manifest["document_id"] = self.document_id
        manifest["last_stage"] = stage
        manifest["last_update"] = datetime.now().isoformat()
        self._atomic_write(self.manifest_path, manifest)

    # ---------- reads ----------

    def has(self, stage: str) -> bool:
        if not self.resume:
            return False
        return self._stage_path(stage).exists()

    def load(self, stage: str) -> Optional[Any]:
        path = self._stage_path(stage)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_governed_kg(self) -> Optional[GovernedKnowledgeGraph]:
        if not self.governed_kg_path.exists():
            return None
        with open(self.governed_kg_path, "r", encoding="utf-8") as f:
            return GovernedKnowledgeGraph.from_dict(json.load(f))

    def load_knowledge_graph(self) -> Optional[KnowledgeGraph]:
        if not self.kg_path.exists():
            return None
        with open(self.kg_path, "r", encoding="utf-8") as f:
            return KnowledgeGraph.from_dict(json.load(f))

    def load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "document_id": self.document_id,
                "completed_stages": [],
                "last_stage": None,
                "last_update": None,
            }
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def completed_stages(self) -> List[str]:
        return list(self.load_manifest().get("completed_stages", []))

    # ---------- lifecycle ----------

    def clear(self) -> None:
        if self.dir.exists():
            shutil.rmtree(self.dir)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------

    @staticmethod
    def _atomic_write(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)


def discover_checkpoints(base_dir: str) -> List[Dict[str, Any]]:
    """List existing per-document checkpoints under base_dir.

    Returns one dict per doc: {document_id, completed_stages, last_update}.
    """
    root = Path(base_dir)
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        manifest = child / CheckpointManager.MANIFEST_NAME
        if not manifest.exists():
            continue
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        out.append(
            {
                "document_id": m.get("document_id", child.name),
                "completed_stages": list(m.get("completed_stages", [])),
                "last_stage": m.get("last_stage"),
                "last_update": m.get("last_update"),
                "dir": str(child),
            }
        )
    return out
