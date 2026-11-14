"""Atomic numeric/JSON embedding storage with stable IDs and exact squared-L2 search."""
from __future__ import annotations
import json
import os
import threading
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


class SkinEmbeddingDatabase:
    FORMAT_VERSION = 2

    def __init__(self, embedding_dim=512, index_type="L2", use_gpu=False, feature_space_id=None):
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if index_type not in ("L2", "IP"):
            raise ValueError("index_type must be L2 or IP")
        self.embedding_dim, self.index_type = embedding_dim, index_type
        self.use_gpu, self.feature_space_id = use_gpu, feature_space_id
        self.embeddings, self.metadata, self.id_to_metadata = [], [], {}
        self.next_id, self.last_update = 0, None
        self._lock = threading.RLock()

    @staticmethod
    def storage_path(filepath):
        path = Path(filepath)
        if path.suffix.lower() in (".pkl", ".pickle"):
            raise ValueError("Legacy pickle is unsupported; choose a new .npz path")
        return path if path.suffix.lower() == ".npz" else Path(str(path) + ".npz")

    def _vector(self, embedding):
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (self.embedding_dim,) or not np.isfinite(vector).all():
            raise ValueError(f"Expected a finite vector of shape ({self.embedding_dim},)")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("Zero embeddings cannot identify a person")
        return vector / norm if self.index_type == "IP" else vector.copy()

    def add_embedding(self, embedding, person_id, person_name, image_path=None, additional_metadata=None):
        vector = self._vector(embedding)
        if not isinstance(person_id, str) or not person_id.strip():
            raise ValueError("person_id must be nonempty")
        if not isinstance(person_name, str) or not person_name.strip():
            raise ValueError("person_name must be nonempty")
        if additional_metadata is not None and not isinstance(additional_metadata, dict):
            raise ValueError("metadata must be a JSON object")
        if {"id", "person_id", "person_name", "timestamp", "image_path"}.intersection(additional_metadata or {}):
            raise ValueError("metadata cannot overwrite identity fields")
        extra = json.loads(json.dumps(additional_metadata or {}, allow_nan=False))
        with self._lock:
            stamp = datetime.now(timezone.utc).isoformat()
            metadata = dict(extra, id=self.next_id, person_id=person_id.strip(),
                            person_name=person_name.strip(), image_path=image_path, timestamp=stamp)
            self.embeddings.append(vector)
            self.metadata.append(metadata)
            self.id_to_metadata[self.next_id] = metadata
            self.next_id += 1
            self.last_update = stamp
            return metadata["id"]

    def search(self, query_embedding, k=5, threshold=None):
        query = self._vector(query_embedding)
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be positive")
        if threshold is not None and not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        with self._lock:
            if not self.embeddings:
                return []
            matrix = np.stack(self.embeddings)
            scores = (np.sum((matrix - query) ** 2, axis=1)
                      if self.index_type == "L2" else matrix @ query)
            rows = np.argsort(scores if self.index_type == "L2" else -scores, kind="stable")[:k]
            results = []
            for row in rows:
                score = float(scores[row])
                if threshold is not None and ((self.index_type == "L2" and score > threshold)
                                               or (self.index_type == "IP" and score < threshold)):
                    continue
                metadata = self.metadata[row]
                results.append((metadata["id"], score, dict(metadata)))
            return results

    def get_person_embeddings(self, person_id):
        with self._lock:
            return [(m["id"], dict(m)) for m in self.metadata if m["person_id"] == person_id]

    def remove_embedding(self, embedding_id):
        with self._lock:
            for row, metadata in enumerate(self.metadata):
                if metadata["id"] == embedding_id:
                    del self.metadata[row]
                    del self.embeddings[row]
                    del self.id_to_metadata[embedding_id]
                    self.last_update = datetime.now(timezone.utc).isoformat()
                    return True
            return False

    def save(self, filepath):
        path = self.storage_path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = dict(format_version=self.FORMAT_VERSION, embedding_dim=self.embedding_dim,
                           index_type=self.index_type, feature_space_id=self.feature_space_id,
                           next_id=self.next_id, last_update=self.last_update, metadata=self.metadata)
            matrix = np.stack(self.embeddings) if self.embeddings else np.empty((0, self.embedding_dim), np.float32)
            fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    np.savez_compressed(stream, embeddings=matrix,
                        payload=np.array(json.dumps(payload, ensure_ascii=False, allow_nan=False)))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def load(self, filepath):
        # Validate before modifying the live object. Never unpickle external data.
        with np.load(self.storage_path(filepath), allow_pickle=False) as archive:
            payload = json.loads(str(archive["payload"].item()))
            matrix = np.asarray(archive["embeddings"], dtype=np.float32).copy()
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("Unsupported database format")
        if payload.get("embedding_dim") != self.embedding_dim or payload.get("index_type") != self.index_type:
            raise ValueError("Incompatible dimension or metric")
        if self.feature_space_id != payload.get("feature_space_id"):
            raise ValueError("Incompatible feature space; re-enrollment required")
        metadata = payload["metadata"]
        if matrix.shape != (len(metadata), self.embedding_dim) or not np.isfinite(matrix).all():
            raise ValueError("Invalid stored embeddings")
        ids = [m["id"] for m in metadata]
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid or duplicate IDs")
        next_id = payload["next_id"]
        if type(next_id) is not int or next_id < max(ids, default=-1) + 1:
            raise ValueError("Invalid next ID")
        for row in matrix:
            self._vector(row)
        for item in metadata:
            if not isinstance(item.get("person_id"), str) or not item["person_id"].strip():
                raise ValueError("Invalid person ID")
            if not isinstance(item.get("person_name"), str) or not item["person_name"].strip():
                raise ValueError("Invalid person name")
        with self._lock:
            self.embeddings = [row.copy() for row in matrix]
            self.metadata, self.id_to_metadata = metadata, {m["id"]: m for m in metadata}
            self.next_id, self.last_update = next_id, payload.get("last_update")

    def get_stats(self):
        with self._lock:
            return dict(total_embeddings=len(self.metadata),
                        total_persons=len({m["person_id"] for m in self.metadata}),
                        last_update=self.last_update, backend="numpy-exact",
                        distance_metric="squared_l2" if self.index_type == "L2" else "cosine_similarity",
                        feature_space_id=self.feature_space_id)

    def clear(self):
        with self._lock:
            self.embeddings.clear()
            self.metadata.clear()
            self.id_to_metadata.clear()
            self.last_update = datetime.now(timezone.utc).isoformat()
