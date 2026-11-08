"""Chroma 向量库封装：余弦空间、按文档幂等写入、按元数据检索。"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.errors import DocumentNotFound
from app.schemas import DocumentInfo

log = logging.getLogger(__name__)


class VectorStore:
    """对 langchain-chroma 的薄封装，集中处理并发写与文档级增删。"""

    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._store: Chroma | None = None
        self._lock = threading.RLock()

    @property
    def store(self) -> Chroma:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    Path(self._settings.chroma_dir).mkdir(parents=True, exist_ok=True)
                    self._store = Chroma(
                        collection_name=self._settings.collection_name,
                        embedding_function=self._embeddings,
                        persist_directory=self._settings.chroma_dir,
                        # 文本向量用余弦距离更稳定；不指定时 Chroma 默认是 L2
                        collection_metadata={"hnsw:space": "cosine"},
                        # 关掉遥测。这是私有文档库，「文档不出机器」得靠默认值兜住，
                        # 不能指望每个部署的人都记得设一个环境变量。
                        # 顺带解决一个更实际的问题：没有外网的机器上，遥测上报会在
                        # 每次写入时阻塞到超时——实测能卡满十分钟，一个字节都没写进库。
                        client_settings=ChromaSettings(anonymized_telemetry=False),
                    )
                    log.info(
                        "向量库就绪：dir=%s collection=%s",
                        self._settings.chroma_dir,
                        self._settings.collection_name,
                    )
        return self._store

    # ------------------------------------------------------------------ 写
    def upsert_document(
        self,
        chunks: list[Document],
        ids: list[str],
        doc_id: str,
        source: str,
    ) -> int:
        """先写入新切片，再清理旧切片。

        顺序必须是「先写后删」：新数据落库后才删除旧切片，这样 Embedding 等外部调用失败时
        不会留下「旧切片已删、新切片没写进来」的空窗——文档不会凭空消失。

        返回被命中的旧切片数，0 表示这是一篇新文档。
        """
        new_ids = set(ids)
        with self._lock:
            existing = self._ids_for(doc_id, source)
            if existing == new_ids:
                # 内容与切片口径完全一致：原样返回，省掉一次重复向量化的开销
                log.info("内容未变化，跳过重复写入：doc_id=%s chunks=%d", doc_id, len(ids))
                return len(existing)

            # 只有「同一 doc_id 但切片口径变了」才会出现 id 冲突，这种情况只能先删后写
            conflicting = existing & new_ids
            if conflicting:
                self.store.delete(ids=sorted(conflicting))

            self.store.add_documents(chunks, ids=ids)

            stale = existing - new_ids
            if stale:
                self.store.delete(ids=sorted(stale))
            return len(existing)

    # ------------------------------------------------------------------ 管理
    def reload(self) -> None:
        """丢掉缓存的 Chroma 客户端，下次访问时重新打开磁盘上的库。

        实测（chromadb 1.5.9）：另一个进程往同一个目录写入后，本进程的计数与向量检索
        **都是实时可见**的，并不存在「索引快照」。所以别拿重载去掩盖「数字不对」——
        重载后数字没变，那就是库真的没变。

        它真正的用途是换库：改了 `CHROMA_DIR`、换了集合名、或整库替换成另一份索引时，
        不必重启服务就能让新库生效；排查问题时也能一次性排除客户端缓存这一项。
        """
        with self._lock:
            self._store = None
            try:
                from chromadb.api.shared_system_client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
            except Exception:  # noqa: BLE001 - 清缓存失败不该阻断重载，但必须留痕
                log.warning("清理 Chroma 系统缓存失败，索引可能仍是旧快照", exc_info=True)
            log.info(
                "向量库已重载：dir=%s collection=%s",
                self._settings.chroma_dir,
                self._settings.collection_name,
            )

    def _ids_for(self, doc_id: str, source: str) -> set[str]:
        """同一篇文档的旧切片：内容相同（doc_id 一致）或文件名相同都算。"""
        ids: set[str] = set()
        for where in ({"doc_id": doc_id}, {"source": source}):
            found = self.store.get(where=where, include=["metadatas"])
            ids.update(found.get("ids") or [])
        return ids

    def delete_document(self, doc_id: str) -> tuple[int, str | None]:
        """删除文档全部切片，返回 (切片数, 原始文件名)，便于调用方一并清理磁盘上的原文件。"""
        with self._lock:
            found = self.store.get(where={"doc_id": doc_id}, include=["metadatas"])
            ids = found.get("ids") or []
            if not ids:
                raise DocumentNotFound(f"文档不存在：{doc_id}")
            metadatas = found.get("metadatas") or []
            source = metadatas[0].get("source") if metadatas else None
            self.store.delete(ids=ids)
            return len(ids), str(source) if source else None

    # ------------------------------------------------------------------ 读
    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """返回 (片段, 余弦相似度)，相似度已归一化到 0~1。"""
        hits = self.store.similarity_search_with_score(query, k=k)
        results: list[tuple[Document, float]] = []
        for document, distance in hits:
            score = min(1.0, max(0.0, 1.0 - float(distance)))
            results.append((document, round(score, 4)))
        return results

    def dump_all(self) -> list[tuple[str, dict, str]]:
        """全量切片 (id, metadata, text)，供 BM25 重建使用。

        用 Chroma 自己的 id 当 key（就是 `doc_id:chunk_index`），两套索引天然对齐：
        融合时按同一个 key 去重，不会出现「向量说是 A、BM25 说是 B」的对不上。

        注意它读的是**当前客户端打开的库**：`/admin/reload` 只重载向量库客户端，
        不会顺带重建 BM25（BM25 有独立指纹，换库后要重启或跑 `scripts/build_bm25.py`）。
        """
        data = self.store.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        return [
            (str(key), dict(metadata), text or "")
            for key, text, metadata in zip(ids, documents, metadatas, strict=False)
            if metadata is not None
        ]

    def list_documents(self) -> list[DocumentInfo]:
        data = self.store.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        grouped: dict[str, DocumentInfo] = {}

        for metadata in metadatas:
            if not metadata:
                continue
            doc_id = str(metadata.get("doc_id") or "")
            if not doc_id:
                continue
            item = grouped.get(doc_id)
            if item is None:
                item = DocumentInfo(
                    doc_id=doc_id,
                    file=str(metadata.get("source") or "未知文件"),
                    file_type=str(metadata.get("file_type") or ""),
                    chunks=0,
                    pages=int(metadata.get("doc_pages") or 0),
                    size_bytes=int(metadata.get("doc_size") or 0),
                    added_at=metadata.get("added_at"),
                )
                grouped[doc_id] = item
            item.chunks += 1

        return sorted(grouped.values(), key=lambda d: (d.added_at or "", d.file), reverse=True)

    def counts(self) -> tuple[int, int]:
        """(文档数, 切片数)。"""
        data = self.store.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        doc_ids = {m.get("doc_id") for m in metadatas if m}
        return len(doc_ids), len(metadatas)
