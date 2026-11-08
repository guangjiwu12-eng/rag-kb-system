"""BM25 稀疏索引：jieba 分词 + rank_bm25 纯内存打分，随入库流程重建并落盘 pkl。

为什么是「随入库重建」而不是独立进程
--------------------------------
语料只有千级切片，全量重建一次大约百毫秒级；独立进程/独立服务换来的是「索引和向量库
可能不一致」这个新故障源。写作路径上（ingest/delete）同步重建，读路径上永远看到同一个
版本的语料，这个取舍在本项目的规模下没有悬念。

重建为什么不是 O(N) 分词
---------------------
分词是唯一贵的步骤（jieba 处理 1 MB 文本约 1 秒）。所以进程内按 `key → (文本指纹,
token)` 缓存一份，重建时只对**新增或内容变了**的切片重新分词，其余直接复用。
上传一篇文档 = 只分词这一篇，其余 1300 片走缓存。

落盘与失效
---------
pkl 里存的是语料指纹（每个切片 key + 文本的 sha1 汇总），不是「切片总数」。
用总数判断失效会在「删一篇加一篇」之后静默拿到旧索引——个数一样，内容全错。
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import Settings
from app.tokenize import TOKENIZER_VERSION, tokenize

log = logging.getLogger(__name__)

INDEX_VERSION = 2


@dataclass(frozen=True)
class Bm25Chunk:
    key: str  # 与向量库 id 同一口径：doc_id:chunk_index
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Bm25Hit:
    chunk: Bm25Chunk
    score: float


def chunk_key(metadata: dict[str, Any]) -> str:
    return f"{metadata.get('doc_id') or ''}:{int(metadata.get('chunk_index') or 0)}"


def fingerprint(rows: list[tuple[str, dict[str, Any], str]]) -> str:
    digest = hashlib.sha1()
    for key, _meta, text in sorted(rows, key=lambda row: row[0]):
        digest.update(key.encode("utf-8"))
        digest.update(hashlib.sha1(text.encode("utf-8")).digest())
    return digest.hexdigest()[:16]


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class BM25Index:
    """线程安全的 BM25 索引。空库时 search 返回空列表，不抛异常。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._path = Path(settings.bm25_index_path)
        self._lock = threading.RLock()
        self._chunks: list[Bm25Chunk] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._fp: str | None = None
        # key → (文本指纹, token)：跨重建复用，避免每次入库把全库重分一遍
        self._token_cache: dict[str, tuple[str, list[str]]] = {}

    # ------------------------------------------------------------------ 状态
    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def ready(self) -> bool:
        return self._bm25 is not None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._chunks),
                "fingerprint": self._fp,
                "persisted": self._path.is_file(),
                "path": str(self._path),
            }

    # ------------------------------------------------------------------ 构建
    def ensure(self, supplier: Any) -> None:
        """首次检索时加载：先比磁盘 pkl，再比内存，都对不上才重新分词。

        `supplier` 是一个返回 `[(key, metadata, text), ...]` 的可调用对象——由调用方
        决定语料从哪来（生产是 Chroma，测试是内存列表），索引本身不依赖向量库。
        """
        if self._bm25 is not None:
            return
        with self._lock:
            if self._bm25 is not None:
                return
            rows = supplier()
            current = fingerprint(rows)
            if self._fp == current:
                return
            payload = self._read_pickle()
            if payload and payload.get("fingerprint") == current:
                self._restore(payload)
                log.info("BM25 索引已从磁盘加载：%d 切片 | %s", len(self._chunks), self._path)
                return
            self.rebuild(rows, persist=True, reason="启动时语料指纹不匹配")

    def rebuild(
        self,
        rows: list[tuple[str, dict[str, Any], str]],
        *,
        persist: bool = True,
        reason: str = "",
    ) -> None:
        """全量重建。分词走缓存，只有新切片真的会被切。"""
        with self._lock:
            fresh: dict[str, tuple[str, list[str]]] = {}
            chunks: list[Bm25Chunk] = []
            tokens: list[list[str]] = []
            reused = 0
            for key, metadata, text in rows:
                digest = _text_hash(text)
                cached = self._token_cache.get(key)
                if cached and cached[0] == digest:
                    token_list = cached[1]
                    reused += 1
                else:
                    token_list = tokenize(text)
                fresh[key] = (digest, token_list)
                chunks.append(Bm25Chunk(key=key, text=text, metadata=metadata))
                tokens.append(token_list)

            self._token_cache = fresh
            self._chunks = chunks
            self._tokens = tokens
            self._fp = fingerprint(rows)
            self._bm25 = (
                BM25Okapi(tokens, k1=self._settings.bm25_k1, b=self._settings.bm25_b)
                if tokens
                else None
            )
            if persist:
                self._write_pickle()
            log.info(
                "BM25 重建完成：%d 切片（复用分词 %d，新分词 %d）%s",
                len(chunks),
                reused,
                len(chunks) - reused,
                f"| {reason}" if reason else "",
            )

    def invalidate(self) -> None:
        """语料被外部改动后强制下次检索重建（测试与脚本用）。"""
        with self._lock:
            self._bm25 = None
            self._fp = None

    # ------------------------------------------------------------------ 检索
    def search(self, question: str, k: int = 20) -> list[Bm25Hit]:
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            query_tokens = tokenize(question)
            if not query_tokens:
                return []
            scores = self._bm25.get_scores(query_tokens)
            # argpartition 只挑 top-k 再排序：1300 片时差别不大，语料上十万时是数量级差别
            order = scores.argsort()[::-1][:k]
            hits = [
                Bm25Hit(chunk=self._chunks[int(index)], score=float(scores[int(index)]))
                for index in order
                if scores[int(index)] > 0
            ]
        return hits

    # ------------------------------------------------------------------ 持久化
    def _payload(self) -> dict[str, Any]:
        return {
            "version": INDEX_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "fingerprint": self._fp,
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "k1": self._settings.bm25_k1,
            "b": self._settings.bm25_b,
            "chunks": [(chunk.key, chunk.metadata, chunk.text) for chunk in self._chunks],
            "tokens": self._tokens,
        }

    def _write_pickle(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(f"{self._path.name}.part")
            tmp.write_bytes(pickle.dumps(self._payload(), protocol=pickle.HIGHEST_PROTOCOL))
            tmp.replace(self._path)  # 原子替换：进程被杀也不会留下半个 pkl
        except OSError as exc:  # 落盘失败不该让入库整体失败，内存索引仍然可用
            log.warning("BM25 索引落盘失败（内存索引仍可用）：%s", exc)

    def _read_pickle(self) -> dict[str, Any] | None:
        if not self._path.is_file():
            return None
        try:
            payload = pickle.loads(self._path.read_bytes())  # noqa: S301 - 本项目自产文件
        except Exception as exc:  # noqa: BLE001 - 版本升级后旧 pkl 反序列化失败属正常
            log.warning("BM25 索引读取失败，将重建：%s", exc)
            return None
        if payload.get("version") != INDEX_VERSION:
            log.info("BM25 索引版本过期（%s → %s），将重建", payload.get("version"), INDEX_VERSION)
            return None
        if payload.get("tokenizer_version") != TOKENIZER_VERSION:
            log.info("分词口径已变，BM25 索引将重建")
            return None
        # k1/b 变了，idf 也就变了，旧索引不能将就
        if payload.get("k1") != self._settings.bm25_k1 or payload.get("b") != self._settings.bm25_b:
            log.info("BM25 参数已变，索引将重建")
            return None
        return payload

    def _restore(self, payload: dict[str, Any]) -> None:
        chunks: list[Bm25Chunk] = []
        tokens: list[list[str]] = []
        cache: dict[str, tuple[str, list[str]]] = {}
        for (key, metadata, text), token_list in zip(
            payload["chunks"], payload["tokens"], strict=False
        ):
            chunks.append(Bm25Chunk(key=key, text=text, metadata=metadata))
            tokens.append(token_list)
            cache[key] = (_text_hash(text), token_list)
        self._chunks = chunks
        self._tokens = tokens
        self._token_cache = cache
        self._fp = str(payload.get("fingerprint") or "")
        self._bm25 = (
            BM25Okapi(tokens, k1=self._settings.bm25_k1, b=self._settings.bm25_b)
            if tokens
            else None
        )
