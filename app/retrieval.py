"""检索流水线：向量 + BM25 双路召回 → 融合 → 重排，六档可按请求切换。

为什么要有「档位」这个概念
------------------------
消融矩阵的说服力全靠一件事：**每一档只改一处**。所以这里的档位不是六份实现，而是同一条
流水线上的三个开关（要不要 BM25、用哪种融合、要不要重排）。切片、向量库、题库、评测口径
全程不动，指标差异才归因得清楚。

```
vector            ─ 向量 top20 ──────────────────────────────► top_k
bm25              ─ BM25  top20 ─────────────────────────────► top_k
hybrid_sum        ─ 向量 + BM25 → 分数归一化加权 ────────────► top_k      （"+BM25"）
hybrid_rrf        ─ 向量 + BM25 → RRF(k=60) ────────────────► top_k      （"+RRF"）
vector_rerank     ─ 向量 top20 → rerank ─────────────────────► top_k      （"+rerank"）
hybrid_rrf_rerank ─ 向量 + BM25 → RRF → rerank ─────────────► top_k
```

深度为什么要对齐
--------------
如果「+BM25」那档因为多了一路召回、候选变成 40，而基线只有 20，那涨的分里有多少是融合
带来的、多少只是候选变多？说不清。所以每一档最后都会被裁到同一个 `depth`。

分数不可比，排名才可比
--------------------
向量给的是余弦，BM25 给的是无上界的词频-逆文档分，RRF 给的是 1/(60+rank) 量级的倒数排名。
把它们硬凑成一个「统一分数」再排序，是这类系统最常见的自欺。RRF 干脆只看名次：不比较
分数大小，只比较「两路是不是都把你排在前面」。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from langchain_core.documents import Document

from app.bm25_index import Bm25Chunk, BM25Index
from app.config import Settings
from app.errors import InvalidRetrievalMode
from app.reranker import CrossEncoderReranker, RerankerUnavailable
from app.vector_store import VectorStore

log = logging.getLogger(__name__)

Mode = Literal["vector", "bm25", "hybrid_sum", "hybrid_rrf", "vector_rerank", "hybrid_rrf_rerank"]

MODES: tuple[str, ...] = (
    "vector",
    "bm25",
    "hybrid_sum",
    "hybrid_rrf",
    "vector_rerank",
    "hybrid_rrf_rerank",
)

NEEDS_BM25 = frozenset({"bm25", "hybrid_sum", "hybrid_rrf", "hybrid_rrf_rerank"})
NEEDS_VECTOR = frozenset({"vector", "hybrid_sum", "hybrid_rrf", "vector_rerank", "hybrid_rrf_rerank"})
NEEDS_RERANK = frozenset({"vector_rerank", "hybrid_rrf_rerank"})


def normalize_mode(mode: str | None, default: str) -> str:
    """空值回落到默认档；拼错的档位直接报错而不是悄悄用默认——评测里静默降级是灾难。"""
    if not mode:
        return default
    candidate = mode.strip().lower()
    if candidate not in MODES:
        raise InvalidRetrievalMode(f"未知检索档位：{mode}（可选：{', '.join(MODES)}）")
    return candidate


@dataclass
class Hit:
    """一条候选。字段按来源分开记，方便在失败清单里回答「它到底是哪一路召回的」。"""

    key: str
    document: Document
    vector_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    rank_vector: int | None = None
    rank_bm25: int | None = None

    @property
    def channels(self) -> tuple[str, ...]:
        names = []
        if self.rank_vector is not None:
            names.append("vector")
        if self.rank_bm25 is not None:
            names.append("bm25")
        return tuple(names)

    @property
    def score(self) -> float:
        """对外展示的分数：谁决定了排序，就报谁的。"""
        for value in (self.rerank_score, self.fused_score, self.vector_score, self.bm25_score):
            if value is not None:
                return round(float(value), 6)
        return 0.0


@dataclass
class RetrievalResult:
    question: str
    mode: str
    hits: list[Hit]
    depth: int
    elapsed_ms: int
    timings: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    # 查询级相关性信号：召回阶段的最高余弦相似度。
    # 注意它取的是**召回池**的最大值，不是最终 top_k 里的最大值——重排把高分片段挤出去
    # 不该反过来触发拒答，那是排序问题，不是「语料里没有」的问题。
    # 纯 BM25 档没有余弦可用，留 None，调用方据此跳过阈值判断而不是拿 0 去比。
    gate_score: float | None = None
    # 实际送进重排的候选数。0 表示这一档不重排；预算不够时会小于 rerank_top_n，
    # 甚至为 0（整档跳过）——评测要能看见「有多少问题真的重排了」，
    # 否则「延迟降了」可能只是「重排没跑」。
    rerank_candidates: int = 0


def _document_of(chunk: Bm25Chunk) -> Document:
    return Document(page_content=chunk.text, metadata=dict(chunk.metadata))


def _minmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-9:
        # 全部同分时给 1.0，避免整列被压成 0 之后权重彻底失效
        return [1.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def _merge_into(target: Hit, extra: Hit) -> None:
    """把另一路的分数并进来。

    这里必须显式判 `is None`，不能用 `a or b`：分数可能是 0.0，而 0.0 是假值——
    用 or 会把「这一路确实召回了、但打了 0 分」悄悄写成「这一路没召回」。
    """
    if target.vector_score is None:
        target.vector_score = extra.vector_score
    if target.bm25_score is None:
        target.bm25_score = extra.bm25_score
    if target.rank_vector is None:
        target.rank_vector = extra.rank_vector
    if target.rank_bm25 is None:
        target.rank_bm25 = extra.rank_bm25


def rrf_fuse(rankings: Sequence[Sequence[Hit]], rrf_k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion：分数 = Σ 1/(k + rank)。

    k=60 的作用是压住「第 1 名对第 2 名」的悬殊差距：k 越大，头部名次之间的分差越小，
    于是「两路都排第 3」会比「一路第 1、另一路没召回」更占优——这正是融合想要的稳健性。
    """
    merged: dict[str, Hit] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.key] = scores.get(hit.key, 0.0) + 1.0 / (rrf_k + rank)
            if hit.key in merged:
                _merge_into(merged[hit.key], hit)  # 合并两路各自的原始分，供展示与排障
            else:
                merged[hit.key] = hit
    for key, hit in merged.items():
        hit.fused_score = round(scores[key], 6)
    return sorted(merged.values(), key=lambda item: item.fused_score or 0.0, reverse=True)


def weighted_fuse(
    vector_hits: Sequence[Hit], bm25_hits: Sequence[Hit], weight: float
) -> list[Hit]:
    """「只用 BM25 但不做 RRF」的那档：两路分数各归一化后加权相加。

    这是 RRF 出现之前的主流做法，也是本矩阵里最该被证伪的一档——归一化是 query 内相对的，
    同一片文档在不同 query 下的「0.8 分」根本不是一回事，加权和于是没有稳定语义。
    把它放进矩阵，就是为了看这个缺陷在真实题库上到底值多少个点。
    """
    merged: dict[str, Hit] = {}

    def absorb(hits: Sequence[Hit]) -> dict[str, float]:
        raw = [hit.vector_score or hit.bm25_score or 0.0 for hit in hits]
        normalized = _minmax(raw)
        result: dict[str, float] = {}
        for hit, value in zip(hits, normalized, strict=False):
            result[hit.key] = value
            if hit.key not in merged:
                merged[hit.key] = hit
            else:
                _merge_into(merged[hit.key], hit)
        return result

    vector_part = absorb(vector_hits)
    bm25_part = absorb(bm25_hits)
    for key, hit in merged.items():
        hit.fused_score = round(
            weight * vector_part.get(key, 0.0) + (1.0 - weight) * bm25_part.get(key, 0.0), 6
        )
    return sorted(merged.values(), key=lambda item: item.fused_score or 0.0, reverse=True)


class Retriever:
    """一条流水线，三个开关。依赖全部注入，便于单测里换成假向量库/假重排器。"""

    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        bm25: BM25Index,
        reranker: CrossEncoderReranker | None = None,
        bm25_supplier: Callable[[], list[tuple[str, dict[str, Any], str]]] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.bm25 = bm25
        self._reranker = reranker
        self._bm25_supplier = bm25_supplier or store.dump_all
        self._rerank_warned = False

    # ------------------------------------------------------------------ 重排器
    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                self.settings.rerank_model,
                onnx_path=self.settings.rerank_onnx_path,
                device=self.settings.rerank_device,
                batch_size=self.settings.rerank_batch_size,
                max_length=self.settings.rerank_max_length,
                num_threads=self.settings.rerank_num_threads,
            )
        return self._reranker

    # ------------------------------------------------------------------ 主流程
    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        mode: str | None = None,
        rerank_budget_ms: int | None = None,
    ) -> RetrievalResult:
        started = perf_counter()
        active = normalize_mode(mode, self.settings.retrieval_mode)
        budget = self.settings.rerank_budget_ms if rerank_budget_ms is None else rerank_budget_ms
        final_k = top_k or self.settings.top_k
        depth = max(final_k, self.settings.recall_candidates)

        timings: dict[str, float] = {}
        degraded: list[str] = []

        vector_hits: list[Hit] = []
        if active in NEEDS_VECTOR:
            mark = perf_counter()
            vector_hits = self._vector_recall(question, depth)
            timings["vector_ms"] = round((perf_counter() - mark) * 1000, 2)

        bm25_hits: list[Hit] = []
        if active in NEEDS_BM25:
            mark = perf_counter()
            bm25_hits = self._bm25_recall(question, depth)
            timings["bm25_ms"] = round((perf_counter() - mark) * 1000, 2)

        mark = perf_counter()
        if active == "vector":
            candidates = vector_hits
        elif active == "bm25":
            candidates = bm25_hits
        elif active == "hybrid_sum":
            candidates = weighted_fuse(
                vector_hits, bm25_hits, self.settings.bm25_weight
            )[:depth]
        else:
            candidates = rrf_fuse([vector_hits, bm25_hits], self.settings.rrf_k)[:depth]
        timings["fuse_ms"] = round((perf_counter() - mark) * 1000, 2)

        rerank_used = 0
        if active in NEEDS_RERANK and candidates:
            mark = perf_counter()
            candidates, note, rerank_used = self._rerank(question, candidates, depth, budget)
            timings["rerank_ms"] = round((perf_counter() - mark) * 1000, 2)
            if note:
                degraded.append(note)

        hits = candidates[:final_k]
        elapsed_ms = int((perf_counter() - started) * 1000)
        timings["total_ms"] = float(elapsed_ms)
        # 召回池里的最高余弦：与最终排序无关，只回答「这个库跟问题沾不沾边」
        gate_score = max((hit.vector_score for hit in vector_hits), default=None)
        log.debug(
            "检索：mode=%s depth=%d 命中=%d 用时=%dms %s",
            active,
            depth,
            len(hits),
            elapsed_ms,
            timings,
        )
        return RetrievalResult(
            question=question,
            mode=active,
            hits=hits,
            depth=depth,
            elapsed_ms=elapsed_ms,
            timings=timings,
            degraded=degraded,
            gate_score=gate_score,
            rerank_candidates=rerank_used,
        )

    # ------------------------------------------------------------------ 两路召回
    def _vector_recall(self, question: str, depth: int) -> list[Hit]:
        hits: list[Hit] = []
        for rank, (document, score) in enumerate(self.store.search(question, depth), start=1):
            metadata = document.metadata or {}
            hits.append(
                Hit(
                    key=f"{metadata.get('doc_id') or ''}:{int(metadata.get('chunk_index') or 0)}",
                    document=document,
                    vector_score=float(score),
                    rank_vector=rank,
                )
            )
        return hits

    def _bm25_recall(self, question: str, depth: int) -> list[Hit]:
        self.bm25.ensure(self._bm25_supplier)
        hits: list[Hit] = []
        for rank, item in enumerate(self.bm25.search(question, depth), start=1):
            hits.append(
                Hit(
                    key=item.chunk.key,
                    document=_document_of(item.chunk),
                    bm25_score=float(item.score),
                    rank_bm25=rank,
                )
            )
        return hits

    # ------------------------------------------------------------------ 重排
    def _rerank(
        self, question: str, candidates: list[Hit], depth: int, budget_ms: int = 0
    ) -> tuple[list[Hit], str, int]:
        """返回 (新排序, 降级说明, 实际重排条数)。

        预算是硬闸门，不是「尽量」：先按实测单条成本换算能排几条，排不动就整档退回融合顺序。
        代价是排序质量，收益是 p95 可预测——交互式问答里，一个 12 秒的「更准」是负分。
        """
        # 只重排前 N 条，尾巴保持融合顺序：N 是延迟预算和排序质量之间唯一的连续旋钮
        head_size = min(self.settings.rerank_top_n, len(candidates))

        if budget_ms > 0:
            affordable = self.reranker.affordable_count(budget_ms)
            head_size = min(head_size, affordable)
            if head_size < self.settings.rerank_min_candidates:
                note = (
                    f"rerank_budget_skip: {budget_ms}ms 只排得动 {head_size} 条"
                    f"（单条 {self.reranker.ms_per_pair or 0:.0f}ms，下限 "
                    f"{self.settings.rerank_min_candidates} 条），改用融合顺序"
                )
                log.info("%s", note)
                return candidates[:depth], note, 0

        head, tail = candidates[:head_size], candidates[head_size:]
        try:
            scored = self.reranker.rerank(question, [hit.document.page_content for hit in head])
        except RerankerUnavailable as exc:
            if not self.settings.rerank_fail_open:
                raise
            # 只喊一次：降级后每次检索都刷一条 warning，日志会被淹掉
            if not self._rerank_warned:
                log.warning("重排不可用，降级为融合顺序：%s", exc)
                self._rerank_warned = True
            return candidates[:depth], f"rerank_unavailable: {exc}", 0

        if self._rerank_warned:
            log.info("重排已恢复")
            self._rerank_warned = False

        reranked = [head[item.index] for item in scored] + tail
        for item in scored:
            head[item.index].rerank_score = item.score
        return reranked[:depth], "", len(head)

    # ------------------------------------------------------------------ 状态
    def status(self) -> dict[str, Any]:
        return {
            "mode": self.settings.retrieval_mode,
            "recall_candidates": self.settings.recall_candidates,
            "rrf_k": self.settings.rrf_k,
            "rerank_top_n": self.settings.rerank_top_n,
            "rerank_budget_ms": self.settings.rerank_budget_ms,
            "bm25": self.bm25.status(),
            # 不加载也要能报出配置：评测报告的抬头必须写「服务端实际用的窗口」，
            # 而不是评测脚本本地的环境变量——这两个值会不一致，而且不一致时报告就错了。
            "rerank": self.reranker.status(),
        }
