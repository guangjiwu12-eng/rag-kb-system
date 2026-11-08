"""合并检索栈与语料栈后的回归测试。

这里钉的是**人工裁决过的语义**，不是实现细节。每一条对应一次真实踩坑：
- `mode` 漏声明 → Pydantic 静默忽略 → 六档消融跑出同一组数，且不报错；
- 融合档沿用逐片阈值 → RRF 分量级 0.016 被 0.30 一次砍光 → 混合检索白接；
- Citation 只留一套字段 → 要么排查不了排序，要么模型分不清「规范要求」和「条文说明」；
- 两套索引各自建 → 融合时按 key 对不上。

少任何一条，将来有人改回去都不会有测试拦他。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.config import Settings
from app.rag_service import RAGService
from app.schemas import SearchRequest
from tests.fakes import FakeChatModel, FakeEmbeddings


def make_chunk(
    text: str = "消防控制室应实行每日 24h 专人值班制度。",
    *,
    article: str = "7.1.1",
    section_type: str = "main",
    chunk_index: int = 0,
) -> Document:
    """`chunk_index` 必须逐条不同：融合按 `doc_id:chunk_index` 去重，撞 key 会被并成一条。"""
    return Document(
        page_content=text,
        metadata={
            "doc_id": "d1",
            "chunk_index": chunk_index,
            "source": "GB25201-2010.txt",
            "page": 12,
            "page_printed": 9,
            "article": article,
            "section_type": section_type,
            "section_path": "检测 > 一般要求",
        },
    )


class StubStore:
    """固定分数的假向量库，同时充当 BM25 的数据源。"""

    def __init__(self, hits: list[tuple[Document, float]]) -> None:
        self._hits = hits

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self._hits[:k]

    def counts(self) -> tuple[int, int]:
        return (1 if self._hits else 0, len(self._hits))

    def dump_all(self) -> list[tuple[str, dict, str]]:
        """用 `doc_id:chunk_index` 当 key，与真实 Chroma 的 id 规则一致。

        两套索引按同一个 key 对齐是硬约束：这里写错就测不出「融合时对不上」的问题。
        """
        return [
            (
                f"{doc.metadata.get('doc_id')}:{doc.metadata.get('chunk_index')}",
                dict(doc.metadata),
                doc.page_content,
            )
            for doc, _ in self._hits
        ]


def build_service(
    tmp_path: Path,
    hits: list[tuple[Document, float]],
    *,
    mode: str = "hybrid_rrf",
    min_score: float = 0.30,
) -> tuple[RAGService, FakeChatModel]:
    settings = Settings(
        _env_file=None,
        data_dir=str(tmp_path / "data"),
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="merge_kb",
        dashscope_api_key="k",
        deepseek_api_key="k",
        min_score=min_score,
        retrieval_mode=mode,
    )
    llm = FakeChatModel(reply="按要求执行 [1]。")
    service = RAGService(settings, embeddings=FakeEmbeddings(), llm=llm)
    service._store = StubStore(hits)  # noqa: SLF001 - 测试替换惰性依赖，避免真起 Chroma
    return service, llm


def test_search_request_carries_mode_and_budget() -> None:
    """`mode` 与 `rerank_budget_ms` 必须在请求模型里显式声明。

    Pydantic 对未声明字段默认忽略：漏了它们，消融脚本照发请求、服务端照返 200，
    但六档跑的是同一档位，报告里的对比全是同一个数。
    """
    payload = SearchRequest(question="值班制度", mode="bm25", rerank_budget_ms=400)

    assert payload.mode == "bm25"
    assert payload.rerank_budget_ms == 400


def test_hybrid_mode_does_not_filter_chunk_by_cosine_threshold(tmp_path: Path) -> None:
    """融合档不再逐片比 MIN_SCORE。

    RRF 分量级是 1/(60+rank)≈0.016，拿 0.30 逐片比会把整批命中一次砍光——
    加 BM25 这一路就白加了。拒答交给查询级 gate，逐片滤噪只在纯向量档保留。
    """
    hits = [
        (make_chunk("高分片段", chunk_index=0), 0.88),
        (make_chunk("词面命中但余弦低", chunk_index=1), 0.20),
    ]
    service, _ = build_service(tmp_path, hits, mode="hybrid_rrf", min_score=0.5)

    response = service.ask("值班制度")

    assert response.abstained is False
    assert len(response.citations) == 2


def test_vector_mode_still_filters_chunk_by_cosine_threshold(tmp_path: Path) -> None:
    """纯向量档保留原来的逐片过滤，这样它和旧 baseline 同口径、数字可比。"""
    hits = [
        (make_chunk("高分片段", chunk_index=0), 0.88),
        (make_chunk("低分片段", chunk_index=1), 0.20),
    ]
    service, _ = build_service(tmp_path, hits, mode="vector", min_score=0.5)

    response = service.ask("值班制度")

    assert len(response.citations) == 1
    assert response.citations[0].content == "高分片段"


def test_citation_carries_channel_scores_and_article_fields(tmp_path: Path) -> None:
    """一条 Citation 必须同时带「哪一路召回的」和「原文里是什么性质」。

    vector_score/bm25_score/channels 用来排查排序；article/section_type/section_path/
    page_printed 用来让模型分清正文条文与条文说明、让人核到印刷页。缺任一套都算漏。
    """
    service, _ = build_service(tmp_path, [(make_chunk(article="7.1.1"), 0.9)], mode="vector")

    citation = service.search("值班制度", top_k=1).hits[0]

    assert citation.vector_score == 0.9
    assert "vector" in citation.channels
    assert citation.article == "7.1.1"
    assert citation.section_type == "main"
    assert citation.section_path == "检测 > 一般要求"
    assert citation.page == 12
    assert citation.page_printed == 9


def test_explanation_chunks_are_labelled_for_the_model(tmp_path: Path) -> None:
    """条文说明必须打上标签再喂给模型，否则模型会把它当成规范要求来引用。

    这个标签只存在于拼给模型的上下文里，不落在 Citation 字段上——所以断言的是
    FakeChatModel 收到的 prompt，而不是返回值。
    """
    hits = [(make_chunk("这是对条文的解释", section_type="explanation"), 0.9)]
    service, llm = build_service(tmp_path, hits, mode="vector")

    service.ask("值班制度")

    assert llm.calls == 1
    assert "【条文说明】" in llm.prompts[0]
