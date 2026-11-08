"""拒答的三条路径 + 阈值边界。

三条路径必须能分开观测，否则会得出「拒答能力为 0」这种结论，然后去调一个本来没问题的阈值：
1. 阈值路径：最高分低于 MIN_SCORE，服务直接拒答，压根不调模型；
2. 模型路径：分数过线，模型自己说「无法回答」；
3. 检索空：库里没有候选。

阈值边界单独钉住：等于阈值必须放行（`>=`），否则 0.30 会变成 0.31，边界上的题全靠运气。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.config import Settings
from app.rag_service import NO_CONTEXT_ANSWER, RAGService
from tests.fakes import FakeChatModel, FakeEmbeddings

MODEL_ABSTAIN_REPLY = "根据现有资料无法回答：资料中没有提到这一点。"


class StubStore:
    """固定分数的假向量库：只控制「检索回来什么」，不碰真实 Chroma。"""

    def __init__(self, hits: list[tuple[Document, float]]) -> None:
        self._hits = hits
        self.searches = 0

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        self.searches += 1
        return self._hits[:k]

    def counts(self) -> tuple[int, int]:
        return (1 if self._hits else 0, len(self._hits))

    def dump_all(self) -> list[tuple[str, dict, str]]:
        """稀疏索引的数据源。检索流水线建对象时就会取它的引用，缺了直接 AttributeError。"""
        return [
            (
                f"{doc.metadata.get('doc_id')}:{doc.metadata.get('chunk_index')}",
                dict(doc.metadata),
                doc.page_content,
            )
            for doc, _ in self._hits
        ]


def make_chunk(text: str = "入职满一年的员工，每年享有 5 天带薪年假。") -> Document:
    return Document(
        page_content=text,
        metadata={"doc_id": "d1", "chunk_index": 0, "source": "handbook.txt", "page": 1},
    )


def build_service(
    tmp_path: Path,
    hits: list[tuple[Document, float]],
    min_score: float,
    reply: str = "员工每年享有 5 天带薪年假 [1]。",
) -> tuple[RAGService, FakeChatModel, StubStore]:
    settings = Settings(
        _env_file=None,
        data_dir=str(tmp_path / "data"),
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="abstain_kb",
        dashscope_api_key="k",
        deepseek_api_key="k",
        min_score=min_score,
        # 这几个用例钉的是「纯向量档的逐片阈值」：这一档的 score 本身就是余弦，
        # 与旧 baseline 同口径。融合档的拒答判据是查询级 gate，见 tests/test_retrieval_merge.py。
        retrieval_mode="vector",
    )
    llm = FakeChatModel(reply=reply)
    service = RAGService(settings, embeddings=FakeEmbeddings(), llm=llm)
    stub = StubStore(hits)
    service._store = stub  # noqa: SLF001 - 测试里替换惰性依赖，避免真起 Chroma
    return service, llm, stub


def test_score_equal_to_threshold_is_answered(tmp_path: Path) -> None:
    """边界必须放行：score == MIN_SCORE 时不能拒答。"""
    service, llm, _ = build_service(tmp_path, [(make_chunk(), 0.5)], min_score=0.5)

    response = service.ask("年假有几天？")

    assert response.abstained is False
    assert len(response.citations) == 1
    assert llm.calls == 1


def test_score_below_threshold_abstains_without_calling_llm(tmp_path: Path) -> None:
    """阈值路径：差 0.01 也算噪声，直接拒答，省一次模型调用。"""
    service, llm, _ = build_service(tmp_path, [(make_chunk(), 0.4999)], min_score=0.5)

    response = service.ask("年假有几天？")

    assert response.abstained is True
    assert response.answer == NO_CONTEXT_ANSWER
    assert response.citations == []
    assert llm.calls == 0


def test_empty_retrieval_abstains(tmp_path: Path) -> None:
    """检索空：库里没东西可给，同样不走模型。"""
    service, llm, stub = build_service(tmp_path, [], min_score=0.3)

    response = service.ask("年假有几天？")

    assert response.abstained is True
    assert response.answer == NO_CONTEXT_ANSWER
    assert stub.searches == 1
    assert llm.calls == 0


def test_model_can_abstain_even_when_threshold_passes(tmp_path: Path) -> None:
    """模型路径：分数过线（abstained=False），但答案里是「无法回答」。

    这正是「阈值拒答率」和「模型拒答率」必须分开统计的原因——
    只看 abstained 会把这一条记成「没有拒答」。
    """
    service, llm, _ = build_service(
        tmp_path, [(make_chunk(), 0.92)], min_score=0.3, reply=MODEL_ABSTAIN_REPLY
    )

    response = service.ask("公司年会在哪办？")

    assert response.abstained is False
    assert "无法回答" in response.answer
    assert llm.calls == 1


def test_zero_threshold_disables_the_gate(tmp_path: Path) -> None:
    """MIN_SCORE=0 等于关闭闸门：分数再低也交给模型判断。"""
    service, llm, _ = build_service(tmp_path, [(make_chunk(), 0.01)], min_score=0.0)

    response = service.ask("年假有几天？")

    assert response.abstained is False
    assert llm.calls == 1


def test_only_hits_above_threshold_become_citations(tmp_path: Path) -> None:
    """过线的进引用、不过线的不进——引用列表本身也是一道闸门。"""
    hits = [(make_chunk("高分片段"), 0.88), (make_chunk("低分片段"), 0.2)]
    service, _, _ = build_service(tmp_path, hits, min_score=0.5)

    response = service.ask("年假有几天？")

    assert len(response.citations) == 1
    assert response.citations[0].content == "高分片段"
