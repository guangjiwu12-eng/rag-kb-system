"""接口出入参模型（Pydantic v2）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """一条引用：来自哪个文件的第几页、第几条，以及原文内容。

    页码有两个口径，都返回：
    - `page` 是 PDF 物理页（第几个页面对象），程序拿它对得上；
    - `page_printed` 是纸面/页脚印的页码，人拿它对得上。
    两者在带封面、扉页的标准里会差一个常数，混用就是页码猜谜。缺页脚时为 null。
    """

    index: int = Field(description="正文中 [n] 的编号")
    doc_id: str
    file: str
    page: int | None = Field(default=None, description="PDF 物理页（从 1 开始）；TXT 为 null")
    page_printed: int | None = Field(default=None, description="印刷页码（页脚），未知为 null")
    chunk_index: int = Field(description="该文件内的切片序号，从 0 开始")
    # 语义随档位变化：纯向量档是余弦，BM25 档是词频分，混合档是 RRF 分，重排档是重排分。
    # 想看「原始余弦是多少」要看 vector_score——混淆这两个是评测归因出错的常见来源。
    score: float = Field(
        description="决定当前排序的分数（随档位变化：余弦 / 词频 / RRF / 重排）"
    )
    content: str = Field(description="命中的原文片段")
    used: bool = Field(default=False, description="答案是否引用了该片段")

    # 下面几个字段只在对应档位有值，用来回答「这条到底是哪一路召回的、各段打多少分」。
    # 排查「明明向量排第 1，重排之后掉出前 4」这类问题时，没有这些字段只能靠猜。
    vector_score: float | None = Field(default=None, description="向量余弦相似度")
    bm25_score: float | None = Field(default=None, description="BM25 原始分（无上界）")
    rrf_score: float | None = Field(default=None, description="RRF 融合分 1/(k+rank) 累加")
    rerank_score: float | None = Field(default=None, description="cross-encoder 重排分（sigmoid）")
    channels: list[str] = Field(default_factory=list, description="命中的召回通道：vector / bm25")
    article: str = Field(default="", description="条文号，如 7.1.1；非条文类文档为空")
    section_type: str = Field(
        default="main",
        description="main=正文条文 / explanation=条文说明 / front=封面前言 / prose=无编号正文",
    )
    section_path: str = Field(default="", description="条文所属层级，如「检测 > 一般要求」")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: str | None = Field(default=None, description="临时覆盖检索档位，便于对比调试")


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    elapsed_ms: int
    abstained: bool = Field(
        default=False, description="是否因检索不到可信资料而拒答（此时未调用大模型）"
    )
    mode: str | None = Field(default=None, description="本次实际使用的检索档位")


class SearchRequest(BaseModel):
    """仅检索请求。`mode` 让消融矩阵不用改配置重启服务。

    `mode` 与 `rerank_budget_ms` 是消融矩阵的入口：缺了它们，六档会跑出同一组数。
    曾因为 Pydantic 静默忽略未声明字段而白跑一轮，所以这两个字段的声明不能省。
    """

    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: str | None = Field(
        default=None,
        description="vector / bm25 / hybrid_sum / hybrid_rrf / vector_rerank / hybrid_rrf_rerank",
    )
    rerank_budget_ms: int | None = Field(
        default=None,
        ge=0,
        le=60000,
        description=(
            "重排时间闸门（毫秒），覆盖服务端配置。0 = 不设限，用于跑「质量上限」那一档；"
            "填具体值则模拟线上预算，排不动会整档跳过并在 degraded 里说明"
        ),
    )


class SearchResponse(BaseModel):
    """top-K 全部命中，**包含低于 MIN_SCORE 的片段**，且不调用大模型。

    这是评测与调阈值用的原始通道：`/query` 只返回过阈值的片段，
    拿它算 recall 会把被阈值拦掉的命中算成「没检索到」，rank 直接失真。
    """

    question: str
    hits: list[Citation]
    elapsed_ms: int
    mode: str | None = Field(default=None, description="本次实际使用的检索档位")
    timings: dict[str, float] = Field(
        default_factory=dict, description="分段耗时：vector_ms / bm25_ms / fuse_ms / rerank_ms"
    )
    degraded: list[str] = Field(
        default_factory=list, description="降级说明（例如重排模型不可用，已退回融合顺序）"
    )
    rerank_candidates: int = Field(
        default=0,
        description="实际送进重排的候选数（0 = 未重排；受 rerank_top_n 与延迟预算双重限制）",
    )


class UploadResponse(BaseModel):
    doc_id: str
    file: str
    file_type: Literal["pdf", "txt"]
    pages: int
    chunks: int
    replaced: bool = Field(description="是否覆盖了同名或同内容的旧切片")
    elapsed_ms: int


class DocumentInfo(BaseModel):
    doc_id: str
    file: str
    file_type: str
    chunks: int
    pages: int
    size_bytes: int
    added_at: str | None = None


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int
    file_removed: bool = Field(default=False, description="是否一并删除了 data 目录下的原始文件")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    collection: str
    documents: int
    chunks: int
    llm_provider: str
    llm_model: str
    embedding_model: str
    missing_env: list[str]
    retrieval_mode: str | None = Field(default=None, description="当前检索档位")
    bm25_chunks: int = Field(default=0, description="BM25 索引的切片数，应与 chunks 一致")
    rerank_loaded: bool = Field(default=False, description="重排模型是否已加载进内存")


class ReloadResponse(BaseModel):
    """重载向量库的结果。前后计数一起返回，是为了让「加载没加载到」可判定。"""

    documents_before: int
    chunks_before: int
    documents_after: int
    chunks_after: int
    changed: bool = Field(description="计数是否变化；False 表示索引本来就是最新的")
