"""集中配置：所有可调参数均来自环境变量或项目根目录的 .env。"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "RAG 知识库问答服务"
    app_version: str = "2.1.0"

    # ---------- 安全 ----------
    api_key: str | None = Field(
        default=None, description="设置后所有写/检索接口必须带 X-API-Key；留空则开放（仅本机）"
    )
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    @property
    def allowed_origin_list(self) -> list[str]:
        """逗号分隔的 CORS 白名单。

        这里刻意用字符串而不是 list[str]：pydantic-settings 对复杂类型会先按 JSON 解析，
        逗号写法会直接把服务启动卡死。
        """
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    # ---------- 模型 ----------
    llm_provider: Literal["deepseek", "dashscope"] = "deepseek"
    deepseek_api_key: str | None = None
    dashscope_api_key: str | None = None
    llm_model: str = "deepseek-chat"

    # ---------- 向量化 ----------
    # dashscope：走云端 text-embedding-*，开箱可用
    # sentence-transformers：本地跑（bge-m3 等），不装 torch 时只有这条分支不可用
    embedding_backend: Literal["dashscope", "sentence-transformers"] = "dashscope"
    embedding_model: str = "text-embedding-v1"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    # torch 的线程数。CPU 推理上这个值影响很大：本机实测 12 → 16 线程，
    # bge-m3 吞吐从 259 字/秒涨到 579 字/秒。留空则用 torch 默认值。
    embedding_num_threads: int | None = Field(default=None, ge=1, le=256)
    embedding_normalize: bool = True
    embedding_max_seq_length: int | None = Field(default=None, ge=64)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout: int = Field(default=60, ge=5, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # ---------- 存储 ----------
    data_dir: str = "./data"
    chroma_dir: str = "./chroma_db"
    collection_name: str = "kb_documents"

    # ---------- 切片与检索 ----------
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0)
    # auto：能认出条文号就按条切（规范类语料），认不出再退回定长切片
    chunk_mode: Literal["auto", "article", "fixed"] = "auto"
    # 条文数少于该值的文档视为「不是规范」，按定长切（实践日志、会议纪要这类）
    article_min_markers: int = Field(default=12, ge=1)
    # 光看条文数不够：实践日志里「1.21 地点……」这种日期行长得也像条文号，
    # 13 个就能过上面那个阈值。规范的行首条文号密度在 10%~20%，日志只有 2% 左右，
    # 用密度区分比用绝对数稳。
    article_min_density: float = Field(default=0.06, ge=0.0, le=1.0)
    # 单条过长才二次切分；规范里 4 位编号的条款通常 100-300 字，1200 已经够宽
    article_max_chars: int = Field(default=1200, ge=200)
    article_sub_overlap: int = Field(default=80, ge=0)
    dedupe_enabled: bool = True
    # 近重复审计阈值：只用来「报出来给人看」，不参与合并（合并只认完全一致）
    dedupe_threshold: float = Field(default=0.90, ge=0.5, le=1.0)
    top_k: int = Field(default=6, ge=1, le=20)
    # 默认 6 而不是 4：实测 n11 的正确答案稳定排第 5，k=4 直接截断 → 模型看不到原文就拒答。
    # 这类失败既不算误拒（阈值口径）也不算幻觉（无答案题），只能靠加宽 k 消除。
    # 代价是 Prompt 变长；用 eval 的 strict/loose 两套数验证这 2 条值不值。
    # 低于该相似度的命中视为「没检索到」，直接拒答，不把噪声喂给大模型
    min_score: float = Field(default=0.30, ge=0.0, le=1.0)
    max_upload_mb: int = Field(default=20, ge=1, le=200)

    # ---------- 多路召回与融合 ----------
    # 当前生效的检索档位。六档对应消融矩阵的六个点，`/search` 可以按请求覆盖，
    # 这样扫一遍矩阵不用改配置重启服务。
    retrieval_mode: Literal[
        "vector",
        "bm25",
        "hybrid_sum",
        "hybrid_rrf",
        "vector_rerank",
        "hybrid_rrf_rerank",
    ] = "hybrid_rrf"
    # 默认不重排：本机 cross-encoder 20 条候选要 7 秒，占不住交互预算（docs/03-retrieval.md §6）。
    # 重排不是「打开就好」的开关，它是一笔延迟买精度的交易，默认档位必须体现这个取舍。
    # 每一路召回取多少条，也是融合/重排后保留的深度。所有档位都对齐这个深度，
    # 消融矩阵比的才是「融合与重排」，而不是「谁拿到的候选更多」。
    recall_candidates: int = Field(default=20, ge=4, le=100)
    # RRF 的平滑常数：60 是原论文的经验值，作用是压住「第 1 名和第 2 名」之间的悬殊，
    # 让多路都认可的结果浮上来。
    rrf_k: int = Field(default=60, ge=1, le=500)
    # BM25 参数：k1 控制词频饱和，b 控制文档长度惩罚，都是 rank_bm25 的默认值
    bm25_k1: float = Field(default=1.5, ge=0.0, le=5.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    # 「+BM25」那档用的分数加权（归一化后相加）。留一半一半，是为了让这档只体现
    # 「多了一路信号」，而不是偷偷调过权重。
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # BM25 索引落盘位置：随入库流程重建，重启直接读，不重新分词
    bm25_index_path: str = "./data/bm25_index.pkl"

    # ---------- 重排 ----------
    # CPU 上的唯一瓶颈就是这一段。max_length 直接决定 20 个候选的总 token 量，
    # 是延迟预算的第一个旋钮，比换模型便宜得多。
    # 默认写成相对路径：仓库在 WSL 与 Windows 两侧都要能跑，绝对路径写死必然有一侧失效，
    # 真正的位置由 .env 的 RERANK_MODEL 给出。
    rerank_model: str = "models/bge-reranker-v2-m3"
    # 指向 int8 量化后的 ONNX 目录时走 onnxruntime，否则回落到 torch
    rerank_onnx_path: str | None = None
    rerank_device: str = "cpu"
    # 窗口必须覆盖整片，不是「能省则省」：切片 token 中位数 502、p90 574，
    # 给 256 会把 63% 的切片砍掉一半以上，重排看不到答案就按套话打分——实测 hit@4
    # 因此从 92.5% 掉到 82.5%（见 docs/03-retrieval.md §6）。640 覆盖最大切片（652）。
    rerank_max_length: int = Field(default=640, ge=32, le=8192)
    rerank_batch_size: int = Field(default=8, ge=1, le=64)
    # 只重排融合结果的前 N 条，其余保持融合顺序垫在后面。
    # cross-encoder 的成本与候选数线性相关，所以这是除 max_length 外最有效的旋钮；
    # 但被砍掉的候选没有第二次机会，N 太小等于白做融合。
    rerank_top_n: int = Field(default=20, ge=1, le=100)
    rerank_num_threads: int | None = Field(default=None, ge=1, le=256)
    # 重排的时间闸门（毫秒）。0 = 不设限（离线评测要的就是「质量上限」那一档）。
    # 非 0 时按实测的单条成本换算「这个预算排得动几条」，排不动就整档跳过——
    # 宁可退回融合顺序（p95 稳在预算内），也不要为了排序多花 10 秒。
    rerank_budget_ms: int = Field(default=0, ge=0, le=60000)
    # 闸门算出「只排得动 2 条」时干脆不排：只重排两三条等于给已经排好的前两名互相调个位，
    # 收益约等于零，却照样付了 2 条的延迟。
    rerank_min_candidates: int = Field(default=5, ge=1, le=100)
    # 模型缺失或推理报错时是否安静降级为「融合顺序」。检索可用性优先于排序质量。
    rerank_fail_open: bool = True

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def effective_llm_model(self) -> str:
        """LLM_MODEL 留空或沿用了另一家的模型名时，给出一个合理默认值。"""
        provider_prefix = {"deepseek": "deepseek", "dashscope": "qwen"}
        if self.llm_model.lower().startswith(provider_prefix[self.llm_provider]):
            return self.llm_model
        return "deepseek-chat" if self.llm_provider == "deepseek" else "qwen-plus"

    def missing_keys(self) -> list[str]:
        """返回尚未配置、且对应功能需要的环境变量名。"""
        missing: list[str] = []
        # 向量化走本地模型时不需要 DashScope 的任何凭据
        if self.embedding_backend == "dashscope" and not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY")
        if self.llm_provider == "deepseek" and not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if self.llm_provider == "dashscope" and not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
