"""FastAPI 入口：REST 接口、静态页面、统一错误处理。"""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, Depends, FastAPI, File, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.errors import AppError, FileTooLarge, Unauthorized
from app.logging_conf import configure_logging
from app.rag_service import RAGService
from app.schemas import (
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReloadResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
READ_CHUNK_SIZE = 1024 * 1024

DESCRIPTION = """
基于 LangChain + Chroma 的私有知识库问答服务。

1. `POST /api/v1/documents` 上传 PDF / TXT，自动切片并写入向量库（同时重建 BM25 索引）
2. `POST /api/v1/search` 只检索不生成：返回 top-K 原始命中，不过阈值、不调模型（评测/调参用）
3. `POST /api/v1/query` 检索相关片段，交由大模型生成带引用编号的回答
4. 回答中的 `[n]` 与 `citations[n-1]` 一一对应，可追溯原文与页码
5. `POST /api/v1/admin/reload` 重载向量库：入库脚本是独立进程时，用它让服务看见新索引

检索档位（`mode` 可按请求覆盖，用于跑消融矩阵而不必重启）：

| 档位 | 召回 | 融合 | 重排 |
| --- | --- | --- | --- |
| `vector` | 向量 top20 | - | - |
| `bm25` | BM25 top20 | - | - |
| `hybrid_sum` | 向量 + BM25 | 归一化加权 | - |
| `hybrid_rrf` | 向量 + BM25 | RRF(k=60) | - |
| `vector_rerank` | 向量 top20 | - | bge-reranker-v2-m3 |
| `hybrid_rrf_rerank` | 向量 + BM25 | RRF(k=60) | bge-reranker-v2-m3 |

注意：`/admin/reload` 只重载向量库客户端，**不重建 BM25**。换库之后要让混合档生效，
必须重启服务或跑 `scripts/build_bm25.py`，否则会拿旧语料的稀疏索引去融合新语料。
"""


def get_service(request: Request) -> RAGService:
    return request.app.state.service


def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """API_KEY 未配置时保持开放（本地开发）；配置后写/检索接口必须携带正确的 X-API-Key。"""
    expected: str | None = request.app.state.settings.api_key
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise Unauthorized("缺少或错误的 X-API-Key")


async def read_limited(file: UploadFile, limit: int) -> bytes:
    """边读边计数，超限立即中断，避免把超大文件整个读进内存。"""
    buffer = bytearray()
    while chunk := await file.read(READ_CHUNK_SIZE):
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise FileTooLarge(f"文件超过 {limit // 1024 // 1024} MB 限制")
    return bytes(buffer)


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        missing = settings.missing_keys()
        if missing:
            log.warning("未配置环境变量 %s，相关接口将返回 503", ", ".join(missing))
        if not settings.api_key:
            log.warning("未设置 API_KEY，写/检索接口处于开放模式，请勿暴露到公网")
        log.info(
            "%s v%s 启动 | 集合=%s | LLM=%s(%s) | Embedding=%s | 检索=%s",
            settings.app_name,
            settings.app_version,
            settings.collection_name,
            settings.llm_provider,
            settings.effective_llm_model,
            settings.embedding_model,
            settings.retrieval_mode,
        )
        # 启动即把索引计数打进日志：别的进程入过库、或换了 chroma 目录之后，
        # 「数字对不上」能当场看见，而不是等评测跑出一堆假指标才发现。
        # 这里不 raise——读不出来也不该让健康检查都起不来，只降级告警。
        try:
            documents, chunks = app.state.service.counts()
            log.info("索引就绪：%d 篇 / %d 片", documents, chunks)
            if documents == 0:
                log.warning("索引为空：向量库里没有任何切片，检索会全部返回空结果")
                data_dir = Path(settings.data_dir)
                pending = list(data_dir.iterdir()) if data_dir.is_dir() else []
                if pending:
                    log.warning(
                        "data 目录里有 %d 个文件但索引为空：多半没跑入库，或入库写到了别的目录",
                        len(pending),
                    )
        except Exception:  # noqa: BLE001 - 启动校验失败只告警，不阻断服务
            log.exception("启动时读取索引失败，检索接口可能不可用")
        yield
        log.info("服务已停止")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = service or RAGService(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origin_list,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        started = perf_counter()
        response = await call_next(request)
        elapsed_ms = (perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        if request.url.path.startswith("/api/"):
            log.debug("%s %s -> %s", request.method, request.url.path, response.status_code)
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("%s %s -> %s %s", request.method, request.url.path, exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code},
        )

    router = APIRouter(prefix="/api/v1", tags=["knowledge-base"])

    @router.get("/health", response_model=HealthResponse, summary="健康检查")
    def health(service: RAGService = Depends(get_service)) -> HealthResponse:
        return service.health()

    @router.get("/retrieval", summary="检索链路状态（档位 / BM25 / 重排后端）")
    def retrieval_status(service: RAGService = Depends(get_service)) -> dict:
        """给压测和排障用：一眼看清 BM25 是否与向量库同步、重排走的是 torch 还是 onnx。"""
        return service.retriever.status()

    @router.post(
        "/documents",
        response_model=UploadResponse,
        status_code=201,
        summary="上传并索引 PDF / TXT",
        dependencies=[Depends(verify_api_key)],
    )
    async def upload_document(
        file: UploadFile = File(...),
        service: RAGService = Depends(get_service),
    ) -> UploadResponse:
        data = await read_limited(file, service.settings.max_upload_bytes)
        return await run_in_threadpool(service.ingest, file.filename or "", data)

    @router.get(
        "/documents",
        response_model=DocumentListResponse,
        summary="文档列表",
        dependencies=[Depends(verify_api_key)],
    )
    def list_documents(service: RAGService = Depends(get_service)) -> DocumentListResponse:
        return service.documents()

    @router.delete(
        "/documents/{doc_id}",
        response_model=DeleteResponse,
        summary="删除文档",
        dependencies=[Depends(verify_api_key)],
    )
    def delete_document(doc_id: str, service: RAGService = Depends(get_service)) -> DeleteResponse:
        return service.delete(doc_id)

    @router.post(
        "/search",
        response_model=SearchResponse,
        summary="仅检索，不生成回答",
        dependencies=[Depends(verify_api_key)],
    )
    def search(
        payload: SearchRequest, service: RAGService = Depends(get_service)
    ) -> SearchResponse:
        """返回 top-K 原始命中（含低于 MIN_SCORE 的），不调用大模型。

        `mode` / `rerank_budget_ms` 必须显式往下传：漏传时 Pydantic 会静默忽略请求里的字段，
        六档消融会跑出同一组数，而报错要到写报告时才发现。
        """
        return service.search(
            payload.question, payload.top_k, payload.mode, payload.rerank_budget_ms
        )

    @router.post(
        "/query",
        response_model=QueryResponse,
        summary="检索并回答问题",
        dependencies=[Depends(verify_api_key)],
    )
    def query(payload: QueryRequest, service: RAGService = Depends(get_service)) -> QueryResponse:
        return service.ask(payload.question, payload.top_k, payload.mode)

    @router.post(
        "/admin/reload",
        response_model=ReloadResponse,
        summary="重载向量库（另一进程入库后用）",
        dependencies=[Depends(verify_api_key)],
    )
    def reload_index(service: RAGService = Depends(get_service)) -> ReloadResponse:
        """丢掉缓存的向量库客户端并重建，返回前后计数。

        用在换库场景：改了 `CHROMA_DIR`、换了集合名、或整库替换成另一份索引之后，
        不重启服务就能生效。实测（chromadb 1.5.9）外部进程的写入本来就是实时可见的，
        所以它不是「看见新文档」的必需品；`changed=false` 说明库确实没变。
        """
        before, after = service.reload()
        return ReloadResponse(
            documents_before=before[0],
            chunks_before=before[1],
            documents_after=after[0],
            chunks_after=after[1],
            changed=after != before,
        )

    app.include_router(router)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        page = STATIC_DIR / "index.html"
        if page.is_file():
            return FileResponse(page, media_type="text/html")
        return JSONResponse({"message": "API 已启动，请访问 /docs"}, status_code=404)

    return app


app = create_app()
