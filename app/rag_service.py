"""RAG 核心服务：文档入库与检索问答，与 Web 框架解耦，便于单测。"""

from __future__ import annotations

import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.bm25_index import BM25Index
from app.config import Settings
from app.document_loader import (
    chunk_documents,
    detect_file_type,
    load_pages,
    printed_offset,
    safe_filename,
    sha256_bytes,
)
from app.errors import InvalidFileContent, InvalidQuestion, ModelInvocationError, NoTextExtracted
from app.llm_factory import build_embeddings, build_llm
from app.retrieval import Retriever
from app.schemas import (
    Citation,
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    QueryResponse,
    SearchResponse,
    UploadResponse,
)
from app.vector_store import VectorStore

log = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "根据现有资料无法回答这个问题。请先上传相关文档，或换一种问法再试。"

SYSTEM_PROMPT = """你是一名严谨的规范/企业知识库问答助手，只能依据【参考资料】回答。

回答前先在内部做一次判断，不要输出这个过程：
把问题拆成它真正要的那个事实——具体数值、名称、条件或做法，然后逐条比对【参考资料】，
看资料是否给出了这个事实。

「给出了」的标准是**原文里有这个内容**，不是「和问题一字不差」。以下都算「给出了」：
- 换个说法的同义表述；
- 需要把相邻的一两句合起来才完整；
- 资料用更宽的口径涵盖了所问的具体情形（问「风压值的允许偏差」，资料写了
  「风压值及断面风速值与设计值的偏差不大于 10%」——这就是答案，不要因为资料
  还多写了别的对象就判成没有）。

以下才算「没有」：
- 需要你自己推断、估算、按常识补全；
- 资料给的是相邻概念（问下降速度，资料只写了下降位置）。

判断结果决定怎么答：

一、资料给出了这个事实 → 只依据资料作答。
    数值、单位、条文号必须与原文一致，不得换算、不得四舍五入、不得改写单位；
    适用条件（场所、前提、例外）要一并写出，不要只摘结论。
    每个结论后标 [编号]，可标多个，例如 [1][3]。

二、资料确实没有 → 只输出「根据现有资料无法回答该问题。」，再用一句话说明缺什么
    （例：资料规定了卷帘的下降位置，但未给出下降速度）。
    不要给推断值，不要把相近条款列出来充数，不要用「一般来说」把答案补全。

三、资料之间互相矛盾 → 指出矛盾并分别标注来源，不要自己裁一个答案。

引用口径：
- 标了【条文说明】的片段是对条文的官方解释。正文条文优先；
  **正文没有规定、只有条文说明写了时，条文说明就是有效依据**——照答，并在措辞上
  标明来源性质（「条文说明指出……」）。不要因为它不是正文就判成「资料没有」。
- 引用规范时带上条文号（如「第 7.1.1 条」），比只给页码更可核。
- 用中文回答，先给结论再给依据，不要整段照抄原文。"""

HUMAN_PROMPT = """【参考资料】
{context}

【问题】
{question}"""


class RAGService:
    """一次实例对应一个配置，内部惰性初始化模型与向量库。"""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
        reranker: Any = None,
    ) -> None:
        self.settings = settings
        self._embeddings = embeddings
        self._llm = llm
        self._store: VectorStore | None = None
        self._bm25: BM25Index | None = None
        self._retriever: Retriever | None = None
        self._chain: Any = None
        self._ingest_lock = threading.Lock()
        self._reranker = reranker

    # ------------------------------------------------------------ 惰性依赖
    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = build_embeddings(self.settings)
        return self._embeddings

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(self.settings, self.embeddings)
        return self._store

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.settings)
        return self._bm25

    @property
    def retriever(self) -> Retriever:
        """一条流水线，六档模式共用。写操作会让它的 BM25 分支自动重建。"""
        if self._retriever is None:
            self._retriever = Retriever(
                self.settings,
                store=self.store,
                bm25=self.bm25,
                reranker=self._reranker,
            )
        return self._retriever

    @property
    def chain(self) -> Any:
        """LCEL 链：填充 prompt → 调模型 → 取纯文本。"""
        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
            )
            self._chain = prompt | self.llm | StrOutputParser()
        return self._chain

    # ------------------------------------------------------------ 写入
    def ingest(self, filename: str, data: bytes) -> UploadResponse:
        started = perf_counter()
        file_type = detect_file_type(filename)
        safe_name = safe_filename(filename)

        if not data:
            raise InvalidFileContent("文件内容为空")
        if len(data) > self.settings.max_upload_bytes:
            raise InvalidFileContent(f"文件超过 {self.settings.max_upload_mb} MB 限制")
        if file_type == "pdf" and not data[:1024].lstrip().startswith(b"%PDF"):
            raise InvalidFileContent("文件头不是合法的 PDF，请确认文件未损坏")

        doc_id = sha256_bytes(data)[:16]
        path = Path(self.settings.data_dir) / safe_name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.part")
        tmp_path.write_bytes(data)
        tmp_path.replace(path)  # 原子替换，避免读到写了一半的文件

        with self._ingest_lock:
            pages = load_pages(path, file_type)
            chunks = chunk_documents(pages, self.settings)
            if not chunks:
                raise NoTextExtracted(
                    "没有从文件中提取到文本。若为扫描版 PDF，请先做 OCR 或改用可复制文本的 PDF。"
                )

            total_pages = (
                max(int(p.metadata.get("page") or 0) for p in pages) + 1
                if file_type == "pdf"
                else 1
            )
            offset = printed_offset(pages)
            added_at = datetime.now(UTC).astimezone().isoformat(timespec="seconds")

            for chunk in chunks:
                raw_page = int(chunk.metadata.get("page") or 0)
                raw_page_end = int(chunk.metadata.get("page_end") or raw_page)
                # 页码两个口径都存：page 是 PDF 物理页，page_printed 是页脚印的印刷页。
                # 缺印刷页时用整篇偏移量补；偏移量也不可信就记 0（0 = 无页码，与 TXT 一致），
                # 绝不拿物理页硬充印刷页——gold 标注看的是印刷页，猜错比没有更费人。
                printed = chunk.metadata.get("page_printed")
                printed = (
                    int(printed)
                    if printed is not None
                    else (raw_page + 1 + offset if offset is not None else 0)
                )
                printed_end = chunk.metadata.get("page_printed_end")
                printed_end = (
                    int(printed_end)
                    if printed_end is not None
                    else (raw_page_end + 1 + offset if offset is not None else printed)
                )
                chunk.metadata.update(
                    {
                        "doc_id": doc_id,
                        "source": safe_name,
                        "file_type": file_type,
                        # 对外统一 1 基页码；TXT 用 0 表示「无页码」
                        "page": raw_page + 1 if file_type == "pdf" else 0,
                        "page_end": raw_page_end + 1 if file_type == "pdf" else 0,
                        "page_printed": printed if file_type == "pdf" else 0,
                        "page_printed_end": printed_end if file_type == "pdf" else 0,
                        "page_offset": offset if offset is not None else 0,
                        "doc_pages": total_pages,
                        "doc_size": len(data),
                        "added_at": added_at,
                    }
                )

            ids = [f"{doc_id}:{chunk.metadata['chunk_index']}" for chunk in chunks]
            replaced = self.store.upsert_document(chunks, ids, doc_id, safe_name) > 0
            # 向量库变了，BM25 必须立刻跟上：两条索引读同一份语料是硬约束。
            # 放在写锁里做，是为了不出现「新切片已入库、BM25 还少一篇」的中间态。
            self._rebuild_bm25(f"入库 {safe_name}")

        elapsed_ms = int((perf_counter() - started) * 1000)
        log.info(
            "文档入库完成：file=%s doc_id=%s chunks=%d pages=%d replaced=%s 用时=%dms",
            safe_name,
            doc_id,
            len(chunks),
            total_pages,
            replaced,
            elapsed_ms,
        )
        return UploadResponse(
            doc_id=doc_id,
            file=safe_name,
            file_type=file_type,  # type: ignore[arg-type]
            pages=total_pages,
            chunks=len(chunks),
            replaced=replaced,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------ 仅检索
    def search(
        self,
        question: str,
        top_k: int | None = None,
        mode: str | None = None,
        rerank_budget_ms: int | None = None,
    ) -> SearchResponse:
        """原始排名通道：不过 MIN_SCORE、不调大模型，返回当前档位的 top-K。

        `/query` 返回的 citations 按阈值筛过，用它算 recall 会把「被阈值拦掉的命中」误判成
        「没检索到」，rank 随之失真。评测、扫阈值、跑消融矩阵都靠这条未经裁剪的通道。

        `mode` 可按请求覆盖档位：消融矩阵有六档 × 50 题，如果每档都要改配置重启，
        跑一轮就是六次冷启动，还容易把「重启忘了改回来」写进报告。
        """
        question = (question or "").strip()
        if not question:
            raise InvalidQuestion("问题不能为空")

        result = self.retriever.retrieve(question, top_k, mode, rerank_budget_ms)
        citations = [
            self._citation(index, hit, used=False)
            for index, hit in enumerate(result.hits, start=1)
        ]
        log.debug(
            "检索完成：mode=%s k=%d 命中=%d 用时=%dms %s",
            result.mode,
            top_k or self.settings.top_k,
            len(citations),
            result.elapsed_ms,
            result.timings,
        )
        return SearchResponse(
            question=question,
            hits=citations,
            elapsed_ms=result.elapsed_ms,
            mode=result.mode,
            timings=result.timings,
            degraded=result.degraded,
            rerank_candidates=result.rerank_candidates,
        )

    # ------------------------------------------------------------ 问答
    def ask(self, question: str, top_k: int | None = None, mode: str | None = None) -> QueryResponse:
        started = perf_counter()
        question = (question or "").strip()
        if not question:
            raise InvalidQuestion("问题不能为空")

        k = top_k or self.settings.top_k
        result = self.retriever.retrieve(question, k, mode)

        # 拒答的判据从「逐片比阈值」改成「问句与语料的最好余弦」。
        # 原因是融合后的分数不再是余弦：RRF 分在 1/(60+1)≈0.016 量级，拿 0.30 去逐片比，
        # 整批命中会被一次性砍光，混合检索等于没接。而这个阈值本来就该回答
        # 「这个问题跟语料沾不沾边」——是查询级判断，不是片段级。
        gate = result.gate_score
        if gate is not None and gate < self.settings.min_score:
            log.info(
                "未检索到可信资料：mode=%s 最高余弦=%.2f < min_score=%.2f -> 拒答",
                result.mode,
                gate,
                self.settings.min_score,
            )
            return QueryResponse(
                answer=NO_CONTEXT_ANSWER,
                citations=[],
                model=self.settings.effective_llm_model,
                elapsed_ms=int((perf_counter() - started) * 1000),
                abstained=True,
                mode=result.mode,
            )

        # 逐片滤噪只在纯向量档生效。
        # 这一档的 score 本身就是余弦，跟旧 baseline 同口径，保留原行为才能让数字可比；
        # 其余档位的 score 是 RRF/重排分，量级与余弦无关，拿 0.30 去比只会把融合的收益
        # 一次性滤掉——BM25 捞回来的那些词面命中恰恰是加这一路的意义所在。
        hits = list(result.hits)
        if result.mode == "vector":
            hits = [hit for hit in hits if hit.score >= self.settings.min_score]

        if not hits:
            log.info("库里没有可用的切片：mode=%s -> 拒答", result.mode)
            return QueryResponse(
                answer=NO_CONTEXT_ANSWER,
                citations=[],
                model=self.settings.effective_llm_model,
                elapsed_ms=int((perf_counter() - started) * 1000),
                abstained=True,
                mode=result.mode,
            )

        citations: list[Citation] = []
        blocks: list[str] = []
        for index, hit in enumerate(hits, start=1):
            citation = self._citation(index, hit)
            citations.append(citation)
            # 喂给模型的标签要带上条文号与「正文/条文说明」：模型据此能分清
            # 「规范要求」和「仅供参考的解释」，也才有东西可引用。
            label = f"{citation.file} 第 {citation.page} 页" if citation.page else citation.file
            if citation.page_printed and citation.page_printed != citation.page:
                label += f"（印刷页 {citation.page_printed}）"
            if citation.article:
                label += f" 第 {citation.article} 条"
            if citation.section_type == "explanation":
                # 只标性质，不下「非规范要求」这个结论。原先那句让模型把条文说明当成禁区：
                # 正文没写、答案只在说明里时，它宁可拒答也不引用（n10 / n23 就是这么丢的）。
                # 性质判断交给 Prompt，标签只负责把事实摆出来。
                label += "【条文说明】"
            blocks.append(f"[{index}] 来源：{label}\n{citation.content}")

        try:
            answer = self.chain.invoke({"context": "\n\n".join(blocks), "question": question})
        except Exception as exc:  # noqa: BLE001 - 统一转成 502，避免泄漏堆栈
            log.exception("大模型调用失败")
            raise ModelInvocationError(f"大模型调用失败：{exc}") from exc

        # 模型回答里的 [n] 视为实际引用，用于区分「检索命中」和「答案真的用了」
        referenced = {int(match) for match in re.findall(r"\[(\d{1,2})\]", answer or "")}
        for citation in citations:
            citation.used = citation.index in referenced

        elapsed_ms = int((perf_counter() - started) * 1000)
        log.info(
            "问答完成：mode=%s k=%d 命中=%d 引用=%d 检索=%dms 总计=%dms",
            result.mode,
            k,
            len(citations),
            sum(c.used for c in citations),
            result.elapsed_ms,
            elapsed_ms,
        )
        return QueryResponse(
            answer=(answer or "").strip(),
            citations=citations,
            model=self.settings.effective_llm_model,
            elapsed_ms=elapsed_ms,
            mode=result.mode,
        )

    # ------------------------------------------------------------ 引用
    @staticmethod
    def _citation(index: int, hit: Any, used: bool = False) -> Citation:
        """检索命中 → 对外的引用对象。

        两套字段缺一不可：分路分数（vector_score/bm25_score/…）回答「这条是哪一路召回的、
        各段打多少分」，排查「向量排第 1、重排后掉出前 4」没有它只能靠猜；条文字段
        （article/section_type/section_path）回答「这段在原文里是什么性质」，模型靠它才能
        分清「规范要求」和「条文说明」。缺字段一律降级，不让前端拿到 500。
        """
        metadata = hit.document.metadata or {}
        printed = int(metadata.get("page_printed") or 0)
        return Citation(
            index=index,
            doc_id=str(metadata.get("doc_id") or ""),
            file=str(metadata.get("source") or "未知文件"),
            page=int(metadata.get("page") or 0) or None,
            page_printed=printed or None,
            chunk_index=int(metadata.get("chunk_index") or 0),
            score=hit.score,
            content=hit.document.page_content,
            used=used,
            vector_score=hit.vector_score,
            bm25_score=hit.bm25_score,
            rrf_score=hit.fused_score,
            rerank_score=hit.rerank_score,
            channels=list(hit.channels),
            article=str(metadata.get("article") or ""),
            section_type=str(metadata.get("section_type") or "main"),
            section_path=str(metadata.get("section_path") or ""),
        )

    # ------------------------------------------------------------ 索引同步
    def _rebuild_bm25(self, reason: str) -> None:
        """语料变更后重建稀疏索引。失败不能让入库整体失败——向量已经写进去了。"""
        try:
            self.bm25.rebuild(self.store.dump_all(), reason=reason)
        except Exception:  # noqa: BLE001
            log.exception("BM25 重建失败，下次检索将重试")
            self.bm25.invalidate()

    # ------------------------------------------------------------ 管理
    def documents(self) -> DocumentListResponse:
        documents = self.store.list_documents()
        return DocumentListResponse(total=len(documents), documents=documents)

    def delete(self, doc_id: str) -> DeleteResponse:
        deleted, source = self.store.delete_document(doc_id)
        self._rebuild_bm25(f"删除 {doc_id}")
        removed = self._remove_file(source)
        log.info("删除文档：doc_id=%s chunks=%d 删除原文件=%s", doc_id, deleted, removed)
        return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted, file_removed=removed)

    def _remove_file(self, source: str | None) -> bool:
        """删除 data 目录下的原始文件；路径越界或文件不存在时静默跳过，避免误删其他目录。"""
        if not source:
            return False
        data_root = Path(self.settings.data_dir).resolve()
        path = (data_root / Path(source).name).resolve()
        if data_root not in path.parents or not path.is_file():
            return False
        path.unlink()
        return True

    def counts(self) -> tuple[int, int]:
        """(文档数, 切片数)。启动校验与重载核对都用这一条，避免两处口径不一致。"""
        return self.store.counts()

    def reload(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """重载向量库，返回 (重载前, 重载后) 的 (文档数, 切片数)。

        用于换了 `CHROMA_DIR` / 集合名 / 整库替换之后，不重启服务就让新库生效。
        前后计数一起返回：数字没变说明库本来就是这个样子，不必再怀疑缓存。

        注意它**不重建 BM25**：稀疏索引有自己的指纹，换库后要重启或跑
        `scripts/build_bm25.py`，否则混合档会拿旧语料的索引去融合新语料。
        """
        before = self.counts()
        self.store.reload()
        after = self.counts()
        log.info("向量库重载：%d 篇/%d 片 -> %d 篇/%d 片", *before, *after)
        return before, after

    def health(self) -> HealthResponse:
        missing = self.settings.missing_keys()
        documents = chunks = 0
        try:
            documents, chunks = self.store.counts()
            store_ok = True
        except Exception:  # noqa: BLE001 - 健康检查本身不能抛异常
            log.exception("向量库健康检查失败")
            store_ok = False

        # BM25 的规模必须跟向量库一致，否则说明有人绕过了入库流程直接改了向量库。
        # 这里主动触发一次 ensure：健康检查的价值就在「能提前发现不一致」，
        # 报一个恒为 0 的空索引等于什么都没检查。
        bm25_size = 0
        bm25_synced = True
        try:
            self.bm25.ensure(self.store.dump_all)
            bm25_size = self.bm25.size
            bm25_synced = bm25_size == chunks or not chunks
        except Exception:  # noqa: BLE001
            log.exception("BM25 索引健康检查失败")
            bm25_synced = False

        return HealthResponse(
            status="ok" if store_ok and not missing and bm25_synced else "degraded",
            version=self.settings.app_version,
            collection=self.settings.collection_name,
            documents=documents,
            chunks=chunks,
            llm_provider=self.settings.llm_provider,
            llm_model=self.settings.effective_llm_model,
            embedding_model=self.settings.embedding_model,
            missing_env=missing,
            retrieval_mode=self.settings.retrieval_mode,
            bm25_chunks=bm25_size,
            # 只在检索流水线已经建好时才问重排模型，否则 /health 会把 2.2GB 权重拉起来
            rerank_loaded=self.retriever.reranker.loaded if self._retriever else False,
        )
