"""文档解析与切片：PDF（逐页）/ TXT（整篇）→ 带元数据的文本块。

切片有两条路：能认出条文号的规范类语料按「条」切（详见 `app/corpus.py` 的取舍说明），
认不出的文档退回中文标点递归定长。两条路产出的元数据字段一致，下游无需分支。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import corpus
from app.config import Settings
from app.errors import EmptyFile, InvalidFileContent, UnsupportedFileType

SUPPORTED_TYPES: dict[str, str] = {".pdf": "pdf", ".txt": "txt"}

# 中文优先的分隔符，逐级退让，尽量避免把一句话切断
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "! ", "? ", "; ", ". ", " ", ""]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_UNSAFE_CHARS = re.compile(r"[^\w.\- ()（）]")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_SOFT_HYPHEN = re.compile(r"-\n(?=[a-z])")


def detect_file_type(filename: str) -> str:
    """根据扩展名判断文档类型，不支持的类型直接拒绝。"""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise UnsupportedFileType(f"仅支持 PDF / TXT 文件，当前为：{suffix or '未知类型'}")
    return SUPPORTED_TYPES[suffix]


def safe_filename(filename: str) -> str:
    """去掉路径与控制字符，避免 ../ 之类的文件名把文件写到数据目录之外。"""
    name = Path((filename or "").replace("\\", "/")).name
    name = unicodedata.normalize("NFC", _CONTROL_CHARS.sub("", name)).strip().strip(".")
    if not name:
        raise UnsupportedFileType("文件名不合法")
    suffix = Path(name).suffix.lower()
    stem = _UNSAFE_CHARS.sub("_", Path(name).stem).strip("_ ") or "document"
    return f"{stem[:80]}{suffix}"


def decode_text(data: bytes) -> str:
    """TXT 编码兜底：UTF-8（含 BOM）→ GB18030 → 带替换的 UTF-8。"""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    text = _SOFT_HYPHEN.sub("", text)  # 英文 PDF 的连字符换行
    text = _TRAILING_SPACES.sub("\n", text)
    text = _EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pages(path: Path, file_type: str) -> list[Document]:
    """把文件读成「页」级别的 Document 列表（TXT 视作一页）。

    页脚页码在这一层剥掉：它既不该进切片正文（「第 5 页」混进正文会稀释向量），
    又是印刷页码的唯一来源。剥掉之后把页脚上的数字单独记进 `page_printed`。
    """
    if file_type == "pdf":
        try:
            raw_pages = PyPDFLoader(str(path)).load()
        except Exception as exc:  # noqa: BLE001 - 第三方解析异常统一转为业务错误
            raise InvalidFileContent(f"PDF 解析失败，文件可能已损坏或被加密：{exc}") from exc

        pages: list[Document] = []
        for page in raw_pages:
            content, printed = corpus.split_page_furniture(clean_text(page.page_content or ""))
            metadata = dict(page.metadata)
            metadata["page"] = int(metadata.get("page") or 0)  # pypdf 的页码从 0 开始
            metadata["page_printed"] = printed
            # 空页也保留：用于统计真实页数，切片阶段会自然跳过
            pages.append(Document(page_content=content, metadata=metadata))
        return pages

    text = clean_text(decode_text(path.read_bytes()))
    if not text:
        raise EmptyFile("TXT 文件没有可读取的文本内容")
    return [Document(page_content=text, metadata={"page": 0, "source": path.name, "page_printed": None})]


def _page_rows(pages: list[Document]) -> list[tuple[str, int, int | None]]:
    """(正文, 物理页 1 基, 印刷页)；TXT 整篇算一页。"""
    rows = []
    for page in pages:
        page_no = int(page.metadata.get("page") or 0) + 1
        rows.append((page.page_content, page_no, page.metadata.get("page_printed")))
    return rows


def printed_offset(pages: list[Document]) -> int | None:
    """整篇的「印刷页 - 物理页」偏移量；页脚缺失或不一致时为 None。"""
    return corpus.printed_page_offset(corpus.to_lines(_page_rows(pages)))


def _fixed_chunks(pages: list[Document], settings: Settings) -> list[Document]:
    """定长兜底：给没有条文结构的文档（实践日志、会议纪要）用。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
        strip_whitespace=True,
        keep_separator=True,
    )

    chunks: list[Document] = []
    for page in pages:
        for piece in splitter.split_text(page.page_content):
            piece = piece.strip()
            if len(piece) < 2:
                continue
            metadata = dict(page.metadata)
            metadata["chunk_mode"] = "fixed"
            metadata["chunk_index"] = len(chunks)
            # 定长切片不做跨页合并，起止页就是本页；补上是为了让元数据字段统一，
            # 下游（评测、前端）不用为两种口径写两套分支。
            metadata.setdefault("page_end", metadata.get("page") or 0)
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks


def _article_chunks(blocks: list[corpus.ArticleBlock], settings: Settings) -> list[Document]:
    """按条切：条文说明单独标类型，重复编号的块直接丢。"""
    chunks: list[Document] = []
    for block in blocks:
        if block.kind == "article" and block.section_type == "duplicate":
            # 同一编号同一段文字出现第二次：目录块或页面重叠，留着只会跟正本抢排名
            continue
        for part, piece in enumerate(
            corpus.split_long(block.text, settings.article_max_chars, settings.article_sub_overlap)
        ):
            piece = piece.strip()
            if len(piece) < 6:
                continue
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "chunk_mode": "article",
                        "chunk_index": len(chunks),
                        # 内部统一用 0 基物理页，出库时再加 1（与历史版本口径一致）
                        "page": block.page - 1,
                        "page_end": block.page_end - 1,
                        "page_printed": block.printed,
                        "page_printed_end": block.printed_end,
                        "article": block.number or "",
                        "section_path": block.section_path,
                        "section_type": block.section_type,
                        "block_kind": block.kind,
                        "part": part,
                        "char_len": len(piece),
                    },
                )
            )
    return chunks


def _drop_duplicates(chunks: list[Document], settings: Settings) -> list[Document]:
    """片内去重：完全重复的整片只留一份（模糊重复不并，见 corpus.dedupe 的说明）。"""
    if not settings.dedupe_enabled or len(chunks) < 2:
        return chunks
    merged = corpus.dedupe([c.page_content for c in chunks])
    kept: list[Document] = []
    for index, chunk in enumerate(chunks):
        target = merged[index]
        if target is None:
            kept.append(chunk)
            continue
        keeper = chunks[target]
        keeper.metadata["dup_in_doc"] = int(keeper.metadata.get("dup_in_doc") or 0) + 1
    for position, chunk in enumerate(kept):
        chunk.metadata["chunk_index"] = position
    return kept


def chunk_documents(pages: list[Document], settings: Settings) -> list[Document]:
    """按配置选择切片口径，产出带 `page / page_printed / article / section_type` 的切片。"""
    if settings.chunk_mode != "fixed":
        lines = corpus.to_lines(_page_rows(pages))
        markers = sum(1 for line in lines if corpus.classify(line.text)[0] == "article")
        density = markers / max(len(lines), 1)
        wanted = settings.chunk_mode == "article" or (
            markers >= settings.article_min_markers and density >= settings.article_min_density
        )
        if wanted and markers:
            blocks = corpus.build_blocks(lines)
            corpus.tag_sections(blocks)
            article_chunks = _article_chunks(blocks, settings)
            if article_chunks:
                return _drop_duplicates(article_chunks, settings)
            if settings.chunk_mode == "article":
                # 显式要求按条切却切不出来：退定长，但把口径写进元数据，别让人以为是条切
                fallback = _fixed_chunks(pages, settings)
                for chunk in fallback:
                    chunk.metadata["chunk_mode"] = "fixed-fallback"
                return fallback

    return _drop_duplicates(_fixed_chunks(pages, settings), settings)
