"""切分、文件名清洗、编码兜底的单元测试。"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.document_loader import chunk_documents, decode_text, safe_filename
from app.errors import UnsupportedFileType


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_safe_filename_strips_directory_traversal() -> None:
    assert safe_filename("../../etc/passwd.txt") == "passwd.txt"
    assert safe_filename(r"C:\tmp\报告.pdf") == "报告.pdf"
    assert safe_filename("a<b>?.txt") == "a_b.txt"


def test_safe_filename_rejects_empty_name() -> None:
    with pytest.raises(UnsupportedFileType):
        safe_filename("...")


def test_decode_text_falls_back_to_gb18030() -> None:
    assert decode_text("员工年假".encode("gb18030")) == "员工年假"
    assert decode_text("员工年假".encode()) == "员工年假"


def test_chunk_documents_keeps_page_and_global_index() -> None:
    pages = [
        Document(page_content="员工依法享有带薪年假。" * 120, metadata={"page": 0}),
        Document(page_content="差旅报销需要提供发票。", metadata={"page": 1}),
    ]
    chunks = chunk_documents(pages, make_settings(chunk_size=120, chunk_overlap=20))

    assert len(chunks) > 2
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.metadata["page"] for chunk in chunks} == {0, 1}
    assert chunks[-1].metadata["page"] == 1
    assert chunks[-1].page_content == "差旅报销需要提供发票。"
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_chunk_documents_drops_blank_fragments() -> None:
    pages = [Document(page_content="\n\n   \n\n只有一句话。\n\n   ", metadata={"page": 0})]
    chunks = chunk_documents(pages, make_settings(chunk_size=100, chunk_overlap=10))

    assert [chunk.page_content for chunk in chunks] == ["只有一句话。"]


def test_chunk_documents_ignores_empty_page() -> None:
    pages = [
        Document(page_content="第一页有内容。", metadata={"page": 0}),
        Document(page_content="", metadata={"page": 1}),
    ]
    chunks = chunk_documents(pages, make_settings(chunk_size=100, chunk_overlap=10))

    assert [chunk.page_content for chunk in chunks] == ["第一页有内容。"]
