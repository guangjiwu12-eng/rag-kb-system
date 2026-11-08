"""向量库热加载：换库后不重启也能生效，且前后计数可见。

注意这里**不**宣称「另一个进程写入后必须重载才能看见」——实测 chromadb 1.5.9 的写入
本来就是实时可见的。重载解决的是换库（改目录 / 换集合 / 整库替换）场景。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag_service import RAGService
from tests.fakes import FakeChatModel, FakeEmbeddings

TXT = "员工手册\n\n入职满一年的员工，每年享有 5 天带薪年假。\n"


def make_client(tmp_path: Path, **overrides) -> tuple[TestClient, RAGService]:
    settings = Settings(
        _env_file=None,
        data_dir=str(tmp_path / "data"),
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="reload_kb",
        dashscope_api_key="k",
        deepseek_api_key="k",
        **overrides,
    )
    service = RAGService(settings, embeddings=FakeEmbeddings(), llm=FakeChatModel())
    return TestClient(create_app(settings, service=service)), service


def test_reload_rebuilds_client_and_reports_counts(tmp_path: Path) -> None:
    client, service = make_client(tmp_path)
    with client:
        client.post("/api/v1/documents", files={"file": ("a.txt", TXT.encode(), "text/plain")})
        client_handle = service.store._store  # noqa: SLF001 - 就是要看底层 Chroma 实例是否换代

        response = client.post("/api/v1/admin/reload")

    assert response.status_code == 200
    body = response.json()
    assert body["documents_before"] == body["documents_after"] == 1
    assert body["chunks_before"] == body["chunks_after"] >= 1
    assert body["changed"] is False
    assert service.store._store is not client_handle  # noqa: SLF001


def test_reload_keeps_documents_searchable(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        client.post("/api/v1/documents", files={"file": ("a.txt", TXT.encode(), "text/plain")})
        client.post("/api/v1/admin/reload")
        hits = client.post("/api/v1/search", json={"question": "年假有几天", "top_k": 3}).json()

    assert hits["hits"], "重载后检索不应变空"


def test_reload_requires_api_key_when_configured(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, api_key="secret")
    with client:
        assert client.post("/api/v1/admin/reload").status_code == 401
        ok = client.post("/api/v1/admin/reload", headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
