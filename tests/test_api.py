"""接口层测试：覆盖上传、检索问答、引用、幂等与错误分支。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag_service import NO_CONTEXT_ANSWER, RAGService
from tests.fakes import CountingEmbeddings, FailingEmbeddings, FakeChatModel, FakeEmbeddings

TXT = """员工手册（节选）

入职满一年的员工，每年享有 5 天带薪年假。年假可以分次使用，但每次不少于半天。

差旅报销需要在出差结束后 10 个工作日内提交发票，逾期需要说明原因。
"""

UPLOAD = {"file": ("handbook.txt", TXT.encode("utf-8"), "text/plain")}


def make_settings(tmp_path: Path, name: str, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=str(tmp_path / name / "data"),
        chroma_dir=str(tmp_path / name / "chroma"),
        collection_name=name,
        dashscope_api_key="k",
        deepseek_api_key="k",
        **overrides,
    )


def build_app(
    settings: Settings,
    embeddings: object | None = None,
    llm: FakeChatModel | None = None,
    **client_kwargs,
) -> TestClient:
    service = RAGService(
        settings, embeddings=embeddings or FakeEmbeddings(), llm=llm or FakeChatModel()
    )
    return TestClient(create_app(settings, service=service), **client_kwargs)


def upload(client: TestClient, name: str = "handbook.txt", content: str = TXT):
    return client.post(
        "/api/v1/documents",
        files={"file": (name, content.encode("utf-8"), "text/plain")},
    )


def test_health_reports_empty_collection(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["collection"] == "test_kb"
    assert body["documents"] == 0
    assert body["chunks"] == 0
    assert body["missing_env"] == []


def test_upload_txt_returns_metadata(client: TestClient) -> None:
    response = upload(client)

    assert response.status_code == 201
    body = response.json()
    assert body["file"] == "handbook.txt"
    assert body["file_type"] == "txt"
    assert body["pages"] == 1
    assert body["chunks"] >= 1
    assert body["replaced"] is False
    assert len(body["doc_id"]) == 16


def test_query_returns_answer_with_citations(client: TestClient, llm: FakeChatModel) -> None:
    upload(client)

    response = client.post("/api/v1/query", json={"question": "年假有几天？"})

    assert response.status_code == 200
    body = response.json()
    assert "年假" in body["answer"]
    assert body["model"] == "deepseek-chat"
    assert llm.calls == 1

    citations = body["citations"]
    assert citations, "应返回检索到的引用片段"
    top = citations[0]
    assert top["index"] == 1
    assert top["file"] == "handbook.txt"
    assert top["page"] is None
    assert "年假" in top["content"]
    assert 0.0 <= top["score"] <= 1.0
    # 假模型的回答里带了 [1]，因此第一条应被标记为已引用
    assert top["used"] is True
    assert any(not item["used"] for item in citations) or len(citations) == 1


def test_query_without_documents_does_not_call_llm(
    client: TestClient, llm: FakeChatModel
) -> None:
    response = client.post("/api/v1/query", json={"question": "年假有几天？"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["citations"] == []
    assert llm.calls == 0


def test_reupload_same_content_is_idempotent(client: TestClient) -> None:
    first = upload(client).json()
    second = upload(client).json()

    assert second["doc_id"] == first["doc_id"]
    assert second["chunks"] == first["chunks"]
    assert second["replaced"] is True

    documents = client.get("/api/v1/documents").json()
    assert documents["total"] == 1
    assert documents["documents"][0]["chunks"] == first["chunks"]


def test_reupload_same_name_with_new_content_replaces_old_chunks(
    client: TestClient,
) -> None:
    upload(client)
    changed = TXT + "\n\n加班需要提前在系统中申请。"
    second = upload(client, content=changed).json()

    documents = client.get("/api/v1/documents").json()
    assert documents["total"] == 1
    assert documents["documents"][0]["doc_id"] == second["doc_id"]
    assert documents["documents"][0]["chunks"] == second["chunks"]


def test_delete_document_removes_chunks(client: TestClient) -> None:
    doc_id = upload(client).json()["doc_id"]

    response = client.delete(f"/api/v1/documents/{doc_id}")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] >= 1
    assert client.get("/api/v1/documents").json()["total"] == 0
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 404


def test_unsupported_file_type_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("report.docx", b"not a real docx", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_file_type"


def test_fake_pdf_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("broken.pdf", b"hello world", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_file_content"


def test_upload_too_large(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=str(tmp_path / "data"),
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="tiny_kb",
        max_upload_mb=1,
        dashscope_api_key="k",
        deepseek_api_key="k",
    )
    service = RAGService(settings, embeddings=FakeEmbeddings(), llm=FakeChatModel())
    app = create_app(settings, service=service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents",
            files={"file": ("big.txt", b"a" * (1024 * 1024 + 10), "text/plain")},
        )

    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


def test_upload_filename_cannot_escape_data_dir(client: TestClient) -> None:
    data_dir = Path(client.app.state.settings.data_dir)

    response = upload(client, name="../../evil.txt")
    body = response.json()

    assert response.status_code == 201
    assert body["file"] == "evil.txt"
    assert (data_dir / "evil.txt").is_file()
    assert not (data_dir.parent / "evil.txt").exists()


def test_blank_question_is_rejected(client: TestClient) -> None:
    assert client.post("/api/v1/query", json={"question": ""}).status_code == 422
    response = client.post("/api/v1/query", json={"question": "   "})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_question"


def test_missing_api_key_returns_503(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=str(tmp_path / "data"),
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="nokey_kb",
    )
    service = RAGService(settings)  # 不注入假模型，触发真实的配置校验
    app = create_app(settings, service=service)

    with TestClient(app) as client:
        health = client.get("/api/v1/health").json()
        assert health["status"] == "degraded"
        assert set(health["missing_env"]) == {"DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY"}

        response = client.post("/api/v1/query", json={"question": "年假有几天？"})
        assert response.status_code == 503
        assert response.json()["code"] == "model_not_configured"


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "RAG 知识库问答" in response.text


def test_process_time_header(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert "X-Process-Time-Ms" in response.headers


# ------------------------------------------------------------------ 安全与健壮性
def test_write_apis_require_api_key(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, "auth_kb", api_key="secret-key")

    with build_app(settings) as client:
        anonymous = client.post("/api/v1/documents", files=UPLOAD)
        assert anonymous.status_code == 401
        assert anonymous.json()["code"] == "unauthorized"

        wrong = client.post("/api/v1/documents", files=UPLOAD, headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401

        correct = client.post(
            "/api/v1/documents", files=UPLOAD, headers={"X-API-Key": "secret-key"}
        )
        assert correct.status_code == 201

        # 健康检查保持开放，便于 Docker HEALTHCHECK 与前端状态探测
        assert client.get("/api/v1/health").status_code == 200

        # 同名但不同内容的第二次上传要能拿到 key
        changed = {"file": ("handbook.txt", (TXT + "\n补充一句。").encode(), "text/plain")}
        assert (
            client.post(
                "/api/v1/documents", files=changed, headers={"X-API-Key": "secret-key"}
            ).status_code
            == 201
        )


def test_cors_only_accepts_localhost_origin(client: TestClient) -> None:
    foreign = client.get("/api/v1/health", headers={"Origin": "http://evil.example"})
    assert foreign.headers.get("access-control-allow-origin") is None

    local = client.get("/api/v1/health", headers={"Origin": "http://localhost:8000"})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:8000"


def test_low_score_hits_are_abstained(tmp_path: Path) -> None:
    """相似度全部低于阈值时拒答，且不调用大模型（省钱、也不编造）。"""
    settings = make_settings(tmp_path, "threshold_kb", min_score=0.99)
    llm = FakeChatModel()

    with build_app(settings, llm=llm) as client:
        client.post("/api/v1/documents", files=UPLOAD)
        response = client.post("/api/v1/query", json={"question": "年假有几天？"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["citations"] == []
    assert body["abstained"] is True
    assert llm.calls == 0


def test_delete_also_removes_source_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, "cleanup_kb")
    stored = Path(settings.data_dir) / "handbook.txt"

    with build_app(settings) as client:
        doc_id = client.post("/api/v1/documents", files=UPLOAD).json()["doc_id"]
        assert stored.is_file()
        response = client.delete(f"/api/v1/documents/{doc_id}")

    assert response.status_code == 200
    assert response.json()["file_removed"] is True
    assert not stored.exists()


def test_identical_reupload_skips_reembedding(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, "idempotent_kb")
    embeddings = CountingEmbeddings()

    with build_app(settings, embeddings=embeddings) as client:
        first = client.post("/api/v1/documents", files=UPLOAD).json()
        assert embeddings.document_calls == 1

        second = client.post("/api/v1/documents", files=UPLOAD).json()

    assert second["doc_id"] == first["doc_id"]
    assert second["chunks"] == first["chunks"]
    assert second["replaced"] is True
    assert embeddings.document_calls == 1, "内容未变化却重新向量化了，白烧 Embedding 的钱"


def test_embedding_failure_keeps_existing_document(tmp_path: Path) -> None:
    """先写后删：新文档向量化失败时，旧的切片不能被提前删掉。"""
    settings = make_settings(tmp_path, "durable_kb")
    embeddings = FailingEmbeddings()

    with build_app(settings, embeddings=embeddings, raise_server_exceptions=False) as client:
        first = client.post("/api/v1/documents", files=UPLOAD).json()
        changed = {"file": ("handbook.txt", (TXT + "\n补充：加班需提前申请。").encode(), "text/plain")}
        failed = client.post("/api/v1/documents", files=changed)
        documents = client.get("/api/v1/documents").json()

    assert failed.status_code == 500
    assert documents["total"] == 1
    assert documents["documents"][0]["doc_id"] == first["doc_id"]


def test_search_returns_raw_hits_without_threshold(tmp_path: Path) -> None:
    """阈值调高到 0.99 时 /query 会拒答，但 /search 必须照样返回原始 top-K。

    这是评测能不能成立的分水岭：拿 /query 的 citations 算召回，被阈值拦掉的命中
    会被误判成「没检索到」，rank 直接失真。
    """
    settings = make_settings(tmp_path, "search_kb", min_score=0.99)
    llm = FakeChatModel()

    with build_app(settings, llm=llm) as client:
        client.post("/api/v1/documents", files=UPLOAD)

        queried = client.post("/api/v1/query", json={"question": "年假有几天？"}).json()
        searched = client.post(
            "/api/v1/search", json={"question": "年假有几天？", "top_k": 20}
        ).json()

    assert queried["abstained"] is True, "阈值 0.99 下 /query 应当拒答"
    assert queried["citations"] == []
    assert searched["hits"], "/search 不能被阈值过滤，否则 recall 无从计算"
    top = searched["hits"][0]
    assert top["index"] == 1
    assert top["file"] == "handbook.txt"
    assert top["used"] is False
    assert "年假" in top["content"]
    assert top["score"] < 0.99, "这条命中本来就低于阈值，正是要验的对象"
    assert llm.calls == 0, "/search 不该调用大模型"


def test_search_respects_top_k_and_validation(client: TestClient) -> None:
    upload(client)

    limited = client.post("/api/v1/search", json={"question": "年假", "top_k": 1})
    too_many = client.post("/api/v1/search", json={"question": "年假", "top_k": 99})
    blank = client.post("/api/v1/search", json={"question": "   "})

    assert limited.status_code == 200
    assert len(limited.json()["hits"]) == 1
    assert too_many.status_code == 422
    assert blank.status_code == 400
    assert blank.json()["code"] == "invalid_question"
