"""测试夹具：每个用例使用独立的临时目录与假模型。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag_service import RAGService
from tests.fakes import FakeChatModel, FakeEmbeddings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=str(tmp_path / "data"),
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="test_kb",
        dashscope_api_key="test-dashscope-key",
        deepseek_api_key="test-deepseek-key",
    )


@pytest.fixture()
def llm() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture()
def service(settings: Settings, llm: FakeChatModel) -> RAGService:
    return RAGService(settings, embeddings=FakeEmbeddings(), llm=llm)


@pytest.fixture()
def client(service: RAGService) -> TestClient:
    app = create_app(service.settings, service=service)
    with TestClient(app) as test_client:
        yield test_client
