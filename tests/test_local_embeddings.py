"""本地 Embedding 的路径判据。

只测「路径不存在要当场报错」这一条——它是踩过的坑，不是理论风险。写的是路径但路径
不存在时，transformers 不会报错，而是当成 HF 仓库名去联网下载，网络不通就卡在重试上。
表现是「加载 Embedding」那一行之后二十分钟没有下文，一个向量都没写。

这些测试**不**加载模型：只调 `_resolve_source`，torch 全程不 import。
"""

from __future__ import annotations

import pytest

from app.local_embeddings import TransformerEmbeddings


def _embeddings(model: str) -> TransformerEmbeddings:
    return TransformerEmbeddings(model_name_or_path=model)


@pytest.mark.parametrize(
    "model",
    [
        "/home/nobody/models/bge-m3",  # WSL 绝对路径，Windows 侧 python 看不到
        "./models/不存在的模型",
        "../上一级/models/bge-m3",
        "~/models/bge-m3",
        "D:/nowhere/bge-m3",
        r"\\server\share\bge-m3",
    ],
)
def test_missing_path_fails_fast(model: str) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        _embeddings(model)._resolve_source()
    message = str(excinfo.value)
    assert "路径不存在" in message
    # 报错要能直接照着改，不能只说「失败了」
    assert "HF_HUB_OFFLINE" in message


def test_existing_path_is_returned() -> None:
    # 项目根目录一定存在，用它当「存在的路径」
    assert _embeddings(".")._resolve_source() == "."


@pytest.mark.parametrize("model", ["BAAI/bge-m3", "bge-m3"])
def test_repo_id_is_left_alone(model: str) -> None:
    """仓库名不能被当成本地路径拦下来：它带斜杠，但那不是目录分隔。"""
    assert _embeddings(model)._resolve_source() == model
