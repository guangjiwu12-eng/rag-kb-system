"""模型工厂：按配置创建 Embedding 与 Chat 模型，缺 Key 时给出明确错误。"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings
from app.errors import ModelNotConfigured

log = logging.getLogger(__name__)


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_backend == "sentence-transformers":
        # 本地模型：不碰网络、不花钱、文档不出机器；代价是首次加载 2 GB 权重
        from app.local_embeddings import TransformerEmbeddings

        log.info(
            "初始化本地 Embedding：%s (device=%s, batch=%d)",
            settings.embedding_model,
            settings.embedding_device,
            settings.embedding_batch_size,
        )
        return TransformerEmbeddings(
            model_name_or_path=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            normalize=settings.embedding_normalize,
            max_seq_length=settings.embedding_max_seq_length,
            num_threads=settings.embedding_num_threads,
        )

    if not settings.dashscope_api_key:
        raise ModelNotConfigured(
            "缺少 DASHSCOPE_API_KEY：向量化依赖通义千问 Embedding，请在 .env 中配置后重启。"
            "（也可以设 EMBEDDING_BACKEND=sentence-transformers 改用本地 bge-m3）"
        )

    # langchain-community 已进入维护模式，但在当前生态中仍是 DashScope 的官方接入点
    from langchain_community.embeddings import DashScopeEmbeddings

    log.info("初始化 Embedding：%s", settings.embedding_model)
    return DashScopeEmbeddings(
        model=settings.embedding_model,
        dashscope_api_key=settings.dashscope_api_key,
        max_retries=settings.llm_max_retries,
    )


def build_llm(settings: Settings) -> BaseChatModel:
    model = settings.effective_llm_model
    log.info("初始化 LLM：provider=%s model=%s", settings.llm_provider, model)

    if settings.llm_provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ModelNotConfigured(
                "缺少 DEEPSEEK_API_KEY：当前 LLM_PROVIDER=deepseek，请在 .env 中配置后重启。"
            )
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=model,
            api_key=settings.deepseek_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    if not settings.dashscope_api_key:
        raise ModelNotConfigured(
            "缺少 DASHSCOPE_API_KEY：当前 LLM_PROVIDER=dashscope，请在 .env 中配置后重启。"
        )
    from langchain_community.chat_models.tongyi import ChatTongyi

    return ChatTongyi(
        model=model,
        dashscope_api_key=settings.dashscope_api_key,
        temperature=settings.llm_temperature,
    )
