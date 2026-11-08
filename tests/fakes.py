"""测试用的假模型：不联网、不花钱，行为确定。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

DIM = 64


class FakeEmbeddings(Embeddings):
    """字符哈希词袋向量：文本重合越多，相似度越高，足以驱动检索断言。"""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIM
        for char in text:
            vector[ord(char) % DIM] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class CountingEmbeddings(FakeEmbeddings):
    """记录 embed_documents 调用次数，用来验证「重复上传不会重新向量化」。"""

    document_calls: int = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return super().embed_documents(texts)


class FailingEmbeddings(FakeEmbeddings):
    """模拟 Embedding 服务挂掉，用来验证入库失败时不会丢掉已有数据。"""

    index: int = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.index += 1
        if self.index > 1:
            raise RuntimeError("Embedding 服务不可用")
        return super().embed_documents(texts)


class FakeChatModel(BaseChatModel):
    """固定回答，用来验证「引用编号 → 引用片段」的映射逻辑。"""

    reply: str = "根据资料，员工每年享有 5 天带薪年假 [1]。"
    calls: int = 0
    # 喂给模型的原文留一份：引用标签（条文号、条文说明）只存在于 prompt 里，
    # 不落在 Citation 上，不记录就断言不到「模型到底看到了什么」。
    prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        self.prompts.append("\n".join(str(message.content) for message in messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])
