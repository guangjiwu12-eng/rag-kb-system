"""本地 Embedding：用 transformers 直接跑 bge-m3，不依赖任何云服务。

为什么不用 sentence-transformers
------------------------------
bge-m3 的处理流程只有三步：xlm-roberta 编码 → 取 CLS → L2 归一化。用 transformers
的 `AutoTokenizer + AutoModel` 十几行就写完了，省掉一个依赖，也少一层版本耦合
（sentence-transformers 对 transformers 的大版本卡得比较紧）。

为什么 import 关在类里
--------------------
默认路线是 DashScope 的云端 Embedding。把 torch / transformers 的 import 放进 `model`
惰性属性里，才能让「不装这两个包的人照常启动服务」成立——否则 requirements.txt 会
平白多出几百 MB，只为一条可选分支。

bge-m3 的关键参数（改了就会和训练口径不一致）
------------------------------------------
- 池化：**CLS**（不是 mean）。取错池化，向量还是一样的维度，但检索质量会悄悄掉下来
- 归一化：embedding 后立刻 L2 归一化，配合 Chroma 的余弦空间
- 维度：1024。换模型一定要重建向量库，Chroma 不接受同一集合里混两种维度
- 不需要 query 前缀指令（bge-v1.5 系列才需要「为这个句子生成表示…」）
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)

DEFAULT_MAX_SEQ_LENGTH = 1024

# 绝对路径 / 家目录 / 相对路径的写法。HF 仓库名（BAAI/bge-m3）不以这些开头，
# 所以这个判据不会把仓库名误判成本地路径。
_PATH_LIKE = re.compile(r"^(/|\\\\|~|\.{1,2}[/\\]|[A-Za-z]:[/\\])")


class TransformerEmbeddings(Embeddings):
    """transformers 后端：CLS 池化 + 可选 L2 归一化。

    懒加载 + 进程内单例：权重 2 GB，启动时不该阻塞健康检查，第一次入库/检索时再载入。
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
        max_seq_length: int | None = None,
        pooling: str = "cls",
        num_threads: int | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.normalize = normalize
        self.max_seq_length = max_seq_length or DEFAULT_MAX_SEQ_LENGTH
        self.pooling = pooling
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 惰性加载
    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "本地 Embedding 需要 torch 与 transformers："
                    "pip install -r requirements-embedding.txt"
                ) from exc

            log.info(
                "加载本地 Embedding：%s (device=%s, max_len=%d)",
                self.model_name_or_path,
                self.device,
                self.max_seq_length,
            )
            self._torch = torch
            if self.num_threads:
                torch.set_num_threads(self.num_threads)
                log.info("Embedding 线程数：%d", self.num_threads)
            source = self._resolve_source()
            self._tokenizer = AutoTokenizer.from_pretrained(source)
            self._model = AutoModel.from_pretrained(source).to(self.device)
            self._model.eval()
            log.info("本地 Embedding 就绪：hidden=%d", int(self._model.config.hidden_size))

    def _resolve_source(self) -> str:
        """把配置里的模型地址解析成「能直接读的路径或仓库名」。

        写的是路径但路径不存在时**当场报错**。不报错的后果很隐蔽：transformers 会把它
        当成仓库名转去联网下载，网络不通就卡在重试上——实测卡满 20 分钟、一个字节都没写，
        而屏幕上只有一行「加载 Embedding」，看起来像在正常干活。

        最容易踩的一种：`.env` 里写的是 WSL 里的绝对路径 `/home/xxx/models/bge-m3`，
        却用 Windows 侧的 python 跑脚本，于是同一个路径两边指向不同地方。
        """
        raw = self.model_name_or_path
        if not _PATH_LIKE.match(raw):
            return raw
        path = Path(raw).expanduser()
        if path.exists():
            return str(path)
        raise RuntimeError(
            f"本地 Embedding 路径不存在：{raw}\n"
            "  · 用 WSL 里的 python 跑，就用 `/home/...` 这种绝对路径；\n"
            "  · 用 Windows 侧的 python 跑，就写相对项目根的 `models/bge-m3`；\n"
            "  · 另外加上 HF_HUB_OFFLINE=1，别让 transformers 转去联网下载权重。"
        )

    @property
    def dimension(self) -> int:
        self._load()
        return int(self._model.config.hidden_size)

    # ------------------------------------------------------------ 编码
    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        torch = self._torch
        vectors: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = self._model(**encoded).last_hidden_state

            if self.pooling == "mean":
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            else:
                pooled = hidden[:, 0]  # CLS：bge-m3 的训练口径

            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.float().cpu().tolist())

        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([text or "" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text or ""])[0]
