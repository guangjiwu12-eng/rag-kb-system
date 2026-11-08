"""Cross-encoder 重排：bge-reranker-v2-m3，torch 与 onnxruntime 两条后端。

它解决的是双塔模型的结构性缺陷
----------------------------
向量检索把「问题」和「切片」分别压成一个点，再用余弦距离比远近，两者从未互相看过一眼。
cross-encoder 把 (问题, 切片) 拼成一句送进同一个 Transformer，让每个 token 都能直接
attend 到对方——精度高得多，代价是**不能预计算**，每个候选都要跑一次前向。

于是所有工程问题都收敛到一件事：20 个候选 × 每个多少 token。
- `rerank_max_length` 是总闸门（只截断第二段，问题永远完整保留）
- 按长度降序排队、分批（batch 内 padding 浪费最小）
- 再不够就上 int8 ONNX（CPU 上有 2~3 倍，本机不保证，见 eval/latency 报告）

两条后端共用一个接口
------------------
`onnx` 目录存在就走 onnxruntime，否则走 torch。上层不需要知道这份差异——消融矩阵跑的
永远是同一条流水线，换后端只影响延迟那一列。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class RerankerUnavailable(RuntimeError):
    """模型缺失或推理失败。是否降级由调用方决定（默认 fail-open）。"""


@dataclass(frozen=True)
class RerankScore:
    index: int  # 在输入 passages 里的下标
    score: float  # sigmoid 后的相关性，0~1


class CrossEncoderReranker:
    """懒加载 + 进程内单例。权重约 2.2 GB，启动时不加载，第一次重排才加载。"""

    def __init__(
        self,
        model_path: str,
        *,
        onnx_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 256,
        num_threads: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self.num_threads = num_threads

        self._tokenizer: Any = None
        self._model: Any = None
        self._session: Any = None
        self._torch: Any = None
        self._lock = threading.Lock()
        # 未加载时也要报对后端：/retrieval 是运维看的那一眼，写「torch」而实际配的是
        # ONNX，会让人在排障时先去怀疑错的东西。
        self.backend = "onnx" if onnx_path and Path(onnx_path).exists() else "torch"
        self.load_seconds: float = 0.0
        # 单条候选（按 max_length 截断）在本机上的实测耗时，由 _calibrate 填。
        # 预算闸门必须用实测值：把别人博客上的「重排 50ms」抄进代码，线上就会变成 12 秒。
        self.ms_per_pair: float | None = None

    # ------------------------------------------------------------------ 加载
    @property
    def loaded(self) -> bool:
        return self._model is not None or self._session is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            started = time.perf_counter()
            if self.onnx_path and Path(self.onnx_path).exists():
                self._load_onnx()
            else:
                self._load_torch()
            self.load_seconds = time.perf_counter() - started
            # 预热一次：第一个真实请求不该为「图/内存池首次分配」买单，
            # 否则 p95 会被一条冷启动样本拖垮。
            self._forward([("预热查询", "预热用的文档内容，长度只为触发一次完整前向。")])
            self._calibrate()
            log.info(
                "重排模型就绪：backend=%s max_length=%d batch=%d 加载 %.1fs 单条 %.0fms",
                self.backend,
                self.max_length,
                self.batch_size,
                self.load_seconds,
                self.ms_per_pair or 0.0,
            )

    def _calibrate(self) -> None:
        """测一次「一条候选跑一遍前向」要多久，作为延迟预算的成本模型。

        为什么在加载时测，而不是写死常量
        ------------------------------
        同一份代码在 M2、在 Xeon、在禁用 AVX-512 的轻薄本上差 20 倍。写死数字的预算闸门
        等于没有闸门：要么永远放行（线上爆），要么永远拦截（重排白装）。这里用真实后端、
        真实 max_length 实测一次，再留 15% 余量——单个样本的抖动不该被当成模型能力。

        成本对候选数近似线性（本机实测 4→20 条，倍数与条数一致），所以只需要这一个系数：
        `预估耗时 = 候选数 × ms_per_pair`。
        """
        # 探针必须真的填满 max_length：如果探针只有 100 token，量出来的是「100 token 的成本」，
        # 用它按 max_length 折算候选数，预算会被高估好几倍——闸门就成了摆设。
        unit = "重排成本标定用的中文句子，长度需要足够填满截断窗口。"
        # 中文约 1 token/字，按 2 倍字数拼，保证一定被截到 max_length
        probe = unit * (self.max_length * 2 // max(1, len(unit)) + 2)
        try:
            warm = self._encode([("标定", probe)])
            actual = int(warm["input_ids"].shape[1])  # type: ignore[index]
            self._forward([("标定", probe)])  # 预热：别把内存池首次分配算进成本
            started = time.perf_counter()
            self._forward([("标定", probe)])
            elapsed_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - 标定失败不该挡住服务启动
            log.warning("重排成本标定失败，预算闸门按「不设限」处理：%s", exc)
            return
        if actual < self.max_length:
            log.warning("标定探针只有 %d token（max_length=%d），成本会被低估", actual, self.max_length)
        self.ms_per_pair = round(elapsed_ms * 1.15, 1)

    def affordable_count(self, budget_ms: int) -> int:
        """这个预算最多排得动几条候选。返回 0 表示一条都排不动。"""
        self.load()
        if not self.ms_per_pair:
            return 0
        return max(0, int(budget_ms // self.ms_per_pair))

    def _load_torch(self) -> None:
        if not Path(self.model_path).exists():
            raise RerankerUnavailable(f"重排模型不存在：{self.model_path}")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RerankerUnavailable(f"缺少 torch/transformers：{exc}") from exc

        self._torch = torch
        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.to(self.device).eval()
        self.backend = "torch"

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RerankerUnavailable(f"缺少 onnxruntime/transformers：{exc}") from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.num_threads:
            options.intra_op_num_threads = self.num_threads
            options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(Path(self.onnx_path) / "model.onnx"),  # type: ignore[arg-type]
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(Path(self.onnx_path) / "tokenizer")  # type: ignore[arg-type]
        )
        self.backend = "onnx"

    @property
    def input_names(self) -> set[str]:
        if self._session is None:
            return {"input_ids", "attention_mask"}
        return {item.name for item in self._session.get_inputs()}

    # ------------------------------------------------------------------ 推理
    def _encode(self, pairs: list[tuple[str, str]]):
        return self._tokenizer(
            [query for query, _ in pairs],
            [passage for _, passage in pairs],
            padding=True,
            truncation="only_second",  # 只砍候选，问题一个 token 都不砍
            max_length=self.max_length,
            return_tensors="pt" if self._session is None else "np",
        )

    def _forward(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        encoded = self._encode(pairs)

        if self._session is not None:
            feeds = {
                name: encoded[name]
                for name in self.input_names
                if name in encoded
            }
            logits = self._session.run(None, feeds)[0]
            import numpy as np

            flat = np.asarray(logits).reshape(-1).astype("float32")
            return (1.0 / (1.0 + np.exp(-flat))).tolist()

        torch = self._torch
        with torch.inference_mode():
            tensor = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self._model(**tensor).logits
            scores = torch.sigmoid(logits.view(-1))
        return scores.float().cpu().tolist()

    def rerank(
        self, query: str, passages: list[str], top_k: int | None = None
    ) -> list[RerankScore]:
        """返回按相关性降序的 (原下标, 分数)。任何异常统一抛 RerankerUnavailable。"""
        if not passages:
            return []
        try:
            self.load()
        except RerankerUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - 加载期的任何异常都不该炸穿检索
            raise RerankerUnavailable(f"重排模型加载失败：{exc}") from exc

        # 按 token 长度降序排队：同一批里长度接近，padding 浪费最小；
        # 分数与批次无关，所以最后可以安心按原下标还原顺序。
        lengths = [len(self._tokenizer.tokenize(text[: self.max_length * 4])) for text in passages]
        order = sorted(range(len(passages)), key=lambda i: lengths[i], reverse=True)

        scores = [0.0] * len(passages)
        try:
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                pairs = [(query, passages[i]) for i in batch]
                for position, value in zip(batch, self._forward(pairs), strict=False):
                    scores[position] = float(value)
        except Exception as exc:  # noqa: BLE001
            raise RerankerUnavailable(f"重排推理失败：{exc}") from exc

        ranked = sorted(
            (RerankScore(index=i, score=round(score, 6)) for i, score in enumerate(scores)),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:top_k] if top_k else ranked

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_path,
            "onnx": self.onnx_path,
            "loaded": self.loaded,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "load_seconds": round(self.load_seconds, 2),
            "ms_per_pair": self.ms_per_pair,
        }
