"""中文分词：BM25 的召回上限几乎全由这一步决定。

三个刻意的选择
--------------
1. **`lcut_for_search` 而不是 `lcut`**
   搜索引擎粒度会把长词再切出子词（「自动喷水灭火系统」→ 自动 / 喷水 / 灭火 / 系统 /
   自动喷水灭火系统）。对本项目的价值很直接：问「喷淋系统设计压力」时，切片里写的是
   「自动喷水灭火系统」，只有切出子词才可能命中。代价是词表变长、idf 被稀释，
   但在「问题短、切片长」的 RAG 场景里，召回收益远大于打分精度损失。

2. **型号/标准号整体保护**
   `GB50084-2017`、`DN100` 这类串如果交给 jieba，会被拆成 `gb`/`50084`/`2017`，
   于是「问 GB50084 能不能召回 GB50084-2017」就没人管了。这里先用正则把它们整段抽出来
   当单一 token，剩余部分才走 jieba。

3. **停用词表刻意短**
   只去掉纯虚词。`不`/`无`/`未` 这类否定词**必须保留**——规范条文里「不应小于 1.8 米」
   和「小于 1.8 米」是两个相反的答案，把它们当停用词抹掉等于主动制造错误命中。
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable

log = logging.getLogger(__name__)

# 分词口径一旦变化，磁盘上的 pkl 就必须失效重建，否则文档侧和查询侧用的不是同一套 token
TOKENIZER_VERSION = 3

# 标准号 / 型号：GB50084-2017、XF-123、DN100。
# 刻意不用 \w——Python 的 \w 含中文，会把「GB50084规定」整串吃成一个 token。
_CODE = re.compile(r"[A-Za-z]{1,12}[-_]?\d[A-Za-z0-9.\-]*")
_ASCII_WORD = re.compile(r"^[a-z]+$")

# 只收纯虚词。见模块 docstring：否定词、量词、数字一律不动。
STOPWORDS: frozenset[str] = frozenset(
    """
    的 了 是 在 和 与 或 及 并 而 但 就 都 也 又 很 更 最 这 那 其 为 对 以 于 中 上 下
    里 个 之 者 所 由 从 到 把 被 让 使 等 如 若 按 应 可 需 有 该 各 每 此 我 你 他 她
    它 们 吗 呢 吧 啊 请 问 说 什么 怎么 如何 哪些 哪个 是否 多少 为什么
    以及 因此 所以 但是 如果 可以 需要 进行 通过 根据 按照 关于 对于 由于 以上 以下
    """.split()
)

_jieba_lock = threading.Lock()
_jieba_ready = False


def _jieba():
    """惰性 import + 关闭 jieba 的启动日志（它会往 stderr 打一行广告）。"""
    global _jieba_ready
    import jieba  # 依赖只有 5 MB，但要等第一次真正用到再 import

    if not _jieba_ready:
        with _jieba_lock:
            if not _jieba_ready:
                jieba.setLogLevel(logging.WARNING)
                jieba.initialize()
                _jieba_ready = True
    return jieba


def _cut(text: str) -> Iterable[str]:
    """交给 jieba，只负责过滤。

    不在分词前把标点替换成空格：`1.2`、`5.0.1` 会被一起拆散，而它们恰恰是规范条文里
    最该被精确命中的东西（「1.2 MPa」和「1.6 MPa」是两个答案）。
    """
    for token in _jieba().lcut_for_search(text):
        token = token.strip().lower()
        if not token or token in STOPWORDS:
            continue
        if not any(char.isalnum() for char in token):
            continue  # 纯标点、纯空白
        if _ASCII_WORD.match(token) and len(token) == 1:
            continue  # 单个英文字母没有区分度（中文单字保留："泵"/"阀" 都是有意义的）
        yield token


def tokenize(text: str) -> list[str]:
    """文本 → token 列表。文档与查询必须走同一个函数，否则 BM25 的 idf 会错位。"""
    if not text:
        return []
    tokens: list[str] = []
    cursor = 0
    for match in _CODE.finditer(text):
        tokens.extend(_cut(text[cursor : match.start()]))
        code = match.group(0).strip().lower()
        if code:
            tokens.append(code)  # 整段保留，"gb50084-2017" 不能被拆成 gb / 50084 / 2017
        cursor = match.end()
    tokens.extend(_cut(text[cursor:]))
    return tokens


def tokenize_batch(texts: Iterable[str]) -> list[list[str]]:
    return [tokenize(text) for text in texts]
