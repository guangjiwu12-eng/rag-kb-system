"""版面还原与条文切片：把「页」还原成「条」，而不是按字数硬切。

为什么不再用 800 字定长
----------------------
定长切片把语义单元交给了字数运气。实测三条失败题（n2 / n16 / n21）都是同一个死法：
答案句被挤到片尾，前半片全是「见表 C.1 中 XX 部分」这类同构噪声，向量被稀释成
「巡查内容」的平均值，检索自然排不上去。n2 的答案句落在第 660 字里的第 589 字——
89% 位置，整片只有最后 56 字有用。

规范类语料的语义单元本来就是「条」。按条切，答案句与它的适用条件、枚举项天然同片，
切片长度也自然落在 100~800 字，不需要 overlap 去救。

正文与条文说明为什么必须分开
----------------------------
这批语料是「正文 + 逐章条文说明」交错排布：1.0.1 条文说完，紧跟一段「条文说明」
把同一条重讲一遍。于是同一事实在库里存了两份，而且**条文说明往往更长、更啰嗦**，
向量更容易赢——检索命中的是解释，不是要求。这既是「互相抢位」，也是答非所问的来源。
本模块给每个切片打 `section_type`（main / explanation / duplicate），让上层能区分
「规范要求」和「仅供参考的解释」。

判定不靠「条文说明」这四个字出现的位置
--------------------------------------
因为那四个字在本批语料里同时是正文分节标签、也出现在每页页眉，还会被导航块重复。
更稳的信号是**编号复用**：条文说明必然重述正文已有的条文号。所以：

    编号首次出现                → main
    编号再次出现且文本差异大    → explanation
    编号再次出现且文本几乎相同  → duplicate（导航块 / 页面重叠造成的真重复）

这条规则与版面无关，扫描件、网页抓取、正规 PDF 都成立。

页码为什么存两个
----------------
PDF 的物理页（第几个 page 对象）和纸面印刷页（页脚印的「第 N 页」）经常差一个常数——
封面、扉页、前言不参与编号。gold 标注时人看的是印刷页，检索返回的是物理页，
两边一旦混用，页码就变成猜谜。所以两个都存：`page` 是物理页，`page_printed` 是页脚页，
并对整篇做偏移一致性校验（不一致就把 page_printed 留空，宁可缺也不给错值）。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

# --------------------------------------------------------------------- 页脚页眉
# 只认独占一行的页码，避免把正文里的「见表 D.1 第 3 页」误当页脚。
_FOOTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[\s\-—－·]*第\s*(\d{1,4})\s*页[\s\-—－·]*$"),
    re.compile(r"^[\s\-—－]+\s*(\d{1,4})\s*[\-—－]+\s*$"),
    re.compile(r"^\s*(\d{1,4})\s*/\s*(\d{1,4})\s*$"),
)

# 页眉：以「条文说明」为首/尾的孤立行、以及国标号独占行，都是版面残留
_RUNNING_HEADER = re.compile(r"^[\s\-—－]*(条\s*文\s*说\s*明|GB[ /]?[\d.]+[\-—－]?\d*)[\s\-—－]*$")

# --------------------------------------------------------------------- 行分类
# 条文号：1.0.1 / 5.0.2 / 附录里的 C.0.1
_ARTICLE = re.compile(r"^(?P<num>[A-Z]?\d{1,2}(?:\.\d{1,3}){1,3})(?![\d.])(?P<gap>[\s\u3000]*)(?P<rest>\S.*)$")
# 章标题：7 检测 / 12 系统调试（两侧留白，标题短且不带句读）
_CHAPTER = re.compile(r"^(?P<num>\d{1,2})(?P<gap>[\s\u3000]+)(?P<title>\S.{0,19})$")
# 附录标题：附录 C 消防用电设备的供配电
_APPENDIX = re.compile(r"^附\s*录\s*(?P<num>[A-Z])(?P<title>.*)$")

# 标题里出现这些字符就说明是正文句子，不是标题。顿号除外——「材料、设备管理」是标准里的正常标题。
_TITLE_CLOSERS = re.compile(r"[。；，：；！？,;:]")
# 句末标点：只有落在句末，后一行才可能是新条文。ASCII 句点不算——「应符合表7.」
# 这种被折断的编号里到处是句点，把它们当句末会让「1.7的规定。」变成一条假条文。
_SENTENCE_TAIL = "。；：！？"
_LABEL = re.compile(r"[。；！？]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
# 条文号后面紧跟这些字符，说明是「2.0m」「1）」「0.5m」这类数值/枚举，不是条文
_NOT_AFTER_NUM = re.compile(r"^[\d.%‰A-Za-z)）、，。；]")


def _norm_space(text: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", text).strip()


@dataclass(slots=True)
class PageLine:
    """一行正文，带页码定位。page 从 1 开始。"""

    text: str
    page: int
    printed: int | None = None


@dataclass(slots=True)
class ArticleBlock:
    """一条（或一段无法识别为条文的连续正文）。"""

    number: str | None
    text: str
    page: int
    page_end: int
    printed: int | None = None
    printed_end: int | None = None
    title: str = ""
    section_path: str = ""
    kind: str = "article"  # article | prose | heading
    section_type: str = "main"  # main | explanation | duplicate | front


# --------------------------------------------------------------------- 页码
def split_page_furniture(text: str) -> tuple[str, int | None]:
    """剥掉页脚页码，返回 (干净正文, 页脚页)。

    印刷页取「本页最后一行是页码」的那一个；没有就返回 None，绝不猜。
    """
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    printed: int | None = None

    for index in range(len(lines) - 1, -1, -1):
        candidate = _norm_space(lines[index])
        if not candidate:
            continue
        for pattern in _FOOTER_PATTERNS:
            match = pattern.match(candidate)
            if match:
                printed = int(match.group(1))
                lines.pop(index)
                break
        else:
            # 只检查最后一个非空行：页码不在那儿就不认，避免误删正文
            pass
        break

    kept = [line for line in lines if not _RUNNING_HEADER.match(_norm_space(line))]
    return "\n".join(kept).strip(), printed


def printed_page_offset(pages: list[PageLine]) -> int | None:
    """整篇共用一个印刷页偏移量；页脚不连续或缺失时返回 None。

    宁可不给，也不给一个错值——错页码比没有页码更费人。
    """
    pairs = {(line.page, line.printed) for line in pages if line.printed is not None}
    by_page: dict[int, int | None] = {}
    for page, printed in pairs:
        if page in by_page and by_page[page] != printed:
            return None  # 同一物理页读出两个不同页脚，不可信
        by_page[page] = printed
    offsets = {printed - page for page, printed in by_page.items() if printed is not None}
    return offsets.pop() if len(offsets) == 1 else None


# --------------------------------------------------------------------- 行流
def to_lines(pages: Iterable[tuple[str, int, int | None]]) -> list[PageLine]:
    """(正文, 物理页, 印刷页) → 非空行序列。"""
    out: list[PageLine] = []
    for text, page, printed in pages:
        for raw in (text or "").split("\n"):
            line = _norm_space(raw)
            if line:
                out.append(PageLine(text=line, page=page, printed=printed))
    return out


def _looks_like_title(rest: str) -> bool:
    """短、无句读、且不是停在顿号/逗号/句点上——停在那儿说明是被折行的正文。

    ASCII 句点尤其要说清楚：「…应符合表7.」这种句子被折在这里，
    看上去像标题，其实是正文，放过去就会把下半句「1.7的规定。」切成一条假条文。
    """
    if len(rest) > 16 or _TITLE_CLOSERS.search(rest):
        return False
    return rest[-1:] not in "、，,；;：:."


def classify(line: str) -> tuple[str, str | None, str]:
    """返回 (kind, 条文号, 标题)。

    kind: article（条文正文）/ heading（章、节、附录标题）/ prose（无编号正文）
    """
    appendix = _APPENDIX.match(line)
    if appendix:
        return "heading", f"附录{appendix.group('num')}", appendix.group("title").strip()

    article = _ARTICLE.match(line)
    if article:
        number = article.group("num")
        rest = article.group("rest").strip()
        head = number.split(".")[0]
        # 「0.5m」「1）室内高度…」这类不是条文号
        if head.isdigit() and int(head) >= 1 or head.startswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
            if not _NOT_AFTER_NUM.match(rest):
                if _looks_like_title(rest):
                    return "heading", number, rest
                return "article", number, ""

    chapter = _CHAPTER.match(line)
    if chapter:
        title = chapter.group("title").strip()
        if _looks_like_title(title):
            return "heading", chapter.group("num"), title

    return "prose", None, ""


def _join(prev: str, nxt: str) -> str:
    """把 PDF 的硬折行接回去。

    中文直接相连；英文只有「上一行以字母数字结尾 + 下一行以大写字母开头」才补空格。
    这个偏心的规则是刻意的：中文 PDF 按字宽折行，英文单词会被拦腰截断
    （实测「5min」被折成「5mi」+「n，」），一律补空格会造出 `5mi n` 这种
    数值校验和锚点匹配都会挂的残句。宁可把两个英文单词粘连，也不把数字和单位拆开。
    """
    if not prev:
        return nxt
    if not nxt:
        return prev
    if prev[-1].isascii() and prev[-1].isalnum() and nxt[0].isascii() and nxt[0].isupper():
        return f"{prev} {nxt}"
    return prev + nxt


# --------------------------------------------------------------------- 分块
def _section_path(headings: dict[str, str], number: str) -> str:
    """按编号前缀拼出层级：「7 检测 > 7.2 检测内容」。

    标题先于条文到达是常态（目录块、版面残留都会提前送一遍标题），所以按编号查表，
    而不是按「最近一次见到的标题」——否则 7.2 的标题会串到 7.1.1 头上。
    """
    if not number or not number[0].isdigit():
        return ""
    parts = number.split(".")
    names = [
        headings[".".join(parts[: i + 1])]
        for i in range(len(parts) - 1)
        if ".".join(parts[: i + 1]) in headings
    ]
    return " > ".join(dict.fromkeys(names))


def _ends_sentence(block: ArticleBlock | None) -> bool:
    """上一块是否落在句末。块还没开（当前是标题/文档开头）也算可以起新条文。"""
    if block is None:
        return True
    return block.text[-1:] in _SENTENCE_TAIL


def _is_continuation(block: ArticleBlock | None, line: str, number: str | None) -> bool:
    """行首的「条文号」其实是上一句被折断的尾巴吗？"""
    if not number or _ends_sentence(block):
        return False
    rest = line[len(number) :].strip()
    # 「1.7的规定。」「4.4.2 条的规定。」这类残句都很短；真正的条文起头会带正文
    return len(rest) < 12


def build_blocks(lines: list[PageLine]) -> list[ArticleBlock]:
    """按条文号切块；无编号的连续正文单独成块（prose）。"""
    blocks: list[ArticleBlock] = []
    headings: dict[str, str] = {}
    current: ArticleBlock | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.text.strip():
            blocks.append(current)
        current = None

    for line in lines:
        kind, number, title = classify(line.text)

        if kind == "heading":
            flush()
            key = number or ""
            if key:
                headings[key] = title or headings.get(key, "")
            continue

        if kind == "article":
            # 折行会造出「表7.」+「1.7的规定。」这种假条文：行首长得像条文号，
            # 其实是上一句的尾巴。判据是「编号后面的内容很短，且上一块没落在句末」——
            # 真正的条文起头总是跟在句号/冒号之后，而残句的上一行通常停在「表7.」。
            if _is_continuation(current, line.text, number):
                kind = "prose"
            else:
                flush()
                current = ArticleBlock(
                    number=number,
                    text=line.text,
                    page=line.page,
                    page_end=line.page,
                    printed=line.printed,
                    printed_end=line.printed,
                    title=title,
                    section_path=_section_path(headings, number or ""),
                    kind="article",
                )
                continue

        if current is None:
            current = ArticleBlock(
                number=None,
                text="",
                page=line.page,
                page_end=line.page,
                printed=line.printed,
                printed_end=line.printed,
                kind="prose",
            )
        current.text = _join(current.text, line.text)
        current.page_end = line.page
        if line.printed is not None:
            current.printed_end = line.printed

    flush()
    kept = []
    for block in blocks:
        text = block.text.strip()
        # 「Ｉ 控制与显示类设备安装」「表B 消防安全管理信息续表B」这类是版面标签，
        # 不是可回答的内容；留着只会占 Top-K 的位置。
        if len(text) < 24 and not _LABEL.search(text):
            continue
        # 「5.5.9（1）、5.5.12、6.1.9（1）、」这种纯条文号清单也不是内容：
        # 它会被当成第 5.5.9 条存进去，跟真正的 5.5.9 抢排名。
        if len(_CJK.findall(text)) < 4:
            continue
        kept.append(block)
    return kept


# --------------------------------------------------------------------- 章节归属
def _shingles(text: str, size: int = 4) -> set[int]:
    clean = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    if len(clean) < size:
        return {hash(clean)}
    return {hash(clean[i : i + size]) for i in range(len(clean) - size + 1)}


def similarity(a: str, b: str) -> float:
    """4-gram Jaccard，用来区分「同号重述」和「同号真重复」。"""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / (len(sa) + len(sb) - inter)


SAME_TEXT = 0.90
SAME_NUMBER_NEW_TEXT = 0.90


def tag_sections(blocks: list[ArticleBlock], same_text: float = SAME_TEXT) -> None:
    """就地判定 section_type：首次出现是 main，同号且文本不同是 explanation，同号同文是 duplicate。

    条文说明之前的无编号正文（封面、前言、编制说明）记为 front；之后的无编号正文仍算正文。
    """
    first: dict[str, ArticleBlock] = {}
    seen_article = False
    for block in blocks:
        if block.kind != "article" or not block.number:
            block.section_type = "main" if seen_article else "front"
            continue
        seen_article = True
        original = first.get(block.number)
        if original is None:
            first[block.number] = block
            block.section_type = "main"
            continue
        score = similarity(block.text, original.text)
        block.section_type = "duplicate" if score >= same_text else "explanation"


# --------------------------------------------------------------------- 长条文再切
_SENTENCE_END = re.compile(r"(?<=[。！？；])")


def split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """条文太长（>max_chars）才二次切分，按句号切，尾部保留 overlap 字以免断言被截断。"""
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    current = ""
    for sentence in (s for s in _SENTENCE_END.split(text) if s):
        if len(sentence) > max_chars:  # 单句超长（无句号的长枚举），硬切
            if current:
                parts.append(current)
                current = ""
            step = max(1, max_chars - overlap)
            parts.extend(sentence[i : i + max_chars] for i in range(0, len(sentence) - overlap, step))
            current = parts.pop() if parts else ""
            continue
        if current and len(current) + len(sentence) > max_chars:
            parts.append(current)
            current = (current[-overlap:] if overlap else "") + sentence
        else:
            current += sentence
    if current.strip():
        parts.append(current)
    return [part for part in parts if part.strip()]


# --------------------------------------------------------------------- 去重
_LEADING_NUMBER = re.compile(r"^\s*[A-Z]?\d{1,2}(?:\.\d{1,3}){1,3}\s*")


def article_body(text: str) -> str:
    """剥掉开头的条文号，只留正文。

    跨标准重复的条文，**条文号一定不同**（同一句话在 GB50166 是 6.0.4，在 GB51309 是 7.0.4），
    所以带号比对永远比不上。剥掉编号后比正文，才认得出「同一句话」。
    """
    return _LEADING_NUMBER.sub("", text or "", count=1)


def fingerprint(text: str, strip_article: bool = True) -> str:
    """精确重复的指纹：NFKC + 去空白，默认忽略开头的条文号。"""
    body = article_body(text) if strip_article else (text or "")
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", body))


def dedupe(texts: list[str]) -> list[int | None]:
    """返回每个元素应归并到哪一个（索引），None 表示它自己就是正本。

    只并**完全一致**的片段，刻意不做模糊合并：规范里「6.2.5 消防供水设施的巡查内容」
    到「6.2.19 其他需要巡查的内容」十九条只差系统名，4-gram 相似度能到 0.8 以上——
    按相似度去重会把十九条不同的要求压成一条，那是删数据，不是去重。
    跨标准的「同义条文」先用 `near_duplicate_pairs` 报出来给人看，确认后再决定删哪条。
    """
    canonical: list[int | None] = [None] * len(texts)
    seen: dict[str, int] = {}
    for index, text in enumerate(texts):
        key = fingerprint(text)
        if len(key) < 8:
            continue
        if key in seen:
            canonical[index] = seen[key]
        else:
            seen[key] = index
    return canonical

def near_duplicate_pairs(
    texts: list[str], threshold: float = 0.9, probes: int = 4
) -> list[tuple[int, int, float]]:
    """近似重复对，只用于审计报告，不参与入库。

    候选生成用「若干个最小哈希做桶键」：只要两段文本高度重合，它们的前几个最小哈希
    大概率撞上同一个桶，于是不必做 O(n^2) 全量比对。单用一个桶键会漏
    （两句只差一个后缀时，最小哈希就可能不同），所以取 4 个。
    """
    signature = [_shingles(text) for text in texts]
    bucket: dict[int, list[int]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int, float]] = []

    for index, grams in enumerate(signature):
        if len(grams) < 8:
            continue
        keys = sorted(grams)[:probes]
        candidates: set[int] = set()
        for key in keys:
            candidates.update(bucket.get(key, []))
        for candidate in candidates:
            pair = (candidate, index)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            score = similarity(texts[index], texts[candidate])
            if score >= threshold:
                pairs.append((candidate, index, score))
        for key in keys:
            bucket.setdefault(key, []).append(index)
    return pairs
