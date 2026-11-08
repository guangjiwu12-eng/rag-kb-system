"""条文切片与版面还原的单元测试（不联网、不依赖 PDF）。"""

from __future__ import annotations

from langchain_core.documents import Document

from app import corpus
from app.config import Settings
from app.document_loader import chunk_documents

PAGE_1 = """4 总则

4.1 一般要求

4.1.1 建筑消防设施应每年至少检测一次，
检测对象包括全部系统设备、组件等。
4.1.2 从事建筑消防设施检测的人员应持有资格证书。
4.1.3 检测应按 GA 503 的要求进行，并填写记录表。
4.1.4 检测发现的问题应及时组织维修。
第 1 页"""

PAGE_2 = """4.1.5 不应擅自关停消防设施。
4.1.6 消防设施投入使用后应处于正常工作状态。
条文说明

4.1.1 本条规定了检测频次。检测频次定为每年一次，
是因为大部分设施的故障在一年内可以暴露。
4.1.2 本条规定了人员资格，与职业资格制度衔接。
第 2 页"""


def make_settings(**kwargs) -> Settings:
    base = {"_env_file": None, "chunk_mode": "auto", "article_min_markers": 6}
    base.update(kwargs)
    return Settings(**base)


def pages(*texts: str) -> list[Document]:
    """按 load_pages 的口径构造页：先剥页脚，再记印刷页。"""
    built = []
    for index, text in enumerate(texts):
        content, printed = corpus.split_page_furniture(text)
        built.append(Document(page_content=content, metadata={"page": index, "page_printed": printed}))
    return built


# --------------------------------------------------------------------- 页脚与页码
def test_footer_is_stripped_and_printed_page_captured() -> None:
    text, printed = corpus.split_page_furniture("正文内容\n第 12 页")
    assert printed == 12
    assert "第 12 页" not in text
    assert text == "正文内容"


def test_footer_variants() -> None:
    assert corpus.split_page_furniture("正文\n- 7 -")[1] == 7
    assert corpus.split_page_furniture("正文\n3 / 40")[1] == 3
    # 页码不在末行时不认，避免误删正文里的「见第 3 页」
    assert corpus.split_page_furniture("见第 3 页\n后续正文")[1] is None


def test_printed_offset_requires_consistency() -> None:
    steady = [
        corpus.PageLine("a", 1, 5),
        corpus.PageLine("b", 2, 6),
        corpus.PageLine("c", 3, 7),
    ]
    assert corpus.printed_page_offset(steady) == 4

    broken = [corpus.PageLine("a", 1, 5), corpus.PageLine("b", 2, 9)]
    assert corpus.printed_page_offset(broken) is None

    assert corpus.printed_page_offset([corpus.PageLine("a", 1, None)]) is None


# --------------------------------------------------------------------- 行分类
def test_classify_article_heading_and_noise() -> None:
    assert corpus.classify("7.1.1 建筑消防设施应每年至少检测一次，")[0] == "article"
    assert corpus.classify("7.1 一般要求")[:2] == ("heading", "7.1")
    assert corpus.classify("2.2 材料、设备进场检查")[0] == "heading"
    assert corpus.classify("7 检测")[:2] == ("heading", "7")
    assert corpus.classify("附录C 消防用电设备的供配电")[:2] == ("heading", "附录C")
    # 数值与枚举不是条文号
    assert corpus.classify("2.0m，其他疏散通道不应小于1.5m。")[0] == "prose"
    assert corpus.classify("1）室内高度大于4.5m的场所，")[0] == "prose"
    # 停在顿号上说明是被折行的正文，不是标题
    assert corpus.classify("2.2.4 系统设备及配件的规格、")[0] == "article"


def test_join_repairs_broken_numbers_without_adding_spaces() -> None:
    assert corpus._join("不应大于0.", "25s；") == "不应大于0.25s；"
    assert corpus._join("时间不应大于5mi", "n，") == "时间不应大于5min，"
    assert corpus._join("符合表11.0", ".15的规定。") == "符合表11.0.15的规定。"
    assert corpus._join("中文", "接排") == "中文接排"
    assert corpus._join("Fire", "Alarm") == "Fire Alarm"


# --------------------------------------------------------------------- 分块与章节
def test_build_blocks_tags_explanation_and_drops_fake_article() -> None:
    lines = corpus.to_lines(
        [(PAGE_1, 1, 1), (PAGE_2, 2, 2)]
    )
    blocks = corpus.build_blocks(lines)
    corpus.tag_sections(blocks)

    numbers = [b.number for b in blocks if b.kind == "article"]
    assert "4.1.1" in numbers
    main = [b for b in blocks if b.number == "4.1.1" and b.section_type == "main"]
    assert main and "每年至少检测一次" in main[0].text

    # 同号、文本不同 → 条文说明
    explanations = [b for b in blocks if b.number == "4.1.1" and b.section_type == "explanation"]
    assert explanations and "本条规定了检测频次" in explanations[0].text
    # 条文说明不能跟正文混在一片里
    assert "本条规定了检测频次" not in main[0].text


def test_broken_article_number_fragment_is_not_a_block() -> None:
    lines = corpus.to_lines(
        [("4.1.7 溅水盘与顶板的距离应符合表7.\n1.7的规定。\n4.1.8 通道上方宜设有喷头。", 1, 1)]
    )
    blocks = corpus.build_blocks(lines)
    corpus.tag_sections(blocks)
    numbers = {b.number for b in blocks}
    assert "1.7" not in numbers, "「1.7的规定。」是上一句的尾巴，不该成为独立条文"
    merged = next(b for b in blocks if b.number == "4.1.7")
    assert "表7.1.7的规定" in merged.text


def test_section_path_follows_number_prefix() -> None:
    text = "7 检测\n7.1 一般要求\n7.2 检测内容\n7.1 一般要求\n7.1.1 建筑消防设施应每年至少检测一次。"
    blocks = corpus.build_blocks(corpus.to_lines([(text, 1, 1)]))
    target = next(b for b in blocks if b.number == "7.1.1")
    assert target.section_path == "检测 > 一般要求"


def test_duplicate_number_with_same_text_is_marked_duplicate() -> None:
    text = "5.1 应设置操作规程。\n5.2 应建立值班制度。\n5.1 应设置操作规程。"
    blocks = corpus.build_blocks(corpus.to_lines([(text, 1, 1)]))
    corpus.tag_sections(blocks)
    kinds = [b.section_type for b in blocks if b.number == "5.1"]
    assert kinds == ["main", "duplicate"]


# --------------------------------------------------------------------- 去重
def test_dedupe_only_merges_exact_text() -> None:
    texts = [
        "消防供水设施的巡查内容见表C.1中“消防供水设施”部分。",
        "消防供水设施的巡查内容见表C.1中“消防供水设施”部分。",
        # 只差系统名：相似度很高，但它是另一条要求，不能并
        "消火栓灭火系统的巡查内容见表C.1中“消火栓灭火系统”部分。",
    ]
    merged = corpus.dedupe(texts)
    assert merged[0] is None
    assert merged[1] == 0
    assert merged[2] is None


def test_near_duplicate_pairs_reports_without_merging() -> None:
    texts = [
        "探测器周围水平距离0.5m内不应有遮挡物。",
        "探测器周围水平距离0.5m内不应有遮挡物，且应便于维护。",
        "完全不相干的一句话，讲的是排烟风机的风速要求。",
    ]
    pairs = corpus.near_duplicate_pairs(texts, threshold=0.6)
    assert (0, 1) in [(a, b) for a, b, _ in pairs]
    assert all(a != b for a, b, _ in pairs)


# --------------------------------------------------------------------- 长条文
def test_split_long_keeps_each_part_under_limit() -> None:
    text = "本条要求如下。" + "第一项要求应当满足相应的技术条件；" * 40
    parts = corpus.split_long(text, max_chars=200, overlap=20)
    assert len(parts) > 1
    assert all(len(part) <= 240 for part in parts)


# --------------------------------------------------------------------- 端到端
def test_chunk_documents_uses_article_mode_for_regulation_like_text() -> None:
    chunks = chunk_documents(pages(PAGE_1, PAGE_2), make_settings())
    assert chunks, "规范类语料应切出多条"
    assert {c.metadata["chunk_mode"] for c in chunks} == {"article"}

    first = next(c for c in chunks if c.page_content.startswith("4.1.1 "))
    assert first.metadata["article"] == "4.1.1"
    assert first.metadata["section_type"] == "main"
    # 内部 0 基，出库时 +1；这里 page 0 表示物理第 1 页
    assert first.metadata["page"] == 0
    assert first.metadata["page_printed"] == 1
    assert "第 1 页" not in first.page_content

    explanation = next(c for c in chunks if "本条规定了检测频次" in c.page_content)
    assert explanation.metadata["section_type"] == "explanation"


def test_chunk_documents_falls_back_to_fixed_for_plain_text() -> None:
    diary = "今天在现场核对了探测器编码。" * 60
    chunks = chunk_documents(pages(diary), make_settings())
    assert chunks
    assert chunks[0].metadata["chunk_mode"] == "fixed"
    assert len(chunks) > 1


def test_chunk_index_is_contiguous_after_dedupe() -> None:
    chunks = chunk_documents(pages(PAGE_1, PAGE_2), make_settings())
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


# --------------------------------------------------------------------- 入库贯通
def _regulation_text() -> str:
    lines = ["4 总则", "", "4.1 一般要求", ""]
    for index in range(1, 9):
        lines.append(f"4.1.{index} 建筑消防设施的检测要求第{index}项应满足本条规定。")
    lines += ["", "4.2 检测内容", ""]
    for index in range(1, 9):
        lines.append(f"4.2.{index} 检测内容第{index}项应按表D.1的规定执行。")
    lines += ["", "条文说明", "", "4.1.1 本条规定了检测的总体要求，是为了统一口径。"]
    return "\n".join(lines)


def test_ingest_carries_article_and_section_type_to_citation(service) -> None:
    """入库 → 检索 → 引用，条文号与「正文/条文说明」要一路带到接口返回。"""
    service.ingest("规范.txt", _regulation_text().encode("utf-8"))

    # FakeEmbeddings 的相似度没有语义，取全量再筛，别让断言依赖假的排名
    hits = service.search("检测的总体要求", top_k=20).hits
    assert hits
    explanation = next(h for h in hits if "统一口径" in h.content)
    assert explanation.article == "4.1.1"
    assert explanation.section_type == "explanation"

    main = next(h for h in hits if "第1项应满足本条规定" in h.content)
    assert main.section_type == "main"
    assert main.section_path == "总则 > 一般要求"
