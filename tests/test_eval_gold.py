"""resolve_gold：一个锚点必须能落到多个切片上。

`chunk_overlap` 会让同一句话同时落在相邻两片里，按页切开的长段也会跨页重复。
第一版只认「第一个匹配的切片」，命中相邻那片就算未命中——这一条 bug 让 hit@4
虚低 17 个百分点，而且看起来像是模型变差了。这里把四档降级与边界都钉住。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# eval/ 不是包（没有 __init__.py），按文件路径加载；必须先登记到 sys.modules，
# 否则 @dataclass 在解析字段类型时取不到模块命名空间（sys.modules[cls.__module__] 为 None）。
_EVAL_FILE = Path(__file__).resolve().parent.parent / "eval" / "eval.py"
_spec = importlib.util.spec_from_file_location("rag_eval", _EVAL_FILE)
assert _spec is not None and _spec.loader is not None
ev = importlib.util.module_from_spec(_spec)
sys.modules["rag_eval"] = ev
_spec.loader.exec_module(ev)

ANCHOR = "不同类型的探测器应有不少于设备总数百分之一的备品"


def chunk(
    doc_id: str, index: int, source: str, page: int, text: str
) -> tuple[dict, str]:
    return (
        {"doc_id": doc_id, "chunk_index": index, "source": source, "page": page},
        text,
    )


def test_overlapping_chunks_are_all_gold() -> None:
    """同一句话落在相邻两片：两片都算 gold，命中任意一片即成功。"""
    rows = [
        chunk("d1", 3, "a.pdf", 7, f"……{ANCHOR}……"),
        chunk("d1", 4, "a.pdf", 7, f"{ANCHOR}，其余要求见下条……"),
        chunk("d2", 0, "b.pdf", 1, "完全无关的另一段"),
    ]
    keys, level = ev.resolve_gold(rows, ev.Gold(anchor=ANCHOR, doc="a.pdf", page=7))

    assert level == 0
    assert keys == [("d1", 3), ("d1", 4)]


def test_pdf_line_breaks_do_not_block_match() -> None:
    """PDF 抽取会塞进换行和缩进，比对前必须去空白。"""
    rows = [chunk("d1", 0, "a.pdf", 7, "不同类型的探测器应有不少于\n  设备总数百分之一的备品")]
    keys, level = ev.resolve_gold(rows, ev.Gold(anchor=ANCHOR, doc="a.pdf", page=7))

    assert level == 0
    assert keys == [("d1", 0)]


def test_wrong_page_same_document_falls_back_to_level_1() -> None:
    """页码标错但文件对：仍然算命中，层级 1 提醒标注要修。"""
    rows = [chunk("d1", 5, "a.pdf", 9, ANCHOR)]
    keys, level = ev.resolve_gold(rows, ev.Gold(anchor=ANCHOR, doc="a.pdf", page=7))

    assert level == 1
    assert keys == [("d1", 5)]


def test_other_document_falls_back_to_level_2() -> None:
    """只出现在别的文件里：命中，但层级 2 说明 gold 文档标错了。"""
    rows = [chunk("d9", 1, "b.pdf", 1, ANCHOR)]
    keys, level = ev.resolve_gold(rows, ev.Gold(anchor=ANCHOR, doc="a.pdf", page=7))

    assert level == 2
    assert keys == [("d9", 1)]


def test_absent_anchor_is_unresolved() -> None:
    rows = [chunk("d1", 0, "a.pdf", 7, "完全无关的另一段")]
    keys, level = ev.resolve_gold(rows, ev.Gold(anchor=ANCHOR, doc="a.pdf", page=7))

    assert keys == []
    assert level == -1


def test_too_short_anchor_is_rejected() -> None:
    """锚点太短会到处误命中，必须直接报错而不是静默匹配一堆切片。"""
    with pytest.raises(ValueError, match="锚点太短"):
        ev.resolve_gold([chunk("d1", 0, "a.pdf", 1, "年假")], ev.Gold(anchor="年假"))


# ---------------------------------------------------------------- 页码区间与别名
def meta_chunk(**meta: object) -> tuple[dict, str]:
    payload = {"doc_id": "d1", "chunk_index": 0, "source": "a.pdf", "page": 7}
    payload.update(meta)
    return payload, str(meta.get("text") or ANCHOR)


def test_page_range_accepts_both_ends() -> None:
    """一条长条文跨两页：标一段区间，两端都算页对得上。"""
    gold = ev.Gold(anchor=ANCHOR, doc="a.pdf", page=88, page_end=89)
    for page in (88, 89):
        _, level = ev.resolve_gold([meta_chunk(page=page)], gold)
        assert level == 0, f"第 {page} 页应落在区间内"


def test_page_range_rejects_outside() -> None:
    gold = ev.Gold(anchor=ANCHOR, doc="a.pdf", page=88, page_end=89)
    _, level = ev.resolve_gold([meta_chunk(page=90)], gold)
    assert level == 1


def test_page_end_defaults_to_page() -> None:
    """不写 page_end 就只看一页，别把区间悄悄放宽。"""
    assert ev.Gold(anchor=ANCHOR, page=88).pages == {88}
    # 写反了也要能收敛成区间
    assert ev.Gold(anchor=ANCHOR, page=89, page_end=88).pages == {88, 89}


def test_printed_page_matches() -> None:
    """标注照印刷页抄：物理页 7、印刷页 12，标 12 也算页对得上。"""
    row = meta_chunk(page=7, page_printed=12)
    _, level = ev.resolve_gold([row], ev.Gold(anchor=ANCHOR, doc="a.pdf", page=12))
    assert level == 0


def test_no_page_given_accepts_any_page() -> None:
    _, level = ev.resolve_gold([meta_chunk(page=7)], ev.Gold(anchor=ANCHOR, doc="a.pdf"))
    assert level == 0


def test_also_in_alias_resolves_merged_duplicate() -> None:
    """跨标准同义条文入库时并成一片：gold 写在被并掉的那部标准上也必须答得对。"""
    row = meta_chunk(
        page=94,
        also_in="GB50084-2017_自动喷水灭火系统设计规范.pdf:73:9.3.1",
        text="另一段合并后保留下来的文字",
    )
    gold = ev.Gold(anchor=ANCHOR, doc="GB50084-2017_自动喷水灭火系统设计规范.pdf", page=73)
    keys, level = ev.resolve_gold([row], gold)

    assert keys == [("d1", 0)]
    assert level == 0


def test_also_in_with_other_page_does_not_match() -> None:
    """别名里页码对不上就不算：同一部标准里别的条文不能顶包。"""
    row = meta_chunk(page=94, also_in="b.pdf:74:9.3.2", text="另一段文字")
    keys, level = ev.resolve_gold([row], ev.Gold(anchor=ANCHOR, doc="b.pdf", page=73))

    assert keys == []
    assert level == -1

