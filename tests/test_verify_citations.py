"""引用自校验的判定逻辑测试：重点是「不放过」和「不误杀」。"""

from __future__ import annotations

from scripts.verify_citations import (
    check_answer,
    extract_facts,
    split_sentences,
    unit_family,
)


def cite(index: int, content: str, article: str = "") -> dict:
    return {"index": index, "content": content, "article": article}


def test_extract_facts_picks_number_and_unit() -> None:
    facts = extract_facts("应确保消防水泵在报警后5.0min内正常工作。")
    assert [(f.number, f.unit) for f in facts] == [("5.0", "min")]


def test_extract_facts_skips_article_numbers() -> None:
    assert extract_facts("依据第 7.1.1 条执行。") == []


def test_extract_facts_skips_bare_four_digit_year() -> None:
    # 不带单位的四位数多半是年份/编号，不是结论里的量
    assert extract_facts("设备编号 2010 已停用。") == []


def test_extract_facts_keeps_percent_and_ranges() -> None:
    facts = extract_facts("偏差不大于10%，压力表量程为2.0倍~2.5倍。")
    numbers = [f.number for f in facts]
    assert "10" in numbers
    assert "2.0" in numbers


# ------------------------------------------------------------ 三类「看起来像数值」的噪声
# 上一轮报告里 9 条「不支持」有 8 条是这三类噪声，真正该看的数值对不上反而被淹掉。
def test_extract_facts_skips_standard_codes() -> None:
    """GB25201-2010、GB/T 40248-2021 是标准编号，不是量。"""
    assert extract_facts("GB25201-2010 第 7.1.1 条规定，每年至少检测一次 [1]。") == []
    assert extract_facts("- GB/T 40248-2021 第 7.6.7 条，每年应至少检查一次 [3]。") == []


def test_extract_facts_skips_list_markers() -> None:
    """Markdown 列表序号 2. 3. 与「直径 3.5 米」的 3 必须分得开。"""
    assert extract_facts("2. **形成书面报告并通报**：应形成书面报告 [4]。") == []
    assert [f.raw for f in extract_facts("直径 3.5 米的管道每季度检查一次。")] == ["3.5 米"]


def test_extract_facts_skips_citation_markers() -> None:
    """[3] 是引用角标，回查它等于没查。"""
    assert extract_facts("应在 5min 内启动[2]。") != []
    assert all("2" != f.number for f in extract_facts("应在 5min 内启动[2]。"))
    assert [f.raw for f in extract_facts("[1] 附录列了 3 个探头。")] == ["3 个"]


def test_unit_family_matches_synonyms() -> None:
    assert unit_family("分钟") == unit_family("min")
    assert unit_family("小时") == unit_family("h")
    assert unit_family("m²") == unit_family("㎡")


def test_supported_when_number_and_unit_present() -> None:
    verdicts = check_answer(
        "n16",
        "应在报警后 5.0min 内正常工作 [1]。",
        [cite(1, "机械应急启动时，应确保消防水泵在报警后5.0min内正常工作。")],
    )
    assert [v.status for v in verdicts] == ["supported"]


def test_unsupported_when_number_absent() -> None:
    verdicts = check_answer(
        "x1",
        "下降速度应不大于 7.5m/min [1]。",
        [cite(1, "防火卷帘应下降至距地面1.8m处。")],
    )
    assert [v.status for v in verdicts] == ["unsupported"]
    assert verdicts[0].missing == ["7.5m/min"]


def test_partial_when_unit_does_not_apply() -> None:
    # 数字在，但单位对不上：模型把 m 写成了 min，属于「部分」，要人看一眼
    verdicts = check_answer(
        "x2",
        "间距不应小于 0.5s [1]。",
        [cite(1, "点型探测器至墙壁的水平距离不应小于0.5m。")],
    )
    assert [v.status for v in verdicts] == ["partial"]


def test_partial_when_article_number_disagrees() -> None:
    verdicts = check_answer(
        "x3",
        "应每年至少检测 1 次 [1]，见第 7.1.2 条。",
        [cite(1, "7.1.1 建筑消防设施应每年至少检测一次。", article="7.1.1")],
    )
    assert verdicts[0].status == "partial"
    assert "条文号不符" in verdicts[0].detail


def test_missing_citation_when_index_out_of_range() -> None:
    verdicts = check_answer("x4", "结论是 5.0min [3]。", [cite(1, "任意内容")])
    assert [v.status for v in verdicts] == ["missing_citation"]


def test_no_fact_when_sentence_has_no_number() -> None:
    verdicts = check_answer("x5", "探测器应水平安装 [1]。", [cite(1, "探测器宜水平安装。")])
    assert [v.status for v in verdicts] == ["no_fact"]


def test_answer_without_any_marker_is_flagged() -> None:
    verdicts = check_answer("x6", "根据现有资料无法回答该问题。", [])
    assert [v.status for v in verdicts] == ["no_fact"]
    assert "没有标任何" in verdicts[0].detail


def test_multiple_citations_in_one_sentence_are_checked_separately() -> None:
    verdicts = check_answer(
        "x7",
        "分别不高于 0.25s [1] 和 5s [2]。",
        [
            cite(1, "高危险场所灯具光源应急点亮的响应时间不应大于0.25s。"),
            cite(2, "其他场所不应大于3s。"),
        ],
    )
    assert verdicts[0].status == "supported"
    # 第二个引用的片段写的是 3s，答案说 5s
    assert verdicts[1].status == "unsupported"


def test_split_sentences_keeps_bullets_together() -> None:
    parts = split_sentences("第一句。[1] 第二句也很长。[2]\n第三句。[3]")
    assert len(parts) == 3
