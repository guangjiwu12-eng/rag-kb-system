#!/usr/bin/env python3
"""引用自校验：答案里每个 [n] 的关键数值，是否真的出现在被引片段里。

解决什么问题
------------
「引用看起来对」是 RAG 最容易骗过自己的一环：模型把 [2] 标在一句结论后面，
片段里可能压根没有那个数——它是在归纳、换算、或者从上一段借来的。
人工抽 20 条能发现，但抽完就完了，下一轮改完 prompt 还得再抽一遍。

这个脚本把那次抽检固化成一条命令：抽句子、抽数值、回查片段、出结论。
它**不判断观点对不对**，只回答一个能用机器判的问题：这个数字在不在你指的这段话里。

三种结论，都保留原样不折算
------------------------
- 支持：句子里每个关键数值都能在被引片段里原样找到
- 部分：数字找得到但单位/口径对不上，或只找到一部分
- 不支持：数值在片段里根本不存在——这是最需要人看的一档
- 无数值：句子没有任何数值，脚本无法判定，单独计数

**不要把「不支持」直接等同于幻觉**：合理的换算（1.5h → 90min）、跨片段合并、
或者引用编号标错，都会落到这一档。脚本的产出是「该看哪几条」，不是判决。

用法
----
    uvicorn app.main:app
    python scripts/verify_citations.py                    # 全部题目
    python scripts/verify_citations.py --ids n2 n16 n21   # 只看某几题
    python scripts/verify_citations.py --sample 20        # 随机抽 20 题
    python scripts/verify_citations.py --fail-on-unsupported   # 有「不支持」就非 0 退出

产物：eval/results/citation_autocheck.md（每次覆盖，可 diff）
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"

# [1] / [1][3] / [1,3]
CITATION_MARK = re.compile(r"\[(\d{1,2}(?:\s*[,，]\s*\d{1,2})*)\]")
# 条文号引用，例如「第 7.1.1 条」
ARTICLE_REF = re.compile(r"第\s*([0-9]+(?:\.[0-9]+){1,3})\s*条")

# 关键数值：数字 + 可选单位。单位选择宁可宽一点——漏抽一个数值，
# 就会把「没核对」记成「核对通过」，这是这个脚本最不该犯的错。
# 拉丁单位放前面并显式支持 a/b 形式（m/min、m3/h），否则「7.5m/min」会被截成「7.5m」，
# 于是「单位对不上」被误判成「数值不存在」。
_UNIT = (
    r"(?:%|‰|℃|°C|°"
    r"|[a-zA-Zµ][a-zA-Z0-9³²]*/[a-zA-Z0-9³²]+"
    r"|[a-zA-Zµ][a-zA-Z0-9³²]*"
    r"|小时|分钟|秒钟|秒|个月|年|月|日|天|人|只|个|次|倍|米|升|吨|元|层|条|款|项|组|台|套|根|处|点|类|级)"
)
KEY_VALUE = re.compile(rf"\d+(?:\.\d+)?(?:\s*[~～\-—]\s*\d+(?:\.\d+)?)?(?:\s*{_UNIT})?")
SENTENCE_SPLIT = re.compile(r"(?<=[。！？；\n])")

# 单位等价类：不同写法指向同一个物理量，比对前统一
_UNIT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("min", ("min", "分钟", "minutes", "minute")),
    ("s", ("s", "秒", "sec", "seconds", "second")),
    ("h", ("h", "hr", "小时", "hour", "hours")),
    ("m", ("m", "米", "meter", "metre")),
    ("mm", ("mm", "毫米")),
    ("cm", ("cm", "厘米")),
    ("m/s", ("m/s", "米/秒", "米每秒")),
    ("%", ("%", "％", "百分")),
    ("℃", ("℃", "°C", "摄氏度")),
    ("m2", ("m2", "m²", "㎡", "平方米")),
    ("m3", ("m3", "m³", "立方米")),
    ("人", ("人", "名")),
    ("h-1", ("次/年", "每年")),
)


def norm(text: str) -> str:
    """比对用归一：全角转半角、去空白、统一波浪号与乘号。"""
    return (
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))
        .replace("～", "~")
        .replace("—", "-")
        .replace("－", "-")
    )


def unit_family(unit: str) -> str:
    key = norm(unit).lower()
    for family, aliases in _UNIT_ALIASES:
        if key in {norm(alias).lower() for alias in aliases}:
            return family
    return key


@dataclass(slots=True)
class Fact:
    """句子里的一个关键数值。"""

    raw: str
    number: str
    unit: str = ""

    @property
    def family(self) -> str:
        return unit_family(self.unit) if self.unit else ""


@dataclass(slots=True)
class Verdict:
    case_id: str
    sentence: str
    citation_index: int
    status: str  # supported | partial | unsupported | no_fact | missing_citation
    detail: str = ""
    facts: list[Fact] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def extract_facts(sentence: str) -> list[Fact]:
    """抽句子里的关键数值。纯年份、条文号、编号不算——那些不是在陈述事实。"""
    facts: list[Fact] = []
    for match in KEY_VALUE.finditer(sentence):
        raw = match.group(0).strip()
        number = re.match(r"\d+(?:\.\d+)?", raw)
        if not number:
            continue
        unit = raw[number.end() :].strip()
        value = number.group(0)
        # 「第 7.1.1 条」里的 7.1.1 是条文号，不是结论里的量
        if _inside_article_ref(sentence, match.start()):
            continue
        # 单独出现的四位数多半是年份
        if not unit and re.fullmatch(r"(19|20)\d{2}", value):
            continue
        # 「第 1 款」「第 5 项」是条款序号，和条文号一样不属于结论里的量
        if unit in ("款", "项"):
            continue
        # 标准编号（GB 25201-2010、GB/T 40248-2021）与行首序号（1. 2. 3.）也不是事实。
        # 这两类噪声能占到「不支持」的一半以上——报告里全是「数值 25201-2010 不在片段里」，
        # 真正该看的「把 2 倍写成 2.5 倍」反而被淹掉。
        if _is_std_code(sentence, match.start(), raw):
            continue
        if not unit and _is_list_marker(sentence, match.start(), match.end()):
            continue
        # 「[3]」是引用角标，回查它等于什么也没查
        if _inside_bracket(sentence, match.start(), match.end()):
            continue
        facts.append(Fact(raw=raw, number=value, unit=unit))
    return facts


# 数字紧跟在这些前缀后面，就是标准编号的一部分，不是一个可比的量
_STD_PREFIX = re.compile(r"(?:GB\s*/?\s*T?|GBT)\s*$")
_STD_BODY = re.compile(r"\d{4,5}\s*[-—–]\s*\d{4}")


def _is_std_code(sentence: str, start: int, raw: str) -> bool:
    if _STD_PREFIX.search(norm(sentence[:start])):
        return True
    return bool(_STD_BODY.fullmatch(norm(raw)))


# 行首（或 - * 之后）的「1.」是 Markdown 列表序号，不是数量
_LIST_PREFIX = re.compile(r"(?:^|[\s\-*—])$")


def _is_list_marker(sentence: str, start: int, end: int) -> bool:
    """「2. **形成书面报告**」里的 2 是列表序号，不是数量。

    两个条件缺一不可：数字后面紧跟 . 、 ) ，**并且**前面是行首/空白/短横线。
    只看后面会把「直径 3.5 米」的 3 误判；只看前面会把「共 2 次」的 2 误判。
    """
    if sentence[end : end + 1] not in (".", "、", ")", "）"):
        return False
    return bool(_LIST_PREFIX.search(sentence[:start]))


def _inside_article_ref(sentence: str, start: int) -> bool:
    return any(match.start() <= start < match.end() for match in ARTICLE_REF.finditer(sentence))


def _inside_bracket(sentence: str, start: int, end: int) -> bool:
    """数字整个落在 [] 里，就是引用角标 [3]，不是数值。"""
    left = sentence.rfind("[", 0, start + 1)
    right = sentence.find("]", end)
    return left != -1 and right != -1 and "]" not in sentence[left:start]


def _find(fact: Fact, haystack: str) -> str:
    """在片段里找这个数值：先找「数字+单位」，再找裸数字。返回命中级别。"""
    compact = norm(haystack)
    if fact.unit:
        for alias in _aliases(fact.unit):
            if norm(fact.number + alias) in compact:
                return "exact"
    if norm(fact.number) in compact:
        return "number_only"
    return ""


def _aliases(unit: str) -> list[str]:
    family = unit_family(unit)
    for name, aliases in _UNIT_ALIASES:
        if name == family:
            return [alias for alias in aliases if alias]
    return [unit]


_ONLY_MARKS = re.compile(r"^(?:\s*\[\d{1,2}(?:\s*[,，]\s*\d{1,2})*\]\s*[。；，]?\s*)+$")


def split_sentences(answer: str) -> list[str]:
    """按句切开，但把「只有 [n] 的碎段」并回上一句。

    模型常把标点写在编号后面（「……一年一次 [1]。」），按句切会切出只剩 `[1]。` 的碎片，
    它没有内容、却会被算成一条「无数值」的核对记录，把统计口径冲稀。
    """
    segments: list[str] = []
    for part in SENTENCE_SPLIT.split(answer or ""):
        part = part.strip()
        if not part:
            continue
        if segments and _ONLY_MARKS.match(part):
            segments[-1] = f"{segments[-1]}{part}"
        else:
            segments.append(part)
    return segments


def check_answer(case_id: str, answer: str, citations: list[dict]) -> list[Verdict]:
    """一条答案 → 逐句逐引用的判定。纯函数，便于单测。"""
    by_index = {int(c.get("index") or 0): c for c in citations}
    verdicts: list[Verdict] = []
    cited_any = False

    for sentence in split_sentences(answer):
        marks = CITATION_MARK.findall(sentence)
        numbers = sorted({int(n) for group in marks for n in re.findall(r"\d+", group)})
        if not numbers:
            continue
        cited_any = True
        facts = extract_facts(CITATION_MARK.sub("", sentence))
        for index in numbers:
            citation = by_index.get(index)
            if citation is None:
                verdicts.append(
                    Verdict(
                        case_id=case_id,
                        sentence=sentence,
                        citation_index=index,
                        status="missing_citation",
                        detail=f"答案引用了 [{index}]，但接口只返回 {len(citations)} 条引用",
                    )
                )
                continue
            if not facts:
                verdicts.append(
                    Verdict(
                        case_id=case_id,
                        sentence=sentence,
                        citation_index=index,
                        status="no_fact",
                        detail="该句没有可核对的数值",
                    )
                )
                continue

            content = str(citation.get("content") or "")
            missing: list[str] = []
            levels: list[str] = []
            for fact in facts:
                level = _find(fact, content)
                if not level:
                    missing.append(fact.raw)
                else:
                    levels.append(level)

            if not missing:
                status = "supported" if all(level == "exact" for level in levels) else "partial"
                detail = "" if status == "supported" else "数值在，但单位/写法对不上"
            elif levels:
                status, detail = "partial", "部分数值不在片段里：" + "、".join(missing)
            else:
                status, detail = "unsupported", "数值均不在片段里：" + "、".join(missing)

            # 顺手核条文号：答案写了「第 X 条」，引用却指向另一条，是引用标错
            article = str(citation.get("article") or "")
            mentioned = ARTICLE_REF.findall(sentence)
            if status in ("supported", "partial") and article and mentioned:
                if not any(item.replace(" ", "") == article for item in mentioned):
                    status = "partial"
                    detail = (detail + "；" if detail else "") + f"条文号不符（答案写 {mentioned[0]}，片段是 {article}）"

            verdicts.append(
                Verdict(
                    case_id=case_id,
                    sentence=sentence,
                    citation_index=index,
                    status=status,
                    detail=detail,
                    facts=facts,
                    missing=missing,
                )
            )

    if not cited_any and (answer or "").strip():
        verdicts.append(
            Verdict(
                case_id=case_id,
                sentence=(answer or "").strip()[:60],
                citation_index=0,
                status="no_fact",
                detail="整段答案没有标任何 [n]",
            )
        )
    return verdicts


# --------------------------------------------------------------------- 取数
# 只打本机服务，一律不走代理：环境里 http_proxy 指向某个本地端口时，urllib 会把
# 127.0.0.1 的请求也塞进去，报出来是 502 —— 看着像服务挂了，其实是代理回的。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_json(url: str, payload: dict, api_key: str | None, timeout: int = 180) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise SystemExit(f"{url} 返回 {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"连不上 {url}：{exc}（先把服务起起来）") from exc


def load_questions(ids: list[str] | None, sample: int | None, seed: int) -> list[dict]:
    import yaml

    path = ROOT / "eval" / "questions.yaml"
    cases = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if ids:
        wanted = {item.lower() for item in ids}
        cases = [case for case in cases if str(case.get("id", "")).lower() in wanted]
        missing = wanted - {str(case.get("id", "")).lower() for case in cases}
        if missing:
            raise SystemExit(f"题库里没有这些题号：{'、'.join(sorted(missing))}")
    if sample and sample < len(cases):
        random.Random(seed).shuffle(cases)
        cases = cases[:sample]
    return cases


# --------------------------------------------------------------------- 报告
STATUS_LABEL = {
    "supported": "支持",
    "partial": "部分",
    "unsupported": "不支持",
    "no_fact": "无数值",
    "missing_citation": "引用缺失",
}


def build_report(verdicts: list[Verdict], model: str, answers: dict[str, str]) -> str:
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.status] = counts.get(verdict.status, 0) + 1

    lines = [
        "# 引用自校验报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}　模型：{model}",
        "",
        "口径：抽答案里每个 `[n]` 所在句的关键数值，回查被引片段。",
        "「不支持」不等于幻觉——换算、跨片段合并、引用标错都会落进这一档，**是给人看的清单，不是判决**。",
        "",
        "| 结论 | 条数 |",
        "| --- | --- |",
    ]
    for status in ("supported", "partial", "unsupported", "no_fact", "missing_citation"):
        lines.append(f"| {STATUS_LABEL[status]} | {counts.get(status, 0)} |")
    lines.append("")

    for status in ("unsupported", "partial", "missing_citation"):
        items = [v for v in verdicts if v.status == status]
        if not items:
            continue
        lines += [f"## {STATUS_LABEL[status]}（{len(items)}）", ""]
        for item in items:
            lines.append(f"- **{item.case_id}** `[{item.citation_index}]` {item.detail}")
            lines.append(f"  - 句子：{item.sentence[:120]}")
            if item.facts:
                lines.append(f"  - 核对数值：{'、'.join(f.raw for f in item.facts)}")
        lines.append("")

    lines += ["## 逐题的答案与引用", ""]
    for case_id, answer in answers.items():
        lines.append(f"### {case_id}")
        lines.append("")
        lines.append(answer.strip() or "（空）")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="引用自校验：答案数值 vs 被引片段")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--ids", nargs="*", default=None, help="只跑指定题号")
    parser.add_argument("--sample", type=int, default=None, help="随机抽 N 题")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(RESULTS_DIR / "citation_autocheck.md"))
    parser.add_argument(
        "--fail-on-unsupported",
        action="store_true",
        help="出现「不支持」时返回非 0，可当 CI 闸门",
    )
    args = parser.parse_args()

    cases = load_questions(args.ids, args.sample, args.seed)
    if not cases:
        raise SystemExit("没有可跑的题")

    verdicts: list[Verdict] = []
    answers: dict[str, str] = {}
    model = "?"

    for position, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or f"#{position}")
        payload = http_json(
            f"{args.base_url}/api/v1/query",
            {"question": case["question"]},
            args.api_key,
        )
        answer = str(payload.get("answer") or "")
        citations = payload.get("citations") or []
        model = str(payload.get("model") or model)
        answers[case_id] = answer

        found = check_answer(case_id, answer, citations)
        verdicts.extend(found)
        worst = min(
            (v.status for v in found),
            key=lambda s: ["unsupported", "missing_citation", "partial", "no_fact", "supported"].index(s),
            default="no_fact",
        )
        print(
            f"  [{position}/{len(cases)}] {case_id:<5} {STATUS_LABEL[worst]:<4} "
            f"核对 {len(found)} 条",
            file=sys.stderr,
        )

    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.status] = counts.get(verdict.status, 0) + 1
    checked = sum(counts.get(k, 0) for k in ("supported", "partial", "unsupported"))

    print("")
    print(f"{'结论':<10}{'条数':>6}")
    print("-" * 18)
    for status in ("supported", "partial", "unsupported", "no_fact", "missing_citation"):
        print(f"{STATUS_LABEL[status]:<10}{counts.get(status, 0):>6}")
    if checked:
        print(
            f"\n可自动核对 {checked} 条：支持 {counts.get('supported', 0)} / "
            f"部分 {counts.get('partial', 0)} / 不支持 {counts.get('unsupported', 0)}"
        )
        print("写进 README 时用绝对条数，不要换算成百分比：样本量的置信区间撑不住百分比。")
    print("「部分 / 不支持」的逐条清单在报告里，人工只需复核这几条。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(verdicts, model, answers), encoding="utf-8")
    print(f"\n报告：{out}")

    if args.fail_on_unsupported and counts.get("unsupported"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
