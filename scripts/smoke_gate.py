#!/usr/bin/env python3
"""smoke 闸门：10 题子集只判绝对下限，不做基线对比。

为什么不复用 `scripts/diff_eval.py`
--------------------------------
diff_eval 做的是「本次 vs 基线」的差值判定，适合 50 题。smoke 只有 7 道有答案题，
**一题就是 14 个百分点**，差值全落在噪声里，比一次红一次绿，最后没人看。
smoke 要回答的只有一个问题：有没有跌破底线。

完整评测（50 题）的回归判定交给 `python scripts/diff_eval.py --fail-on-regression`。

用法
----
    python scripts/smoke_gate.py                       # 用 eval/results/latest.json
    python scripts/smoke_gate.py --candidate x.json --floors eval/smoke_floors.json

退出码：0 达标、1 未达下限、3 缺文件
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FLOORS = ROOT / "eval" / "smoke_floors.json"
DEFAULT_CANDIDATE = ROOT / "eval" / "results" / "latest.json"


def load(path: Path) -> dict:
    if not path.is_file():
        print(f"读不到文件：{path}", file=sys.stderr)
        raise SystemExit(3)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="按绝对下限判定 smoke 结果")
    parser.add_argument("--floors", default=str(DEFAULT_FLOORS))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    args = parser.parse_args()

    floors = load(Path(args.floors))
    candidate = load(Path(args.candidate))
    failures: list[str] = []

    expect = floors.get("suite") or {}
    actual = candidate.get("counts") or {}
    print(f"下限：{Path(args.floors).name}   本次：{Path(args.candidate).name}\n")
    for key in ("answerable", "unanswerable"):
        if expect.get(key) is not None and expect[key] != actual.get(key):
            failures.append(
                f"题集不符：期望 {key}={expect[key]}，实际 {actual.get(key)}"
                "（smoke 必须跑 eval/smoke_questions.yaml）"
            )
    if failures:
        for item in failures:
            print(f"  ✗ {item}")
        return 1

    metrics = candidate.get("metrics") or {}
    print(f"{'指标':<18}{'下限':>10}{'本次':>10}   判定")
    print("-" * 50)
    for name, floor in (floors.get("metrics") or {}).items():
        if name not in metrics:
            failures.append(f"{name} 没有产出，无法判定")
            print(f"{name:<18}{'>=' + format(float(floor), '.2f'):>10}{'缺失':>10}   无法判定")
            continue
        value = float(metrics[name])
        ok = value >= float(floor)
        print(f"{name:<18}{float(floor):>10.4f}{value:>10.4f}   {'ok' if ok else '低于下限'}")
        if not ok:
            failures.append(f"{name}: {value:.4f} < 下限 {float(floor):.4f}")

    if "false_answer_rate" not in metrics:
        print(
            "\n[warn] 只测了检索层（--no-query），拒答/幻觉指标没产出；"
            "那部分由 tests/test_abstain_paths.py 守"
        )

    if failures:
        print("\n未达下限：")
        for item in failures:
            print(f"  ✗ {item}")
        return 1
    print("\n通过：达到 smoke 下限。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
