#!/usr/bin/env bash
# 10 题 smoke：起服务 → 只测检索层 → 关服务 → 交给 eval_gate 判定。
#
# 只测检索层（--no-query）是刻意的：不花 token、不依赖模型 key，提交前和 CI 都能跑。
# 拒答行为由单测守（tests/test_abstain_paths.py），不靠这套 smoke。
#
# 用法：
#   bash scripts/smoke_eval.sh                      # 与 eval/baseline-smoke.json 比
#   PY=venv/bin/python PORT=8051 bash scripts/smoke_eval.sh
#   bash scripts/smoke_eval.sh --write-baseline      # 语料变了之后重定基线
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
PORT=${PORT:-8041}
WORK=${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/rag-smoke.XXXXXX")}
SNAP=$WORK/chroma
LOG=$WORK/server.log

cd "$ROOT" || exit 1

# 用向量库副本：跑 smoke 时别被别的进程入库干扰，也别让评测去写正式库。
# 每次一个全新临时目录（mktemp -d），不用 rm -rf，既省事也不会被安全策略拦。
cp -r chroma_db "$SNAP" || exit 1
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; rm -rf "$WORK" 2>/dev/null' EXIT

"$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SRV=$!

# bge-m3 首次加载要 30~120 秒，机器同时在跑别的任务时更久。
# 这里给足 6 分钟，并且每轮把加载进度打出来，免得误判成「服务起不来」。
for i in $(seq 1 180); do
  if curl -s --max-time 3 "http://127.0.0.1:$PORT/api/v1/health" | grep -q '"status"'; then
    echo "服务就绪（等待 $((i * 2)) 秒）"
    break
  fi
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "服务进程已退出，日志："; tail -20 "$LOG"; exit 1
  fi
  [ $((i % 15)) -eq 0 ] && echo "  等待模型加载… $((i * 2)) 秒"
  sleep 2
done

curl -s --max-time 5 "http://127.0.0.1:$PORT/api/v1/health" | grep -q '"status"' || {
  echo "服务 6 分钟没就绪，日志："; tail -20 "$LOG"; exit 1;
}

echo "=== 10 题 smoke（仅检索层）==="
"$PY" eval/eval.py \
  --questions eval/smoke_questions.yaml \
  --chroma-dir "$SNAP" \
  --base-url "http://127.0.0.1:$PORT" \
  --no-query

echo
echo "=== smoke 闸门（绝对下限）==="
"$PY" scripts/smoke_gate.py --floors "$ROOT/eval/smoke_floors.json" \
  --candidate "$ROOT/eval/results/latest.json" "$@"
