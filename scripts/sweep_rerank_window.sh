#!/usr/bin/env bash
# 扫重排截断窗口：RERANK_MAX_LENGTH = 384 / 640 各跑一遍重排档。
#
# 为什么要这一步
# -------------
# 消融里「+RRF+rerank」比「+RRF」低 10 个点，看着像「重排没用」。但把 6 道被踢出前四的题
# 逐题剖开，答案锚点全在切片第 268~332 个 token —— 而窗口是 256。也就是说重排根本没看到
# 答案，它是按切片前半段（大多是套话）打的分。
#
# 于是这里只改窗口，别的都不动，看分数会不会回来。回来 = 问题在截断；不回来 = 模型本身
# 不适合这份语料。这两种结论对应完全不同的改法，不能靠猜。
set -uo pipefail
cd "$(dirname "$0")/.."

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
PORT=8032
PY=../.venv-rag/Scripts/python.exe

for ML in 384 640; do
  echo "=== RERANK_MAX_LENGTH=$ML ==="
  RERANK_MAX_LENGTH=$ML $PY -m uvicorn app.main:app --host 127.0.0.1 --port $PORT \
    > "/d/workbuddy-space/rag-work/server-ml$ML.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 40); do
    sleep 2
    curl -s --noproxy '*' -m 2 "http://127.0.0.1:$PORT/api/v1/health" | grep -q '"status":"ok"' && break
  done
  $PY -m eval.ablation --base-url "http://127.0.0.1:$PORT" \
    --modes hybrid_rrf_rerank --rerank-budget 0 --timeout 900 \
    --out-dir "/d/workbuddy-space/rag-work/window-$ML"
  kill $PID 2>/dev/null
  sleep 3
done
echo "done"
