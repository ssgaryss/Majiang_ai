#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""value_head_eval.py - 价值头的验收指标（**不许用平均误差** ✗ 因标签零和 ⇒ 均值恒 0）。

指标：
  ① **分桶单调性**：按预测值分成 K 桶 ⇒ 报每桶的**实际均分** ✓（应单调递增 ✓）
  ② **Spearman(pred, actual)** ＋ bootstrap 95% CI ✓（要求 > 0 且 CI 下界 > 0 ✓）
用法：
  python tools/value_head_eval.py --data _tmp/value_data.jsonl --pred <preds.jsonl>
  python tools/value_head_eval.py --selftest      # 用"神谕/随机"验证指标本身 ✓
"""
import argparse, json, math, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0

    def rank(v):
        idx = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def buckets(preds, ys, k=5):
    idx = sorted(range(len(preds)), key=lambda i: preds[i])
    out = []
    for b in range(k):
        lo = b * len(idx) // k
        hi = (b + 1) * len(idx) // k
        if hi <= lo:
            continue
        sel = idx[lo:hi]
        out.append((len(sel), sum(preds[i] for i in sel) / len(sel),
                    sum(ys[i] for i in sel) / len(sel)))
    return out


def boot_ci(preds, ys, n=300, seed=7):
    rnd = random.Random(seed)
    vals = []
    N = len(preds)
    for _ in range(n):
        s = [rnd.randrange(N) for _ in range(N)]
        vals.append(spearman([preds[i] for i in s], [ys[i] for i in s]))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def load(paths):
    P, Y = [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                d = json.loads(ln)
                if "p" in d:
                    P.append(float(d["p"]))
                Y.append(float(d["y"]))
    return P, Y


def report(preds, ys, tag):
    rho = spearman(preds, ys)
    lo, hi = boot_ci(preds, ys)
    print("  [%s] n=%d ｜ Spearman = %+.4f ｜ 95%%CI [%+.4f, %+.4f] ⇒ %s"
          % (tag, len(preds), rho, lo, hi, "**有效** ✓" if lo > 0 else "未分辨/无效 ✗"))
    for i, (cnt, pm, ym) in enumerate(buckets(preds, ys, 5)):
        print("      桶%d n=%-5d 预测均 %+8.3f ｜ **实际均分 %+8.3f**" % (i + 1, cnt, pm, ym))
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "_tmp", "value_data.jsonl"))
    ap.add_argument("--pred", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        ys = []
        with open(a.data, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    ys.append(float(json.loads(ln)["y"]))
        print("=== 指标自检（数据 %d 条）===" % len(ys))
        rnd = random.Random(1)
        report([rnd.random() for _ in ys], ys, "随机预测（应 ≈0 ✗）")
        report([float(v) for v in ys], ys, "神谕=标签本身（应 =1 ✓）")
        print("  反序预测："); report([-v for v in ys], ys, "反号（应 =-1 ✓）")
        return
    P, Y = load([a.data] if not a.pred else [a.pred])
    print("=== 价值头验收 ===")
    report(P, Y, os.path.basename(a.pred or a.data))


if __name__ == "__main__":
    main()