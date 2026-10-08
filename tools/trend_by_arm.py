# -*- coding: utf-8 -*-
"""_tmp/trend.py -- 逐日 x 逐臂的【房均】趋势（回答「模型有没有变强」用 ✓ 只读 ✓）

★ 纪律 ✓：单位＝分/局，按【房】聚合 ✓；只算**完整观测**局（`deltas_base` 全零 ✓）；
   n < 2 时**不给 CI** ✓；样本 SD ÷(n−1) ＋ t 乘子 ✓（§Z.1373 ✓）。
"""
import collections
import io
import json
import math
import statistics as st
import time

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(n):
    return T95.get(max(1, n - 1), 1.959964)


rows = []          # (day, arm, room_mean, games)
# ★★★ 2026-10-07（§Z.1757）：★ **扫所有 `live_r16*.jsonl`** ✓
#   ✗ 旧写法只读 `live_r16.jsonl` ⇒ ★★ **换过日志名的新房静默漏掉** ✗（★ 实测漏 14 房 ✓ §Z.1755 ✓）
import glob as _glob
_lines = []
for _p in sorted(_glob.glob("live_r16*.jsonl")):
    _lines.extend(io.open(_p, encoding="utf-8", errors="replace").read().splitlines())
print("（★ 已扫 %d 个 `live_r16*.jsonl` ✓）" % len(_glob.glob("live_r16*.jsonl")))
for ln in _lines:
    ln = ln.strip()
    if not ln or '"a_' not in ln:
        continue
    try:
        d = json.loads(ln)
    except Exception:
        continue
    if not str(d.get("room") or "").startswith("a_"):
        continue
    g = (d.get("genome") or "").lstrip("\ufeff").strip()
    mine = []
    for r in (d.get("rows") or []):
        base = r.get("deltas_base") or []
        if sum(abs(float(x)) for x in base) == 0 and r.get("mine") is not None:
            mine.append(r["mine"])
    if not mine:
        continue
    day = time.strftime("%m-%d", time.localtime(d.get("ended") or 0))
    rows.append((day, g, sum(mine) / float(len(mine)), len(mine)))

print("  逐日 x 逐臂（房均为单位：分/局 ✓）")
per = collections.defaultdict(list)
for day, g, m, n in rows:
    per[(day, g)].append((m, n))
for key in sorted(per):
    day, g = key
    vals = [m for m, _ in per[key]]
    games = sum(n for _, n in per[key])
    n = len(vals)
    if n < 2:
        print("    %s  %-24s 房=%2d 局=%3d ｜ 房均 %+8.2f ｜ **n<2 ⇒ 不给 CI**" % (day, g[:24], n, games, vals[0]))
        continue
    m = st.mean(vals)
    sd = st.stdev(vals)
    se = sd / math.sqrt(n)
    t = t95(n)
    print("    %s  %-24s 房=%2d 局=%3d ｜ 房均 %+8.2f ｜ SD %6.2f ｜ 95%%CI [%+7.2f, %+7.2f]"
          % (day, g[:24], n, games, m, sd, m - t * se, m + t * se))

print("\n  全期合计（按臂 ✓ 仅完整观测局）")
by = collections.defaultdict(list)
for day, g, m, n in rows:
    by[g].append(m)
for g in sorted(by, key=lambda k: -len(by[k])):
    vals = by[g]
    n = len(vals)
    m = st.mean(vals)
    if n < 2:
        print("    %-24s 房=%2d ｜ 房均 %+8.2f ｜ n<2" % (g[:24], n, m))
        continue
    sd = st.stdev(vals)
    se = sd / math.sqrt(n)
    t = t95(n)
    print("    %-24s 房=%2d ｜ 房均 %+8.2f ｜ SD %6.2f ｜ 95%%CI [%+7.2f, %+7.2f]"
          % (g[:24], n, m, sd, m - t * se, m + t * se))
