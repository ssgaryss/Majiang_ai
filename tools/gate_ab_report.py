#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gate_ab_report.py - 门控 A/B 的专用读数（只做诊断，不作判据）。

用法: python tools/gate_ab_report.py [window_hours]
输出:
  1) 每个臂的 missed_hu / 房（预注册的**诊断量**：期望 4.32 -> ~2.2）
  2) 每个臂的房均与房数（供人工对照）
  3) 提醒：**判据仍必须用 analyze_live_paired.py 的主口径 β 与 CI**（本脚本不给判据）
"""
import json, os, sys, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = float(sys.argv[1]) if len(sys.argv) > 1 else 48.0
cut = time.time() - WIN * 3600.0

agg = defaultdict(lambda: {"rooms": 0, "missed": 0, "sum": 0.0, "t": 0.0, "tl": 0.0})
tot_r = 0
for ln in open(os.path.join(ROOT, "live_r16.jsonl"), encoding="utf-8"):
    ln = ln.strip()
    if not ln:
        continue
    try:
        d = json.loads(ln)
    except Exception:
        continue
    if not d.get("room", "").startswith("a_"):
        continue
    if float(d.get("ended") or d.get("started") or 0) < cut:
        continue
    g = d.get("genome") or "?"
    a = agg[g]
    a["rooms"] += 1
    a["missed"] += int(d.get("missed_hu") or 0)
    a["sum"] += float(d.get("total") or 0)
    r = d.get("rows") or []
    a["t"] += len(r)
    a["tl"] += sum(int(x.get("rounds") or 0) for x in r if isinstance(x, dict))
    tot_r += 1

print("=== 门控 A/B 诊断（窗口 %.1f 小时；正式房 %d 房）===" % (WIN, tot_r))
print("%-26s %6s %8s %10s %10s  %s" % ("genome", "rooms", "missed", "missed/房", "房均", "小局"))
for g, a in sorted(agg.items(), key=lambda kv: -kv[1]["rooms"]):
    if a["rooms"] < 5:
        continue
    print("%-26s %6d %8d %10.2f %10.2f  %d"
          % (g[:26], a["rooms"], a["missed"], a["missed"] / max(1, a["rooms"]),
             a["sum"] / max(1, a["rooms"]), a["tl"]))
print("")
print("⚠ 本脚本只给**诊断量**。判据必须用: python analyze_live_paired.py --jsonl live_r16.jsonl --window-hours %d" % int(WIN))
print("   预注册: 48h 若 β 的 CI 上界 < 0 => 回滚; 成功需 CI 下界 > 0 且 β >= +1.0")