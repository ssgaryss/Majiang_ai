#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage1_fan_ev.py - Stage 1：从真实牌谱量化「大牌对期望的贡献」。

只读 field_cache 的 round_ended.data（含 fan / detail / scores ✓）⇒ 无需重建牌局 ✓
回答两问：① 大牌占全部得分的百分之多少？② 赢大牌多的人，净分是不是更高？
"""
import glob, json, os, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 250

fan_hist = Counter(); detail_hist = Counter()
pts_by_fan = defaultdict(float); n_by_fan = Counter()
per_user = defaultdict(lambda: {"hands": 0, "wins": 0, "big": 0, "fan_sum": 0.0, "net": 0.0})
tot_abs_pts = 0.0; n_rounds = 0; n_draw = 0

files = sorted(glob.glob(os.path.join(ROOT, "field_cache", "*.json")))[:N]
for fp in files:
    try: d = json.load(open(fp, encoding="utf-8"))
    except Exception: continue
    for ev in d.get("evs", []):
        seats = ev.get("seats") or []
        names = [s.get("user_id") for s in seats] if isinstance(seats, list) else []
        for blk in ev.get("blocks", []):
            for e in blk.get("events", []):
                if e.get("type") != "round_ended": continue
                dd = e.get("data") or {}
                sc = dd.get("scores") or []
                if len(sc) != 4: continue
                n_rounds += 1
                fan = int(dd.get("fan") or 0); det = dd.get("detail") or []
                win = e.get("seat")
                if dd.get("draw"): n_draw += 1
                if isinstance(win, int) and 0 <= win < 4 and fan > 0:
                    fan_hist[fan] += 1
                    for x in det: detail_hist[x] += 1
                    pts_by_fan[fan] += sc[win]; n_by_fan[fan] += 1
                    if names and win < len(names) and names[win]:
                        u = per_user[names[win]]
                        u["hands"] += 1; u["wins"] += 1; u["fan_sum"] += fan
                        if fan >= 2 or any(k in det for k in ("七对", "爆头", "清一色", "碰碰胡")):
                            u["big"] += 1
                for i, v in enumerate(sc):
                    tot_abs_pts += abs(v)
                    if names and i < len(names) and names[i]:
                        per_user[names[i]]["net"] += v

print("=== Stage 1：大牌对期望的贡献（真实牌谱 %d 文件 / %d 局）===" % (len(files), n_rounds))
print("  流局 = %d (%.1f%%)" % (n_draw, 100.0 * n_draw / max(1, n_rounds)))
print("--- fan 分布与得分占比 ---")
tot_win_pts = sum(pts_by_fan.values()) or 1.0
for f in sorted(fan_hist, reverse=True):
    print("  fan=%d  次数=%6d (%.1f%%)  赢家得分合计=%9.0f (%.1f%%)  均分=%6.1f"
          % (f, fan_hist[f], 100.0 * fan_hist[f] / max(1, sum(fan_hist.values())),
             pts_by_fan[f], 100.0 * pts_by_fan[f] / tot_win_pts, pts_by_fan[f] / max(1, n_by_fan[f])))
big_pts = sum(v for k, v in pts_by_fan.items() if k >= 2)
print("  ★ fan>=2 占比 = %.1f%% 次数 ／ **%.1f%% 得分**" % (
    100.0 * sum(v for k, v in fan_hist.items() if k >= 2) / max(1, sum(fan_hist.values())),
    100.0 * big_pts / tot_win_pts))
print("  牌型明细 =", dict(detail_hist.most_common(10)))
print("--- 玩家层面：赢大牌多的人净分更高吗（净分合计 != 0 说明样本未闭合）---")
rows = [(u["net"], u["wins"], u["big"], u["fan_sum"] / max(1, u["wins"]), k) for k, u in per_user.items() if u["wins"] >= 20]
rows.sort(reverse=True)
print("  样本玩家数(>=20 次胡) = %d" % len(rows))
print("  %-18s %8s %6s %6s %7s" % ("user_id", "净分", "胡", "大牌", "均番"))
for net, w, b, af, k in rows[:8]:
    print("  %-18s %8.0f %6d %6d %7.2f" % (k[:18], net, w, b, af))
print("  ...")
for net, w, b, af, k in rows[-5:]:
    print("  %-18s %8.0f %6d %6d %7.2f" % (k[:18], net, w, b, af))
if len(rows) >= 6:
    hi = rows[:max(1, len(rows)//4)]; lo = rows[-max(1, len(rows)//4):]
    print("  ★ 净分前 25%% 平均大牌率 = %.3f ｜ 后 25%% 平均大牌率 = %.3f" % (
        sum(r[2] / max(1, r[1]) for r in hi) / len(hi),
        sum(r[2] / max(1, r[1]) for r in lo) / len(lo)))