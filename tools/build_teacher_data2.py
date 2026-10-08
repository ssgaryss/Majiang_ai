#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_teacher_data2.py - 提取「强手决策」数据集（含防守决策）。

与 v1（build_teacher_data.py）的区别：
  1) 先算每个 user 的净分/手数 ⇒ 只保留**老师**（net>0 且 hands>=H）
  2) **保留防守决策**（pass / 危险局面下的 discard）—— 见 §Z.1159：
     赢家与输家的大牌率**一样高**（0.29~0.30），差异在**支付端/防守** ⇒ 必须学防守 ✓
  3) 剔除 timeout（占 48% ✗）与 tile_drawn（非选择 ✗）

用法: python tools/build_teacher_data2.py [n_files] [min_hands]
输出: _tmp/teacher_strong.jsonl  ＋ 屏幕清点
"""
import glob, json, os, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
H = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SKIP = {"timeout", "tile_drawn", "round_ended", "game_ended", "gang_ended"}


def uid_of(seats, i):
    if isinstance(seats, list) and 0 <= i < len(seats) and isinstance(seats[i], dict):
        return seats[i].get("user_id")
    return None


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "field_cache", "*.json")))[:N]
    # ---- pass 1: 强度 ----
    st = defaultdict(lambda: {"net": 0.0, "hands": 0})
    for fp in files:
        try: d = json.load(open(fp, encoding="utf-8"))
        except Exception: continue
        for ev in d.get("evs", []):
            seats = ev.get("seats") or []
            for blk in ev.get("blocks", []):
                for e in blk.get("events", []):
                    if e.get("type") != "round_ended": continue
                    sc = (e.get("data") or {}).get("scores") or []
                    if len(sc) != 4: continue
                    for i in range(4):
                        u = uid_of(seats, i)
                        if u: st[u]["net"] += sc[i]; st[u]["hands"] += 1
    teach = {u for u, v in st.items() if v["net"] > 0 and v["hands"] >= H}
    print("pass1: 玩家 %d ｜ 老师(net>0, hands>=%d) = **%d**" % (len(st), H, len(teach)))

    # ---- pass 2: 决策 ----
    os.makedirs(os.path.join(ROOT, "_tmp"), exist_ok=True)
    out_path = os.path.join(ROOT, "_tmp", "teacher_strong.jsonl")
    kinds = Counter(); n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for fp in files:
            try: d = json.load(open(fp, encoding="utf-8"))
            except Exception: continue
            for ev in d.get("evs", []):
                seats = ev.get("seats") or []
                for blk in ev.get("blocks", []):
                    for e in blk.get("events", []):
                        t = e.get("type"); seat = e.get("seat")
                        if t in SKIP or not isinstance(seat, int): continue
                        u = uid_of(seats, seat)
                        if u not in teach: continue
                        kinds[t] += 1; n += 1
                        out.write(json.dumps({
                            "file": os.path.basename(fp), "game_id": ev.get("game_id"),
                            "round_no": blk.get("round_no"), "dealer": blk.get("dealer"),
                            "seat": seat, "seq": e.get("seq"), "type": t,
                            "tile": e.get("tile"),
                            "teacher": u,
                        }, ensure_ascii=False) + "\n")
    print("pass2: 强手决策点 = **%d**" % n)
    print("  动作构成 =", dict(kinds.most_common(10)))
    print("  ★ 含防守类（pass/chi/peng 之外的响应）= pass %d" % kinds.get("pass", 0))
    print("  输出 =", out_path, "%.1f MB" % (os.path.getsize(out_path) / 1e6))


if __name__ == "__main__":
    main()