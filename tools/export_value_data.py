#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_value_data.py - 导出 ③ 价值/排序头的监督数据（标签=真实结果，不是老师动作）。

与 export_distill.py 的差别：
  · **不筛老师**（覆盖全部座位 ✓ —— 价值头要学"任何局面值多少分"）
  · 标签 = **该局该席的真实得分**（round_ended.data.scores[seat] ✓）
  · 先扫一遍 block 取结局（结局在决策之后出现 ⇒ 必须两遍扫 ✓）
输出 _tmp/value_data.jsonl：每行 {"s":[...370...], "seat":n, "y":<该局得分>, "fan":n, "k":"discard"}
"""
import glob, json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rl_action_space as A
from mj_bot import View

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NBS = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def snap_of(hands, melds, discards, seat, dealer, wall, last):
    return {"phase": "turn", "seat": seat, "turn": 0,
            "my_hand": sorted(hands[seat].elements()),
            "melds": [list(m) for m in melds], "last_discard": last,
            "wall_remaining": wall, "discards": [list(d) for d in discards],
            "god": {}, "scores": [0, 0, 0, 0], "dealer": dealer}


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "field_cache", "*.json")))[:NBS]
    os.makedirs(os.path.join(ROOT, "_tmp"), exist_ok=True)
    outp = os.path.join(ROOT, "_tmp", "value_data.jsonl")
    n = skip = noout = 0
    ymin, ymax = 10 ** 9, -10 ** 9
    with open(outp, "w", encoding="utf-8") as out:
        for fp in files:
            if n >= CAP:
                break
            try:
                d = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            for ev in d.get("evs", []):
                for blk in ev.get("blocks", []):
                    if n >= CAP:
                        break
                    sh = blk.get("start_hands")
                    if not sh or len(sh) != 4 or any(h is None for h in sh):
                        skip += 1
                        continue
                    evs = blk.get("events") or []
                    # ---- 第 1 遍：取结局 ----
                    sc = None; fan = 0; wseat = None
                    for e in evs:
                        if e.get("type") == "round_ended":
                            dd = e.get("data") or {}
                            s2 = dd.get("scores")
                            if isinstance(s2, list) and len(s2) == 4:
                                sc = s2; fan = int(dd.get("fan") or 0); wseat = e.get("seat")
                    if sc is None:
                        noout += 1
                        continue
                    # ---- 第 2 遍：发样本 ----
                    dealer = int(blk.get("dealer") or 0)
                    hands = [Counter(h) for h in sh]
                    melds = [[], [], [], []]
                    discards = [[], [], [], []]
                    wall, last = 83, ""
                    for e in evs:
                        if n >= CAP:
                            break
                        t = e.get("type"); seat = e.get("seat"); tl = e.get("tile")
                        if t == "tile_drawn":
                            if isinstance(seat, int) and 0 <= seat < 4 and tl:
                                hands[seat][tl] += 1
                            wall -= 1
                        elif t == "tile_discarded":
                            if not (isinstance(seat, int) and 0 <= seat < 4 and tl):
                                continue
                            if hands[seat][tl] > 0:
                                try:
                                    v = View(snap_of(hands, melds, discards, seat, dealer, wall, last), seat)
                                    s = list(A.encode_state(v))
                                    out.write(json.dumps({"s": s, "seat": int(seat),
                                                          "y": float(sc[seat]), "fan": int(fan),
                                                          "y_win": 1.0 if (wseat == int(seat) and fan > 0) else 0.0,
                                                          "k": "discard"}) + "\n")
                                    n += 1
                                    y = float(sc[seat])
                                    if y < ymin: ymin = y
                                    if y > ymax: ymax = y
                                except Exception:
                                    pass
                                hands[seat][tl] -= 1
                                discards[seat].append(tl)
                                last = tl
                        elif t in ("chi", "peng", "gang"):
                            if not (isinstance(seat, int) and 0 <= seat < 4 and tl):
                                continue
                            melds[seat].append({"kind": t, "tiles": [tl]})
                            if t == "peng":
                                hands[seat][tl] = max(0, hands[seat][tl] - 2)
                            elif t == "gang":
                                hands[seat][tl] = max(0, hands[seat][tl] - 3)
                            else:
                                ti = A.tile_index(tl); done = False
                                for dd in ((-2, -1), (-1, 1), (1, 2)):
                                    ca, cb = A.tile_code(ti + dd[0]), A.tile_code(ti + dd[1])
                                    if hands[seat][ca] > 0 and hands[seat][cb] > 0:
                                        hands[seat][ca] -= 1; hands[seat][cb] -= 1; done = True; break
                                if not done:
                                    hands[seat][tl] = max(0, hands[seat][tl] - 2)
    print("=== 价值数据导出（标签=真实结果）===")
    print("  导出 = **%d** 条（上限 %d）｜ 截断小局跳过 = %d ｜ 无结局跳过 = %d"
          % (n, CAP, skip, noout))
    print("  y(该局得分) 范围 = [%g, %g]" % (ymin if n else 0, ymax if n else 0))
    print("  输出 = %s ｜ %.1f MB" % (outp, os.path.getsize(outp) / 1e6))
    if n:
        with open(outp, encoding="utf-8") as f:
            first = json.loads(f.readline())
        print("  样例：维度 %d ｜ seat %d ｜ y %g ｜ fan %d"
              % (len(first["s"]), first["seat"], first["y"], first["fan"]))


if __name__ == "__main__":
    main()