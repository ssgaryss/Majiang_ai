#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""replay_window_check.py - 验收窗口决策（chi/peng/gang 的合法性 + pass 计数）。

窗口快照与回合快照的差别：phase 用 response_peng / response_chi，且带 last_discard。
判据：老师做的 chi/peng 动作**必须出现在** window_candidates 里（目标 100%）。
pass 不做合法性判定（它总是候选之一），只计数并报告。
"""
import glob, json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rl_action_space as A
from mj_bot import View

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1
KINDMAP = {"chi": "chi", "peng": "peng", "gang": "gang"}


def snap_of(hands, melds, discards, seat, dealer, wall, last, phase):
    return {"phase": phase, "seat": seat, "turn": 0,
            "my_hand": sorted(hands[seat].elements()),
            "melds": [list(m) for m in melds],
            "last_discard": last, "wall_remaining": wall,
            "discards": [list(d) for d in discards],
            "god": {}, "scores": [0, 0, 0, 0], "dealer": dealer}


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "field_cache", "*.json")))[:NB]
    hit = miss = npass = err = 0
    ex = []
    for fp in files:
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for ev in d.get("evs", []):
            for blk in ev.get("blocks", []):
                sh = blk.get("start_hands")
                if not sh or len(sh) != 4 or any(h is None for h in sh):   # ★ 截断小局 ⇒ 跳过
                    continue
                dealer = int(blk.get("dealer") or 0)
                hands = [Counter(h) for h in sh]
                melds = [[], [], [], []]
                discards = [[], [], [], []]
                wall, last = 83, ""
                for e in blk.get("events", []):
                    t = e.get("type"); seat = e.get("seat"); tl = e.get("tile")
                    if t == "tile_drawn":
                        if isinstance(seat, int) and 0 <= seat < 4 and tl:
                            hands[seat][tl] += 1
                        wall -= 1
                    elif t == "tile_discarded":
                        if not (isinstance(seat, int) and 0 <= seat < 4 and tl):
                            continue
                        if hands[seat][tl] > 0:
                            hands[seat][tl] -= 1
                            discards[seat].append(tl)
                            last = tl
                    elif t in ("chi", "peng", "gang"):
                        if not (isinstance(seat, int) and 0 <= seat < 4 and tl):
                            continue
                        ph = "response_chi" if t == "chi" else "response_peng"
                        okc = False
                        try:
                            v = View(snap_of(hands, melds, discards, seat, dealer, wall, last, ph), seat)
                            cc = list(v.counts)
                            ti = A.tile_index(tl)
                            mset = int(v.meld_sets or 0)
                            if t == "chi":
                                okc = bool(A.chi_tiles(cc, tl, mset))
                            elif t == "peng":
                                okc = cc[ti] >= 2
                            else:
                                okc = cc[ti] >= 3
                        except Exception as exx:
                            err += 1
                            if len(ex) < 3:
                                ex.append(("ERR", type(exx).__name__, str(exx)[:70]))
                        if okc:
                            hit += 1
                        else:
                            miss += 1
                            if len(ex) < 6:
                                ex.append((t, seat, tl, sum(hands[seat].values()), "last=" + str(last)))
                        melds[seat].append({"kind": t, "tiles": [tl]})
                        if t == "peng":
                            hands[seat][tl] = max(0, hands[seat][tl] - 2)
                        elif t == "gang":
                            hands[seat][tl] = max(0, hands[seat][tl] - 3)
                        else:
                            ti = A.tile_index(tl); used = 0
                            for dd in ((-2, -1), (-1, 1), (1, 2)):
                                ca, cb = A.tile_code(ti + dd[0]), A.tile_code(ti + dd[1])
                                if hands[seat][ca] > 0 and hands[seat][cb] > 0:
                                    hands[seat][ca] -= 1; hands[seat][cb] -= 1; used = 2; break
                            if used == 0:
                                hands[seat][tl] = max(0, hands[seat][tl] - 2)
                    elif t == "pass":
                        npass += 1
    tot = hit + miss
    print("=== 窗口决策验收（文件 %d 个）===" % len(files))
    print("  chi/peng/gang 合计 = %d ｜ 命中 = %d ｜ 未命中 = %d ｜ 异常 = %d" % (tot, hit, miss, err))
    print("  ★ 窗口合法性覆盖率 = %.4f%%（目标 100%%）" % (100.0 * hit / max(1, tot)))
    print("  pass 事件计数（不判合法性）= %d" % npass)
    for x in ex:
        print("   样例:", x)


if __name__ == "__main__":
    main()