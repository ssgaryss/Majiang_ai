#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_distill.py - 导出蒸馏样本（弃牌侧试点）：(状态, 老师动作索引)。

要点：
  · 老师集合从 _tmp/teacher_strong.jsonl 里取（前 20 万行内的 distinct teacher ✓）
  · 只导**弃牌**决策（已验证覆盖率 100% ✓ 干净子集 ✓）
  · 跳过 start_hands 含 None 的截断小局 ✓（与验收工具同一守卫 ✓）
  · 上限 cap 条（默认 2000 ✓ 先小后大 ✓）
输出：_tmp/distill_sample.jsonl，每行 {"s":[...370 floats...], "a":<候选索引>, "k":"discard"}
"""
import glob, json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rl_action_space as A
from mj_bot import View

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NBS = int(sys.argv[2]) if len(sys.argv) > 2 else 2

TEACH = set()
with open(os.path.join(ROOT, "_tmp", "teacher_strong.jsonl"), encoding="utf-8") as f:
    for ln in f:
        try:
            TEACH.add(json.loads(ln)["teacher"])
        except Exception:
            pass


def snap_of(hands, melds, discards, seat, dealer, wall, last):
    return {"phase": "turn", "seat": seat, "turn": 0,
            "my_hand": sorted(hands[seat].elements()),
            "melds": [list(m) for m in melds], "last_discard": last,
            "wall_remaining": wall, "discards": [list(d) for d in discards],
            "god": {}, "scores": [0, 0, 0, 0], "dealer": dealer}


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "field_cache", "*.json")))[:NBS]
    os.makedirs(os.path.join(ROOT, "_tmp"), exist_ok=True)
    outp = os.path.join(ROOT, "_tmp", "distill_sample.jsonl")
    n = skip = nonteacher = 0
    with open(outp, "w", encoding="utf-8") as out:
        for fp in files:
            if n >= CAP:
                break
            try:
                d = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            for ev in d.get("evs", []):
                seats = ev.get("seats") or []
                for blk in ev.get("blocks", []):
                    if n >= CAP:
                        break
                    sh = blk.get("start_hands")
                    if not sh or len(sh) != 4 or any(h is None for h in sh):
                        skip += 1
                        continue
                    dealer = int(blk.get("dealer") or 0)
                    hands = [Counter(h) for h in sh]
                    melds = [[], [], [], []]
                    discards = [[], [], [], []]
                    wall, last = 83, ""
                    for e in blk.get("events", []):
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
                                uid = None
                                if seat < len(seats) and isinstance(seats[seat], dict):
                                    uid = seats[seat].get("user_id")
                                if uid in TEACH:
                                    try:
                                        v = View(snap_of(hands, melds, discards, seat, dealer, wall, last), seat)
                                        cands, _f = A.turn_candidates(v)
                                        s = list(A.encode_state(v))
                                        idx = [i for i, c in enumerate(cands)
                                               if c.get("kind") == A.K_DISCARD and c.get("tile") == A.tile_index(tl)]
                                        if idx:
                                            out.write(json.dumps({"s": s, "a": idx[0], "k": "discard"}) + "\n")
                                            n += 1
                                    except Exception:
                                        pass
                                else:
                                    nonteacher += 1
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
    mb = os.path.getsize(outp) / 1e6
    print("=== 蒸馏样本导出（弃牌侧试点）===")
    print("  老师数 = %d ｜ 导出 = **%d** 条（上限 %d）｜ 截断小局跳过 = %d ｜ 非老师弃牌 = %d"
          % (len(TEACH), n, CAP, skip, nonteacher))
    print("  输出 = %s ｜ %.1f MB" % (outp, mb))
    if n:
        with open(outp, encoding="utf-8") as f:
            first = json.loads(f.readline())
        print("  样例：状态维度 = %d ｜ 动作索引 = %d ｜ 状态前5 = %s"
              % (len(first["s"]), first["a"], [round(x, 3) for x in first["s"][:5]]))


if __name__ == "__main__":
    main()