#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""replay_teacher_states.py - 用 start_hands + 事件流构造 View，验收「老师动作在候选集合内」。

阶段一（本文件）：只验**弃牌决策**（最大类，占强手决策 361,849 条）。
  逐步：start_hands 建手牌 -> tile_drawn 加入 -> 到 tile_discarded 时**先构造快照**再移除该牌
  验收：老师弃的那张牌是否落在 turn_candidates(v) 的 discard 候选里（目标 100%）。
不依赖 seed / Game（原因见 GOAL_STATUS Z.1171/Z.1172）。
"""
import glob, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rl_action_space as A
from mj_bot import View

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def snap_from(hands, melds, discards, seat, dealer, wall, last_discard):
    return {"phase": "turn", "seat": seat, "turn": 0,
            "my_hand": sorted(hands[seat].elements()),
            "melds": [list(m) for m in melds],
            "last_discard": last_discard,
            "wall_remaining": wall,
            "discards": [list(d) for d in discards],
            "god": {}, "scores": [0, 0, 0, 0], "dealer": dealer}


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "field_cache", "*.json")))[:NB]
    n_disc = ok = miss = err = resync = 0
    miss_ex = []
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
                wall = 83
                last = ""
                for e in blk.get("events", []):
                    t = e.get("type"); seat = e.get("seat")
                    if t == "tile_drawn":
                        if isinstance(seat, int) and 0 <= seat < 4 and e.get("tile"):
                            hands[seat][e["tile"]] += 1
                        wall -= 1
                    elif t == "tile_discarded":
                        if not (isinstance(seat, int) and 0 <= seat < 4 and e.get("tile")):
                            continue
                        if hands[seat][e["tile"]] <= 0:
                            resync += 1
                            discards[seat].append(e["tile"])
                            last = e["tile"]
                            continue
                        n_disc += 1
                        try:
                            v = View(snap_from(hands, melds, discards, seat, dealer, wall, last), seat)
                            cands, _f = A.turn_candidates(v)
                            want = A.tile_index(e["tile"])
                            got = {c["tile"] for c in cands if c.get("kind") == A.K_DISCARD}   # 候选已是 index ✓
                            if want in got:
                                ok += 1
                            else:
                                miss += 1
                                if len(miss_ex) < 5:
                                    miss_ex.append((os.path.basename(fp), seat, e["tile"],
                                                    len(got), sorted(hands[seat].elements())[:14]))
                        except Exception as ex:
                            err += 1
                            if err == 1:
                                import traceback
                                print("--- 首次异常完整 traceback ---")
                                traceback.print_exc()
                                print("--- 当时快照 ---")
                                print({k: (v if k not in ("my_hand", "discards", "melds") else str(v)[:200])
                                       for k, v in snap_from(hands, melds, discards, seat, dealer, wall, last).items()})
                            if len(miss_ex) < 3:
                                miss_ex.append(("ERR", type(ex).__name__, str(ex)[:80]))
                        hands[seat][e["tile"]] -= 1
                        discards[seat].append(e["tile"])
                        last = e["tile"]
                    elif t in ("chi", "peng", "gang"):
                        if isinstance(seat, int) and 0 <= seat < 4 and e.get("tile"):
                            tl = e["tile"]
                            melds[seat].append({"kind": t, "tiles": [tl]})
                            # ★ 正确扣自家用牌：碰扣2 / 杠扣3 / 吃扣另两张（第三张来自他家）
                            if t == "peng":
                                hands[seat][tl] = max(0, hands[seat][tl] - 2)
                            elif t == "gang":
                                hands[seat][tl] = max(0, hands[seat][tl] - 3)
                            else:                                   # chi：从手牌找与 tl 成顺的两张
                                ti = A.tile_index(tl)
                                used = 0
                                for d in ((-2, -1), (-1, 1), (1, 2)):
                                    a_, b_ = ti + d[0], ti + d[1]
                                    ca, cb = A.tile_code(a_), A.tile_code(b_)
                                    if hands[seat][ca] > 0 and hands[seat][cb] > 0:
                                        hands[seat][ca] -= 1
                                        hands[seat][cb] -= 1
                                        used = 2
                                        break
                                if used == 0:
                                    hands[seat][tl] = max(0, hands[seat][tl] - 2)
    tot = ok + miss
    print("=== 弃牌决策覆盖率（blocks=%d 文件 %d 个）===" % (NB, len(files)))
    print("  弃牌决策 = %d ｜ 命中 = %d ｜ 未命中 = %d ｜ 异常 = %d ｜ resync跳过 = %d" % (n_disc, ok, miss, err, resync))
    print("  ★ 覆盖率 = %.4f%%（目标 100%%）" % (100.0 * ok / max(1, tot)))
    for x in miss_ex:
        print("   样例:", x)


if __name__ == "__main__":
    main()