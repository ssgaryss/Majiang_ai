#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""drift_probe.py - 定位手牌漂移：找出每个 block 里**第一个把某张牌计数打成负数**的事件。

为什么用"负数"作指纹：Counter 允许负值，而 elements() 对负项不产出 ⇒ 手牌看起来变空
⇒ 正是弃牌 resync(16%) 与窗口非法(41%) 的直接原因。所以"谁把计数打成负"就是元凶。
注意：不变量 sum(hand)+3*melds 并不成立（副露只从手牌拿 2~3 张），故不用它。
"""
import glob, json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rl_action_space as A

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TILES = [A.tile_code(i) for i in range(34)]


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "field_cache", "*.json")))[:NB]
    blocks = bad = 0
    culprit = Counter(); prev_seen = Counter()
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
                blocks += 1
                hands = [Counter(h) for h in sh]
                prev = ""
                for e in blk.get("events", []):
                    t = e.get("type"); seat = e.get("seat"); tl = e.get("tile")
                    neg = False
                    if t == "tile_drawn" and isinstance(seat, int) and 0 <= seat < 4 and tl:
                        hands[seat][tl] += 1
                    elif t == "tile_discarded" and isinstance(seat, int) and 0 <= seat < 4 and tl:
                        hands[seat][tl] -= 1
                        neg = hands[seat][tl] < 0
                    elif t in ("chi", "peng", "gang") and isinstance(seat, int) and 0 <= seat < 4 and tl:
                        if t == "peng":
                            hands[seat][tl] -= 2
                        elif t == "gang":
                            hands[seat][tl] -= 3
                        else:
                            ti = A.tile_index(tl); done = False
                            for dd in ((-2, -1), (-1, 1), (1, 2)):
                                ca, cb = A.tile_code(ti + dd[0]), A.tile_code(ti + dd[1])
                                if hands[seat][ca] > 0 and hands[seat][cb] > 0:
                                    hands[seat][ca] -= 1; hands[seat][cb] -= 1; done = True; break
                            if not done:
                                hands[seat][tl] -= 2
                        neg = any(hands[seat][x] < 0 for x in TILES)
                    if neg:
                        bad += 1
                        culprit[prev or "(block首事件)"] += 1
                        break
                    prev = t
    print("=== 漂移定位（文件 %d 个 ｜ block %d 个）===" % (len(files), blocks))
    print("  出现负计数的 block = %d（%.1f%%）" % (bad, 100.0 * bad / max(1, blocks)))
    print("  ★ 元凶事件类型（负计数前的最后一个事件）:")
    for k, v in culprit.most_common():
        print("     %-16s %d" % (k, v))


if __name__ == "__main__":
    main()