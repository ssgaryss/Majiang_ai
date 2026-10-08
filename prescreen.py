#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分歧率预筛 —— 跑对局之前，先免费量一下候选改动会不会改变行为。

动机（round 13 的教训）：本项目的结构性改动按行为分歧率分成泾渭分明的两类：
  · 分歧率 ~60%（value 规则）        -> 30 局就对拍出显著结论（自摸率 27.0% vs 3.3%）
  · 分歧率 0.0~2.9%（七对承诺/智能碰）-> 烧掉整个世代也测不出任何东西
离线量分歧率是零成本的，能把不可测的改动直接筛掉，省下整轮对局时间。

判据（经验阈值，非硬性）：
  >= 20%   值得跑对局
  5%~20%   效应量可能偏小，需要更多局，谨慎投入
  < 5%     基本不可测，别浪费房间

用法：
  python prescreen.py --cand value:1:1:0:0:0:0
  python prescreen.py --cand A --cand B --base X --n 3000
  python prescreen.py --sweep              # 对预置候选批量预筛
"""
import argparse
import random
import sys
import time

import shanten_ref as S
from mj_bot import View
from mj_core import NTILE, WILD, tile_code
from mj_policy import Evolved, GENE_SCHEMA, parse_genome

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DEFAULT = "shanten:1:1:0:0:0:0"
SWEEP = [
    "shanten:2:1:0:0:0:0",       # 智能碰
    "shanten:1:1:0:0:1:0",       # 未见牌修正
    "value:1:1:0:0:0:0",         # 胜率期望骨架
    "shanten:1:1:1:0:0:0",       # 杠
    "shanten:1:1:0:1:0:0",       # 安全飘
    "shanten:1:1:0:0:0:1",       # 七对承诺
]


def build_win(rng):
    """构造一副成型手（4 面子 + 1 将，再掺若干财神）。

    必须包含成型手：decline（弃胡博财飘）只在 can_win 为真时才有行为差异，
    而随机 14 张手牌几乎从不是成型手 —— 早先的贫瘠采样因此把它误判成 0% 分歧。
    """
    pool = [4] * NTILE
    pool[WILD] = 0
    avail = [t for t in range(NTILE) if t != WILD]
    tiles = []
    for _ in range(4):
        for _try in range(200):
            if rng.random() < 0.5:
                t = rng.choice(avail)
                if pool[t] >= 3:
                    pool[t] -= 3
                    tiles += [t] * 3
                    break
            else:
                b = rng.choice((0, 9, 18))
                s = b + rng.randint(0, 6)
                if pool[s] and pool[s + 1] and pool[s + 2]:
                    pool[s] -= 1
                    pool[s + 1] -= 1
                    pool[s + 2] -= 1
                    tiles += [s, s + 1, s + 2]
                    break
    for _try in range(200):
        t = rng.choice(avail)
        if pool[t] >= 2:
            pool[t] -= 2
            tiles += [t] * 2
            break
    for i in rng.sample(range(len(tiles)), min(rng.randint(0, 3), len(tiles))):
        tiles[i] = WILD
    c = [0] * NTILE
    for t in tiles:
        c[t] += 1
    return c


def snap_from_counts(c, rng=None, seat=0):
    """把一份手牌包成一个**带真实上下文**的合法快照。

    上下文不是装饰：ukeire 只有在弃牌堆/副露非空时才与朴素口径不同；
    decline 只有 can_win 为真时才触发。早先造空快照导致这两个
    已被证实有效的基因都被误判成「0% 分歧、不可测」。
    """
    hand = [tile_code(t) for t in range(NTILE) for _ in range(c[t])]
    discs = [[], [], [], []]
    melds = [[], [], [], []]
    wall = 84
    if rng is not None:
        dp = [t for t in range(NTILE) for _ in range(4)]
        rng.shuffle(dp)
        for i in range(rng.randint(0, 60)):
            discs[i % 4].append(tile_code(dp[i]))
        if rng.random() < 0.4:                      # 有时给别家一副碰
            row = rng.randrange(4)
            t = tile_code(rng.randrange(27))
            melds[row] = [{"kind": "peng", "tiles": [t, t, t]}]
        wall = rng.randint(24, 84)                  # 覆盖牌墙将尽/充裕两档
    return {
        "phase": "draw", "turn": seat, "seat": seat, "round_no": 1,
        "hand_counts": [len(hand), 13, 13, 13],
        "my_hand": hand, "drawn_tile": "",
        "melds": melds,
        "god": {"baotou": False, "chain_count": 0, "catch_play": False,
                "god_discarder_seat": -1},
        "discards": discs, "last_discard": "",
        "scores": [0, 0, 0, 0], "wall_remaining": wall,
    }


def snap_window(c, rng, seat=0):
    """造一个碰/吃响应窗口的快照。

    必须单独覆盖：peng / chi 是 on_window 的决策，只测 on_turn 的话
    智能碰一类改动永远显示 0% 分歧（这正是早先漏判的原因）。
    """
    hand = [tile_code(t) for t in range(NTILE) for _ in range(c[t])]
    dp = [t for t in range(NTILE) for _ in range(4)]
    rng.shuffle(dp)
    discs = [[], [], [], []]
    for i in range(rng.randint(0, 40)):
        discs[i % 4].append(tile_code(dp[i]))
    # 上家打出的牌：优先挑一张我手上有对子（开碰窗）或能成顺（开吃窗）的
    owned = [t for t in range(NTILE) if c[t] >= 2]
    near = [t for t in range(27) if c[t] and (
        (t % 9 <= 7 and c[t + 1]) or (t % 9 <= 6 and c[t + 2])
        or (t % 9 >= 1 and c[t - 1]) or (t % 9 >= 2 and c[t - 2]))]
    if owned and (not near or rng.random() < 0.5):
        last = tile_code(rng.choice(owned))
        phase = "response_peng"
    elif near:
        last = tile_code(rng.choice(near))
        phase = "response_chi" if rng.random() < 0.5 else "response_peng"
    else:
        last = tile_code(rng.randrange(NTILE))
        phase = "response_peng"
    return {
        "phase": phase, "turn": (seat - 1) % 4, "seat": seat, "round_no": 1,
        "hand_counts": [len(hand), 13, 13, 13],
        "my_hand": hand, "drawn_tile": "",
        "melds": [[], [], [], []],
        "god": {"baotou": False, "chain_count": 0, "catch_play": False,
                "god_discarder_seat": -1},
        "discards": discs, "last_discard": last,
        "responding_seats": [seat],
        "scores": [0, 0, 0, 0], "wall_remaining": rng.randint(24, 84),
    }


def action_of(strategy, snap):
    return strategy.on_turn(View(snap, 0))


def measure(base_name, cand_name, n, rng, win_frac=0.25):
    """同时覆盖出牌决策与碰/吃窗口决策，两者任一有分歧都算。"""
    sb = Evolved(parse_genome(base_name))
    sc = Evolved(parse_genome(cand_name))
    pool = []
    for t in range(NTILE):
        pool += [t] * 4
    same = diff = 0
    turn_diff = win_diff = 0
    examples = []
    for i in range(n):
        if rng.random() < win_frac:
            c = build_win(rng)
        else:
            pick = rng.sample(pool, 14)
            c = [0] * NTILE
            for t in pick:
                c[t] += 1
        snap = snap_from_counts(c, rng=rng)
        a, b = action_of(sb, snap), action_of(sc, snap)
        if a != b:
            diff += 1
            turn_diff += 1
            if len(examples) < 2:
                examples.append(("出牌", c, a, b))
            continue
        # 回合决策一致时，再看窗口决策
        wc = list(c)
        if rng.random() < 0.5:                     # 一半走 13 张的窗口态
            for t in range(NTILE):
                if wc[t] > 0:
                    wc[t] -= 1
                    break
        wsnap = snap_window(wc, rng)
        wa = sb.on_window(View(wsnap, 0), wsnap["phase"])
        wb = sc.on_window(View(wsnap, 0), wsnap["phase"])
        if wa != wb:
            diff += 1
            win_diff += 1
            if len(examples) < 2:
                examples.append(("窗口", wc, wa, wb))
        else:
            same += 1
    return same, diff, examples, turn_diff, win_diff


def verdict(rate):
    if rate >= 20.0:
        return "值得跑对局"
    if rate >= 5.0:
        return "效应可能偏小，谨慎"
    return "不可测，别浪费房间"


def main():
    p = argparse.ArgumentParser(description="行为分歧率预筛")
    p.add_argument("--base", default=BASE_DEFAULT)
    p.add_argument("--cand", action="append", default=None,
                   help="候选基因组（可重复）")
    p.add_argument("--n", type=int, default=1500, help="抽样手牌数")
    p.add_argument("--win-frac", type=float, default=0.25,
                   help="样本中成型手的比例（decline 等基因只在成型手上才有差异）")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--sweep", action="store_true", help="对预置候批量预筛")
    a = p.parse_args()

    cands = list(SWEEP) if a.sweep else (a.cand or [])
    if not cands:
        raise SystemExit("需要 --cand 或 --sweep")

    say = print
    say("=== 分歧率预筛（零成本，跑对局之前先做）===")
    say("  基准 = %s   抽样 = %d 副手牌   成型手占比 = %.0f%%"
        % (a.base, a.n, a.win_frac * 100))
    say("  基因语义版本 = v%d" % GENE_SCHEMA)
    say("")
    say("  %-26s %6s %6s %9s %7s %7s  %s"
        % ("候选", "相同", "不同", "分歧率", "出牌", "窗口", "判据"))
    t0 = time.time()
    for cn in cands:
        rng = random.Random(a.seed)
        try:
            same, diff, ex, td, wd = measure(a.base, cn, a.n, rng, a.win_frac)
        except Exception as e:                 # noqa: BLE001
            say("  %-26s  预筛失败: %r" % (cn, e))
            continue
        rate = diff / max(1, same + diff) * 100
        say("  %-26s %6d %6d %8.2f%% %7d %7d  %s"
            % (cn, same, diff, rate, td, wd, verdict(rate)))
        if ex and rate >= 5.0:
            kind, c, act_b, act_c = ex[0]
            say("      例（%s）：%s" % (kind, " ".join(
                tile_code(t) for t in range(NTILE) for _ in range(c[t]))))
            say("        基准 -> %s      候选 -> %s" % (act_b, act_c))
    say("")
    say("  总用时 %.1f 秒（对比：一轮 10 局对局约 200 秒）" % (time.time() - t0))


if __name__ == "__main__":
    main()
