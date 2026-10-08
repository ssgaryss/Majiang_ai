#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL 环境包装 —— 让**任意策略对象**驱动 `mj_sim` 自对弈，并把决策轨迹交出来。

为什么需要这一层
----------------
`mj_sim.Game.play(strategies)` 已经接受任意实现了 `on_turn(v)` / `on_window(v, phase)`
的对象，所以"注入一个可学习策略"本身并不难。**难的是保证这层包装没有偷偷改变环境。**
本项目吃过这个亏：`run_self_test.py` 曾经一直用玩具策略（`my_bot.Player`）跑自对战，
所有结论都建立在错的策略上（§Q）。所以这一层的**第一件事是自证清白**，不是学东西。

正确性闸门（`--gate`，默认就跑）
--------------------------------
把 live 基因组包成策略，跑同样的 seed，要求**逐 seed 与既有夹具完全一致**
（不是"统计上接近"，是 `abs(diff) < 1e-9`）。基准取 `probe_posture_frontier._play`，
因为波 1~5 的**全部**结论都是它产出的 —— 环境对不上它，学出来的东西就没法跟历史比。

座位/庄家约定（必须与夹具逐字一致，否则配对是全错的）
----------------------------------------------------
    my_seat = seed % 4          被评估的那一席
    dealer  = (seed // 4) % 4
这是 `probe_posture_frontier._play` 的约定，本文件**照抄**，不自创。

用法
----
    python rl_env.py --gate             # 只跑正确性闸门
    python rl_env.py --gate --n 400
    python rl_env.py --trace 4          # 打印前几局的决策轨迹（看接口对不对）
"""
import argparse
import sys

sys.path.insert(0, ".")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_GENOME = "shanten:1:1:2:2:1:1:0"      # 线上基因组


def seat_and_dealer(seed):
    """被评估席位与庄家 —— 与 `probe_posture_frontier._play` 完全一致。"""
    return seed % 4, (seed // 4) % 4


class Policy(object):
    """RL 策略接口。要学的东西只需要覆写这两个方法。

    `on_turn(v)`   -> {"action": "discard"/"peng"/"chi"/"gang", ...} 或 None
    `on_window(v, phase)` -> 是否响应（碰/吃 窗口）
    """

    def on_turn(self, v):                                      # noqa: ARG002
        raise NotImplementedError

    def on_window(self, v, phase):                             # noqa: ARG002
        return None


class GenomePolicy(Policy):
    """把既有基因组包成 Policy，用于**对齐夹具**和当基线/对手。"""

    def __init__(self, genome=BASE_GENOME, fresh=True):
        self.genome = genome
        if fresh:
            from mj_arena import make_strategy
            self.inner = make_strategy(genome)
        else:
            from probe_posture_frontier import _strat
            self.inner = _strat(genome)

    def on_turn(self, v):
        return self.inner.on_turn(v)

    def on_window(self, v, phase):
        return self.inner.on_window(v, phase)

    def __repr__(self):
        return "GenomePolicy(%s)" % self.genome


class RecordingPolicy(Policy):
    """透明包装：记录每一手（view 特征、动作），用于 RL 的轨迹收集。

    只记录"被包装策略实际做的决定"，不改变它 —— 所以它天然可用于闸门自检。
    """

    def __init__(self, inner, seat=None):
        self.inner = inner
        self.seat = seat
        self.turns = []      # [(seat, action_dict)]
        self.windows = []    # [(seat, phase, action)]

    def on_turn(self, v):
        act = self.inner.on_turn(v)
        self.turns.append((getattr(v, "seat", None), act))
        return act

    def on_window(self, v, phase):
        act = self.inner.on_window(v, phase)
        self.windows.append((getattr(v, "seat", None), phase, act))
        return act


def episode(seed, policies):
    """跑一局。返回 `mj_sim.Game.result()`（含 scores / winner / fan / draw）。"""
    from mj_sim import Game
    _seat, dealer = seat_and_dealer(seed)
    return Game(seed=seed, dealer=dealer).play(policies)


def four(policy_factory):
    """四个座位各一份**独立**策略实例（共享实例会让状态串味）。"""
    return [policy_factory() for _ in range(4)]


# ---------------------------------------------------------------- 闸门

def gate(n=200, verbose=True):
    """逐 seed 对拍 `probe_posture_frontier._play`。返回 (ok, bad_list)。"""
    from probe_posture_frontier import _play

    bad = []
    for s in range(n):
        r = episode(s, four(lambda: GenomePolicy(BASE_GENOME)))
        seat, _d = seat_and_dealer(s)
        mine = float(r["scores"][seat])
        ref = float(_play(s, BASE_GENOME)["score"])
        if abs(mine - ref) > 1e-9:
            bad.append((s, mine, ref))
            if verbose and len(bad) <= 5:
                print("  ✗ seed=%d 本环境 %+g  vs  夹具 %+g" % (s, mine, ref))

    print()
    print("=== 正确性闸门：RL 环境 vs 既有夹具（%d seed，逐 seed 精确对拍）===" % n)
    if bad:
        print("  ✗ **不一致 %d / %d** ⇒ 这层包装改变了环境，先修再谈学习。" % (len(bad), n))
        print("    任意一个小差异都会让 RL 的结论与波 1~5 无法比较。")
        return False, bad
    print("  ✅ %d/%d 逐 seed **完全一致**（|diff| < 1e-9）" % (n, n))
    print("     ⇒ 这层包装是透明的：可学习策略可以挂在这里，结论与历史可比。")
    return True, bad


def trace(n=3, genome=BASE_GENOME):
    """打印前 n 局的决策轨迹，确认接口/动作形状是对的。"""
    for s in range(n):
        recs = four(lambda: RecordingPolicy(GenomePolicy(genome)))
        r = episode(s, recs)
        seat, dealer = seat_and_dealer(s)
        print("--- seed=%d  被评估席位=%d  庄家=%d ---" % (s, seat, dealer))
        print("    scores=%s winner=%s fan=%s draw=%s"
              % (r["scores"], r.get("winner"), r.get("fan"), r.get("draw")))
        mine = recs[seat]
        print("    本席 on_turn %d 次，on_window %d 次"
              % (len(mine.turns), len(mine.windows)))
        for _s, a in mine.turns[:4]:
            print("      turn   -> %s" % (a,))
        for _s, ph, a in mine.windows[:4]:
            print("      window(%s) -> %s" % (ph, a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="跑正确性闸门")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--trace", type=int, default=0)
    a = ap.parse_args()

    if not a.gate and not a.trace:
        a.gate = True                     # 默认就跑闸门：它是这层存在的理由

    ok = True
    if a.gate:
        ok, _bad = gate(a.n)
    if a.trace:
        print()
        trace(a.trace)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
