#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**房间层**：按真实赛制把多局串起来（庄位继承）。

为什么需要它（第 32 轮）
------------------------
`probe_posture_frontier._play` 一直是 `Game(seed, dealer=(seed//4)%4)` ——
**每局庄家强制独立**。但真实赛制经**逐局对拍**确认（`check_dealer_succession.py`）：

    「下一局庄家 == 本局赢家」   437/437 = **100.00%**（流局连庄）

⇒ 夹具里**局与局之间没有任何状态传递**，而真实过程有：
赢 → 做庄 → 期望更高（`MODEL_SETTLEMENT.md` §3.2：做庄 +0.40 vs 做闲 −0.13）
→ 更容易再赢。这是**正反馈**。

**这对什么有影响**（务必分清）：
  · 波 1~5 的**逐局** Δμ/Δσ 对比 —— **不受影响**，那些实验要的就是边际量；
  · **赛段级** σ 与位置/晋级模型 —— 可能有影响（自相关）；
  · **RL** —— **确定有影响**：夹具里学不到"赢下这局换来下局庄位"这个期权。

所以这一层不是"更真实的模拟"这种锦上添花，而是 **RL 的前置条件**。

设计约定（与既有夹具保持一致，不自创）
--------------------------------------
    seat 约定：调用方自己决定哪一席是"我们"（夹具用 `seed % 4`）；
    初始庄家：`seed % 4`（`audit_stage_sigma.py` 用 `(seed//4)%4` 时自行传 `first_dealer`）；
    每局牌山：`wall_seed(room_seed, t)` —— 房间之间不重叠，房间内每局不同。

用法
----
    python mj_room.py --self-test
"""
import argparse
import sys

sys.path.insert(0, ".")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mj_sim import Game                                        # noqa: E402

BASE_GENOME = "shanten:1:1:2:2:1:1:0"
DEFAULT_ROUNDS = 8          # 服务端 auto 房写死的 Rounds=8


def wall_seed(room_seed, t, salt=0):
    """房间内第 t 局的牌山 seed。

    ⚠ 房间之间必须不重叠：`room_seed * 1000 + t` 要求 `t < 1000`
    （Rounds 最大 16，绰绰有余）。`salt` 用于同房间多组对照（如 succ/iid）
    共用**同一批牌** —— 配对对照的关键。
    """
    if t >= 1000:
        raise ValueError("t=%d 超出 1000，会让房间之间的牌山重叠" % t)
    return (room_seed * 1000 + t) * 16 + (salt % 16)


# ★★ 第 164 轮：`wall_seed` 的**逆**。为什么必须有它（而且必须放在这里）：
#   `Xctx` 在 room 模式下**不再是常量**（§Z.142），所以 room 数据回来之后，
#   **同一房间里的多局共享**同一批累计分 / 连庄链 / 对手 ⇒ 它们**不是独立的**。
#   而 `analyze_decline_cate.py` / `train_decline.py` 的自举与"按局切 train/val"
#   都按 `gid` 聚类 ⇒ 在 room 数据上会
#     ① **把 CI 算小**（同房间的多局被当独立），
#     ② **让同一房间的局漏进验证集**（泄漏）—— 后者更危险，因为它让 val 指标**偏乐观**。
#   ⇒ 聚类单元必须能升到**房间**。而 id 是 `wall_seed` 造的，
#     **逆函数放在生成它的同一个模块里**才不会再分叉出一份口径。
def room_of_gid(gid):
    """从局 id 反推房间号（`salt` 构造上 < 16）。"""
    return int(gid) // 16 // 1000


def round_of_gid(gid):
    """从局 id 反推它是该房间的第几局（0 起）。"""
    return (int(gid) // 16) % 1000


def is_room_level(gid):
    """这份数据的 id 是不是 room 模式造的？

    room 模式：`gid = (room_seed*1000 + t)*16 + salt`，`room_seed ≥ 1e6`
      ⇒ `gid ≥ 1.6e10`；game 模式：`gid` 就是种子里的**小整数**（5e6 ~ 9.1e7）。
    ⇒ 用 `1e9` 作阈值能干净分开 **当前用到的所有种子范围**。
    ⚠ 这是**启发式**，所以调用方一律**打印判定结果**，允许 `--cluster` 显式覆盖。
    """
    import numpy as _np
    a = _np.asarray(gid)
    return bool(a.size and int(a.max()) >= 10 ** 9)


def play_room(seed, policies, rounds=DEFAULT_ROUNDS, mode="succ",
              first_dealer=None, salt=0):
    """跑一个房间，返回逐局结果列表。

    mode="succ"：真实规则，`dealer ← winner`（流局连庄）。
    mode="iid" ：庄家每局均匀轮转（= 旧夹具的行为），用于**配对对照**。
    """
    dealer = first_dealer if first_dealer is not None else seed % 4
    out = []
    # ★ 第 145 轮：**累计分**要带进下一局，否则策略在决策点上永远只看到 [0,0,0,0]
    #   （`self.scores` 只在结算时写）⇒ 扩写后的 goal §3 要求的
    #   「当前局分数 / 与其他玩家的分差」就**根本不存在**。
    total = [0.0, 0.0, 0.0, 0.0]
    streak = 0                      # 连庄长度：庄家连续坐庄的局数
    for t in range(rounds):
        d = dealer if mode == "succ" else (seed + t) % 4
        ws = wall_seed(seed, t, salt)
        r = Game(seed=ws, dealer=d, carry_scores=list(total),
                 dealer_streak=streak, round_no=t + 1,
                 rounds=rounds).play(policies)
        total = [total[i] + float(r["scores"][i]) for i in range(4)]
        out.append({"round": t, "wall_seed": ws, "dealer": d,
                    "winner": r.get("winner"), "scores": list(r["scores"]),
                    "total": list(total), "dealer_streak": streak,
                    "fan": r.get("fan"), "draw": bool(r.get("draw")),
                    # ★★ 第 347 轮补：番型分解项。**这些量 `mj_sim` 早就在算了**
                    #   （`mj_sim.py:606-615` 的结果字典里就有 fan/baotou/chiitoi/
                    #    win_wild/chain/piao/melds），只是 `play_room` **没有透传**。
                    #   为什么必须补：§Z.347 用线上真值发现**我方几乎不财飘**
                    #   （P(财飘|胡) = 0.0008 vs 真场 0.0090），而 `decline` 同时是
                    #   财飘闸门（`mj_policy.py:145`）⇒ 若要筛 decline=6/7，
                    #   **必须有 piao 这个直接读数**；否则"番没动"这一结果
                    #   无法区分"闸门没开"与"闸门开了但不值钱"（= 歧义）。
                    #   ⚠ 只加字段、不改任何判定逻辑 ⇒ 对既有脚本逐字无害
                    #     （它们都按 key 取值，且不比较 row 的全等）。
                    "piao": r.get("piao"), "chain": r.get("chain"),
                    "baotou": r.get("baotou"), "chiitoi": r.get("chiitoi"),
                    "win_wild": r.get("win_wild"), "melds": r.get("melds")})
        if mode == "succ":
            w = r.get("winner")
            # ★★★ 第 299 轮修一个**真 bug**（由 subagent 报告，我从源码复核确认）：
            #   `mj_sim` 流局时 `winner` 是 **-1**，**不是 None** ——
            #   `mj_sim.py:124` 初始化 `self.winner = -1`，而 `_settle_draw`
            #   （`:453-457`）只写 `draw=True` 与 `phase="finished"`，
            #   **从不写 winner** ⇒ `winner` 保持 -1。
            #   旧写法只判 `w is None` ⇒ **流局那一支永远进不去** ⇒
            #   落到 else 把 `streak = 0`（流局本该连庄 +1），并且
            #   `dealer = w if w is not None else dealer` ⇒ **`dealer = -1`**
            #   被带进下一手 `Game(seed=ws, dealer=-1, ...)`。
            #   `Game(dealer=-1)` 的后果（读 `mj_sim._settle_win`）：
            #   `if seat == self.dealer` 永不成立 ⇒ 全走 else，
            #   `mult = 8 if s == self.dealer else 1` ⇒ **每家只付 1f，
            #   庄家那条 8f 整条消失**；且 `self.turn = -1` 会让 `hands[-1]`（3 号位）
            #   白拿第 14 张。**继承链从那一手起全错。**
            #   ⇒ 正确判据是「winner < 0 即流局」（规则原文：流局庄家连庄）。
            try:
                wi = int(w)
            except (TypeError, ValueError):
                wi = -1
            if wi < 0:                          # 流局（winner = -1 或 None）
                streak += 1                     # 流局连庄：庄家不变、连庄 +1
            elif wi == d:
                streak += 1                     # 庄家胡 ⇒ 连庄
                dealer = wi
            else:
                streak = 0                      # 换庄
                dealer = wi
    return out


def test_room_scores_zero_sum(rows):
    """每个房间每局的四家得分必须零和（恒等式的前提）。"""
    for r in rows:
        if abs(sum(r["scores"])) > 1e-9:
            return False, r
    return True, None


def self_test(rooms=30, rounds=8, mode="succ"):
    from mj_arena import make_strategy

    succ = iid = tot = 0
    zero_bad = []
    dealer_hist = [0, 0, 0, 0]
    outside = 0          # ★ 第 299 轮：dealer 不在 0..3 的手数（旧版**静默漏掉**）
    n_draw = 0
    for s in range(rooms):
        pols = [make_strategy(BASE_GENOME) for _ in range(4)]
        rows = play_room(s, pols, rounds, mode=mode)
        ok, bad = test_room_scores_zero_sum(rows)
        if not ok:
            zero_bad.append((s, bad))
        for i, r in enumerate(rows):
            d = r["dealer"]
            # ★★ 旧版写的是 `dealer_hist[r["dealer"]] += 1` ——
            #   一旦 `dealer == -1`，Python 会把 -1 **静默算成最后一个桶（座位 3）**
            #   ⇒ 流局的错庄被伪装成"座位 3 坐庄"，直方图看起来仍然正常。
            if 0 <= d < 4:
                dealer_hist[d] += 1
            else:
                outside += 1
            if r.get("draw"):
                n_draw += 1
        for i in range(len(rows) - 1):
            tot += 1
            w, d = rows[i]["winner"], rows[i]["dealer"]
            nxt = rows[i + 1]["dealer"]
            # ★★ 旧判据 `(w is None and nxt == d) or (w is not None and nxt == w)`
            #   在流局时 `w = -1`、`nxt = -1` ⇒ `-1 is not None` 为真且 `-1 == -1`
            #   为真 ⇒ **被判成"继承正确"** ⇒ 自测 100% 通过是**假通过**。
            try:
                wi = int(w)
            except (TypeError, ValueError):
                wi = -1
            if wi < 0:
                good = (nxt == d)               # 流局 ⇒ 连庄（庄家不变）
            else:
                good = (nxt == wi)              # 赢者坐庄
            if good:
                succ += 1

    print("=== 房间层自测（%d 房 × %d 局，mode=%s）===" % (rooms, rounds, mode))
    print("  零和检查：%s" % ("✅ 全部零和" if not zero_bad
                             else "✗ %d 局不零和" % len(zero_bad)))
    print("  流局手数：%d / %d（%.3f%%）" % (n_draw, rooms * rounds,
                                         100.0 * n_draw / max(1, rooms * rounds)))
    print("  ★ dealer 不在 0..3 的手数 = **%d**（必须为 0；旧版漏掉这个检查）" % outside)
    print("  庄位继承命中率：%d/%d = %.2f%%"
          % (succ, tot, 100.0 * succ / max(1, tot)))
    print("  庄家分布：%s（均匀应约各 25%%）"
          % [round(100.0 * x / (rooms * rounds), 1) for x in dealer_hist])
    ok = (not zero_bad) and (succ == tot) and (outside == 0)
    if mode == "succ":
        print("  %s" % ("✅ 继承规则成立（应恰好 100%）且无越界庄家"
                        if ok else "✗ 继承规则没成立 —— 房间层有 bug"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--rooms", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--mode", default="succ", choices=["succ", "iid"])
    a = ap.parse_args()
    if a.self_test or True:
        return self_test(a.rooms, a.rounds, a.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
