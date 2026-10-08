#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**完备信息**的决策点编码 —— 当前牌型 + 场内打出牌 + **逐对手**的出牌/副露（第 137 轮）。

为什么必须另开一个编码器
------------------------
目标里明确要求："**一定要有当前牌型、场内打出牌、对手出牌信息完备的数据**"。
而原来的 `features.py`（84 维）**不满足**这一条：

    f[0:34]  自己的手牌                      ✅ 当前牌型
    f[34:68] **四家弃牌汇总之后的"还剩几张"**  ⚠ **把"谁打的"合并掉了**
    其余    座位/庄/副露数/牌墙/…            ⚠ 只有一个**总数** `meld_sets`，
                                              没有"哪家、什么牌"的副露

⇒ 对麻将来说这是**信息缺口**：同样"3w 已经打出 2 张"，
   是**上家打的**还是**下家打的**，含义完全不同（下家打过 3w ⇒ 他不太可能要 3w）；
   而对手的**副露牌**直接暴露他的花色/听口方向。
   `features.py` 把这些**结构信息全都抹平成了一个计数**。

⚠ 硬约束（与 `features.py` 同一条，不能破）
    所有信息**只能来自 `View.snap`**（= 线上 `GET /api/games/{id}/state` 的快照），
    **不许碰模拟器专属属性**。否则线上要么崩、要么行为完全不同，
    而且**对拍也看不出来**（对拍也跑在模拟器上）。

编码（F2 = 360 维）
    自己手牌        34
    未见牌          34
    **逐家弃牌**     4 × 34 = 136      ← 新增：谁打过哪些牌
    **逐家副露牌**   4 × 34 = 136      ← 新增：谁副露了哪些牌
    逐家副露数       4
    座位 one-hot     4
    局面标量        12
"""
import os

from mj_core import NTILE, WILD, can_win, is_baotou, tile_index
import shanten_ref as S

# ★★ 第 564 轮：`dealer_streak` 的**训练侧对齐**开关（默认关）。
#
# 为什么需要它（这是一处**已用代码证实的训练/线上不一致**）：
#   · 训练走房间层 `mj_room.play_room(...)`，它把 `dealer_streak` 传进 `Game`，
#     而 `mj_sim.snapshot()` 会把它暴露出来（`mj_sim.py:588`）⇒ **训练时该维会非零**。
#   · 线上服务器的快照**没有这个键**（`crash_dump_snapshot.json` 的顶层键里没有它）
#     ⇒ `context()` 恒得 0。
#   ⇒ 同一个输入维度在训练与线上**语义不同**，而这类"输入分布不一致"**在仿真里永远看不出来**。
#
# 修法为什么选"训练侧对齐"（训练时也置 0），而不是"线上补上"：
#   线上补需要改客户端 ⇒ **必须重启客户端**，而本仓规矩是"等自然退出、绝不 kill"
#   （`run_live.py:458-461` 写明重启会丢掉正在打的 ~10 个房间）。
#   ⇒ 所以先把训练侧对齐（一行开关），把"用错的信息"换成"用不了的信息"。
#   ⚠ 代价：**丢掉连庄这一路信息**（10 维 CTX 少 1 维有效信息）。等客户端重启窗口
#     再考虑真正把它补一致。
# ⚠ **默认关** ⇒ 不改变既有行为；既有实验与线上都不受影响。
NO_STREAK = bool(int(os.environ.get("MJ_NO_STREAK", "0") or 0))

NPLAY = 4
F = 34 + 34 + NPLAY * NTILE * 2 + NPLAY + NPLAY + 12


# ---------------------------------------------------------------------------
# 候选弃牌的特征（**唯一实现**）
#
# ⚠ 这里为什么必须"唯一"：原来 `explore_discard_shard.py` 里有一份**手抄的**
#   `ukeire`，它与策略真正用的 `shanten_ref.ukeire` **不是同一个东西** ——
#   手抄版按"假设 4 张都在"算（`unseen=None`），而策略用快照算出**真正还剩几张**。
#   两份实现在"候选排序"这件事上会给出不同的名次 ⇒
#   ① 用旧数据估的"模型 vs ukeire 分歧率"量的**不是**规则用的那把尺子；
#   ② 一旦策略侧再抄一份，必然分叉（`mj_core.chi_tiles` 的教训）。
#   ⇒ 第 145 轮把它们**搬到这里**（`features_full` 只依赖 mj_core + shanten_ref，
#     不会被 mj_policy ↔ mj_arena 的循环导入卡住），`explore_discard_shard`
#     改为从本模块 import，**逐字不改语义**。
# ---------------------------------------------------------------------------
def ukeire(counts, melds, unseen=None):
    """打完之后还有多少张有效牌（**旧口径**：能降低向听的牌数，`unseen=None` 时按 4 张算）。

    ⚠ 与 `shanten_ref.ukeire` 的差别是**故意的**：这里保留旧口径，
    因为历史数据集（`dd*`/`cf*`/`ce*`）与它们的分析脚本都按这个口径写的，
    改语义会让那些结论**静默失真**。新代码要规则口径请直接用 `S.ukeire`。
    """
    base = S.shanten(counts, melds)
    n = 0
    for t in range(NTILE):
        if counts[t] >= 4:
            continue
        c = list(counts)
        c[t] += 1
        if S.shanten(c, melds) < base:
            left = 4 - counts[t]
            if unseen is not None:
                left = unseen[t]
            n += max(0, left)
    return n


def cand_scalars(v, c, t):
    """候选的 6 个标量 —— 足以**精确重建** `cand_vec` 的 39 维。

    ⚠ 第 133 轮踩的两个坑（都记在这里）：
    1. 本来只想存"牌号"以省空间，但 `is_drawn` **不在 84 维状态特征里**
       ⇒ 评估时重建不出来、训练与评估的输入会不一致。⇒ 存 6 个标量。
    2. ★ **ukeire 不能截断**：原来写成 `min(uk, 40)/40`，而 shanten 高的手牌
       ukeire 常超过 40 ⇒ **好几个候选被压成同一个值**，而
       "并列候选里 ukeire 有没有区分力"**正是本实验要检验的东西**
       ⇒ 截断会把这个信号自己毁掉（实测会虚报"ukeire 并列 75%"）。
       ⇒ 改成 `min(uk, 400)/400`（实际范围 ~0-70，等于不截断）。
    """
    sh = S.shanten(c, v.meld_sets)
    uk = ukeire(c, v.meld_sets)
    try:
        bt = 1.0 if is_baotou(c, v.meld_sets) else 0.0
    except Exception:                                             # noqa: BLE001
        bt = 0.0
    return [sh / 9.0, min(uk, 400) / 400.0, bt,
            1.0 if t == WILD else 0.0,
            1.0 if (v.drawn and t == tile_index(v.drawn)) else 0.0,
            float(t)]


CAND_DIM = 6


def features(v, counts=None):
    """完备编码。`counts` 给定时用它替换手牌（用于评估候选弃牌）。"""
    f = [0.0] * F
    c = list(counts) if counts is not None else v.counts
    snap = v.snap or {}
    k = 0
    for t in range(NTILE):
        f[k + t] = c[t] * 0.25
    k += NTILE

    # ---- 逐家弃牌 / 逐家副露（**核心新增**）----
    disc = [[0] * NTILE for _ in range(NPLAY)]
    meld = [[0] * NTILE for _ in range(NPLAY)]
    for seat, row in enumerate((snap.get("discards") or [])[:NPLAY]):
        for code in (row or []):
            try:
                disc[seat][tile_index(code)] += 1
            except (ValueError, AttributeError, TypeError):
                pass
    for seat, row in enumerate((snap.get("melds") or [])[:NPLAY]):
        for m in (row or []):
            for code in ((m or {}).get("tiles") or []):
                try:
                    meld[seat][tile_index(code)] += 1
                except (ValueError, AttributeError, TypeError):
                    pass

    # 未见牌 = 4 - 自己手牌 - 四家弃牌 - 已副露用掉的
    dead = [0] * NTILE
    me = v.seat if v.seat is not None and 0 <= v.seat < NPLAY else 0
    for seat in range(NPLAY):
        for t in range(NTILE):
            dead[t] += disc[seat][t] + meld[seat][t]
    for t in range(NTILE):
        left = 4 - c[t] - dead[t]
        f[k + t] = (left if left > 0 else 0) / 4.0
    k += NTILE

    for seat in range(NPLAY):
        for t in range(NTILE):
            f[k + t] = disc[seat][t] * 0.25
        k += NTILE
    for seat in range(NPLAY):
        for t in range(NTILE):
            f[k + t] = meld[seat][t] * 0.25
        k += NTILE
    for seat in range(NPLAY):
        f[k] = len(snap.get("melds", [[], [], [], []])[seat] or []) / 4.0 \
            if seat < len(snap.get("melds") or []) else 0.0
        k += 1

    # ---- 座位 / 局面标量（与旧编码保持同一批语义，便于对照）----
    f[k + me] = 1.0
    k += NPLAY

    dealer = snap.get("dealer")
    disc_total = sum(len(r or []) for r in (snap.get("discards") or []))
    f[k] = 1.0 if dealer is not None and me == dealer else 0.0
    f[k + 1] = v.meld_sets / 4.0
    f[k + 2] = v.meld_chi / 2.0
    f[k + 3] = (v.wall or 0) / 83.0
    f[k + 4] = disc_total / 64.0
    f[k + 5] = min(1.0, (v.chain or 0) / 4.0)
    f[k + 6] = 1.0 if v.baotou else 0.0
    f[k + 7] = 1.0 if v.catch_play else 0.0
    f[k + 8] = 1.0 if v.restricted else 0.0
    f[k + 9] = (S.shanten(c, v.meld_sets) + 1) / 9.0
    f[k + 10] = sum(c) / 14.0
    f[k + 11] = 1.0 if can_win(c, v.meld_sets) else 0.0
    return f


# ---------------------------------------------------------------------------
# 局面上下文旁路（第 145 轮）—— **不在 360 维主体里**，单独一条 `Xctx`
#
# 为什么单开一路而不并进 `features()`：
#   改 `F` 会让**所有历史数据集**（`du*`/`dg*`/`ds*` 的 `X360`）的维度与含义失效，
#   而它们是对抗"过拟合/假结论"主张的**证据链**。⇒ 与 `Xseq` / `cands` 同样的做法：
#   加**旁路**，老数据没有这一路就按缺失处理（读的时候允许没有 `Xctx`）。
#
# 为什么必须有它（对上用户第 145 轮扩写的 goal）：
#   §3 明确要求策略感知「**当前局分数**」与「**与其他玩家的分差**」——
#   这是"风险感知 / 什么时候该停止贪番"的**唯一输入**。
#   而 `features()` 的 12 个标量里**一个分数字段都没有**（实测：字段是
#   庄==我/meld数/chi数/牌墙/弃牌总数/链长/爆头/抓打圈/受限/向听/手牌数/能胡）。
#   ★ 数据其实是**现成的**：`Game.snapshot()` 里有 `scores`，
#     线上 `mj_bot.View.__init__` 也把它读进来（`self.scores = snap.get("scores") or []`）
#     ⇒ 只是**编码时没用**。
# ---------------------------------------------------------------------------
CTX = 10

# ★ 第 689 轮（GOAL_STATUS §Z.1089–§Z.1093）：**庄位身份**（`is_dealer`）—— 门控新增维。
#   动机（全部为**线上**实测）：
#     · 坐庄 Δμ **−4.05** vs 做闲 **−2.03**（CI 不含 0 ✓ 规则臂 n=875/2673 ✓ §Z.1087）
#     · 我方坐庄**和牌率更高**（0.2567 vs 0.2337）却净分更差 ⇒ 亏在**付款结构** ✓ §Z.1088
#     ⇒ 正确行为是"庄位更保守/降低被铳代价" = **位置条件化**行为 ✗
#     ⇒ 但 `context()` 只有"连庄长度"（`dealer_streak` ✓），**没有"庄家是不是我"** ✗ §Z.1089
#   ★ 安全约束（**必须**，否则打破线上 ✗ §Z.1093）：
#     `CTX` 是全局常量 ⇒ 直接改它会让**所有**调用方（含 `run_live.py` 线上客户端）
#     的 STATE_DIM 变化 ⇒ 线上现役 370 维权重**立刻加载失败** ✗✗
#     ⇒ 所以新增维**必须 env 门控**：只有设了 `MJ_CTX_DEALER=1` 才生效 ✓；默认关闭 ⇒ 线上零影响 ✓
CTX_DEALER = 1
USE_CTX_DEALER = bool(int(os.environ.get("MJ_CTX_DEALER", "0") or 0))

# ★ 第 698 轮（GOAL_STATUS §Z.1098–§Z.1099）：**牌墙分段**（`wall_remaining` 归一）—— 门控新增维。
#   前置条件（§Z.1098 ✓ 已通过）：
#     `mj_sim.py:560 snapshot()` 的 L571 已带 `"wall_remaining": self.wall_left` ✓
#     （自检目标集合 L834 亦含它 ✓；`DEADWALL = 20` ⇒ 有效摸牌 83 → 20 ✓）
#     ⇒ **非常数** ✓（不同于 `dealer_streak` 单局恒 0 的那个坑 ✗）
#   为何有用：`context()` 已有 `round_no`（**第几小局** ✓），但那是"局序号"✗，
#     不是"这副牌摸到多深"✓ —— 防守/弃胡决策依赖的是后者 ✓
#   语义：`(wall_remaining − DEADWALL) / (83 − DEADWALL)` ∈ [0,1]（1=刚开局 ✓ 0=将流局 ✓）
CTX_WALL = 1
USE_CTX_WALL = bool(int(os.environ.get("MJ_CTX_WALL", "0") or 0))



def context(v):
    """(CTX,) 的局面上下文：分数 / 分差 / 名次 / 连庄长度 / 局号。

    ⚠ `dealer_streak`（连庄长度）走 `snap["dealer_streak"]`：
    `mj_sim.snapshot()` 目前**没有**这个键（单局模拟 `round_no` 恒为 1，
    连庄由 `mj_room.py` 的 `succ` 模式在**房间层**维护）⇒ 单局数据里它恒为 0。
    这正是本函数要暴露的那个缺口：**不补房间层，agent 学不到连庄相关的行为。**
    """
    snap = v.snap or {}
    me = v.seat if v.seat is not None and 0 <= v.seat < NPLAY else 0
    sc = list(v.scores or [])
    while len(sc) < NPLAY:
        sc.append(0.0)
    sc = [float(x) for x in sc[:NPLAY]]
    out = [0.0] * CTX
    for s in range(NPLAY):                      # 0-3 四家累计分
        out[s] = max(-2.0, min(2.0, sc[s] / 40.0))
    k = NPLAY
    for s in range(NPLAY):                      # 4-6 我与三家的分差
        if s == me:
            continue
        out[k] = max(-2.0, min(2.0, (sc[me] - sc[s]) / 40.0))
        k += 1
    order = sorted(range(NPLAY), key=lambda s: -sc[s])
    out[k] = order.index(me) / 3.0              # 7 当前名次（0=第一）
    k += 1
    try:
        streak = float(snap.get("dealer_streak") or 0)
    except (TypeError, ValueError):
        streak = 0.0
    if NO_STREAK:                               # ★ 第 564 轮：训练侧对齐（见文件头注释）
        streak = 0.0
    out[k] = min(1.0, streak / 5.0)             # 8 连庄长度（单局数据恒 0）
    k += 1
    try:
        rn = float(snap.get("round_no") or 1)
        rr = float(snap.get("rounds") or snap.get("total_rounds") or 8)
    except (TypeError, ValueError):
        rn, rr = 1.0, 8.0
    out[k] = min(1.0, rn / max(rr, 1.0))        # 9 局号进度
    if USE_CTX_DEALER:
        # ★ 第 689 轮：**庄位身份**（唯一新增维 ✓ 默认关闭 ✓ §Z.1093）
        #   `dealer` 由 `mj_bot.py:364` 的 `snap["dealer"]` 提供 ✓
        try:
            dl = int(snap.get("dealer", -1))
        except (TypeError, ValueError):
            dl = -1
        out.append(1.0 if dl == me else 0.0)
    if USE_CTX_WALL:
        # ★ 第 698 轮：**牌墙分段**（唯一新增维 ✓ 默认关闭 ✓ §Z.1098）
        #   `mj_sim.py:571` 的 `snap["wall_remaining"]` ✓；有效区间 [DEADWALL, 83] ✓
        try:
            wr = float(snap.get("wall_remaining", 83.0))
        except (TypeError, ValueError):
            wr = 83.0
        out.append(max(0.0, min(1.0, (wr - 20.0) / 63.0)))
    return out


# ---------------------------------------------------------------------------
# ★ 第 549 轮：**对手副露信息**（`CTX2` 旁路）—— goal §3 里明确要求、而当前表示缺失的一路
# ---------------------------------------------------------------------------
# 为什么它确实是"缺失"的（读代码得到的证据，不是感觉）：
#   `features()` 里**已经**读了 `snap["melds"]`，但只把它**累加进 `dead`**（未见牌统计）——
#   见本文件 `for seat, row in enumerate(...melds...)` 那段。
#   ⇒ 网络因此知道"**哪些牌被用掉了**"，但**不知道"谁副露了什么"**；
#     而"某家已经 3 副露"与"某家从未副露"是完全不同的威胁信号（goal §3「对手副露信息」）。
# 为什么单开旁路而不并进 `features()` / `context()`（沿用本文件第 145 轮写下的同一条理由）：
#   改 `F` 或 `CTX` 会让**所有历史数据集**（`du*`/`dg*`/`ds*` 的 `X360`/`Xctx`）的维度与含义失效，
#   而那些数据是对抗"过拟合/假结论"主张的**证据链**。
# ⇒ 做法与 `Xctx` 一致：**加旁路**，老数据没有这一路就按缺失处理。
# ⚠ **默认不启用**：启用会改变 `STATE_DIM`，而**线上现役的 `rl_bc6` 是 370 维** ——
#   全局改维度会让它**加载不了**。因此这里**只提供函数与常量**，
#   是否把它接进状态由调用方用**显式开关**决定（默认关，线上逐字不受影响）。
CTX2 = 6


def context_extra(v):
    """(CTX2,) 对手副露摘要：3 家各自的 **副露数** 与 **是否含杠**。

    维度安排（全部归一化到 [0,1]）：
        0-2  三个对手的副露数 / 4
        3-5  三个对手是否有杠（0/1）
    ⚠ 只描述**对手**：自己的副露数**已经在 `F` 里**（`len(melds[me])/4`）⇒ 不重复放。
    ⚠ 字段缺失（老数据 / 单局模拟没有 `melds`）⇒ **返回全 0**，不抛异常、不猜。
    """
    snap = v.snap or {}
    me = v.seat if v.seat is not None and 0 <= v.seat < NPLAY else 0
    melds_all = snap.get("melds") or []
    out = [0.0] * CTX2
    j = 0
    for seat in range(NPLAY):
        if seat == me:
            continue
        row = melds_all[seat] if seat < len(melds_all) else []
        row = row or []
        if j >= NPLAY - 1:
            break
        out[j] = min(1.0, len(row) / 4.0)
        has_gang = 0.0
        for m in row:
            k_ = str((m or {}).get("kind") or "")
            if "gang" in k_ or "杠" in k_:
                has_gang = 1.0
                break
        out[NPLAY - 1 + j] = has_gang
        j += 1
    return out


# ---------------------------------------------------------------------------
# 紧凑表征（第 160 轮）：**给"胡 / 弃"这一个二值决策**用的少量充分统计量
# ---------------------------------------------------------------------------
# 为什么需要它（这是 `decline_net.pt` 第一版**必然过拟合**的直接原因）：
#   第一版直接把 `features(v)`（360 维）+ `context(v)`（10 维）= **370 维**
#   塞进 hidden 256 / depth 2 的 TNet ⇒ 约 **16 万参数**，而"胡/弃"决策点
#   全数据集只有 **883 条无偏样本**。实测 train/val gap **+150.1**、val loss 318.5，
#   且退化成"建议弃 100%"—— 那不是"学不到"，是**容量比样本多了两个数量级**。
# 而 `analyze_decline_cate.py` 的**分层 ATE** 已经指出：τ 的异质性主要来自
#   庄/闲、财神张数、自家副露数、牌墙、动作链 —— **五个量**。
#   ⇒ 用它 + `context(v)` 做 18 维输入，参数降到几十个，问题才是适定的。
# ⚠ 训练侧从 `.npz` 的记录字段重建、推理侧从 `v` 现场重建 ⇒ **两侧必须共用本函数**
#   （本仓纪律：允许副本但必须有对拍；共用就不存在副本）。
COMPACT_DIM = 18                                    # 8 标量 + CTX(10)


def compact_ctx(wall, chain, melds, is_dealer, wild, branch_b, ctx):
    """紧凑特征的**唯一布局定义**。训练与推理都必须走这里。

    8 个标量 = CATE 识别出的异质性来源 + 一个交互项：
      0 庄位      —— CATE：庄 +13.04 / 闲 +6.63（规则却无条件禁止庄家弃胡）
      1 财神张数  —— CATE 单调：1/2/3 张 = +5.16/+9.58/+13.23
      2 自家副露  —— CATE 单调递减：0/1/2/3 = +15.2/+10.4/+6.2/+4.6
      3 牌墙/70   —— CATE：24~39 +7.53 / 40~59 +6.12 / >=60 +10.65
      4 残墙>=60  —— 上面那个 U 形的右端，线性项表达不了
      5 动作链    —— §2.6 财飘/杠链，Multiplier=2^n
      6 分支      —— B=打一张仍爆头 / C=safe_piao 打白板（两种弃法风险不同）
      7 庄×财神   —— 交互项：两个 CATE 上都是正，线性模型需要它才能表达"更该弃"
    """
    w = float(wall or 0.0)
    return [
        float(is_dealer),
        float(wild),
        float(melds),
        w / 70.0,
        1.0 if w >= 60.0 else 0.0,
        float(chain),
        float(branch_b),
        float(wild) * float(is_dealer),
    ] + [float(x) for x in ctx]


def compact_from_view(v, branch_b):
    """推理侧：从 `View` 现场算紧凑特征。

    ⚠ 每个字段都**逐字对齐** `gen_decline_shard.py:132-136` 的记录口径
      （那是训练数据的来源）—— 换了口径就等于换了一个输入空间。
    """
    return compact_ctx(
        v.wall or 0.0,
        v.chain or 0.0,
        int(v.meld_sets or 0),
        1 if (v.snap or {}).get("dealer") == v.seat else 0,
        int(v.counts[WILD]) if v.counts else 0,
        branch_b,
        context(v),
    )


def discard_seq(v, K=20):
    """**逐家的弃牌序列**（第 142 轮）—— 有序、取最近 K 张，左补 -1。

    为什么需要它：第 141 轮把逐家信息摊成 `4×34` 的**计数**喂全连接 MLP，
    四个目标上都没有增益。当时的判断是**归纳偏置不对**：
    计数丢掉了**顺序**（"谁在什么时候打了什么"才是读牌依据）与**交互**。
    ⇒ 这个函数把"顺序"原样交出来，供序列模型（1D 卷积 / 小 transformer）使用，
      从而能**只换结构、不换信息**地做对照。

    ⚠ 取**最近** K 张：读牌看的是"刚刚发生了什么"；早期弃牌的信息大多
      已经体现在"还剩几张"（未见牌段）里了。
    ⚠ 只依赖 `v.snap`（与 `features` 同一条硬约束）。
    """
    out = [[-1] * K for _ in range(NPLAY)]
    snap = v.snap or {}
    for seat, row in enumerate((snap.get("discards") or [])[:NPLAY]):
        idxs = []
        for code in (row or []):
            try:
                idxs.append(tile_index(code))
            except (ValueError, AttributeError, TypeError):
                pass
        idxs = idxs[-K:]                        # 最近 K 张
        for j, t in enumerate(idxs):
            out[seat][K - len(idxs) + j] = t    # 左补 -1
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    print("=== features_full ===")
    print("  维度 F = %d（旧编码 84）" % F)
    print("  逐家弃牌 %d 维 + 逐家副露 %d 维 = 新增 %d 维"
          % (NPLAY * NTILE, NPLAY * NTILE, NPLAY * NTILE * 2))
    print("  ⚠ 完整自检请跑 `python check_features_full.py`")
