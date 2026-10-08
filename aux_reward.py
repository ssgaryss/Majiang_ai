#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""goal **§4 Auxiliary Reward** —— 四个辅助信号的**唯一实现**。

目标的原文（逐字）：

    R_t = R_terminal + λ₁·R_hand + λ₂·R_efficiency + λ₃·R_multiplier − λ₄·R_risk

    Hand Progress：Shanten 改善 / 听牌 / 有效牌增加 / 牌型完成度
    Efficiency   ：Ukeire / 有效进张 / 牌型灵活性 / 未来路线数量
    Multiplier   ：爆头潜力 / 财飘潜力 / 杠链潜力 / 七对·豪华七对潜力 / 4 白板潜力
    Risk         ：延迟风险 / 流局风险 / 对手高价值胡牌风险 / 机会成本

    ⚠ 辅助 reward **必须显著低于最终真实得分的影响**，
      避免智能体通过刷中间 reward 而偏离真实麻将目标。

为什么必须先把公式写死、并**逐个验证它真的预测终局得分**（本仓纪律）
------------------------------------------------------------------
本仓已经吃过"代理指标不对目标指标验证"的亏。辅助 reward 的**唯一危险**是
**reward hacking**：模型去刷那个中间量，而那个量与真实得分无关甚至负相关。
⇒ 所以本模块交付两样东西：
  1. 四个信号的**显式公式**（下面每个函数都写清每一项的来历与量纲）；
  2. `verify_aux.py`：在**真实决策点**上量
     **每个信号与终局 ΔScore 的相关系数**，并据此给出 λ 的建议量级。
     ⚠ **不相关（或符号反）的信号不许进 reward** —— 那正是 hacking 的入口。

设计约束
--------
* 只读 **`View` 上的字段**（与 `features_full` 同一条硬约束：只依赖线上快照），
  ⇒ 训练（模拟器）与部署（线上）能算出同一批数；
* 每个信号**逐个决策点**定义（单步增量），不依赖未来 ⇒ 可以在仿真里在线算；
* 量纲一律 **"分"**：每个信号都按"它大致值多少分"来定标，
  这样 λ 才是无量纲的"信任度"，而不是一个凑出来的魔数。

用法（本模块是库；验证脚本见 `verify_aux.py`）：
    from aux_reward import hand_progress, efficiency, multiplier_potential, risk
"""
from mj_core import NTILE, WILD, is_baotou, melds_of
import shanten_ref as S

# 默认权重（**先给保守值**；`verify_aux.py` 会按"辅助总和 ≪ σ_terminal"校准）
LAMBDA = {"hand": 0.10, "efficiency": 0.10, "multiplier": 0.05, "risk": 0.10}


# --------------------------------------------------------------------- 工具
def _pairs(counts):
    """对子数（含财神可补的落单：财神算"万能"，见 `seven_pairs_mult` 的口径）。"""
    n = sum(1 for t in range(NTILE) if t != WILD and counts[t] >= 2)
    return n


def _quads(counts):
    return sum(1 for t in range(NTILE) if t != WILD and counts[t] >= 4)


def _ukeire(counts, melds, unseen=None):
    """规则同款口径：能降低向听的牌数（`shanten_ref.ukeire`）。"""
    try:
        return S.ukeire(counts, melds, unseen=unseen)[0]
    except Exception:                                                # noqa: BLE001
        return 0


def _shanten(counts, melds):
    try:
        return S.shanten(counts, melds)
    except Exception:                                                # noqa: BLE001
        return 8


# ------------------------------------------------------- 1. Hand Progress
def hand_progress(counts_before, melds, counts_after, unseen=None):
    """**手牌进度**（单步，单位≈分）。

    = 2.0·(向听下降张数) + 3.0·[这一步进听] + 0.05·(有效牌变化)

    权重来历（摆出来，不含魔数）：
      · 向听下降 1 步 → 大约把"还需摸几次"砍掉一截。本仓实测**有效牌每多 1 张值
        +0.0403 分**（§Z 的随机化估计），而进听一步的经验价值远大于此 ⇒ 取 2.0 分/步
        作为**保守下界**（真实值由 `verify_aux.py` 与终局得分的相关性反查）。
      · 进听 = 从"还要再进一次"变成"只等胡" ⇒ 取 3.0 分。
      · 有效牌每张 0.05 分（比实测 0.0403 略大，留一点余量）。
    ⚠ 三者都只是**信号**，最终由 λ 与验证决定是否留下。
    """
    s0 = _shanten(counts_before, melds)
    s1 = _shanten(counts_after, melds)
    u0 = _ukeire(counts_before, melds, unseen)
    u1 = _ukeire(counts_after, melds, unseen)
    v = 2.0 * (s0 - s1)
    if s1 == 0 and s0 != 0:
        v += 3.0
    v += 0.05 * (u1 - u0)
    return v


# ---------------------------------------------------------- 2. Efficiency
def efficiency(counts, melds, unseen=None):
    """**效率**（状态量，单位≈分）：有效牌 + 牌型灵活性 + 未来路线数。

    = 0.05·有效牌枚数 + 0.5·(有效牌**种类**数 / 8) + 1.0·[七对路线仍可行]

    · 有效牌枚数：直接对应"下一次摸牌能进听的概率"；
    · **种类**数：同样 10 张有效牌，摊在 6 种里比挤在 2 种里更抗"被别人摸走"
      ⇒ 这是"灵活性"，取 0.5 分满标；
    · 七对路线仍可行（对子数 ≥ 4 且无副露）：保留一条未来路线，取 1.0 分。
    ⚠ 刻意**不**把"最低向听"写进来 —— 它是主键，不该在辅助里重复计一次。
    """
    u = _ukeire(counts, melds, unseen)
    kinds = 0
    try:
        kinds = S.ukeire(counts, melds, unseen=unseen, count_kinds=True)[0]
    except Exception:                                                # noqa: BLE001
        kinds = 0
    v = 0.05 * u + 0.5 * min(1.0, kinds / 8.0)
    if melds == 0 and _pairs(counts) >= 4:
        v += 1.0
    return v


# -------------------------------------------------- 3. Multiplier Potential
def multiplier_potential(counts, melds, wall=None, chain=0, piao=0, baotou=False):
    """**倍率潜力**（状态量，单位≈分）：爆头 / 财飘 / 杠链 / 七对 · 豪华 / 4 白板。

    = 4.0·[已爆头]                      ← 爆头是把 ×1 变 ×2，一次就值 4 分左右
    + 2.0·chain + 1.0·piao              ← 链每 +1 就是再乘 2（`Multiplier = 2^n`）
    + 1.5·[可杠张数 ≥ 1]                ← 杠 = 一次额外摸牌 + 链 +1
    + 1.0·[对子 ≥ 5] + 1.0·[四张 ≥ 1]   ← 七对 / 豪华七对的门槛
    + 2.0·[手上白板 ≥ 2]                ← 4 白板（×2）× 财飘的复合潜力

    权重来历：**倍率是乘性的**（`分支 × 2^链 × 4白板 × 爆头`），
    所以"已经在倍率位上"值多少钱应当用**当前胡牌期望**衡量：
    本仓实测均番 ≈1.3、庄家 pay ≈ 8×番 ⇒ 一次胡约 10~20 分，
    乘以"这一步真能兑现"的概率 ⇒ 取 4 分为爆头的上限估值（保守）。
    ⚠ 这个信号**最容易诱发 hacking**（模型会为了保爆头而不胡）
    ⇒ `verify_aux.py` 必须单独看它与终局得分的相关性**符号**。
    """
    v = 0.0
    if baotou:
        v += 4.0
    v += 2.0 * float(chain or 0) + 1.0 * float(piao or 0)
    if any(counts[t] >= 4 for t in range(NTILE) if t != WILD):
        v += 1.5
    if _pairs(counts) >= 5:
        v += 1.0
    if _quads(counts) >= 1:
        v += 1.0
    if counts[WILD] >= 2:
        v += 2.0
    return v


# ------------------------------------------------------------------- 4. Risk
def risk(declined_win=False, wall=None, tenpai=False, opp_melds_max=0,
         shanten=8):
    """**风险**（单步，单位≈分；**越大越糟，进 reward 时取负号**）。

    = 6.0·[这一步弃掉了已能胡的牌]        ← 机会成本，最大的那一项
    + 3.0·[牌墙 ≤ 28 且未听牌]            ← 延迟/流局风险（最后 10 墩不摸）
    + 1.0·(对手最大副露数 / 2)            ← 对手高价值胡牌风险（只能自摸，故按副露代理）

    权重来历：
      · 弃胡的机会成本：本仓实测弃胡局兑现率 q≈0.784（`mj_policy.py:1055`），
        而一次爆头胡约值 10~20 分 ⇒ 失败那部分 ≈ (1−q)×15 ≈ 3 分；
        但弃胡**还**保留了继续做大的期权 ⇒ 取 6.0 分作为"这一步风险"的上限标尺。
      · 牌墙 ≤28（≈还剩不到 2 巡）：`wall-20 >= 4` 是策略自己"还能再摸一张"的判据
        ⇒ 跨过这条线直接进入流局区，取 3.0 分。
      · 对手副露：本仓已证"副露是速度的加速器"（§ 的 peng ATE +0.951）
        ⇒ 用最大副露数做代理，满标 1.0 分（**刻意压低**：它是代理，不是因果量）。
    ⚠ "对手听牌概率 / 对手潜在高番"**不可观测** ⇒ 只用副露做代理，并**明说这是代理**；
      这正是 §2.7 说的"要靠学习"的部分，不该由我手写进去当因果量。
    """
    v = 0.0
    if declined_win:
        v += 6.0
    if wall is not None and wall <= 28 and not tenpai:
        v += 3.0
    v += 1.0 * min(1.0, float(opp_melds_max or 0) / 2.0)
    return v


def composite(hand=0.0, eff=0.0, mult=0.0, risk_v=0.0, lam=None):
    """按 §4 的组合式合成（**只含辅助项**；`R_terminal` 由调用方在局末加）。

    ⚠ 刻意把 `R_terminal` 排除在外：这样调用方必须**显式**决定
    "终局分成什么时候加"，避免把它误当成每步都加一次的项。
    """
    L = dict(LAMBDA)
    if lam:
        L.update(lam)
    return (L["hand"] * hand + L["efficiency"] * eff
            + L["multiplier"] * mult - L["risk"] * risk_v)
