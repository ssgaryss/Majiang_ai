#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""姿态前沿：**方差到底买不买得到？**（`MODEL_STAGE_FIELD.md` §5.4 缺的那一半）

为什么必须先做这个
------------------
`MODEL_STAGE_FIELD.md` §5 证明了**需求侧**：

    落后 ⇒ 方差是资产；领先 ⇒ 方差是负债。
    交换率：1 单位 σ(分/局) ≡ −z/√r 单位 μ(分/局)

但这是**需求**。它只在"σ 买得到"时才有用 —— 如果每一个能把 σ 抬高的改动
都同时把 μ 抬得更高，那么"永远最大化 μ"仍然是最优的，位置感知就是**空的**。
所以必须量**供给侧**：把姿态旋钮扫一遍，画出 (μ, σ) 前沿。

本项目的两个天然旋钮（都已在基因组里）
--------------------------------------
* `decline` —— 弃胡换爆头（放弃确定的小胡，博 ×2）
* `route`   —— 七对承诺（×4 路线，窄而深）

设计
----
**非对称桌**（`MODEL.md` 指定的唯一可信夹具形态）：
我们那一席换成候选姿态，其余三席**固定为线上基因组**。
逐 `seed` 配对（同一 `seed` ⇒ 同一副起手牌），座位与庄家用 seed 的两个不同位。

★ 并行方式（本沙箱专用）
------------------------
`multiprocessing.Pool` 在这里**直接被拒**（`connection.Pipe` 需要命名管道）。
所以并行只能**外部起 N 个独立 python 进程**，各自认领一段互不相交的 seed：

    python probe_posture_frontier.py --shard 0 --nshards 6 --seeds 1200
    ...
    python probe_posture_frontier.py --merge

每片写 `probe_posture_shard<i>.jsonl`（逐局原始记录），merge 时统一分析。

输出
----
每个姿态的 μ / σ / 偏度 / 自摸率 / 爆头率 / 均值番、相对基准的**配对 Δμ（含 CI）**
与 Δσ，最后按交换率判定在哪些 (d, r) 上该用哪个姿态。
"""
import argparse
import glob
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, ".")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "shanten:1:1:2:2:1:1:0"      # 线上基因组 discard:peng:chi:gang:decline:ukeire:route:feed
BASE_NAME = "base(decline2,route1)"
MELD_BASE_NAME = "base(peng1,chi1)"
# ★ 只动第 5 位(decline) 与第 7 位(route)，其余保持线上值。
#
# ⚠ 这里踩过一次坑（第 4 轮）：**8 个字段 = discard + 7 个数字**。
#   第一版每个姿态都多写了一位（如 `shanten:1:1:2:2:0:1:1:0`，9 个字段），
#   而 `parse_genome` 当时用 `zip` 按短的截断 ⇒ 后面基因整体错位一位、
#   且**不报错**。结果七个姿态里三对统计量逐位完全相同。
#   现在 `parse_genome` 已改为「多写就抛错」，本文件也加了 `_assert_genome`。
DECLINE_ROUTE_POSTURES = [
    (BASE_NAME, BASE),
    # 字段顺序: discard:peng:chi:gang:decline:ukeire:route:feed
    ("decline0,route1", "shanten:1:1:2:0:1:1:0"),
    ("decline1,route1", "shanten:1:1:2:1:1:1:0"),
    ("decline3,route1", "shanten:1:1:2:3:1:1:0"),
    ("decline2,route0", "shanten:1:1:2:2:1:0:0"),
    ("decline0,route0", "shanten:1:1:2:0:1:0:0"),
    ("decline3,route0", "shanten:1:1:2:3:1:0:0"),
]

# ★★ 波 2：**副露激进度**（`MODEL_STAGE_FIELD.md` §5.4.1 的修正）
#
# 为什么换旋钮：`decline`/`route` 虽然活着，但**分叉太稀有** ——
# 财飘只占胡牌 0.7%、七对 2.5%，做不出可用的 (μ,σ) 前沿。
# 而 `peng`/`chi` 同时决定**听牌速度**与**爆头（×2）路线**，是本项目里
# 唯一被反复证明"一动就变"的旋钮（peng=0 显著更差、peng=3 显著更差、倒 U 形）。
#
# 档位语义（读 mj_policy.GENES 的注释确认）：
#   0 = 完全不碰/不吃（最保守）
#   5 = 已有 ≥1 摊就不再副露
#   4 = 已有 ≥2 摊就不再副露
#   2 = 只在"副露后向听不变差"时才副露
#   8 = 已听牌就不再副露（时机门控）
#   3 = 见碰就碰，但手上有白板就不碰
#   1 / 7 = 见碰就碰（无条件下接受，最激进）
MELDING_POSTURES = [
    (MELD_BASE_NAME, BASE),
    ("peng0,chi0(从不副露)", "shanten:0:0:2:2:1:1:0"),
    ("peng5,chi5(≤1摊)", "shanten:5:5:2:2:1:1:0"),
    ("peng4,chi4(≤2摊)", "shanten:4:4:2:2:1:1:0"),
    ("peng2,chi2(向听门控)", "shanten:2:2:2:2:1:1:0"),
    ("peng8,chi8(听牌即停)", "shanten:8:8:2:2:1:1:0"),
    ("peng3,chi3(有白不副露)", "shanten:3:3:2:2:1:1:0"),
    ("peng7,chi7(最激进)", "shanten:7:7:2:2:1:1:0"),
]

POSTURE_SETS = {"decline_route": DECLINE_ROUTE_POSTURES,
                "melding": MELDING_POSTURES}

# ★★ 波 3：**「route=1 七对承诺」在 peng=1（见碰就碰）下是不是白开的？**
#
# 动机（第 14 轮线上真值）：我方线上七对 = **0/100**，而对手 1.7%，
# 且 `route=1` 是开着的。既有的 `probe_chiitoi2.txt` 显示 route=1 把
# "无副露七对听牌态"从 3.25% 抬到 11.00%，**但两边均番都是 1.20** ——
# 也就是说**七对听牌没有转化成七对胡牌**。
#
# 机制上的怀疑：`GENES` 的注释自己写着"副露……**永久关闭七对分支**"，
# 而线上基因组是 `peng=1, chi=1`（见碰就碰）⇒ 两个基因在**互相拆台**。
#
# 这个 2×2（route × peng）就是去分清：**七对胡牌率到底是被 route 决定的，
# 还是被 peng 决定的？**
CHIITOI_POSTURES = [
    (MELD_BASE_NAME, BASE),                              # route=1, peng=1（现线上）
    ("route1,peng0(不副露)", "shanten:0:0:2:2:1:1:0"),
    ("route0,peng1(关承诺)", "shanten:1:1:2:2:1:0:0"),
    ("route0,peng0", "shanten:0:0:2:2:1:0:0"),
    ("route1,peng5(≤1摊)", "shanten:5:5:2:2:1:1:0"),
    ("route1,peng2(向听门控)", "shanten:2:2:2:2:1:1:0"),
]
POSTURE_SETS["chiitoi"] = CHIITOI_POSTURES

# ★★ 波 4：**把基因组剩下的三个旋钮补齐**（第 25 轮）
#
# 前三波扫了 `decline` / `route` / `peng` / `chi`。按 `GENE_ORDER`
# （discard:peng:chi:gang:decline:ukeire:route:feed）清点，**还有三位从没被系统扫过**：
#
#   gang   ∈ {0,1,2}   0=不杠 / 1=只暗杠 / 2=暗杠+补杠（现线上）
#   ukeire ∈ {0,1}     0=不修正未见牌 / 1=按弃牌+副露修正（现线上）
#   feed   ∈ {0,1}     0=关 / 1=开「喂牌风险」修正（现线上是 0）
#
# 为什么这一波比前三波**更可能出结果**：前三波回答的是"能不能买到 σ"（答案：买不到），
# 而这一波问的是最朴素的问题 —— **现线上在这三个基因上是不是最优的**。
# 任一位若更好，就是一个**可以直接上线的改动**（不需要位置感知那套前提）。
#
# 设计：以现线上为基准做**单因子 + 少量交互**（OAT），共 7 个姿态。
REST_POSTURES = [
    ("base(gang2,ukeire1,feed0)", BASE),                 # 现线上
    ("gang0(不杠)", "shanten:1:1:0:2:1:1:0"),
    ("gang1(只暗杠)", "shanten:1:1:1:2:1:1:0"),
    ("ukeire0(不修正未见牌)", "shanten:1:1:2:2:0:1:0"),
    ("feed1(开喂牌修正)", "shanten:1:1:2:2:1:1:1"),
    ("ukeire0+feed1", "shanten:1:1:2:2:0:1:1"),
    ("gang0+ukeire0", "shanten:1:1:0:2:0:1:0"),
]
POSTURE_SETS["rest"] = REST_POSTURES

# ★★ 波 5（第 31 轮）：**决定性单点复核** —— `decline3,route1` 到底有没有 μ 代价？
#
# 动机：`analyze_sigma_ci.py` 的配对 bootstrap（2000 次重采样）显示，
# `decline3,route1` 是四波 26 个姿态里**唯一**一个：
#
#     Δμ = -0.047  CI [-0.148, +0.055]   ⇒ μ **未分辨**
#     Δσ = -0.175  CI [-0.278, -0.081]   ⇒ σ **被分辨**（CI 不含 0）
#
# ⇒ 也就是说：**"降方差"的旋钮一直存在，第 20 轮漏掉了它** ——
#   因为预注册的判据 ① 写的是"Δμ 跨 0 且 **Δσ > 0**"，只覆盖了
#   "追分者要更高方差"那**一个方向**，是**不对称**的。判据 ③
#   （"所有姿态同向 ⇒ 没有取舍前沿"）又把它一起扫进了"同向下降"里。
#
# ⚠ 但**决策表现在仍然六格全负**（−0.2 ~ −0.8 pp），
#   因为那张表是**按点估计 Δμ = −0.048 算的**。所以唯一待定的是：
#
#        Δμ 到底是真 −0.048，还是 0？
#
#   · 若真是 −0.048 ⇒ 决策表六格全负成立，`decline3` 不能用；
#   · 若 Δμ = 0     ⇒ 决策表要重算，`decline3,route1` 在**领先**时为正、
#                     在**落后**时为负（决策表里 d=−40 那格 −0.3、d=+40 那格 −0.8
#                     的差就是纯 σ 的贡献）⇒ **这将是本项目第一个真正的位置感知旋钮**。
#
# bootstrap 反推：分辨它只需 ~**5679 seed**（不是 `ukeire` 那种 48k）。
# 所以直接跑，并且**把波 1 的 1200 个 seed 也并进来**（PREFIXES_OF 里含
# `probe_posture_shard`，merge 按 (seed, g) 去重）⇒ 合计 17200 seed。
DECLINE3_FOCUS_POSTURES = [
    (BASE_NAME, BASE),                                   # decline2,route1（现线上）
    ("decline3,route1", "shanten:1:1:2:3:1:1:0"),        # decline2 + 威胁门控
]
POSTURE_SETS["decline3_focus"] = DECLINE3_FOCUS_POSTURES

# ★★ 波 6（第 32 轮）：**`peng` 与 `chi` 从来没被分开测过**
#
# 怎么发现的：`rl_meld_probe.py` 的**逐决策反事实**（5436 个配对样本，
# 做法是"正常跑一局 vs 强制放弃某一个副露窗口"，两局在那一刻之前逐手相同）：
#
#     response_peng   n=2169   放弃一次的代价 Δ = **-1.091** 分/局
#     response_chi    n=3267   放弃一次的代价 Δ = **-0.082** 分/局
#
# ⇒ **差一个量级（约 13 倍）**：**"碰"是有价值的，"吃"几乎免费。**
#
# 而**波 2 的每一个姿态都把 peng 与 chi 设成同一个值**
# （(peng0,chi0)、(peng5,chi5)、(peng8,chi8)… 只走了 9×9 网格的 8 个**对角格**）。
# ⇒ "只动 chi、不动 peng"这一整片区域**从未被测过**。
#
# 为什么这一波值得跑：波 2 里"限制副露"是**降 σ** 的
# （`peng8` Δσ=-0.399、`peng5` Δσ=-1.362），但 μ 代价太大（`peng5` Δμ=-1.407）。
# 若 chi 的边际价值真的 ≈0，那么"**只收 chi、保留 peng**"就可能是一个
# **低 μ 成本的降 σ 旋钮** —— 正是 §W.6 / §Z 一直在找的判据 ①
# （μ 未分辨 **且** σ 被分辨）。这不是 RL 的活，是**直接改基因组**就能验的假设。
PENG_CHI_POSTURES = [
    (MELD_BASE_NAME, BASE),                             # peng=1, chi=1（现线上）
    ("peng1,chi0(从不吃)", "shanten:1:0:2:2:1:1:0"),
    ("peng1,chi4(吃≤2摊)", "shanten:1:4:2:2:1:1:0"),
    ("peng1,chi5(吃≤1摊)", "shanten:1:5:2:2:1:1:0"),
    ("peng1,chi8(听牌不吃)", "shanten:1:8:2:2:1:1:0"),
    ("peng2,chi1(向听门控碰)", "shanten:2:1:2:2:1:1:0"),
]
POSTURE_SETS["peng_chi"] = PENG_CHI_POSTURES

# ★★ 波 7（第 36 轮）：**`chi8`（已听牌就不再吃）的高功效确认**
#
# 波 6 的结果里，`peng1,chi8` 是**六波以来第一个**同时满足这两条的姿态：
#   · 决策表**六格全正**（+1.2 ~ +2.1 个百分点，这是最大的正行之一）；
#   · 分叉率 **38.58%**（够活跃，不是"稀有到测不出"）。
# 而它的 Δμ = **+0.138**、Δσ = −0.010（**不付出方差代价**）。
#
# ⚠ 但它**还不显著**：CI [−0.476, +0.726]，t = 0.45，
#   bootstrap 反推要 **~23000 seed** 才能分辨 —— 1200 远远不够。
#
# **它为什么值得花这个算力**：所有已验证的旋钮都在 0.1~0.3 分/局，
# 而**没有一个方向是正的**。chi8 是第一个"看着为正且不买方差"的。
# 若 +0.138 是真的 ⇒ 晋级概率在中性位置 **+2.1 个百分点**（决策表）。
#
# ★ 机制上也讲得通，而且刚好是波 6 想找的**不对称**：
#   波 2 的 `peng8,chi8`（碰也停）是 **−0.859**（显著更差），
#   而这里 `peng1,chi8`（只停吃）是 **+0.138** ⇒
#   **"听牌即停"这个时机门控对「碰」是坏的、对「吃」是好的。**
#   波 2 一直是把 peng 与 chi 设成同一个值测的，所以这个不对称从没被看见。
#
# 判据（**先写死**）：
#   · Δμ 的 CI **不含** 0 且为正 ⇒ **可以上线**（现线上 `chi=1` 改成 `chi=8`）；
#   · CI 含 0 ⇒ **未分辨**，不许上线（"没测出来"不是"没差别"，也不是"有好处"）。
#
# 合并时**并进波 6 的 1200 seed**（两者 seed 不重叠：波 6 是 0~1199，
# 本波从 1200 起）⇒ 合计 ~25200 seed。
CHI8_FOCUS_POSTURES = [
    (MELD_BASE_NAME, BASE),                             # peng=1, chi=1（现线上）
    ("peng1,chi8(听牌不吃)", "shanten:1:8:2:2:1:1:0"),
]
POSTURE_SETS["chi8_focus"] = CHI8_FOCUS_POSTURES

# ★★ 波 8（第 37 轮）：**`chi5` 的 Δμ 到底是不是真的负？**
#
# 波 6 的 bootstrap 给出了**第二类**的判据 ① 满足者：
#   `peng1,chi5`  Δμ = −0.449（CI [−0.957,+0.069]，**跨 0**）
#                Δσ = −0.647（CI [−1.100,−0.205]，**不含 0**）
# ⇒ 它是目前**降 σ 幅度最大**的旋钮（σ 的 5.6%，是 `decline3` 的 3.7 倍）。
#
# 但它和 `decline3` 面对**同一个待决问题**：Δμ 是真的负、还是 0？
# 而波 5 刚刚用 **12776 seed** 把 `decline3` 判掉了：
#   `decline3` 的 Δμ = **−0.155 CI[−0.191,−0.121]**（**显著为负**）
#   ⇒ 它的"降 σ"是**有代价的**，决策表六格全负成立、**不能用**。
#
# ⇒ 于是 `chi5` 就成了**最后一个还没定价的 σ 旋钮**，而且它**最便宜**：
#   bootstrap 反推只需 **1635 seed**（已有波 6 的 1200，再补几千即可）。
#
# 判据（**先写死**，与波 5 一致）：
#   · Δμ 的 CI **不含 0** ⇒ 有真实 μ 代价 ⇒ 决策表成立、**不能用**；
#   · Δμ 的 CI **含 0**   ⇒ **第一个可用的位置感知旋钮**（领先时降 5.6% 方差）。
#
# ⚠ `chi4` 与 `chi5` 在夹具里**行为完全相同**（波 6 两者逐项相同：
#   Δμ −0.449、Δσ −0.647、分叉 313/1200）⇒ 只测 `chi5` 一个即可。
CHI5_FOCUS_POSTURES = [
    (MELD_BASE_NAME, BASE),                             # peng=1, chi=1（现线上）
    ("peng1,chi5(吃≤1摊)", "shanten:1:5:2:2:1:1:0"),
]
POSTURE_SETS["chi5_focus"] = CHI5_FOCUS_POSTURES

# ★★ 波 9（第 42 轮，**查代码查出来的空白**）：`chi=2` / `chi=6` 从没被单独扫过
#
# 缘起：用户的"很少做大牌"引我去读 `mj_policy` 的档位分支，结果发现两件事：
#
# 1. **`peng=6` 与 `peng=2` 是同一段代码**（`on_window` 碰分支最后那段：
#    "peng=2 / peng=6：只在「碰后向听不变差」时才碰"）⇒ **行为逐字节相同**。
#    所以历史战绩里那个 `shanten:6:6:2:2:1:1:0`（20 场 +6.75/场）**不是新地面**
#    —— 它等价于波 2 已经测过的 `peng2,chi2`（−0.364，不显著）。纯噪声。
#
# 2. **但 `chi=6` 是它自己的分支**（吃得让向听**严格变小**才吃，
#    `if shanten(after) >= before: PASS`），比 `chi=2` 严、也与 `peng=6` 不同。
#    而**波 2 与波 6 都没有扫过 `chi=6`**（波 2 用 {0,1,2,3,4,5,7,8}、波 6 用 {0,1,4,5,8}）。
#    同理 **`chi=2` 从没和线上的 `peng=1` 组合测过**（只在 `peng=2` 下测过）。
#
# ⇒ 这两个格子是**真实的空白**，而且它们正好落在"限制吃"这条轴上 ——
#   而该轴上唯一看着为正的就是 `chi8`（+0.138）。值得补测。
#
# 判据（与波 6 一致）：`Δμ` 的 CI **不含 0 且为正** ⇒ 可上线；含 0 ⇒ 未分辨。
CHI_GAP_POSTURES = [
    (MELD_BASE_NAME, BASE),                             # peng=1, chi=1（现线上）
    ("peng1,chi2(留白不吃·未测)", "shanten:1:2:2:2:1:1:0"),
    ("peng1,chi6(吃须降向听·未测)", "shanten:1:6:2:2:1:1:0"),
]
POSTURE_SETS["chi_gap"] = CHI_GAP_POSTURES

# ★★ 波 10（第 43 轮）：**解除「庄家不许弃胡」这条禁令** —— 把假设变成数据
#
# 起因：读 `_dealer_blocks_decline` 时发现它的依据**从没被直接检验过** ——
# 那条禁令让庄家**永远没有弃胡机会**，所以线上 97 次弃胡机会**全是闲家**
# （`audit_decline_clean.py`），**"庄家弃胡的兑现率"这个数根本不存在**。
#
# 而模型说这笔账划得来（含庄位期权的盈亏平衡）：
#     闲家 q* = (10f+Δ)/(20f+Δ) ≈ 0.54~0.58   实测闲家 q = **0.784**
#     庄家 q* = (24f+Δ)/(48f+Δ) ≈ 0.52~0.54   ← **比闲家那条还低**
# ⇒ 若庄家的 q 也在 0.7 以上，这条禁令就在白扔钱。
#
# `mj_policy` 已加 `decline=4` = "3 + 解除庄位禁令"（1/2/3 行为**完全不变**）。
# 判据（先写死）：`Δμ` 的 CI **不含 0 且为正** ⇒ 可上线；含 0 ⇒ 未分辨。
#
# ⚠ 反向证据也要记住（所以这是**实验**不是"显然该做"）：
# `diag_dealer_chain.py` 测出**我们的连庄链比对手短**（1.275 vs ~1.46，
# 差 −0.187，CI [−0.30,−0.06]）⇒ 我们本来就在庄位上赢得少，
# 再主动让出庄位可能雪上加霜。**这正是要用配对实验回答的问题。**
DECLINE4_POSTURES = [
    (BASE_NAME, BASE),                                  # decline=2（现线上）
    ("decline4,route1(解除庄位禁令)", "shanten:1:1:2:4:1:1:0"),
]
POSTURE_SETS["decline4"] = DECLINE4_POSTURES

# ★★★★★ 第 118 轮：**座位条件化的庄位弃胡**（`decline=5`）
#
# 由来：第 118 轮先把线上 μ 按**结算类别**拆开（`decompose_live_mu.py`），
# 发现我们的劣势**完全集中在做庄**：
#     番 | 做庄胡：我们 1.1864 vs 场上 1.3122（差 −0.1258，房间级 CI 不含 0）
#     番 | 做闲胡：我们 1.3708 vs 场上 1.3207（我们**更好**）
#     而场上自己**没有**这个庄/闲落差（1.3122 vs 1.3207）。
# ⇒ 不是"我们整体牌小"，是"**我们做庄时牌小**"。
#
# 再把 `decline4` 的 Δμ **按座位拆开**（`analyze_seat_conditional.py`）：
#     做庄 +0.663 [+0.44,+0.89] / 做闲 −0.198 [−0.26,−0.14] / 全部 +0.017
#   三个**独立 seed 段**复现（+0.663 / +0.803 / +0.536）。
#   ⇒ 全基因组的 +0.017 是**两个大数相消**的结果（0.25×0.663 − 0.75×0.198）。
#   负对照：`chi8` 在同一套拆分下**没有**翻转（−0.253 / −0.066，两个都含 0）。
#
# ⚠ 但 `decline4` **读不出"只有我们自己改"的效果**：
#   4 = 3 + 解除禁令 = **带威胁门控**，相对线上的 2 改了**两件事**，
#   而自对局里陪打的三个座位用的是**同一个基因组**。
#   `_dealer_blocks_decline(v)` 是**纯庄位**判定（`v.dealer == v.seat`）
#   ⇒ 新增 **`decline=5` = 2 + 解除庄位禁令（不带威胁门控）**：
#   自对局里 **5 相对 2 只改庄家** ⇒ 「做庄那一列」就是**干净的**
#   "我们做庄时可以弃胡"的收益（对手全在闲家位，行为逐字不变）。
#   ⇒ **可动用的收益 = Δμ(做庄) × P(我们做庄) ≈ Δμ(做庄) × 0.25。**
SEAT_DECLINE_POSTURES = [
    (BASE_NAME, BASE),                                  # decline=2（现线上）
    ("decline5(庄位可弃胡,无威胁门控)", "shanten:1:1:2:5:1:1:0"),
]
POSTURE_SETS["seat_decline"] = SEAT_DECLINE_POSTURES

# ★★★ 第 80 轮：**小效应搜索**（依据 §Z.73/§Z.74 与 `PREREG_SMALL_EFFECT.md`）
#
# 为什么要再扫一遍：§Z.50 的动作空间体检是"策略侧已空"的**承重墙**，
# 它在**大效应**上站得住（μ 的 MDE 0.83 < 晋级缺口 1.36），但第 79 轮
# **逐候选**算 σ_配对之后发现：对 `ukeire=0` / `gang=0,ukeire=0`
# 这两个候选，体检当时的 MDE 有 **0.67~0.73** —— 而**有用的那一档只有 ~+0.2**。
#
# 为什么 +0.2 就值得跑（§Z.64(d)，σ_局=12.621、N=4218）：
#     μ = −0.1097 ⇒ P(当天为正) = **28.6%**
#     μ = +0.14   ⇒ P(当天为正) = **76.4%**
#     μ = +0.25   ⇒ **90.1%**
# ⇒ **我们不需要 +1.36，只需要把 μ 的符号翻过来。**
#
# 候选 = 体检里 **Δμ>0 且 n≥1200 的全部 5 个**（不扩大：网格很大，
# 先只把体检**点过名**的这几个收敛掉）。
#
# 预注册判据（见 `PREREG_SMALL_EFFECT.md`，**事后不改**）：
#   Bonferroni zc≈2.6（5 个比较）
#   · CI 下界 > 0 且 Δμ ≥ **+0.15** ⇒ 进**线上配对 A/B**；
#   · CI 含 0 ⇒ **不许上线**，并**报出该候选的实际 MDE**（别把"没测出来"说成"无效应"）；
#   · CI 上界 < 0 ⇒ 关档；
#   · 5 个全"未分辨/否掉" ⇒ **这一次可以有功效地**把"策略侧已空"重新升级为已证结论
#     （而不是 §Z.50 那个被 §Z.73 降级过的版本）。
#
# 代价（`probe_small_effect_design.py`，σ_配对 中位 3.239）：
#   MDE=0.10 ⇒ 每候选 7,091 seed、合计 35,457 ⇒ 约 **1.5 小时 × 4 进程**。
SMALL_EFFECT_POSTURES = [
    (BASE_NAME, BASE),                                  # 线上（基准）
    ("ukeire0", "shanten:1:1:2:2:0:1:0"),
    ("gang0,ukeire0", "shanten:1:1:0:2:0:1:0"),
    ("decline4", "shanten:1:1:2:4:1:1:0"),
    ("peng2", "shanten:2:1:2:2:1:1:0"),
    ("gang1", "shanten:1:1:1:2:1:1:0"),   # ⚠ 第 81 轮更正标签：这一位是 **gang 2→1**，不是 chi
]
POSTURE_SETS["small_effect"] = SMALL_EFFECT_POSTURES


def _dedup_by_genome(*sets):
    """把多组姿态按**基因组**去重（名字保留第一次出现的那个）。

    两组的基准基因组是同一个，所以合并后是 7 + 8 − 1 = 14 个点，
    共用同一个基准 ⇒ 可以直接画在**同一张 (μ,σ) 图**上。
    """
    out, seen = [], set()
    for ps in sets:
        for name, g in ps:
            if g in seen:
                continue
            seen.add(g)
            out.append((name, g))
    return out


POSTURE_SETS["combined"] = _dedup_by_genome(DECLINE_ROUTE_POSTURES,
                                            MELDING_POSTURES)

# 基准姿态的**名字**随姿态组变化 —— 第一版只写了 decline 组的名字，
# 扫 melding 组时 `S[BASE_NAME]` 直接 KeyError（两组的基准名不同，但基准**基因组**相同）。
BASE_NAME_OF = {"decline_route": BASE_NAME, "melding": MELD_BASE_NAME,
                "combined": BASE_NAME, "chiitoi": MELD_BASE_NAME,
                "rest": "base(gang2,ukeire1,feed0)",
                "decline3_focus": BASE_NAME,
                "peng_chi": MELD_BASE_NAME,
                "chi8_focus": MELD_BASE_NAME,
                "chi5_focus": MELD_BASE_NAME,
                "chi_gap": MELD_BASE_NAME,
                "decline4": BASE_NAME,
                "seat_decline": BASE_NAME,
                "small_effect": BASE_NAME}
# 每个姿态组对应哪些分片前缀（merge 时按这个取文件）。
# ⚠ 波 5 故意把 `probe_posture_shard`（波 1）也放进来：两者都用
#   `decline2,route1` 作基准、都含 `decline3,route1`，且 seed 不重叠
#   （波 1 是 0~1199，波 5 从 seed-base 1200 起）⇒ merge 直接**合并样本量**。
PREFIXES_OF = {"decline_route": ["probe_posture_shard"],
               "melding": ["probe_meld_shard"],
               "combined": ["probe_posture_shard", "probe_meld_shard"],
               "chiitoi": ["probe_chiitoi_shard"],
               "rest": ["probe_rest_shard"],
               "decline3_focus": ["probe_d3_shard", "probe_d3b_shard",
                                  "probe_posture_shard"],
               "peng_chi": ["probe_pc_shard", "probe_pc2_shard"],
               "chi8_focus": ["probe_chi8_shard", "probe_chi8b_shard",
                               "probe_pc2_shard"],
               "chi5_focus": ["probe_chi5_shard", "probe_pc2_shard"],
               "chi_gap": ["probe_gap_shard"],
               "decline4": ["probe_d4_shard", "probe_d4b_shard"],
               # 第 118 轮：座位条件化的庄位弃胡（`decline=5`）。
               # **只用自己的前缀**：这是一次**新的**实验，
               # 不能与 `probe_d4_*`（那个是 decline=4，含威胁门控）合并样本 ——
               # 两者的基因组不同，混在一起会把"门控"和"禁令"两件事搅在一块。
               "seat_decline": ["probe_sd_shard"],
               # 第 80 轮：小效应搜索。**只用自己的前缀**（条件统一）；
               # 第 115 轮加 `probe_se_shard_b`：wave 1 冻结判决后追加的**尾巴**
               #   （`--seed-base 7091 --seeds 2684`，seed 区间 7091~9774，
               #    **与 wave 1 的 0~7090 不相交**，见 §Z.96(d)）。
               #   目的：把 `ukeire0` / `gang0,ukeire0` 的 MDE 从 0.27/0.29
               #   压到 ≈0.25 以下，让"5 条全部定案"（§Z.96(b)、§Z.101(d)）。
               "small_effect": ["probe_se_shard", "probe_se_shard_b"]}
POSTURES = DECLINE_ROUTE_POSTURES        # 默认；main() 按 --postures 覆盖
CUR_BASE_NAME = BASE_NAME                # 默认；main() 按 --postures 覆盖


def _assert_genome(s):
    """构造期就把字段数钉死 —— 不要等到"统计量一模一样"才发现。"""
    n = len(str(s).split(":"))
    if n != 8:
        raise ValueError("姿态基因组 %r 有 %d 个字段，必须是 8 个"
                         "（discard:peng:chi:gang:decline:ukeire:route:feed）" % (s, n))
    return s


def _check_postures(ps):
    for _n, _g in ps:
        _assert_genome(_g)
    assert len({g for _, g in ps}) == len(ps), "姿态基因组有重复：%s" % ps
    assert len({n for n, _ in ps}) == len(ps), "姿态名有重复：%s" % ps
    return ps


_check_postures(POSTURES)
DECIDE_GRID = ((0, 20), (10, 10), (20, 10), (40, 20), (-20, 10), (-40, 20))

_CACHE = {}


def _strat(gname):
    """按名字缓存策略对象。

    ⚠ "可复用"是**推断**（`mj_sim`/`mj_policy` 全仓没有 `reset`），
    所以带 `--check-cache` 自检：缓存 vs 每局新建，逐 seed 逐姿态比对。
    实测 56/56 项一致。注意瓶颈不在策略构造，在 `Game.play`。
    """
    s = _CACHE.get(gname)
    if s is None:
        from mj_arena import make_strategy
        s = make_strategy(gname)
        _CACHE[gname] = s
    return s


def _play(seed, gname, fresh=False):
    from mj_sim import Game
    if fresh:
        from mj_arena import make_strategy
        me = make_strategy(gname)
        opp = make_strategy(BASE)
    else:
        me = _strat(gname)
        opp = _strat(BASE)
    my = seed % 4
    dealer = (seed // 4) % 4
    strats = [opp] * 4
    strats[my] = me
    r = Game(seed=seed, dealer=dealer).play(strats)
    w = r.get("winner")
    return {"seed": seed, "g": gname, "seat": my, "dealer": dealer,
            "score": r["scores"][my],
            "win": 1 if w == my else 0,
            "baotou": 1 if (w == my and r.get("baotou")) else 0,
            "fan": r.get("fan") if w == my else 0,
            "draw": 1 if r.get("draw") else 0,
            # ★ 七对分支倍数（0 = 平胡）。`mj_sim.Game.result()` 直接给，
            #   加进来是为了查清"线上七对 0/100"到底是不是被副露关掉的。
            "chiitoi": int(r.get("chiitoi") or 0) if w == my else 0,
            "melds_n": int(r.get("melds") or 0)}


def check_cache(seeds=40):
    print("=== 缓存自检：缓存策略 vs 每局新建（%d seed × %d 姿态）==="
          % (seeds, len(POSTURES)))
    bad = n = 0
    for s in range(seeds):
        cached = {r["g"]: r["score"] for r in (_play(s, g) for _, g in POSTURES)}
        for _, g in POSTURES:
            fresh = _play(s, g, fresh=True)["score"]
            n += 1
            if cached.get(g) != fresh:
                bad += 1
                if bad <= 5:
                    print("  ✗ seed=%d %s: 缓存 %s vs 新建 %s"
                          % (s, g, cached.get(g), fresh))
    print("  比对 %d 项，不一致 %d 项" % (n, bad))
    print("  %s" % ("✅ 缓存安全" if bad == 0 else "★ 缓存**不安全** ⇒ 去掉缓存重跑"))
    return 0 if bad == 0 else 1


def shard_run(a):
    mine = [a.seed_base + s for s in range(a.seeds) if s % a.nshards == a.shard]
    out = a.out or ("%s%d.jsonl" % (a.out_prefix, a.shard))
    print("=== 分片 %d/%d：%d 个 seed × %d 姿态 = %d 局 → %s ==="
          % (a.shard, a.nshards, len(mine), len(POSTURES),
             len(mine) * len(POSTURES), out))
    step = max(1, len(mine) // 10)
    with open(out, "w", encoding="utf-8") as f:
        for i, s in enumerate(mine):
            for _, g in POSTURES:
                f.write(json.dumps(_play(s, g), ensure_ascii=False) + "\n")
            # ★ 每个 seed 落一次盘。第一版只靠 Python 的块缓冲（8KB），
            #   实测跑了 3 分钟、写了 ~35KB，文件仍然是 **0 字节** ——
            #   也就是说**进程一被杀就全丢**。跑一小时的实验不该有这个风险。
            f.flush()
            if i % step == 0:
                print("  ... %d/%d" % (i, len(mine)), flush=True)
    print("  分片 %d 完成 → %s" % (a.shard, out))
    return 0


def _erf(x):
    """Abramowitz-Stegun 7.1.26 近似，精度足够。"""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * pow(2.718281828459045, -x * x)
    return sign * y


def _phi(z):
    return 0.5 * (1.0 + _erf(z / (2 ** 0.5)))


def boot_ci(xs, iters=4000, seed=13):
    rng = random.Random(seed)
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return ms[int(0.025 * iters)], ms[int(0.975 * iters)]


def analyze(rows):
    by = {}
    for r in rows:
        by.setdefault(r["g"], []).append(r)

    def stats(rs):
        sc = [x["score"] for x in rs]
        n = len(sc)
        mu = st.mean(sc)
        sd = st.pstdev(sc)
        sk = (sum((v - mu) ** 3 for v in sc) / n / (sd ** 3)) if sd else 0.0
        fans = [x["fan"] for x in rs if x["win"]]
        # ⚠ `chiitoi` / `melds_n` 是第 15 轮才加进记录的。**波 1/2 的旧记录里没有这两个键** ——
        #   若直接用 .get(...,0) 平均，会打印出一个**看起来像真的 0.00%**，
        #   而真相是"这一列根本没测"。所以：整组都没有这个键时返回 None，
        #   打印成 `—`（"未测"），绝不打印 0。
        has_chi = any("chiitoi" in x for x in rs)
        has_meld = any("melds_n" in x for x in rs)
        return {"n": n, "mu": mu, "sd": sd, "skew": sk,
                "win": sum(x["win"] for x in rs) / n,
                "baotou": sum(x["baotou"] for x in rs) / n,
                "chiitoi": (sum(x.get("chiitoi", 0) > 0 for x in rs) / n)
                           if has_chi else None,
                "melds": (st.mean([x.get("melds_n", 0) for x in rs]) if rs else 0.0)
                         if has_meld else None,
                "fan": st.mean(fans) if fans else 0.0}

    def _pc(v, fmt="%7.2f%%"):
        return "      —" if v is None else (fmt % (100 * v))

    print("%-26s %7s %10s %10s %8s %8s %8s %8s %8s"
          % ("姿态", "局数", "μ(分/局)", "σ(分/局)", "偏度", "自摸率", "爆头率",
             "七对率", "副露/局"))
    S = {}
    for name, g in POSTURES:
        if g not in by:
            continue
        s = stats(by[g])
        S[name] = s
        print("%-26s %7d %+10.3f %10.3f %+8.2f %7.1f%% %7.1f%% %s %s"
              % (name, s["n"], s["mu"], s["sd"], s["skew"],
                 100 * s["win"], 100 * s["baotou"], _pc(s["chiitoi"]),
                 "      —" if s["melds"] is None else "%8.2f" % s["melds"]))
    if any(s["chiitoi"] is None for s in S.values()):
        print("  （`七对率` = `—` 表示**这批记录里没有这个字段**（旧批次），"
              "不是 0 —— 第 15 轮才加进夹具记录的）")
    if len({s["n"] for s in S.values()}) > 1:
        print("  ⚠ 各姿态的**局数不同**（分片先后完成）⇒ 上面的 μ/σ **不可直接横向比较**；"
              "结论一律看下面的**配对**段（它在共同的 seed 子集上算）。")

    print("\n--- 相对基准的**配对**差（同 seed ⇒ 同起手牌）---")
    base = {x["seed"]: x for x in by.get(BASE, [])}
    print("%-22s %11s %22s %11s %s"
          % ("姿态", "Δμ(分/局)", "Δμ 95%CI", "Δσ(分/局)", "σ 方向"))
    frontier = []
    for name, g in POSTURES:
        if g == BASE or g not in by:
            continue
        cur = {x["seed"]: x for x in by[g]}
        common = sorted(set(base) & set(cur))
        if not common:
            continue
        dmu = [cur[s]["score"] - base[s]["score"] for s in common]
        lo, hi = boot_ci(dmu)
        # ★★ Δσ 必须在**同一批 seed** 上算！
        #
        # 第一版写成 `S[name]["sd"] - S[BASE]["sd"]` —— 也就是拿"该姿态全部 seed 的 σ"
        # 减"基准全部 seed 的 σ"。当两者的 seed 集合不同（合并两波时必然如此：
        # 基准在两波里都有、其它姿态只有一波）时，**这把"起手牌不同"记成了"波动不同"**。
        # 实测后果：`peng7,chi7` 与基准**逐局完全相同**（分叉率 0.00%），
        # 却报出 Δσ = **+0.194**。一个逐局等价的姿态不可能有 σ 差异 ——
        # 那 0.194 全是种子集合差。
        sd_cur = st.pstdev([cur[s]["score"] for s in common])
        sd_base = st.pstdev([base[s]["score"] for s in common])
        dsig = sd_cur - sd_base
        frontier.append((name, st.mean(dmu), lo, hi, dsig))
        print("%-22s %+11.3f  [%+.3f,%+.3f] %+11.3f   %s"
              % (name, st.mean(dmu), lo, hi, dsig,
                 "σ 更高" if dsig > 0.05 else ("σ 更低" if dsig < -0.05 else "σ 基本不变")))

    # ---- ★ 档位活性：每个姿态与基准的**逐 seed 分叉率** ----
    #
    # 为什么必须随报告一起打：本项目在这一点上栽过两次 ——
    #   ① `decline=0` 与 `decline=1` 在 n=1195 上统计量逐位相同（实为**从不分叉**）；
    #   ② `peng7` 与 `peng1` 同效（代码注释写明），姿态集里多了一个**重复点**。
    # 两者都**不会**被"统计量"看出来，只有逐 seed 比对才发现。
    # 分叉率 0% 的姿态**不是"效果相同"，而是"这个旋钮没接上"**，
    # 不该被当成前沿上一个独立的点来解读。
    print("\n--- ★ 档位活性（与基准逐 seed 比对得分）---")
    print("  %-24s %8s %10s %9s  %s" % ("姿态", "可比seed", "得分不同", "分叉率", "判读"))
    for name, g in POSTURES:
        if g == BASE or g not in by:
            continue
        cur = {x["seed"]: x for x in by[g]}      # by[g] 是**列表**，要按 seed 建索引
        common = sorted(set(base) & set(cur))
        if not common:
            continue
        diff = sum(1 for s in common if cur[s]["score"] != base[s]["score"])
        rate = 100.0 * diff / len(common)
        tagv = ("★ 从不分叉 ⇒ **与基准行为相同**，不可当前沿上的独立点"
                if diff == 0 else
                ("分叉很稀有（<2%）⇒ 该轴分辨率有限" if rate < 2 else "正常分叉"))
        print("  %-24s %8d %10d %8.2f%%  %s" % (name[:24], len(common), diff, rate, tagv))

    print("\n--- ★ 供给侧判定：这些旋钮能不能「用 μ 买 σ」？---")
    print("  判据（第 2 轮先写死的，防事后解释）：")
    print("   ① Δμ 的 CI 跨 0 且 Δσ 明显 > 0 ⇒ **σ 免费** ⇒ 位置感知该做；")
    print("   ② Δμ 显著为负、Δσ 才勉强为正 ⇒ σ 很贵 ⇒ 只在足够落后时用；")
    print("   ③ 所有姿态 Δμ 与 Δσ 同向 ⇒ **没有取舍前沿 ⇒ 结论作废**。")
    print()
    free = [f for f in frontier if f[2] <= 0 <= f[3] and f[4] > 0.05]
    pricey = [f for f in frontier if f[3] < 0 and f[4] > 0.05]
    same_dir_up = [f for f in frontier if f[1] > 0 and f[4] > 0]
    same_dir_dn = [f for f in frontier if f[1] < 0 and f[4] < 0]
    if free:
        print("  ✅ σ 免费（μ 未分辨且 σ 更高）：")
        for f in free:
            print("     %-22s Δμ %+.3f [%+.3f,%+.3f]   Δσ %+.3f" % f)
    else:
        print("  ⚠ 没有找到「μ 未分辨且 σ 更高」的姿态。")
    if pricey:
        print("  ⚠ σ 有价（μ 显著下降才换来 σ 上升）：")
        for f in pricey:
            print("     %-22s Δμ %+.3f [%+.3f,%+.3f]   Δσ %+.3f" % f)
    if same_dir_up:
        print("  ⚠ Δμ 与 Δσ **同向上升**（说明该旋钮不是取舍，是纯粹的改进或纯粹的加剧）：")
        for f in same_dir_up:
            print("     %-22s Δμ %+.3f   Δσ %+.3f" % (f[0], f[1], f[4]))
    if same_dir_dn:
        print("  ⚠ Δμ 与 Δσ **同向下降**：")
        for f in same_dir_dn:
            print("     %-22s Δμ %+.3f   Δσ %+.3f" % (f[0], f[1], f[4]))

    print("\n--- ★ 结论：哪些姿态在哪些 (d, r) 上该用 ---")
    mu0, sd0 = S[CUR_BASE_NAME]["mu"], S[CUR_BASE_NAME]["sd"]
    print("  基准: μ=%+.3f σ=%.3f（n=%d）" % (mu0, sd0, S[CUR_BASE_NAME]["n"]))
    print()
    print("  %-22s %s" % ("姿态", "".join("  d=%+d,r=%-3d" % (d, r) for d, r in DECIDE_GRID)))
    for name, dmu, lo, hi, dsig in frontier:
        cells = []
        for d, r in DECIDE_GRID:
            m1, s1 = mu0 + dmu, max(0.1, sd0 + dsig)
            z1 = (r * m1 - d) / (s1 * (r ** 0.5))
            z0 = (r * mu0 - d) / (sd0 * (r ** 0.5))
            cells.append("%+9.1f" % (100 * (_phi(z1) - _phi(z0))))
        print("  %-22s %s" % (name, "".join(cells)))
    print()
    print("  读法：数字 = 该姿态在该位置相对基准的晋级概率变化（百分点）。")
    print("  **同一行出现正负翻转 ⇒ 位置感知成立**（同一改动在不同名次上取舍不同）。")
    print("  若某行全为正 ⇒ 该姿态**无条件更好**，不需要位置感知。")
    print("  若某行全为负 ⇒ 无条件更差。")
    return 0


def merge_shards(prefixes=("probe_posture_shard",)):
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    files = []
    for pre in prefixes:
        files.extend(sorted(glob.glob(pre + "*.jsonl")))
    files = sorted(set(files))
    if not files:
        print("找不到分片文件：%s" % (list(prefixes),))
        return 1
    rows = []
    seen = set()
    dup = 0
    for fn in files:
        n0 = len(rows)
        for line in open(fn, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            key = (r.get("seed"), r.get("g"))
            if key in seen:                 # 多波追加时防重叠
                dup += 1
                continue
            seen.add(key)
            rows.append(r)
        print("  %s: %d 条" % (fn, len(rows) - n0))
    if dup:
        print("  （去重丢弃 %d 条重复的 seed×姿态）" % dup)
    print("  合计 %d 条逐局记录（%d 个 seed × %d 姿态）\n"
          % (len(rows), len({r["seed"] for r in rows}), len(POSTURES)))
    return analyze(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1200)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="本波 seed 的起始偏移。多波追加时用不同的 base，"
                         "避免 seed 重叠（merge 也会按 (seed, 姿态) 去重兜底）。")
    ap.add_argument("--nproc", type=int, default=1,
                    help="沙箱内无效（Pool 被禁）；保留只为沙箱外兼容")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--out-prefix", default="probe_posture_shard",
                    help="分片输出前缀。不同波次用不同前缀，merge 时也按前缀取。")
    ap.add_argument("--postures", default="decline_route",
                    choices=sorted(POSTURE_SETS.keys()),
                    help="扫哪一组姿态。decline_route=第 2 轮那组"
                         "（分叉稀有、天花板低）；"
                         "melding=副露激进度 peng/chi（§5.4.1 推荐的真正方差旋钮）。")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--merge-all", action="store_true",
                    help="合并**所有波次**（decline_route + melding），"
                         "按基因组去重成 14 个姿态，共用同一个基准 ⇒ "
                         "一张 (μ,σ) 前沿图。这是最终交付的那条命令。")
    ap.add_argument("--check-cache", action="store_true")
    ap.add_argument("--single", action="store_true",
                    help="单进程跑完全部 seed 并直接出报告（小规模用）")
    a = ap.parse_args()

    # 选定姿态组（全局，analyze/_play 都用它）。
    # ⚠ 先定组、再打横幅 —— 第一版在 --merge-all 分支里才改组，
    #   于是横幅打的是 decline_route，而实际分析用的是 14 个合并姿态
    #   （"日志说的和你做的不是一回事"这类最容易被信任的错）。
    global POSTURES, CUR_BASE_NAME
    grp = "combined" if a.merge_all else a.postures
    POSTURES = _check_postures(POSTURE_SETS[grp])
    CUR_BASE_NAME = BASE_NAME_OF[grp]
    print("姿态组 = %s（%d 个，基准=%s）" % (grp, len(POSTURES), CUR_BASE_NAME))

    if a.check_cache:
        return check_cache()
    if a.merge_all:
        return merge_shards(PREFIXES_OF["combined"])
    if a.merge:
        return merge_shards(PREFIXES_OF[a.postures])
    if a.single:
        rows = [_play(s, g) for s in range(a.seeds) for _, g in POSTURES]
        return analyze(rows)
    return shard_run(a)


if __name__ == "__main__":
    sys.exit(main())
