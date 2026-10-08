#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略池 + 基因组 —— 竞技场做淘汰/变异搜索的材料。

出牌准则从「粗糙连接度启发式」升级到「向听数 + 有效牌」：
  · 候选 = 手牌里每张可打的牌
  · 主键 = 打完之后的手牌向听数（越小越好）
  · 次键 = 该向听下的有效牌枚数（越多越好）；只对并列最优候选计算，省时间
  · shanten_ref 已用「穷举全部胡牌型」+ mj_core.can_win 双向交叉验证

基因组 = 可演化的离散参数组合：
    discard   出牌准则   first | heuristic | shanten
    peng      是否碰      0/1
    chi       是否吃      0/1
    gang      是否杠      0/1
    decline   弃胡博财飘  0/1
编码串形如 "shanten:1:1:0:0"。
"""
import math
import random
import sys
from collections import OrderedDict

import shanten_ref as S
from mj_bot import Strategy, View
from mj_core import (NTILE, WILD, best_discard, is_baotou, melds_of,
                     tile_code, tile_index, to_counts, winning_tiles)
from mj_core import chi_tiles as _core_chi_tiles

PASS = {"action": "pass", "tile": ""}

DISCARD_MODES = ("first", "heuristic", "shanten", "shanten2", "shanten_fan",
                 "value", "baotou", "net", "baotou2", "rank", "rank2")

GENES = OrderedDict([
    ("discard", DISCARD_MODES),
    # peng：0=不碰  1=手上有对子就碰（原语义）  2=只在碰后向听不变差时才碰
    #       3=**手上有白板时不碰** —— 复盘数据直接推出来的规则，见下
    # 线上实测（房间 a_5a17de403f70，10 场 80 局，按 user_id 拆开）：
    #   副露  87 次 -> 爆头率 30.8%  场均 +19.10
    #   副露  92 次 -> 爆头率 33.3%  场均 +11.60
    #   副露 150 次 -> 爆头率 10.0%  场均  -7.20   ← 我的 bot
    #   副露  11 次 -> 爆头率  0.0%  场均 -23.50
    # 呈清晰倒 U：几乎不副露和过度副露都亏。机制是规则本身 ——
    # **爆头 = 4 个自然面子 + 一张余白**，白板正是那条路线的钥匙，
    # 手里有白板时碰吃等于亲手把 ×2 变成 ×1。
    # 注意：这与 10000 局循环赛「peng=0 显著更差」并不矛盾 —— 那是另一个极端。
    #   4=副露总数达到 2 后不再碰   5=达到 1 后不再碰
    # 4/5 是「速率旋钮」：真实对局里副露量与胜场呈倒 U ——
    #   房间 a_5a17de403f70（10 场 80 局，按 user_id 拆开）
    #     副露  11 次 -> 10 胜  -235 分   （几乎不副露）
    #     副露  87 次 -> 26 胜  +191 分   <- 最优区
    #     副露  92 次 -> 24 胜  +116 分
    #     副露 150 次 -> 20 胜   -72 分   <- 我的 bot（peng=1 有对子就碰）
    # 换算成每次：最优 ~1.1 次/局，我打到 1.72 次/局。
    # 而 10000 局循环赛里「peng=0 显著更差」测的是**另一个极端**（0 次/局），
    # 中间这段从没被测过 —— 4/5 就是去测它。
    # 副露速率/门控。**5/6 是 2026-09-17 新增的"向听门控"**，依据是全场普查：
    #
    #   survey_field.py + probe_peer.py（服务器真值，n≥150 局才排名）：
    #     排名 5  EdGrass  自摸 28.12%  均番 1.444  副露 0.856
    #                        副露分布 0摊 37.5% / 1摊 43.1% / 2摊 16.2% / 3+ 3.1%
    #     我方            自摸 24.19%  均番 1.289  副露 1.687
    #                        副露分布 0摊 13.9% / 1摊 26.9% / 2摊 38.8% / 3+ 20.4%
    #   而且两者的**速度完全相同**（出牌巡数 9.4 vs 9.1、首次听牌巡中位都是 6）
    #   —— 也就是说我们多出来的那 0.83 次副露**没有换来任何速度**，
    #   只换来了更低的番数（1.289 vs 1.444）和更少的自摸（24.19% vs 28.12%）。
    #
    #   根因：碰/吃各消耗手上 2~3 张牌。牌少则摸牌的有效牌面变窄，
    #   而且副露把牌型暴露给对手、永久关闭七对分支（EdGrass 七对 6.7%，我们 0%）。
    #   所以门控应当是"**副露必须严格让我更接近胡牌**"，而不是"有机会就吃"。
    #
    # ★ 5 与 7 是 2026-09-17 补上的**反方向**档位（此前基因空间只有"更保守"）：
    #   · 5 = 已有 ≥1 摊就不再碰（**最保守**的速率闸）
    #   · 7 = **最激进**：不受任何速率闸约束，见碰就碰
    #   为什么要补 7：`probe_peer_loss.py` 显示同级桌上我方自摸 22.4%
    #   而对手 25.9%（速度劣势），我们副露却比对手多 52%。两种解释都成立 ——
    #   "副露拖慢了速度"（→ 该更保守）或"我们还不够激进、被对手先听牌"。
    #   保守方向的 2/6 档已测完（B 无差别、D 显著更差），所以必须把
    #   **激进方向**也测掉，否则无法判断副露这个旋钮到底该往哪拧。
    #
    # ⚠ 各档位的**完整语义表**（自测 [7] 逐条钉住，别再靠猜）：
    #     1 → 见碰就碰（无速率闸）
    #     2/6 → 只在"碰后向听不变差"时碰
    #     3 → 见碰就碰，但**手上有白板就不碰**
    #     4 → 已有 ≥2 摊就不再碰
    #     5 → 已有 ≥1 摊就不再碰
    #     7 → 见碰就碰（与 1 同效，但**必须放在所有速率闸之前**才成立）
    #   7 与 1 在当前实现下同效；保留 7 是为了让"最激进"这个语义在基因空间里
    #   独立可寻址 —— 将来若把 1 改成有条件，7 仍代表"无条件下接受"。
    #
    # ★ 8 是 2026-09-17 依据**杭州麻将主流打法经验**补的"**时机**门控"
    #   （不是数量门控）：**已听牌就不再副露**。
    #   出处：《杭州麻将高手打牌思路》「若是已经进入了听牌的阶段，那就不要随意碰牌了，
    #   以免耽误了后续的胡牌」；以及「三吃不如一摸」（边锋游戏 / 浙江游戏大厅）。
    #   机制上也自洽：副露跳过摸牌（diag_meld_draw_cost.py：副露 0.015→1.759 时
    #   全场巡数 40.69→36.40），而听牌之后唯一的进度来源就是摸到那张牌。
    #   4/5 的闸看"已有几摊"，8 的闸看"当前是否听牌" —— 两者必须分开可寻址，
    #   否则测不出"副露量"与"副露时机"哪个才是关键。
    ("peng", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
    # chi 同 peng 的语义，另有两点差别：
    #   · 规则硬上限：**最多吃 2 摊**（服务端也会拒第 3 次），所有档位都受它约束；
    #   · 4/5 的闸按**已有总摊数**算（与吃了几次无关），7 则绕过这条闸、
    #     只受上面那条硬上限约束。
    ("chi", (0, 1, 2, 3, 4, 5, 6, 7, 8)),
    # 杠。0 = 不杠；1 = 只暗杠（历史语义，保持不变 —— 之前的实验结论继续可比）；
    # 2 = 暗杠 + 补杠（碰出的面子 + 手上第 4 张）。
    # 补杠为什么单独占一档：它是**另一个动作**，不是同一个旋钮的值。
    # 线上对手杠率 0.03~0.09 次/局，与我们实测的补杠机会 0.043 次/局吻合；
    # 而模拟器直到 2026-09-17 才实现这条通路（此前策略若返回 gang 会被
    # _try_gang 拒掉并落进随机弃牌 —— 那是比不实现更糟的静默失真）。
    ("gang", (0, 1, 2)),
    # decline：弃胡博财飘/爆头。
    #   0/1 = 不弃胡（1 = safe_piao，实测从不触发）
    #   2   = 弃胡（基本条件：爆头形 + 牌墙够 + 非受限）
    #   3   = 2 + **威胁门控**（只在 threats(v) <= 1 时才弃）
    #   ★ 4  = 3 + **解除庄位禁令**（第 43 轮新增）
    #
    # 为什么加 4：`_dealer_blocks_decline` 一直**无条件**禁止庄家弃胡，
    # 依据是"让出庄位很贵"（`diag_dealer_chain.py`：庄位持有价值 ≈4.57 分/局）。
    # 但那条依据**从没被直接检验过** —— 因为禁令本身让庄家**永远没有弃胡机会**：
    # 线上 97 次弃胡机会**全是闲家**（`audit_decline_clean.py`），
    # 所以"庄家弃胡的兑现率 q"这个数**根本不存在**。
    #
    # 而模型说这笔账是划得来的：含庄位期权的盈亏平衡
    #   闲家 q* = (10f+Δ)/(20f+Δ) ≈ 0.54~0.58，**实测闲家 q = 0.784**；
    #   庄家 q* = (24f+Δ)/(48f+Δ) ≈ 0.52~0.54 —— **比闲家那条还低**。
    # ⇒ 若庄家的 q 也在 0.7 以上，这条禁令每年都在白扔钱。
    # 4 就是那个"把禁令打开"的格子：**把假设变成数据**。
    # ★ 5 = **2 + 解除庄位禁令**（第 118 轮新增）。与 4 的关键区别：
    #   4 = 3 + 解除禁令 = **带威胁门控**；5 = **不带**威胁门控。
    #   为什么要单独有 5：第 118 轮把 `decline4` 的 Δμ **按座位拆开**，得到
    #   做庄 **+0.663 [+0.44,+0.89]** / 做闲 **−0.198 [−0.26,−0.14]**
    #   （两边 CI 都不含 0，且三个**独立 seed 段**复现：+0.663/+0.803/+0.536）。
    #   但 4 相对 2 改了**两件事**（威胁门控 + 庄位禁令）⇒ 那两列**读不出
    #   "只有我们自己改"的效果** —— 自对局里陪打的三个座位用同一个基因组。
    #   而 `_dealer_blocks_decline(v)` 是**纯庄位**判定（`v.dealer == v.seat`，第 800 行）
    #   ⇒ **5 相对 2 只改庄家**：自对局里"做庄那一列"就是**干净的**
    #   "我们做庄时可以弃胡"的收益（对手全在闲家位，行为逐字不变）。
    #   ★ 第 159 轮更正：**"恒弃"是个空档位** —— `decline=2` 本来就是恒弃
    #   （`decline_tile_for` 逐字复刻它的出口 ⇒ 两者恒等，实测配对 SD=0.00）。
    #   那个 73% 是**兑现率 q** 不是决策率。⇒ 这条轴在决策层面已拉满，
    #   唯一剩下的杠杆是**闸门**，也就是 4/5（庄位禁令）与墙量闸。
    #   ★ 第 162 轮新增 6/7 —— `pre_baotou` 守卫的两档手术（见 `on_turn` 里的长注释）：
    #     6 = 完整移除守卫（B+C 都放开）；7 = **只**放开分支 C（safe_piao 打财神博财飘）。
    #     ★★ **已按上界关闭，不要跑配对**。`probe_prebaotou_piao.py`（500 局实测）：
    #        能胡决策点 0.324/局，其中「摸牌前已爆头」**0.062/局**（守卫确实在触发），
    #        但那些局面里 **有分支 B 的只有 0.008/局**、**有分支 C 的只有 0.004/局**。
    #        ⇒ `decline=7` 的机会率 **0.0040/局**、`decline=6` 是 **0.0080/局**。
    #        要凑 30 次机会分别需 **7,500 / 3,750 局**，而每次机会的效应必须
    #        **> 50 / > 25 分**才够得着 +0.2 分/局 —— 一局的最大摆动也就这个量级
    #        ⇒ **上界就够不着门槛**。留着是把它变成可复核的封闭记录，不是待办。
    #     另记（解释了一个老疑点）：**"非 pre 且只有 C"的机会 = 0.0000/局**
    #        ⇒ `safe_piao` 那一支在规则里几乎永远轮不到，因为分支 B 总先命中。
    ("decline", (0, 1, 2, 3, 4, 5, 6, 7, 9)),
    # ukeire 计算是否做「未见牌修正」。
    #   0 = 旧口径：left = 4 - 自己手牌张数（忽略别人打出的与副露用掉的）
    #   1 = 用快照的 discards + melds 算出真正还剩几张可摸
    # 依据（技术报告 §2.6）：不修正会把 8 枚的有效牌当成全都在牌山里，
    # 而实际约只有一半会落到自己的摸牌位 —— 等于系统性高估了某些牌的价值。
    # 这一位原先放的是 baotou（爆头偏好）。它已被 120 局干净隔离实验证伪：
    # 得分 +0.06、CI [-2.58, 2.59]，且自摸率是全部 shanten 变体里最低的（22.5%）。
    # 换掉它时末位取 0 的旧基因组语义完全不变，连续性得以保留。
    ("ukeire", (0, 1)),
    # route：弃牌时的路线承诺。
    #   0 = 现有：每条候选路都取 min(平胡向听, 七对向听)，会在两条路线间摇摆
    #   1 = 承诺：七对向听不落后平胡太多、且对子够多时，整条路线按七对推进
    # 动机：自摸率已被证明接近四方均势的必然值（25%），提升空间在**番数**。
    # 七对 ×2 / 豪华七对 ×4 / 双豪华 ×8，相对平胡 ×1 是成倍差。
    # 代价是七对更难成牌，可能压低自摸率 —— 这正是要对拍的权衡。
    ("route", (0, 1)),
    # feed：弃牌时的**喂牌风险**修正（0 关 / 1 开）。
    #
    # 这一维此前完全没被考虑过：本游戏**没有点炮**，所以传统麻将里的"安全牌"概念
    # 不适用 —— 但弃牌仍然会**喂给对手碰/吃**，而副露正是对手推进（乃至走爆头）的
    # 加速器。也就是说"打哪张"除了影响自己，还影响对手的速度。
    # 唯一可用的公开代理是**没见过几张**：某张牌若已有 3 张在弃牌堆/副露里，
    # 对手就不可能拿它碰；而一张"全新"的牌，对手手里有两张的概率最高。
    # 开启后：在（向听、有效牌）都并列的候选里，优先打**没见过张数最少**的那张。
    ("feed", (0, 1)),
])

GENE_ORDER = list(GENES.keys())

# 基因组语义版本：末位基因含义改变时必须 +1（stats.py 用它隔离旧数据）
#   v1 末位 = baotou    v2 末位 = ukeire    v3 末位 = route（多一位，故名字长度也变）
GENE_SCHEMA = 4

BASELINE_GENOME = {"discard": "first", "peng": 0, "chi": 0, "gang": 0,
                   "decline": 0, "ukeire": 0, "route": 0, "feed": 0}
# ⚠⚠ 第 145 轮审计：`CHAMPION_GENOME` 是**历史残留，不是线上基因组**，别用它。
#   两个问题：① **缺 `feed` 键** ⇒ `genome_id()` 会输出 "…:None"；
#   ② 内容（gang=0 / decline=0 / route=0）与**线上全不同** ——
#   线上是 `shanten:1:1:2:2:1:1:0`（gang=2 / decline=2 / route=1 / feed=0）。
#   保留而不删，是因为它记录了当年那一代的冠军；补上 `feed` 只是修好那个 None。
#   ★ **凡要"当前线上基因组"，一律以 `run_live.py:52` 为准。**
CHAMPION_GENOME = {"discard": "shanten", "peng": 1, "chi": 1, "gang": 0,
                   "decline": 0, "ukeire": 1, "route": 0, "feed": 0}


def peng_tiles(v):
    """本人**碰出**的面子对应的牌索引集合（杠不算 —— 杠不能再补）。

    线上快照 melds[seat] = [{"kind":"peng"|"chi"|"gang", "tiles":[...]}, ...]，
    与模拟器同构。补杠只对 peng 成立，所以必须按 kind 过滤。
    """
    out = set()
    row = []
    melds = v.snap.get("melds")
    if isinstance(melds, dict):
        row = melds.get(str(v.seat)) or melds.get(v.seat) or []
    elif isinstance(melds, list) and 0 <= v.seat < len(melds):
        row = melds[v.seat] or []
    for m in row:
        if not isinstance(m, dict):
            continue
        if (m.get("kind") or "").lower() != "peng":
            continue
        tiles = m.get("tiles") or []
        if not tiles:
            continue
        try:
            out.add(tile_index(tiles[0]))
        except (ValueError, TypeError):
            pass
    return out


def threats(v):
    """有几家对手"看起来快听牌了" —— 用副露组数当代理。

    线上的听牌状态不可见，唯一与"离胡牌还有多远"相关的公开信息就是对手的副露
    与弃牌数。副露 2 组以上意味着他的手牌只剩 7 张要凑 2 个面子 + 1 将，
    离听牌明显更近。这是**刻意做粗**的一阶代理 —— 它唯一要回答的是
    "现在弃胡，被别人抢先的概率是不是明显偏高"。
    """
    n = 0
    melds = v.snap.get("melds")
    if not isinstance(melds, list):
        return 0
    for s, row in enumerate(melds):
        if s == v.seat:
            continue
        if isinstance(row, list) and len(row) >= 2:
            n += 1
    return n


def unseen_from_view(v):
    """算出每种牌「还有几张没见过」，供有效牌枚数修正。

    没见过的 = 4 - 已打出的（四家弃牌堆）- 已副露用掉的 - 自己手上的。
    不修正的话，ukeire 会假设 4 张全在牌山，系统性高估有效牌价值。
    """
    dead = [0] * NTILE
    for row in (v.discards or []):
        for code in row:
            try:
                dead[tile_index(code)] += 1
            except (ValueError, TypeError):
                pass
    for row in (v.snap.get("melds") or []):
        for m in row:
            if not isinstance(m, dict):
                continue
            for code in (m.get("tiles") or []):
                try:
                    dead[tile_index(code)] += 1
                except (ValueError, TypeError):
                    pass
    out = []
    for t in range(NTILE):
        left = 4 - dead[t] - v.counts[t]
        out.append(left if left > 0 else 0)
    return out


# ---------------------------------------------------------------- 基因组操作

def genome_id(g):
    return ":".join(str(g.get(k)) for k in GENE_ORDER)


def parse_genome(s):
    """'shanten:1:1:0:0' -> dict。字段数不足的按基线补。

    ★ **字段数过多必须报错，不能静默截断**（2026-09-18 补）。

    事故形状：写基因组时多写了一位（`shanten:1:1:2:2:0:1:1:0`，9 个字段），
    而 `zip(GENE_ORDER, parts)` **按短的截断** ⇒ 多出来的那位被丢掉、
    **后面所有基因整体错位一位**，并且没有任何报错。
    实测后果：一轮"扫 decline×route 姿态前沿"的实验里，七个姿态里有
    三对统计量**逐位完全相同**（μ/σ/偏度/自摸率/爆头率/均番全一样到三位小数）
    —— 因为错位后它们其实解析成了同一批基因组。**如果没有"一模一样就怀疑"
    这一步，整轮实验会以"扫完了、没差别"收尾。**

    这是本项目第 7 次踩"基因编码静默失真"这一类坑（前 6 次见 STATE.md）。
    所以：**少写**仍然容忍（按基线补，历史兼容），**多写**一律抛错。
    """
    parts = [x.strip() for x in str(s).split(":")]
    if len(parts) > len(GENE_ORDER):
        raise ValueError(
            "基因组字段过多：收到 %d 个，最多 %d 个（%s）。"
            "多写会被 zip 静默截断并让后面所有基因错位 —— 拒绝解析。收到 %r"
            % (len(parts), len(GENE_ORDER), ":".join(GENE_ORDER), s))
    g = dict(BASELINE_GENOME)
    for k, v in zip(GENE_ORDER, parts):
        choices = GENES[k]
        if k == "discard":
            if v not in choices:
                raise ValueError("discard 只能取 %s，收到 %r" % (list(choices), v))
            g[k] = v
        else:
            g[k] = int(v)
    return g


def random_genome(rng):
    return {k: rng.choice(list(v)) for k, v in GENES.items()}


# 分基因变异率。discard 已被强证据钉死（60 局配对检验：first 基线 -5.45/场、
# heuristic 也显著弱于 shanten），均匀变异会让大量子代一出生就是废的 ——
# 上一代就引入了 first:1:1:1:1:0 和 first:1:0:0:0:0 这种，白白烧掉整个世代。
# 其余基因尚无定论，保持较高探索率。
MUTATE_RATE = {"discard": 0.06, "peng": 0.35, "chi": 0.35,
               "gang": 0.30, "decline": 0.30, "ukeire": 0.40, "route": 0.45,
               "feed": 0.40}


def mutate(g, rng, rate=None):
    """变异。rate=None 用分基因速率；给标量则所有基因统一用该速率（测试用）。"""
    out = dict(g)
    for k, choices in GENES.items():
        r = MUTATE_RATE.get(k, 0.34) if rate is None else rate
        if rng.random() < r:
            alt = [c for c in choices if c != out.get(k)]
            if alt:
                out[k] = rng.choice(alt)
    return out


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENE_ORDER}


# ---------------------------------------------------------------- 向听数出牌

def commit_chiitoi(counts, meld_count):
    """是否该整手承诺七对路线。

    实测发现：**弃牌层面这条路没有可操作空间** —— min(平胡向听, 七对向听)
    已经在七对更优时自动走七对，4000 副随机手的分歧率为 0.0%。
    真正有行为差异的地方在副露窗口：规则规定「七对子禁止吃碰明杠」，
    所以承诺七对时必须放弃碰/吃，否则亲手把七对路线拆掉。
    """
    if meld_count > 0:
        return False                      # 已有副露即与七对互斥
    sc = S.shanten_chiitoi(counts, 0)
    if sc >= 99:
        return False
    pairs = sum(1 for x in counts if x >= 2)
    return sc <= S.shanten_regular(counts, 0) and pairs >= 5


def win_prob(s, u, draws_left, unseen_total=70.0):
    """粗估「从 s 向听、有效牌 u 枚，在 draws_left 次摸牌内胡牌」的概率。

    模型：每次摸牌以 u/U 的概率推进向听；从 s 走到胡需 (s+1) 次推进，
    故平均耗时 (s+1)·U/u 次摸牌，用指数分布近似。

    这是**刻意做粗**的一阶模型。它唯一要捕捉的是「向听数」与「有效牌数」的
    互换关系 —— 而现有的字典序规则（先比向听、再比有效牌）做不到：
    向听 1 但只有 4 张有效牌，未必优于向听 2 但有 30 张，字典序永远选前者。
    """
    if s < 0:
        return 1.0
    if u <= 0 or draws_left <= 0:
        return 0.0
    need = (s + 1) * (unseen_total / float(u))
    return 1.0 - math.exp(-draws_left / max(0.5, need))


def best_discard_value(counts, meld_count, forbid=(), unseen=None, draws_left=16,
                       span=1):
    """按估计胜率最大的弃牌（而非向听数最小的）。

    与 best_discard_shanten 的区别：后者是字典序（向听优先），本函数把
    两者放进同一个标量里比较，允许「用 1 个向听换更多有效牌」。

    span 限制互换范围：只在该手最优向听 +span 以内的候选里比较。
    不限制的话实测分歧率 60.5% 且**全部**是拿向听换有效牌（1815:0）——
    因为一阶模型假设有效牌率在推进全程恒定，高向听那侧会被系统性高估。
    限制到 ±1 既更合理，也把每回合要算有效牌的候选从 14 个降到约 3 个。
    """
    cands = []
    for t in range(NTILE):
        if counts[t] <= 0 or t in forbid:
            continue
        c = list(counts)
        c[t] -= 1
        cands.append((S.shanten(c, meld_count), t, c))
    if not cands:
        return None
    best_s = min(x[0] for x in cands)
    out = []
    for s, t, c in cands:
        if s > best_s + span:
            continue
        u = S.ukeire(c, meld_count, unseen=unseen)[0]
        out.append((win_prob(s, u, draws_left), u, s, t))
    if not out:
        return None
    return max(out, key=lambda x: (x[0], x[1]))[3]


def wait_width(counts, meld_count, unseen=None):
    """**已听牌**时的有效胡牌张数：仍未见过、能让我们胡的牌一共几张。

    为什么必须单独写一个（第 52 轮 §Z.45）：
    `S.ukeire` 数的是"能让**向听下降**的牌"（`if s < base`）。
    在 `shanten == 0` 时**没有任何牌能让向听低于 0** ⇒ 它对**每一个候选都返回 0**
    ⇒ `best_discard_shanten` 里那段并列裁决退化成"只在第一个候选处赋值"
    ⇒ **`return top[0]`，即按"牌码顺序"挑听口，从来没看过"能胡几张"。**

    这正是 §Z.44 测到的"同样时间里我们更摸不到"（兑现缺口 −3.75pp，格内效应）
    最自然的机制解释。
    """
    tot = 0
    for code in winning_tiles(counts, meld_count):
        ti = tile_index(code)
        left = (4 - counts[ti]) if unseen is None else unseen[ti]
        if left > 0:
            tot += left
    return tot


def best_discard_shanten(counts, meld_count, forbid=(), prefer_baotou=False,
                         unseen=None, route=0, feed=0, wait_aware=False,
                         fan_aware=0, tie_hook=None):
    """向听数最小 -> （可选）爆头优先 -> 有效牌最多 的弃牌。返回牌索引，无牌可打返回 None。

    unseen 给定时，有效牌枚数按「真正没见过几张」算，而不是假设牌山里有 4 张。
    route=1 时启用七对路线承诺：**先按整手定路线，再在同一把尺子上比较所有候选**。
    feed=1 时，在（向听、有效牌）并列的候选里优先打"没见过张数最少"的那张 ——
    本游戏没有点炮，但弃牌会喂对手碰/吃，而副露是对手的加速器（见 GENES 的说明）。

    ⚠ `tie_hook`（第 144 轮新增）**只给诊断探针用，默认 None**：
    给了它就在「向听 + 有效牌都并列」的那个池子里再问一次
    `tie_hook(counts, meld_count, top, unseen, use_chiitoi) -> 牌索引 | None`。
    ★ 它插在**规则自己的裁决层**上，所以用它可以量出
    「(a) 保持向听/有效牌主键不变、(b) 只把**逐家信息**加进并列裁决」
    这条路的**上界** —— 探针拿模拟器里已知的四家手牌当 oracle 喂给它。
    ★★ 生产路径**一个字节都没动**：`tie_hook is None` 时下面每一次 return 都与原来逐字一致。

    早先的实现是对每个候选各自决定「用七对向听还是 min」，导致不同候选用了
    不同的量纲 —— 两个数不可比，会把真正更差的选项选出来（实测分歧率仅 0.1%，
    且选出的常是两个指标都更差的那张）。路线必须整手一次定，否则没有意义。
    """
    use_chiitoi = False
    if route and meld_count == 0:
        use_chiitoi = commit_chiitoi(counts, meld_count)

    cands = []
    for t in range(NTILE):
        if counts[t] <= 0 or t in forbid:
            continue
        c = list(counts)
        c[t] -= 1
        if route and use_chiitoi:
            s = S.shanten_chiitoi(c, 0)       # 承诺七对：整手统一用七对尺子
        else:
            s = S.shanten(c, meld_count)      # 未承诺：与原规则逐字一致
        cands.append((s, t, c))
    if not cands:
        return None
    best_s = min(x[0] for x in cands)
    top = [x for x in cands if x[0] == best_s]

    if prefer_baotou and best_s == 0:
        bt = [x for x in top if is_baotou(x[2], meld_count)]
        if bt:
            top = bt                      # 有爆头形就只在爆头形里挑

    if len(top) == 1:
        return top[0][1]
    # ★ 承诺七对时，并列候选必须用**七对口径**的有效牌来分胜负。
    #
    # 这是一个真 bug 的修复，`probe_meld_fan.py` 把它暴露了出来：
    #   承诺七对（route=1）之后，2500 局同质桌里七对胡率只有 **0.20%**
    #   （不承诺时 0.00%，线上对手是 2.7%）。
    # 根因：候选的向听是用`shanten_chiitoi`算的（口径对），但并列之后
    # 却用**平胡口径**的 `ukeire` 去挑 —— 而七对的有效牌（能凑成对子的牌）
    # 总是比平胡的两面/嵌张少。于是"向听并列 → 平胡 ukeire 更大 → 选平胡那条"，
    # 路线承诺在弃牌层面被**系统性推翻**。所以 route 基因此前只有"禁副露"那一半生效。
    #
    # 正确做法：承诺哪条路线，就用哪条路线的有效牌。
    if route and use_chiitoi:
        best = None
        for _s, t, c in top:
            tot = S.ukeire(c, 0, unseen=unseen, chiitoi=True)[0]
            if best is None or tot > best[0]:
                best = (tot, t)
        if tie_hook is not None:                    # ← 诊断钩子（默认不进来）
            alt = tie_hook(counts, meld_count, top, unseen, True)
            if alt is not None:
                return alt
        return best[1]
    if fan_aware:
        # ★★ `discard=shanten_fan`（第 58 轮）：**把番型并进并列裁决**。
        #
        # 为什么这是"换表示"而不是"再调一个参数"：
        #   现有的裁决只有一把尺子 —— `S.ukeire`（"能让向听下降的牌有几张"）。
        #   这把尺子**对番数完全盲**（`best_discard_baotou` 的注释自己写着这一点）。
        #
        # ⚠⚠ 第一版我用的特征是「手里的**白板张数**」——**那是退化特征**：
        #   打掉白板一定会抬高向听 ⇒ 白板永远不会被打掉
        #   ⇒ **所有并列候选的白板张数完全相同**，这一维加了个常数，等于没加。
        #   实测分歧 **0 / 2656**（`check_fan_active.py`，自校验 0 不一致）。
        #
        # ✅ 正确的特征是「**不含白板的那部分牌，离 4 个面子有多近**」——
        #   因为**爆头 = 4 个自然面子 + 一张白板当将**，所以"白板不用来凑面子"
        #   才是爆头可达的关键；而这一项**随候选不同而不同**（换牌会改变非白板牌集）。
        #   口径：把白板全部置 0 后再算一次向听。因为白板张数在各候选间是常数，
        #   这个数**在候选之间可比**（手牌张数相同）。
        #
        # ⚠ 主键仍是**最小向听**（`best_s`），这一层只在并列候选之间起作用 ——
        #   §Z.47 已用真实对手证明全场强手都打最小向听，主键不能动。
        # ⚠ 权重是**人为选的**，没有理论依据 ⇒ 若结果为"没动"，
        #   只说明"这个权重不行"，不能一口咬定"这条路不行"。
        # ⚠⚠ 第二版教训：我用 `value = ukeire + 3*(8-nw_no)` 加权，
        #   实测仍然 **0/2656 分歧**。逐候选打表（`check_fan_debug.py`）才看清：
        #   在 3087 个并列局面里 **`ukeire` 有 81.8% 会变，而 `nw_no` 只有 1.0% 会变**
        #   （丢一张牌几乎不改变"不含白板部分"的向听）。
        #   ⇒ 小权重永远翻不动（差距动辄 10~20 张有效牌）；
        #     大权重则会**盖过 ukeire**，那就不是"再加一维"，而是换掉了主准则。
        #   ⇒ 正确形态是**字典序**：先把 ukeire 用尽，**只在它打平时**才让番型投票。
        #     这也说明了一件事：**并列裁决那一层并不是空的 —— ukeire 已经占掉了 82%。**
        best = None
        for _s, t, c in top:
            u = S.ukeire(c, meld_count, unseen=unseen)[0]
            c2 = list(c)
            c2[WILD] = 0
            try:
                nw_no = S.shanten(c2, meld_count)
            except Exception:                                    # noqa: BLE001
                nw_no = 8
            v = (u, -nw_no)
            if best is None or v > best[0]:
                best = (v, t)
        return best[1]
    if wait_aware and best_s == 0:
        # ★★ `discard=shanten2`：**只在已听牌时**换一个并列裁决的尺子。
        #
        # 原来的尺子（`S.ukeire`，数"能让向听下降的牌"）在 `shanten == 0` 时
        # 对每个候选都恒为 0 ⇒ 等价于"按牌码顺序挑"。已听牌时唯一有意义的
        # 次准则就是**还剩几张能胡** —— 也就是听口宽度。
        #
        # ⚠ 刻意放在**七对承诺分支之后**：承诺七对时保持原语义不变，
        #   把改动面收窄到"没有承诺七对、且已经听牌"的那部分局面。
        best = None
        for _s, t, c in top:
            tot = wait_width(c, meld_count, unseen)
            if best is None or tot > best[0]:
                best = (tot, t)
        return best[1]
    best = None
    for _s, t, c in top:                      # 仍并列才付 ukeire 的代价
        tot = S.ukeire(c, meld_count, unseen=unseen)[0]
        if best is None or tot > best[0]:
            best = (tot, t)
    if tie_hook is not None:                  # ← 诊断钩子（默认不进来）
        # 插在**规则自己的裁决层**上：池子 = 向听并列 ∩ 有效牌并列。
        # 钩子自己重算池子（它拿得到全部输入），这样生产路径一个字都不用改。
        alt = tie_hook(counts, meld_count, top, unseen, False)
        if alt is not None:
            return alt
    if feed:
        # 有效牌并列时再看喂牌风险：没见过越少 = 对手越难拿它碰
        u_best = best[0]
        pool = []
        for _s, t, c in top:
            tot = S.ukeire(c, meld_count, unseen=unseen)[0]
            if tot == u_best:
                pool.append(t)
        if len(pool) > 1:
            if unseen is not None:
                left = [unseen[t] for t in pool]
            else:
                left = [4 - counts[t] for t in pool]
            best = (u_best, pool[left.index(min(left))])
    return best[1]


def best_discard_baotou(counts, meld_count, forbid=(), unseen=None):
    """新方案：为爆头留门 —— 向听不变的前提下，尽量别把白板嵌进面子。

    为什么值得单独做一条骨架（实测依据见 probe_highfan.py，250 局 7950 个决策点）：
      · 有白板的决策点占 41.6%；**胡牌时手留白 >=1 张的占 82.8%**
      · **留白板的向听代价 100% 为 0** —— 留着白板从来不吃亏
      · 「白板全部作废后向听 <= 1」的局面占 32.0%，入口很宽
      · 但爆头态出现 **0 次**，番数 299/300 是 x1
    根因：shanten/ukeire **对番数完全盲**。白板嵌在面子里与白板当将向听相同，
    于是它永远选前者，把 ×2 的爆头出口堵死。这不是参数没调好，是表示问题。

    规则（严格不牺牲向听，只在并列候选里重排）：
      1. 向听最小
      2. 若手上还有白板：只在「打完仍留着白板」的候选里，取白板全作废后
         向听最小者（= 自然牌自己越接近成型，那张白板越接近「多余」）
      3. 打完即爆头形
      4. 有效牌最多
    第 2 步先于有效牌，是本方案与 shanten 骨架的唯一实质差别 —— 它可能牺牲
    一点有效牌去换番数，这个权衡正是要对拍的东西。
    """
    cands = []
    for t in range(NTILE):
        if counts[t] <= 0 or t in forbid:
            continue
        c = list(counts)
        c[t] -= 1
        s = S.shanten(c, meld_count)
        cn = list(c)
        cn[WILD] = 0
        nat = S.shanten(cn, meld_count)
        cands.append((s, nat, t, c))
    if not cands:
        return None
    best_s = min(x[0] for x in cands)
    top = [x for x in cands if x[0] == best_s]

    if counts[WILD] > 0:
        keep = [x for x in top if x[3][WILD] > 0]
        if keep:
            best_nat = min(x[1] for x in keep)
            top = [x for x in keep if x[1] == best_nat]

    if best_s == 0:
        bt = [x for x in top if is_baotou(x[3], meld_count)]
        if bt:
            top = bt
    if len(top) == 1:
        return top[0][2]
    best = None
    for _s, _n, t, c in top:
        tot = S.ukeire(c, meld_count, unseen=unseen)[0]
        if best is None or tot > best[0]:
            best = (tot, t)
    return best[1]


def best_discard_baotou2(counts, meld_count, forbid=(), unseen=None):
    """自然牌优先的爆头塑形：**允许用 1 个向听换自然牌部分成型**。

    与 best_discard_baotou 的唯一实质差别是"能不能付这个价"：
      · baotou  ：向听并列时才比较自然牌（严格不牺牲向听）
      · baotou2 ：若某一手能让自然牌部分**严格更接近成型**，允许它的真实向听
                  比最优差 1

    依据是爆头代数（MODEL.md §4）：有 1 张白板时，爆头 = 自然牌部分恰好
    (4−m) 个完整面子。也就是说**白板不进面子**才是终点，而"自然牌向听"
    （把白板全部作废后再算向听）就是这个终点的距离。真实牌桌上爆头率最高的
    玩家（走马 50%）打的正是这条路线：让白板当那条"余白"，用碰/吃把面子数
    堆上去，而不是让白板去凑面子。

    风险与它是同一个硬币的两面：真实向听差 1 意味着慢一巡，所以这是
    "用速度换爆发"的取舍，必须在同一张牌桌上与 baotou 对拍（不允许凭论证上线）。
    """
    cands = []
    for t in range(NTILE):
        if counts[t] <= 0 or t in forbid:
            continue
        c = list(counts)
        c[t] -= 1
        s = S.shanten(c, meld_count)
        cn = list(c)
        cn[WILD] = 0
        nat = S.shanten(cn, meld_count)
        cands.append((s, nat, t, c))
    if not cands:
        return None
    best_s = min(x[0] for x in cands)
    pool = [x for x in cands if x[0] == best_s]

    if counts[WILD] > 0:
        nat0 = min(x[1] for x in pool)
        # 只有"真的推进了自然牌"才值得付这一向听；否则等于白慢一巡
        trade = [x for x in cands if x[0] == best_s + 1 and x[1] < nat0]
        if trade:
            pool = trade
        keep = [x for x in pool if x[3][WILD] > 0]
        if keep:
            pool = keep                     # 白板是那条余白，不能亲手用掉

    nat_min = min(x[1] for x in pool)
    top = [x for x in pool if x[1] == nat_min]

    if min(x[0] for x in top) == 0:
        bt = [x for x in top if is_baotou(x[3], meld_count)]
        if bt:
            top = bt
    if len(top) == 1:
        return top[0][2]
    best = None
    for _s, _n, t, c in top:
        tot = S.ukeire(c, meld_count, unseen=unseen)[0]
        if best is None or tot > best[0]:
            best = (tot, t)
    return best[1]


# ------------------------------------------------------------ 价值网络决策
_NET = {"net": None, "mu": None, "sd": None, "path": None}


def load_value_net(path="value_net.npz"):
    """载入由 train_value.py 训出的价值网络。惰性调用，纯 numpy 无额外依赖。"""
    from train_value import Net
    net, mu, sd = Net.load(path)
    _NET.update(net=net, mu=mu, sd=sd, path=path)
    return True


def net_win_prob(v, counts=None):
    """网络对「这一席本局会胡」的预测。counts 给定时按该手牌评估。"""
    import numpy as np
    from features import features
    if _NET["net"] is None:
        load_value_net()
    X = np.asarray([features(v, counts)], dtype=np.float32)
    X = (X - _NET["mu"]) / _NET["sd"]
    logit, _s, _ = _NET["net"].forward(X)
    z = float(logit[0])
    return 1.0 / (1.0 + math.exp(-z)) if -30 < z < 30 else (0.0 if z <= -30 else 1.0)


def best_discard_net(v, forbid=(), shrink_to_min_shanten=True):
    """一步前瞻：枚举弃牌，取「打完之后网络预测胡牌概率最高」的那张。

    为什么要先用向听过滤（shrink_to_min_shanten）：
      第一版让网络在**全部**合法弃牌里挑，结果配对对拍 -8.08 分/局、p=0.0002，
      而同一个网络的跨状态 AUC 却有 0.75。原因不是网络没学到东西，而是
      **AUC 衡量「跨状态排序」，策略需要的是「同一状态内对候选动作排序」**。
      候选之间只差一张牌（84 维里动 2 维），网络的输出差异极小、极易被噪声主导，
      于是它会把向听更差的牌选出来。

      先按向听收敛候选，问题就变成一个干净、可证伪的问法：
      **在向听并列的候选里，网络能不能比 ukeire 挑得更好？**
      加分就是真加分，不加分就老实用 ukeire —— 不留模糊地带。

    用胡牌概率而不是期望得分：实测番数在 99.6% 的胡牌局里都是 ×1
    （probe_highfan.py），故期望得分 ≈ P(胡) × 常数；而得分回归头本身几乎
    无预测力（验证集相关仅 0.20，MSE 105 对 SD 8.7）。用更准的那个头。
    """
    import numpy as np
    from features import features
    if _NET["net"] is None:
        load_value_net()
    cands = []
    for t in range(NTILE):
        if v.counts[t] <= 0 or t in forbid:
            continue
        c = list(v.counts)
        c[t] -= 1
        cands.append((t, c))
    if not cands:
        return None
    if shrink_to_min_shanten:
        sh = [(t, c, S.shanten(c, v.meld_sets)) for t, c in cands]
        best_s = min(x[2] for x in sh)
        sh = [x for x in sh if x[2] == best_s]
        if len(sh) == 1:
            return sh[0][0]
        cands = [(t, c) for t, c, _s in sh]
    X = np.asarray([features(v, c) for _t, c in cands], dtype=np.float32)
    X = (X - _NET["mu"]) / _NET["sd"]
    logit, _s, _ = _NET["net"].forward(X)
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    # 概率并列时退回向听，保证确定性（对拍必须可复现）
    best_i, best_key = 0, None
    for i, (t, c) in enumerate(cands):
        key = (round(float(p[i]), 6), -S.shanten(c, v.meld_sets))
        if best_key is None or key > best_key:
            best_key, best_i = key, i
    return cands[best_i][0]


# --------------------------------------------------- 学到的候选排序器（第 145 轮）
_RANK = {"net": None, "mu": None, "sd": None, "path": None, "K": 10,
         "arch": "flat"}


def load_cand_ranker(path="cand_rank.pt"):
    """载入「候选排序器」。

    ⚠ torch 是**惰性导入**的：线上客户端不跑 `discard=rank` 就永远不会碰它。

    ★ 第 182 轮加 `candseq` 分支：`train_cand_ranker_seq.py` 训出的模型**输入不同**
      （状态 360 ＋**逐家弃牌序列 4×20** ＋ 候选 6 维，且**不做标准化**），
      架构也不同（`SeqRanker`）。⇒ 必须按 checkpoint 里的 `features` 分派，
      否则会拿错架构去 `load_state_dict`（那会报错，但**报在哪一步**取决于键名是否恰好兼容）。
    """
    sys.path.insert(0, "_pkgs")
    import torch                                                     # noqa: F401
    ck = torch.load(path, map_location="cpu", weights_only=False)
    feats = ck.get("features", "flat")
    if feats == "candseq":
        from train_cand_ranker_seq import SeqRanker
        net = SeqRanker(ck["d_state"], ck["hidden"], ck["seq_h"], ck["emb"],
                        ck.get("dropout", 0.0))
        net.load_state_dict(ck["sd"])
        net.eval()
        _RANK.update(net=net, mu=None, sd=None, path=path, arch="candseq",
                     K=int(ck.get("K", 10)))
        return True
    from train_cand_ranker import Ranker
    net = Ranker(ck["d_in"], ck["hidden"], ck["depth"], ck.get("dropout", 0.0))
    net.load_state_dict(ck["sd"])
    net.eval()
    _RANK.update(net=net, mu=ck["mu"], sd=ck["sd_norm"], path=path, arch="flat",
                 K=int(ck.get("K", 10)))
    return True


def best_discard_rank(v, forbid=(), route=0, feed=0, use_unseen=1,
                      only_ties=False):
    """在**向听并列**的候选里，用学到的排序器挑一张。返回牌索引。

    ★★ `only_ties=True` = **只在"规则自己也没偏好"时才让模型出手**（第 150 轮新增）。
    为什么这是最该先测的用法：
      · 规则的裁决是**字典序**：向听 → ukeire → **牌码顺序**。
        最后那一步是**任意的** —— 并列时它只是"挑牌码最小的那张"，没有任何依据。
      · 本仓实测：向听并列的决策点里，**ukeire 也只能定掉 56.3%**，
        剩 **43.7%** 是 ukeire 也并列的（§Z.126 实测 `len(pool)>1` 占 43.7%）。
      · 在那些点上，规则的选择≈抛硬币 ⇒ **模型只要不是更差，就是纯赚**；
        而在 ukeire 有偏好的点上，规则已经是对的（§Z.130 证明覆盖它会亏）。
      ⇒ `only_ties=True` 把模型的改动面**限制在规则无偏好的子集**，
        这是"用上模型"与"不亏分"同时成立的那个用法。

    `only_ties=False`（默认）沿用旧语义：在整个向听并列池里按模型 argmax 重排
    （⚠ 已实测**显著更差 −1.03 / −0.81**，见 §Z.130，**不要上线**）。
    """
    unseen = unseen_from_view(v) if use_unseen else None
    if _RANK["net"] is None:
        try:
            # ★ 允许用环境变量换 checkpoint（评测不同目标训出的模型时必需，
            #   否则要来回覆盖 `cand_rank.pt`，很容易把默认模型弄丢）。
            import os
            load_cand_ranker(os.environ.get("MJ_RANK_PATH") or "cand_rank.pt")
        except Exception as e:                                       # noqa: BLE001
            # ★ 优雅降级（与 `q_peng_policy` 缺模型时同款）：
            #   模型文件不在就退回**同一套基因的规则**，绝不抛异常。
            #   理由：`on_turn` 抛异常会被上层吞成"随便打一张"，
            #   那是比不用模型**更糟**的静默失真（`mj_sim._fallback_discard`）。
            if not _RANK.get("warned"):
                _RANK["warned"] = True
                sys.stderr.write(
                    "[mj_policy] discard=rank 载入失败（%s: %s）⇒ 退回向听规则\n"
                    % (type(e).__name__, e))
            return best_discard_shanten(v.counts, v.meld_sets, forbid,
                                        unseen=unseen, route=route, feed=feed)

    # ⚠ 这三行 import 必须留在这里（而不是文件顶部）：torch 很重，
    #   线上客户端不跑 `discard=rank` 时不该为它付出启动代价。
    #   ⚠⚠ 第 145 轮踩过：把降级 try/except 插进来时**顺手把这三行删掉了**，
    #   于是 `ffull` 未定义 ⇒ 每次取牌 `NameError`（被上层吞成"随便打一张"）。
    #   抓它的方式是**逐决策点调用一次**（`_rank` 自检），而不是只跑对局看胜率。
    import numpy as np
    import torch
    from features_full import cand_scalars
    from features_full import features as ffull
    cands = []
    for t in range(NTILE):
        if v.counts[t] <= 0 or t in forbid:
            continue
        c = list(v.counts)
        c[t] -= 1
        cands.append((t, c, S.shanten(c, v.meld_sets)))
    if not cands:
        return None
    best_s = min(x[2] for x in cands)
    tied = [(t, c) for t, c, s in cands if s == best_s]
    if len(tied) == 1:
        return tied[0][0]
    tied = tied[:_RANK["K"]]
    # ★★ `only_ties`：先把池子收窄到**ukeire 也并列**的那一层。
    #   理由见 docstring —— 规则在那一层是按**牌码顺序**挑的，**没有任何依据**；
    #   而在 ukeire 有偏好的地方规则已经是对的（§Z.130 证明覆盖它会亏）。
    uk = [S.ukeire(c, v.meld_sets, unseen=unseen)[0] for _t, c in tied]
    if only_ties:
        u_best = max(uk)
        ties = [i for i, u in enumerate(uk) if u == u_best]
        if len(ties) == 1:
            return tied[ties[0]][0]        # ukeire 有唯一最优 ⇒ 交给规则，模型不动手
        tied = [tied[i] for i in ties]
        uk = [uk[i] for i in ties]
        _RANK["n_tie_fired"] = _RANK.get("n_tie_fired", 0) + 1
    rows = []
    # ★★★ 第 182 轮修掉一个**训练/推理输入不一致**的 bug（影响 `rank`/`rank2` 的全部历史判决）：
    #
    #   训练侧（`train_cand_ranker.py:187` 与 `:267`）是
    #       `np.repeat(X[idx], K, axis=0)` —— **同一个 X 重复 K 次**，
    #   也就是"**摸牌后的 14 张状态**"（`gen_cand_shard.py:135` 记的是 `features(v)`），
    #   候选之间**只靠那 6 个 `cand_scalars` 区分**。
    #
    #   而这里原来写的是 `ffull(v, c)`，`c` 是**打掉候选之后**的 13 张
    #   ⇒ **X 随候选变化**，与训练时的输入分布**不是同一个**。
    #
    #   ⇒ 于是 `discard=rank`（−1.03）与 `rank2`（−0.297）测的是
    #     "**在没训练过的输入上**用模型"，不是模型本身的能力。**那两个判决必须重测。**
    #   ⇒ 修法：X 用 `ffull(v)`（不带 `c`，回到 14 张），候选特征仍用打完后的 `c`
    #     （`gen_cand_shard.py:129` 就是这么记的）。
    state_x = ffull(v)
    for t, c in tied:
        row = state_x + cand_scalars(v, c, t) + [float(len(tied))]
        rows.append(row)
    if _RANK.get("arch") == "candseq":
        # ★★ 第 182 轮：`candseq` 分支 —— **用上逐家弃牌序列**。
        #   口径必须与 `train_cand_ranker_seq.py` 的 `pick()` **逐字一致**：
        #     状态 = `features(v, counts=c)`（360，**不标准化**）
        #     序列 = `discard_seq(v, 20)`（4×20，-1 补位；同一决策点内所有候选共用）
        #     候选 = `cand_scalars(v, c, t)`（6，**不标准化**）
        import torch
        from features_full import CAND_DIM, F as _F, discard_seq as _dseq
        n_t = len(tied)
        Xs = np.asarray([r[:_F] for r in rows], dtype=np.float32)
        Cs = np.asarray([r[_F:_F + CAND_DIM] for r in rows], dtype=np.float32)
        Q = np.asarray(_dseq(v, 20), dtype=np.int8)
        Qs = np.repeat(Q[None, :, :], n_t, axis=0)
        with torch.no_grad():
            sc = _RANK["net"](torch.from_numpy(Xs), torch.from_numpy(Qs),
                              torch.from_numpy(Cs)).numpy()
        best_i, best_key = 0, None
        for i, (t, _c) in enumerate(tied):
            key = (round(float(sc[i]), 6), uk[i], -t)
            if best_key is None or key > best_key:
                best_key, best_i = key, i
        return tied[best_i][0]
    X = np.asarray(rows, dtype=np.float32)
    assert X.shape[1] == len(_RANK["mu"]), \
        "输入维度 %d != 模型 %d（数据与策略的候选特征口径不一致）" % (
            X.shape[1], len(_RANK["mu"]))
    X = (X - _RANK["mu"]) / _RANK["sd"]
    with torch.no_grad():
        sc = _RANK["net"](torch.from_numpy(X)).numpy()
    best_i, best_key = 0, None
    for i, (t, _c) in enumerate(tied):
        key = (round(float(sc[i]), 6), uk[i], -t)
        if best_key is None or key > best_key:
            best_key, best_i = key, i
    return tied[best_i][0]


# ------------------------------------------------- 学到的「胡/弃」决策器（第 152 轮）
_DECLINE = {"net": None, "mu": None, "sd": None, "path": None, "warned": False,
            "features": "full"}
_DECLINE_GENOME = "shanten:1:1:2:2:1:1:0"      # 采集数据时用的基因组（decline=2）


def load_decline_net(path="decline_net.pt"):
    """载入 `train_decline.py` 训出的「胡/弃」决策器。

    ⚠ torch 惰性导入：线上不跑 `decline=9` 就永远不碰它。
    ⚠ 两种口径的**输入不同**，必须按 checkpoint 里的 `objective` 分派：
      · `tlearner`（默认，推荐）：`ŷ = b(s) + a·τ(s)`，输入**不含**动作列；
      · `mse`：输入 = 状态 ⊕ 动作。
    """
    sys.path.insert(0, "_pkgs")
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    obj = ck.get("objective", "mse")
    if obj == "tlearner":
        from train_decline import TNet
        net = TNet(ck["d_in"], ck["hidden"], ck["depth"], ck.get("dropout", 0.0))
    else:
        from train_decline import Q
        net = Q(ck["d_in"], ck["hidden"], ck["depth"], ck.get("dropout", 0.0))
    net.load_state_dict(ck["sd"])
    net.eval()
    _DECLINE.update(net=net, mu=ck["mu"], sd=ck["sd_norm"], obj=obj, path=path,
                    features=ck.get("features", "full"))
    return True


def decline_tile_for(v):
    """**逐字复刻 `decline=2` 的弃胡出口**，返回 (牌码, 分支) 或 (None, None)。

    ⚠ 这是一份**刻意的副本**（原逻辑写在 `on_turn` 里、且被三道闸裹着，不好提取）。
    `check_decline_hook.py` 用**真实局面**逐点对拍它与规则本身 ⇒ 分叉会被当场发现
    （本仓的纪律：**允许副本，但必须有对拍**）。
    顺序与规则一致：**先分支 B（打一张仍爆头），再分支 C（safe_piao 打白板）**。
    """
    drawn_i = tile_index(v.drawn) if v.drawn else -1
    for x in range(NTILE):
        if v.counts[x] <= 0 or x == WILD or x == drawn_i:
            continue
        c2 = list(v.counts)
        c2[x] -= 1
        try:
            if is_baotou(c2, v.meld_sets):
                return tile_code(x), "B"
        except Exception:                                            # noqa: BLE001
            pass
    try:
        if safe_piao(v.counts, v.meld_sets):
            return "白", "C"
    except Exception:                                                # noqa: BLE001
        pass
    return None, None


def decline_gate_ok(v, allow_dealer=False):
    """`decline=2` 的三道闸（数据就是在这个支撑集上采的 ⇒ `decline=9` 必须同闸）。

    ① 非抓打圈受限；② 牌墙还够自己再摸一张（`wall-20 >= 4`）；③ 非庄位禁令。
    ⚠ **不含** `pre_baotou` —— 那一支是"上行空间为零、必胡"的**已证明**情形，
    规则与模型都不该在那里弃胡（`on_turn` 会先拦掉）。

    ★ 第 160 轮：`allow_dealer=True` 时**去掉第③道闸**。
    为什么这是"让模型落地"的关键一步，而不是放宽标准：
      · 采集数据时用的生成器闸**本来就不含庄位禁令**（`gen_decline_shard.py`
        只查 `restricted` 与"存在可弃牌"）⇒ **庄位样本本来就在训练集里**；
      · 但推理时挂上第③道闸，模型就**永远看不到它学过的那批庄位局面**
        ⇒ 学到的 τ(庄) 被白白扔掉，模型等于被自己的闸门阉了。
      · 分层 ATE：庄 **+13.041 [+6.680,+19.139]** / 闲 +6.628（`analyze_decline_cate.py`）。
      · 折算成每局（`probe_decline_rate.py` 实测机会率）：
        庄位机会 **0.0175 次/局** × 13.041 = **+0.228 分/局**，
        与夹具实测 `decline=5 − 2` = +0.293 同向同量级（1.3σ）。
      ⇒ 所以 `decline=9` 走 `allow_dealer=True`：**让模型自己决定**。
        它若认为庄位不该弃，τ(s) 会给出负值，行为自动退回规则。
    """
    try:
        if getattr(v, "restricted", False):
            return False
        if (v.wall or 0) - 20 < 4:
            return False
        if not allow_dealer and _dealer_blocks_decline(v):
            return False
        return True
    except Exception:                                                # noqa: BLE001
        return False


def decline_model_action(v):
    """`decline=9`：让模型在 **胡** 与 **弃** 之间选。返回 action；不可用时返回 None。

    模型口径与 `train_decline.py` **逐项对齐**：输入 = 状态 360 ⊕ ctx 10 ⊕ 动作 1。
    ⚠ `pre_baotou` 时**不看模型**，直接胡（理由见 `mj_policy.py` 的原始注释：
    摸牌前已爆头 ⇒ 摸任意牌都能胡，再弃一次上行空间为零）。
    """
    import numpy as np
    if _DECLINE["net"] is None:
        try:
            import os
            load_decline_net(os.environ.get("MJ_DECLINE_PATH") or "decline_net.pt")
        except Exception as e:                                       # noqa: BLE001
            if not _DECLINE["warned"]:
                _DECLINE["warned"] = True
                sys.stderr.write(
                    "[mj_policy] decline=9 载入失败（%s: %s）⇒ 退回 decline=2 规则\n"
                    % (type(e).__name__, e))
            return None
    drawn_i = tile_index(v.drawn) if v.drawn else -1
    if drawn_i >= 0:
        c13 = list(v.counts)
        c13[drawn_i] -= 1
        try:
            if is_baotou(c13, v.meld_sets):
                return {"action": "hu", "tile": ""}      # 上行空间为零 ⇒ 必胡
        except Exception:                                        # noqa: BLE001
            pass
    tile, _branch = decline_tile_for(v)
    if tile is None:
        return {"action": "hu", "tile": ""}              # 没有可弃的选择
    from features_full import context as _ctx
    from features_full import features as _ff
    try:
        if _DECLINE.get("features") == "compact":
            # ★ 第 160 轮：紧凑口径（18 维）。布局**唯一**定义在
            #   `features_full.compact_ctx`，这里只负责把 branch 传进去。
            #   `branch` 就是上面 `decline_tile_for` 已经返回的那一个 ——
            #   不再重算，避免与它分叉。
            from features_full import compact_from_view as _cfv
            base = _cfv(v, 1 if _branch == "B" else 0)
        else:
            base = _ff(v) + _ctx(v)
    except Exception:                                                # noqa: BLE001
        return None
    import torch
    import os as _os
    # ★★★ 第 190 轮修掉一个**符号反了**的 bug（影响：`decline=9` 的历史判决作废）
    #
    #   数据里的动作编码在 `gen_decline_shard.py` 写死为 **`A_HU, A_DECLINE = 1, 0`**
    #   ⇒ **a=1 是"胡"、a=0 是"弃"**。
    #   T-learner 的口径是 `ŷ = b(s) + a·τ(s)`（`train_decline.py:144-147`）⇒
    #       ŷ(a=1) = b + τ = **胡** 的期望分
    #       ŷ(a=0) = b     = **弃** 的期望分
    #   而这里原来写的是 `q_hu = net(x, zeros)` / `q_dc = net(x, ones)`
    #   ⇒ **两个名字各自拿到了对方的值** ⇒ `if q_dc > q_hu: 弃` 实际是
    #     "**胡更好就弃**" ⇒ 策略≈**永不弃胡**。
    #
    #   ★ 症状是**可复核的**，不是推的：修好数据后重训（`ge0..2`，5,494 局唯一），
    #     `train_decline.py` 打印 **"模型建议弃的比例 0.000"**，而**同一份数据**上
    #     无偏 **ATE(弃 − 胡) = +8.382 [+7.226,+9.536] ★** —— 两者不可能同时为真。
    #     `check_decline_sign.py` 把它钉死：模型学到的
    #     **τ = ŷ(a=1) − ŷ(a=0) = −8.986**，与数据 ATE **同号同量级**
    #     ⇒ **模型是对的，是这里的标签写反了**。
    #
    #   ⚠ 为什么以前没被发现：旧的 `decline_net.pt` / `decline_c.pt` 是在**另一个口径**下
    #     训的（它们记录的 `decline_rate_val` 分别是 **1.0** / **0.6446**），
    #     **恰好与这里的反向标签相抵** ⇒ 症状被掩盖；换成按现口径重训的 checkpoint
    #     立刻现形（0.000）。⇒ 这也是 §Z.169 那条"数据/口径没对齐就会互相掩盖"的又一例。
    #
    #   `MJ_DECLINE_LEGACY_SIGN=1` 复现**旧的反向**行为 —— **只为把这次修复本身
    #   做成同 seed 配对测量**（`PREREG_DECLINE_SIGN.md`），**不要用于生产**。
    #
    #   ⚠⚠ 第 190 轮踩到：**`cmd` 的 `set VAR=1 && cmd` 会把值设成 `"1 "`（带尾空格）**，
    #     于是 `== "1"` 恒为 False ⇒ 开关**静默失效**，两个臂给出**逐字相同**的 Δμ。
    #     症状是"不同策略却同结果" ⇒ 一律 `.strip()`，并且启动时用
    #     `set "VAR=1"`（带引号）那种写法。
    _legacy_sign = (_os.environ.get("MJ_DECLINE_LEGACY_SIGN") or "").strip() == "1"
    obj = _DECLINE.get("obj", "mse")
    if obj == "tlearner":
        X = np.asarray([base], dtype=np.float32)
        if X.shape[1] != len(_DECLINE["mu"]):
            return None
        X = (X - _DECLINE["mu"]) / _DECLINE["sd"]
        with torch.no_grad():
            xt = torch.from_numpy(X)
            q_dc = float(_DECLINE["net"](xt, torch.zeros(1)))     # a=0 ⇒ **弃**
            q_hu = float(_DECLINE["net"](xt, torch.ones(1)))      # a=1 ⇒ **胡**
    else:
        X = np.asarray([base + [0.0], base + [1.0]], dtype=np.float32)
        if X.shape[1] != len(_DECLINE["mu"]):
            return None
        X = (X - _DECLINE["mu"]) / _DECLINE["sd"]
        with torch.no_grad():
            q = _DECLINE["net"](torch.from_numpy(X)).numpy()
        q_dc, q_hu = float(q[0]), float(q[1])                      # 行 0 = a=0 = 弃
    if _legacy_sign:
        q_hu, q_dc = q_dc, q_hu
    if q_dc > q_hu:
        return {"action": "discard", "tile": tile}
    return {"action": "hu", "tile": ""}


def chi_tiles(counts, tile, meld_count=None):
    """用手上哪两张吃这张牌（返回牌码对），不可吃返回 None。财神不能参与吃。

    ★ 2026-09-18：实现搬到 `mj_core.chi_tiles`，因为**模拟器也要用同一份**。
      留这个薄壳只为兼容旧调用点（`test_policy.py` 等按 `mj_policy.chi_tiles`
      的名字导入）。

    行为变化（仅当传入 `meld_count` 时）：不再无条件取"第一个可行组合"，
    而是取**吃完后向听最小**的那个；向听并列仍取第一个。
    详见 `mj_core.CHI_PICK` 上方那段频率实测与诚实的"未验证"标注。
    """
    return _core_chi_tiles(counts, tile, meld_count)


# ---------------------------------------------------------------- 策略

def safe_piao(counts, meld_count):
    """打出财神后是否「仍然是爆头」—— 飘的唯一安全门控。

    依据（后台研究交付的技术报告）：
      · 连飘 k 次倍率 2^k，故只要 P(飘后仍能胡) > 2^-k 就值得飘；
      · 飘完仍是爆头时 P≈1，**无条件该执行**；
      · 反之（飘完只是普通听牌）就是拿确定的胡牌去赌小概率，负期望。
      · 区分度极大：爆头听 34 种，普通听牌只约 3.9 种，宽 8.7 倍。

    早先的门控写成「爆头 and 手上有财神 and 链未满」——太松，实测 S3-财飘 垫底
    （-38，均分 -3.80）。那条「弃胡是负期望」的结论测的是错误门控，不成立。
    """
    if counts[WILD] < 1 or meld_count > 4:
        return False
    c = list(counts)
    c[WILD] -= 1
    if sum(c) != 13 - 3 * meld_count:
        return False
    return is_baotou(c, meld_count)


def _dealer_blocks_decline(v):
    """庄位上**不许弃胡** —— 这条不是调参，是算术。

    规则（线上 250/250 次转移实测）：**下一局庄家 = 本局赢家**，流局庄家连庄。
    于是庄位是**可继承**的，而庄家胡一局值 24f、闲家只值 10f。

    用线上真值算（`diag_dealer_chain.py`，37 房 2926 局）：
        庄位自摸率 28.6%，闲家自摸率 22.9%
        ⇒ 每局坐在庄位的期望收入 = 0.286 × 24 = **6.86 分**
           每局坐在闲家的期望收入 = 0.229 × 10 = **2.29 分**
        ⇒ 庄位的**持有价值** ≈ 4.57 分/局（相对闲家）

    ★ 而现场测到的**结构性缺陷**（同一个脚本，逐座位建链）：

        我方连庄链均值        1.275（n=149）
        对手座位 0/1/2/3      1.462 / 1.476 / 1.451 / 1.462
        我方 − 对手 = −0.187  95% CI [−0.30, −0.06]  ⇒ **对手更长，显著**

    连庄 = 庄家连续获胜，所以连庄更短直接意味着"**我们在庄位上赢得更少**"
    （反推庄位自摸率：估计式 mean = 1/(1−p) 给出我们 ≈28.3%、对手 ≈31.6%，
    与我们直接测到的 28.6% 一致）。亏 3 个百分点 × 24 分 = **0.7 分/局**，
    而当前每局总优势只有 +0.32 分 —— 也就是说这一条比我们**全部的优势**还大两倍。

    弃胡正是让出庄位最直接的方式：庄家弃胡失败 = 对手胡 = 庄位必然易主。
    所以庄位一律**立即胡**：把确定的 24f 和庄位一起收下。
    """
    return v.dealer == v.seat


def _dealer_win_bonus(counts, meld_sets):
    """庄位专用的碰/吃门控（**目前未启用**，留作下一步实验的钩子）。

    为什么不许庄位随便副露：
      · 庄家输一局要给**三家各 8f**（闲家输只给庄 8f + 两个闲各 f），
        所以庄家的"手牌张数"比闲家更值钱 —— 副露少 2 张、暴露信息、
        且丢掉七对分支，这些代价在庄位上是加倍的；
      · 庄家本来就多一张牌、先摸牌，**速度优势已经内置**，
        不需要用副露再去换速度。

    ⚠ 刻意**不接入基因空间**（不想为一个未验证的假设动基因组语义）：
    要先用夹具同 seed 配对对照"庄位保守副露"值多少分，通过再谈上线。

    ⛔⛔ **第 119 轮：这个假设已经被已有数据否掉了 —— 不要实现它。**
    用 `hunt_seat_flips.py` 把**已有的** `peng` 档位按座位拆开：

        peng=5（≥1 摊不再碰）, chi=1 : 做庄 **−0.773 [−1.49,−0.05]** n=1500
                                        做闲   −0.136 [−0.35,+0.08] n=4500
        peng=8（听牌后不再碰）, chi=1 : 做庄 **−0.621 [−1.22,−0.02]** n=3000
                                        做闲   −0.116 [−0.29,+0.06] n=9000
        peng=4,chi=4（≥2 摊不再碰）   : 做庄 **−1.733 [−3.38,−0.08]** n=300
                                        做闲   −0.148 [−0.61,+0.31] n=900

    ⇒ **把副露改保守，代价几乎全部落在庄家身上**（两个最干净的**单因子**
    档位都是"做庄显著为负、做闲不显著"）。
    ⇒ **方向是反的**：庄位要的不是更保守，而是**至少和现在一样激进**。
    这与下面那段"手牌张数在庄位更值钱"的推理相反，而
    **`MODEL_SETTLEMENT.md §3.1`（庄家更该为速度付代价）那一侧赢了**。
    ⇒ 该测的反方向是"**庄位更激进**"（不是这个函数）。
    """
    return S.shanten(counts, meld_sets)


class Evolved(Strategy):
    """基因组驱动的策略。一个类覆盖整个搜索空间。"""

    def __init__(self, genome=None):
        self.g = dict(genome or BASELINE_GENOME)
        self.name = "G[%s]" % genome_id(self.g)
        self.desc = "基因组策略 " + genome_id(self.g)
        # ★ 诊断钩子（第 144 轮）：默认 None ⇒ 生产行为**逐字不变**。
        #   只有 `probe_feed_oracle.py` 会把它设成一个用 oracle 四家手牌
        #   挑"最不会被吃碰"的并列候选的函数，用来量这条路的上界。
        self.tie_hook = None

    # --- 出牌回合 ---
    def on_turn(self, v):
        if v.can_win():
            # ★★ 第 162 轮：把 decline 档位提到这里，因为**两处**要用它 ——
            #   规则块里的 `pre_baotou` 守卫，以及**块外**的 `safe_piao` 出口。
            _dc = int(self.g.get("decline") or 0)
            # 「杠爆」：能胡的时候**先杠再胡**，番数 ×2（手册 §番型 明确列了这一条）。
            #
            # 规则原文：`杠爆 | 爆头状态下杠，杠补牌胡 = 爆头 × 杠开 | ×4`。
            # 触发形状：手牌 = 3 个面子 + 某张牌的 4 张 + 1 张财神（14 张，已能胡）。
            #   · 现在直接胡：爆头 ×2
            #   · 先杠那 4 张（补牌后剩 3 面子 + 财神，而财神与任意补牌都能成将）
            #     -> 补牌必胡，且链 +1（杠开 ×2）再叠爆头 ×2 = ×4
            # 判据：把 4 张 t 拿掉之后，手里（10−3m 张、副露 m+1）是不是"听任意"
            # —— 那正是 is_baotou 的定义（它要的正是 13−3(m+1) = 10−3m 张）。
            # 所以这是**确定性**的 ×2，不是赌博：杠完那张补牌必胡。
            # 现有代码永远先返回 hu，这一条从来没被执行过。
            if int(self.g["gang"] or 0) >= 3 and not v.restricted \
                    and (v.wall or 0) > 20:
                for t in range(NTILE):
                    if t == WILD or v.counts[t] < 4:
                        continue
                    c2 = list(v.counts)
                    c2[t] -= 4
                    if is_baotou(c2, v.meld_sets + 1):
                        return {"action": "gang", "tile": tile_code(t)}
            # decline=2：**用确定的 ×1 换下一巡的 ×2**。
            # 触发条件：手上这 14 张已经能胡，但打掉其中一张非财神牌之后，
            # 剩下的 13 张是爆头形（听任意）—— 那下一巡摸到任何牌都胡，且 ×2。
            #
            # 依据来自真实事件流（房间 a_4e85ac7dbc5f，玩家 glm-flash）：
            #   seq=659  打 2t   打出前 14 张 可胡=True  打完是否爆头形=True
            #   seq=682  摸 7w  -> 胡，fan=2，detail=['平胡','爆头']
            # 也就是**真实玩家确实会弃掉一手确定的胡去走爆头**（走马 79 局 13 次）。
            #
            # ⚠ 关键判据：`x != 刚摸的那张`，缺了它这个基因会**无限弃胡**：
            #   在爆头形 H 上摸到任意 t，打掉 t 恰好还原成 H —— 条件永远成立，
            #   于是每巡都弃胡，直到牌墙耗尽流局。第一版就是这个 bug，
            #   所以线上玩家这么打能赢，模拟器里却把它测成 −1.43 分/局。
            #   加上这一条之后语义才对：能变成爆头的弃牌若就是刚摸的那张，
            #   说明你摸牌前**已经是**爆头形，那就该胡，不该再弃。
            #
            # 安全性：打完就是听任意，只要牌墙还够自己再摸一张就必胡
            # （牌墙保留最后 10 墩，过 3 家轮一次，故需 >= 4 张）。
            # 唯一代价是这中间别人先胡 —— 这是该权衡的全部风险。
            # decline=3：在 decline=2 之上加**威胁门控**。
            #
            # 为什么要加：线上实测（diag_decline.py，8 房 50 次弃胡事件）——
            #   兑现成我们的胡牌 68.6%（按事件算 q≈0.75）
            #   被对手抢先胡掉   31.4%
            # 也就是说"下一巡必胡"这句话在**三家人各走一巡**之后只剩约 3/4 的概率。
            # 代价是双向的：失败时不但丢了这一胡，还要付给对手。
            #   E[弃胡] = q × 2P − (1−q) × c ，P≈10.5、c≈4.2
            #   q=0.75 → 15.8−1.05 = 14.7   vs  E[直接胡] = 10.5   仍然赚，但只赚 ~40%
            # 所以值得用**可见的威胁信号**把 q 抬上去：对手副露越多，离听牌越近。
            # 门控：最多允许一家对手有 >=2 组副露。
            #
            # ★★ 2026-09-17 修复：**摸牌前就已经是爆头形时，必须直接胡，不得弃胡**。
            #
            # `audit_decline_prebaotou.py`（29 个健康房间，143 次弃胡动作）测出：
            #   37.1% 的弃胡动作发生在"**摸牌前**已是爆头形"的局面；
            #   而这些局的兑现率只有 **60.0%**，正常的弃胡局是 **84.7%**。
            #
            # 机制上这是纯亏：摸牌前已爆头 ⇒ 摸到任意 t 都**已经能胡**，
            # 且胡了就是爆头 ×2。再弃一张 x(≠t) 保持爆头，**下一巡仍然只是 ×2** ——
            # 上行空间为零，却白担一次被别人先胡的风险。
            # 原判据只有 `x != drawn_i`（防"摸 t 打 t 回到原形"的无限弃胡），
            # 漏掉了这一情形。
            #
            # ⚠ 同时修正一个测量口径错误：此前 `audit_decline_clean.py` 把
            # **副露之后那个出牌点**也算成弃胡（那里手牌形状确实"完整"，
            # 但服务端规则是 `meld is not self-draw`，**本来就不许胡**）。
            # 把那些点剔除后，真实弃胡局 97 局、兑现率 **q = 0.784**
            # （不是 0.556），**高于盈亏平衡 0.583** ⇒ decline=2 本身是正期望，
            # 与夹具结论一致（`probe_rel.py`：decline0 比现行显著更差 −0.358）。
            # ★★★ 第 159 轮：**"恒弃"是一个空档位 —— 已证明 `decline=2` 本来就是恒弃**。
            #
            # 我原本要加 `decline=8=恒弃`，动机是随机化数据的 CATE：弃 − 胡 的 ATE
            # 在**每一层**都显著为正（总体 **+7.765 [+5.655,+9.884]**，见 `analyze_decline_cate.py`），
            # 于是推断"规则只弃了 73% 的机会，还有 27% 可挖"。
            # **这个推断是错的**，而这是**构造性**的、不是测出来的：
            #   `decline_tile_for(v)`（第 915 行）逐字复刻了下面这段规则的出口 ——
            #   同一个 `for x in range(NTILE)`、同一套跳过条件
            #   (`counts[x] <= 0 or x == WILD or x == drawn_i`)、同一分支顺序（先 B 后 C）。
            #   因此当 `decline_gate_ok(v)` 为真且非 `pre_baotou` 时：
            #     规则返回"弃"  ⟺  `decline_tile_for` 非 None，**且是同一张牌**。
            #   ⇒ `decline=8` 与 `decline=2` **恒等**。
            # 实测印证：`probe_rel.py --rounds 300 -a 恒弃8=...` 的**配对 SD = 0.00**
            #   （逐局差恒为 0，不是"没测出来"）。
            # ⇒ 那个 73% **不是决策率，是兑现率 q**（弃了之后被别人先胡掉的比例，
            #   `audit_decline_clean.py` 实测 q=0.784）。
            #   也就是说**决策层面这条轴已经拉满**，剩下的唯一杠杆是**闸门**
            #   （哪些局面根本不许进这个决策）—— 见 §Z.159 与 `decline=5`。
            # 所以 8 **不进 GENES**（传了会立刻报"非法基因值"，而不是静默无效）。

            # ★★ 第 152 轮：`decline=9` = **胡/弃交给学到的模型**（与 `peng=9` /
            #   `discard=rank` 同款的"模型值"）。
            #   ⚠ 只在 `decline_gate_ok(v)` 成立时进 —— 那正是**采数据时的支撑集**
            #     （数据是用 `decline=2` 的基因组采的），出支撑集就是外推。
            #   ⚠ 模型不可用（返回 None）时**继续往下走原有规则**，不抛异常。
            if (int(self.g.get("decline") or 0) == 9
                    and decline_gate_ok(v, allow_dealer=True)):
                _act = decline_model_action(v)
                if _act is not None:
                    return _act
            # ⚠ 第 159 轮收紧：这段规则分支**只服务 2/3/4/5**。
            #   8（恒弃）与 9（模型）的语义已经完全由上面两个早退分支表达；
            #   若让它们掉进这里，`>=4 解除庄位禁令` 与"威胁<=1"这两条会
            #   **顺手改掉别的东西** —— 8 就不再是"只有弃胡比例变了"。
            #   对 0/1/2/3/4/5 逐字不变（线上现用 decline=2 在集合内）。
            # ★★★ 第 265 轮：入口集合**补上 6/7** —— 它们此前**进不来**。
            #   `GENES` 第 155 行写着 `("decline", (0,1,2,3,4,5,6,7,9))`（**合法基因值**），
            #   而第 1401/1403 行专门为 `_dc in (6,7)` / `_dc == 6` 写了语义 ⇒
            #   入口若不含 6/7，那两行就是**死代码**，而 `decline=6/7` 的真实含义会变成
            #   "掉出整个规则分支" = **完全不弃胡**，而不是文档说的"移除 `pre_baotou` 守卫"。
            #   ⚠ 第 159 轮的注释只点了 8/9，**没提 6/7**；6/7 的语义是**第 162 轮**加在
            #     分支体**内部**的，加的时候没回头改入口 ⇒ 这两个臂**从出生起就跑不出设计语义**。
            #   ⚠ 影响面：`STRATEGY.md:358` 记的 §2.4 关闭依据是**机会率上界**
            #     （B 0.008/局 + C 0.004/局 ⇒ 上界 ≤0.08 分/局），**不依赖**这两个臂是否可达
            #     ⇒ **§2.4 的关闭结论不变**；但它把 6/7 "保留为可复核记录" ⇒ 修好之后复核才有效。
            #   ⚠ 线上零风险：线上是 `decline=5`，且 6/7 此前不可达 ⇒ 无任何已部署行为改变。
            if (int(self.g.get("decline") or 0) in (2, 3, 4, 5, 6, 7)
                    and not v.restricted
                    and (v.wall or 0) - 20 >= 4
                    # 6/7 的语义是"= 5 但放开 pre_baotou 守卫" ⇒ 必须**与 5 同闸**，
                    # 否则它们会被 `threats(v) <= 1` 额外收紧，就不是"只改守卫"了。
                    and (int(self.g["decline"]) < 3
                         or int(self.g["decline"]) in (5, 6, 7)
                         or threats(v) <= 1)
                    # ★ decline>=4 = **解除庄位禁令**（见 GENES 里 decline 的说明）。
                    #   1/2/3 一律仍然禁止庄家弃胡（保持历史行为不变）。
                    and (int(self.g["decline"]) >= 4
                         or not _dealer_blocks_decline(v))):
                drawn_i = tile_index(v.drawn) if v.drawn else -1
                # 摸牌前的手牌 = 现在这 14 张去掉刚摸的那张
                pre_baotou = False
                if drawn_i >= 0:
                    c13 = list(v.counts)
                    c13[drawn_i] -= 1
                    try:
                        pre_baotou = bool(is_baotou(c13, v.meld_sets))
                    except Exception:                            # noqa: BLE001
                        pre_baotou = False
                # ★★★ 第 162 轮：`pre_baotou` 守卫**从未被配对检验过** ——
                #   而 goal §2.4/§5 明确说「×2 vs ×4/×8 **不能人为规定哪个更好**」。
                #
                #   机制拆开看，这道守卫对**两条分支**的效果完全不同：
                #     · **分支 B**（打一张别的牌、保持爆头）：
                #       ⚠⚠ **第 359 轮更正：下面这句「上行空间确实为零」是错的。**
                #       原文写的是"你已经爆头（×2），再弃一张保持爆头下一巡仍是 ×2" ——
                #       那句话描述的是 **`pre_baotou == True`** 的局面，
                #       而**那种局面根本走不到这一行**（上面第 1415 行的守卫已经 `return hu`）。
                #       ⇒ 分支 B **真正执行**的点全都满足 `pre_baotou == False`，
                #         也就是：**此刻手上是平胡（×1）**，打掉某张 x 之后才变成爆头
                #         ⇒ **上行空间就是 ×2**，不是零。
                #       ★ 证据（真场，`analyze_decline_field.py`，39,464 局 / 1,763 个弃胡事件）：
                #         `missed_detail.hand` 重算"若当场胡"的番 = **1.1429**，
                #         弃胡后真胡到手的番 = **2.2818**，**比值 1.996**
                #         —— 正好是爆头那一档，一次干净的 ×2。
                #         定价：弃胡 **+0.3212 分/局**（闲 +6.33 / 庄 +16.26 per 事件）；
                #         牌墙分层单调且**每一档都为正**（+2.31 / +5.46 / +8.35）。
                #       ⇒ **守卫是对的**（它封掉的是真正零上行的 `pre_baotou` 局面），
                #         **行为是对的**，**只有这句注释是错的**。不要照着注释去改代码。
                #     · **分支 C**（`safe_piao` 打财神 ⇒ 财飘 +1 ⇒ 下次胡 ×4）：
                #       **有真实上行空间**，却被"已是爆头就直接胡"**一起关掉了**。
                #       ⚠ 但注意：分支 C 在 `decline=5` 下**同样只可能出现在
                #         `pre_baotou == False` 的点**（`pre_baotou` 时守卫已先返回胡）。
                #   证据只支持前半：`audit_decline_prebaotou.py` 测的是**兑现率**
                #   （60.0% vs 84.7%），那是**分支 B 的风险**，**不是**分支 C 的收益。
                #   ⇒ 用一个没被测过的守卫把一条有上行的路线也封了，正是 §5 要避免的事。
                #   ★ 第 359 轮补：分支 C 的上行（×4）**没有**被证伪；但本轮量出
                #     分支 B 自己的上行（×2）已经值 +0.32 分/局 ⇒ **不要**用
                #     "放开守卫"（6/7）去换它 —— 那会同时放开真正零上行的
                #     `pre_baotou` 局面。（6/7 的机会率上界见 `GENES` 里 decline 那段。）
                #
                #   `decline=6` = **两条都放开**（完整移除守卫，用来量守卫的总代价）
                #   `decline=7` = **只放开分支 C**（外科手术式，直接问 §2.4 那个问题）
                if pre_baotou and _dc not in (6, 7):
                    return {"action": "hu", "tile": ""}      # ← 已是爆头，直接胡
                if (not pre_baotou) or _dc == 6:
                    for x in range(NTILE):
                        if v.counts[x] <= 0 or x == WILD or x == drawn_i:
                            continue
                        c2 = list(v.counts)
                        c2[x] -= 1
                        if is_baotou(c2, v.meld_sets):
                            return {"action": "discard", "tile": tile_code(x)}
                # （`pre_baotou` 且 `_dc == 7` ⇒ 跳过分支 B，直接落到下面的 safe_piao）
            # 弃胡博财飘，两个条件缺一不可：
            #   ① 飘完仍是爆头（safe_piao）—— 此时下一次摸牌任意牌都胡，飘是白赚 ×2
            #   ② 牌墙还够自己再摸一张 —— 少了这条，牌墙将尽时飘会把一手
            #      确定的胡牌换成流局 0 分，这是「安全飘」唯一可能亏钱的路径
            # 牌墙保留最后 10 墩（20 张）不摸；飘后要过 3 家才轮回自己，故需 >= 4 张。
            # ⚠ 第 159 轮：safety_piao 只属于 1~5 号语义（8/9 由上面的早退分支全权表达）。
            # ★ 第 162 轮：加上 **6/7** —— 它们正是"允许在 pre_baotou 时走分支 C"的档位，
            #   所以这一支**必须**对它们开放，否则 7 就等于没改。
            if (_dc in (1, 2, 3, 4, 5, 6, 7)
                    and not v.restricted
                    and (v.wall or 0) - 20 >= 4
                    and not _dealer_blocks_decline(v)
                    and safe_piao(v.counts, v.meld_sets)):
                return {"action": "discard", "tile": "白"}
            return {"action": "hu", "tile": ""}

        # 杠：两种通路，都由 gang 基因控制
        #   gang=1 只暗杠（历史语义，保持不变 —— 之前的实验结论继续可比）
        #   gang=2 暗杠 + 补杠
        # ★★ 第 410 轮：曾试图加 `gang=3`＝"承诺七对时不杠"（保豪华七对），
        #   **实测证明那是死代码，已撤回**，理由记在这里免得再写一次：
        #     `shanten_chiitoi` 把 4 张同牌只算**1 对**（`shanten_ref.py:367`），
        #     所以"4 张 + 5 对"这种手牌的七对向听是 **1**（而不是 −1）
        #     ⇒ `commit_chiitoi` 返回 **False** ⇒ **握有 4 张同牌时永远不可能"承诺七对"**
        #     ⇒ 那条"承诺了就不杠"的分支**结构上永远不会触发**。
        #   ⚠ 顺带暴露一个**真 bug**（见 `shanten_ref.shanten_chiitoi` 的注释）：
        #     同一手牌 `can_win=True`（且是**豪华七对 ×4**）而 `shanten_chiitoi=1`
        #     —— 两者口径分叉。**修它之前，一切"为豪华七对让路"的改动都是空中楼阁。**
        # 补杠 = 已碰出的面子 + 手上第 4 张。它的本质是**用一张几乎无用的牌
        # （同牌另外 3 张都在自己的面子锁死，它只能靠两侧搭子成顺）换一次摸牌，
        # 顺带 +1 链。线上对手杠率 0.03~0.09 次/局，而补杠机会实测 0.094 次/局
        # （probe_kong_ops.py）—— 对手基本都吃掉了，我们此前一次不做。
        if self.g["gang"] and not v.restricted and (v.wall or 0) > 20:
            for t in range(NTILE):
                if t != WILD and v.counts[t] >= 4:
                    return {"action": "gang", "tile": tile_code(t)}
            if int(self.g["gang"]) >= 2:
                for t in peng_tiles(v):
                    if v.counts[t] >= 1:
                        return {"action": "gang", "tile": tile_code(t)}

        forbid = set(range(NTILE)) - v.allowed_discards()
        mode = self.g["discard"]
        if mode == "value":
            unseen = unseen_from_view(v) if self.g.get("ukeire") else None
            # 自己还能摸几次：牌墙保留最后 10 墩（20 张）不摸，过 3 家轮一次
            draws_left = max(1, ((v.wall or 20) - 20) // 4)
            t = best_discard_value(v.counts, v.meld_sets, forbid,
                                   unseen=unseen, draws_left=draws_left)
            if t is None:
                return None
            return {"action": "discard", "tile": tile_code(t)}
        if mode == "net":
            t = best_discard_net(v, forbid)
            if t is None:
                return None
            return {"action": "discard", "tile": tile_code(t)}
        if mode in ("rank", "rank2"):
            # ★ 第 145/150 轮：学到的「候选排序器」，输入是 **360 维完备编码**
            #   （`net` 那一支用的是汇总过的 84 维）。
            #   `rank2` = **只在 ukeire 也并列时**才让模型出手（规则的裁决在那一层
            #   是按牌码顺序挑的，没有依据）⇒ 改动面最小、风险最低。
            #   缺失模型时 `best_discard_rank` 内部优雅退回同一套基因的规则。
            t = best_discard_rank(v, forbid,
                                  route=int(self.g.get("route") or 0),
                                  feed=int(self.g.get("feed") or 0),
                                  use_unseen=1 if self.g.get("ukeire") else 0,
                                  only_ties=(mode == "rank2"))
            if t is None:
                return None
            return {"action": "discard", "tile": tile_code(t)}
        if mode == "baotou" or mode == "baotou2":
            unseen = unseen_from_view(v) if self.g.get("ukeire") else None
            fn = best_discard_baotou2 if mode == "baotou2" else best_discard_baotou
            t = fn(v.counts, v.meld_sets, forbid, unseen=unseen)
            if t is None:
                return None
            return {"action": "discard", "tile": tile_code(t)}
        if mode in ("shanten", "shanten2", "shanten_fan"):
            unseen = unseen_from_view(v) if self.g.get("ukeire") else None
            t = best_discard_shanten(v.counts, v.meld_sets, forbid,
                                     unseen=unseen,
                                     route=int(self.g.get("route") or 0),
                                     feed=int(self.g.get("feed") or 0),
                                     wait_aware=(mode == "shanten2"),
                                     fan_aware=(3 if mode == "shanten_fan" else 0),
                                     # ⚠ 用 getattr 兜底：线上客户端一次跑 8 小时，
                                     #   若哪天有别的构造路径（copy/反序列化）绕过
                                     #   __init__，硬取属性会把整局打成异常。
                                     tie_hook=getattr(self, "tie_hook", None))
            if t is None:
                return None
            return {"action": "discard", "tile": tile_code(t)}
        if mode == "heuristic":
            _t, code = best_discard(v.counts, v.meld_sets, forbid=forbid)
            if code is None:
                return None
            return {"action": "discard", "tile": code}
        # first：复刻官方模板
        if v.restricted and v.drawn:
            return {"action": "discard", "tile": v.drawn}
        if not v.hand:
            return None
        return {"action": "discard", "tile": v.hand[0]}

    # --- 碰/吃窗口 ---
    def on_window(self, v, phase):
        if v.restricted:                      # 抓打圈：非豁免方不能吃碰明杠
            return PASS
        # 承诺七对时禁止副露 —— 吃碰会亲手拆掉七对路线
        #
        # ★★★ 第 126 轮修一个**部署接线缺陷**：`peng=9`（碰不碰交给学到的 Q）
        #   原先排在这个门**之后** ⇒ **模型永远够不着被 route 否决的碰机会**，
        #   于是"模型决定碰不碰"实际上是"模型只在 route 放行时决定"，
        #   评测出来的 Δμ≈0（§Z.109(e)）有一半是**接线造成的假象**。
        #   而训练数据是对的：`explore_peng_shard` 的包装器在**策略外层**，
        #   它记录并改写不受 route 影响 ⇒ 反事实本来就在数据里。
        #   ⇒ `peng=9` 时**不由 route 硬规则否决**，交给 Q 决定。
        if (self.g.get("route") and commit_chiitoi(v.counts, v.meld_sets)
                and int(self.g.get("peng") or 0) != 9):
            return PASS
        if phase == "response_peng" and self.g["peng"] and v.last_discard:
            t = tile_index(v.last_discard)
            if t != WILD and v.counts[t] >= 2:
                pv = int(self.g["peng"])
                # ★★★ 第 124 轮：`peng=9` = **碰不碰交给学到的 Q**（强化学习策略）。
                #   为什么单独占一个档位：`make_strategy` 对未知名字会回落到
                #   `Evolved(parse_genome(...))` ⇒ 用基因值接入就**不需要注册新类**，
                #   全项目现有评测工具（配对夹具 / 座位拆分 / run_shards）立即能用。
                #   ⚠ 模型缺失时 `decide()` 返回 None ⇒ **退回本函数原本的逻辑**，
                #     绝不抛异常（线上循环用同一个工厂函数）。
                if pv == 9:
                    legal = [34, 38]
                    if v.counts[t] >= 3 and (v.wall or 0) > 20:
                        legal.append(36)
                    try:
                        import q_peng_policy as QP
                        pick = QP.decide(v, legal)
                    except Exception:                             # noqa: BLE001
                        pick = None
                    if pick == 38:
                        return PASS
                    if pick == 36:
                        return {"action": "gang", "tile": v.last_discard}
                    if pick == 34:
                        return {"action": "peng", "tile": v.last_discard}
                    # pick is None ⇒ 落到下面原有的基因组逻辑（优雅降级）
                # ★★★ 2026-09-17 新增：**明杠优先于碰**（被支配决策，属"纯增益"类）
                #
                # 发现（`audit_minggang.py`，29 个健康房间 2072 局，事件流真值）：
                #     我方  杠合计 145 = 暗杠 20 + **明杠 0** + 补杠 125
                #     对手  杠合计 311 = 暗杠 42 + **明杠 150** + 补杠 119
                #   即**我们一次明杠都没做过**，而对手人均 0.0241 次/局。
                #   根因：本函数此前只有 peng / chi 两个分支，全文没有 `response_gang`。
                #
                # 为什么明杠**严格优于**碰（不是取舍，是支配关系）：
                #   · 手上已有 3 张 t、别人打出第 4 张时：
                #       碰  = 用 2 张 → 得 1 面子，**不摸牌**，且**剩下一张 t 永久死牌**
                #             （4 张全在场上：2 张在自己面子、1 张已打出、剩 1 张在手，
                #              再也凑不成对子）
                #       明杠 = 用 3 张 → 得 1 面子，**且摸一张补牌**，链 +1（可叠杠开 ×2）
                #   · 规则明确 **禁止抢杠胡**（guide §规则：「只能自摸…禁止抢杠胡」），
                #     所以明杠**没有**额外风险 —— 与"弃胡博爆头"那种赌博完全不同。
                #   · guide 亦明确「**碰（含明杠）窗口先于吃窗口**」、
                #     响应阶段动作为「可碰/吃/明杠/pass」⇒ 服务端接受在此窗口提交 gang。
                #
                # ⚠ 必须排在所有 `pv` 速率闸**之前**（与 peng=7 同一个教训）：
                #   否则它会被"已有 N 摊就不碰"的闸吞掉，表现成"看起来更激进、
                #   实际更保守"。
                # 门控：`gang >= 1` 且 `peng >= 1`（明杠也是一种"接别人打出的牌"，
                #       所以尊重 peng=0「不接牌」的语义）；牌墙 >20（最后 10 墩禁杠）。
                if (int(self.g.get("gang") or 0) >= 1 and pv >= 1
                        and v.counts[t] >= 3 and (v.wall or 0) > 20):
                    return {"action": "gang", "tile": v.last_discard}
                # ⚠ 顺序至关重要：**"更激进"的档位必须在速率闸之前**。
                #   7 曾经被写成和 1 同一个分支、且排在下面这个 `pv >= 4` 闸之后，
                #   于是 7 反而变成"1 摊后就不碰"（比 1 更保守）——
                #   实测表现为 `peng7chi7` 与 `peng5chi1` 逐局同分。
                #   这类"看起来更激进、实际更保守"的错最难靠看代码发现，
                #   只能靠不同档位出现**不合理的相同结果**来暴露。
                if pv == 7:
                    return {"action": "peng", "tile": v.last_discard}
                if pv == 8:
                    # ★ peng=8（2026-09-17 新增，依据杭州麻将主流打法经验）：
                    #   「已经进入听牌阶段，那就不要随意碰牌了，以免耽误了后续的胡牌」
                    #   （《杭州麻将高手打牌思路》/《杭州麻将怎么玩 碰牌吃牌有一套》）
                    #   机制上自洽：副露跳过摸牌（diag_meld_draw_cost.py：副露
                    #   0.015→1.759 时全场巡数 40.69→36.40），而听牌之后唯一的
                    #   进度来源就是**摸到那张牌** —— 此时副露纯亏一摸。
                    #
                    #   ⚠ 判据为什么是"**碰完还在等牌**"而不是"碰之前已听牌"：
                    #     ① 看"碰之前"几乎永不触发 —— 听牌时手上通常没有可碰的对子；
                    #     ② "碰进听"在数学上近乎不可能：碰后是 11 张 + 1 摊，
                    #        需要 3 个面子 + 1 将；若这 11 张里含被碰的那对，
                    #        剩下 9 张就要装下 9 张面子 **加** 2 张将 = 11 张，装不下。
                    #        （穷举检索也确认没有这种局面，见 probe_meld_gates.py）
                    #     所以真正可执行的语义是：**碰完仍未听牌（after>0）就不碰** ——
                    #     这正是"别为了一个面子把摸牌机会用掉"，与文章一致。
                    #   （与 pv=4/5 的数量闸不同：8 只看时机，不看已有几摊）
                    before8 = S.shanten(v.counts, v.meld_sets)
                    if before8 <= 0:
                        return PASS
                    c8 = list(v.counts)
                    c8[t] -= 2
                    after8 = S.shanten(c8, v.meld_sets + 1)
                    if after8 > 0:
                        return PASS
                    return {"action": "peng", "tile": v.last_discard}
                if pv >= 4 and v.meld_sets >= (2 if pv == 4 else 1):
                    return PASS               # 副露速率旋钮
                if pv == 3:
                    # 手上有白板就不碰：白板是爆头路线（4 自然面子 + 余白）的钥匙
                    if v.counts[WILD] > 0:
                        return PASS
                    return {"action": "peng", "tile": v.last_discard}
                if pv == 7:
                    # 见上：7 已提前返回，这里不会到达。保留分支只为可读性。
                    return {"action": "peng", "tile": v.last_discard}
                if pv in (1, 4, 5):
                    return {"action": "peng", "tile": v.last_discard}
                # peng=2 / peng=6：只在「碰后向听不变差」时才碰。
                # 碰 = 用手上两张凑一组面子、副露 +1，手牌少 2 张。
                # 若那对是唯一的将牌，碰完就再没有对子可用作将，向听可能倒退。
                #
                # ★ peng=6 是 2026-09-17 依据全场普查新增的门控档位（见 GENES 的说明）：
                #   与 peng=2 同源，但在基因空间里与"碰保守"分开命名，便于线上 A/B
                #   把"向听不倒退"这条门控单独拿出来与"有机会就碰"对拍。
                before = S.shanten(v.counts, v.meld_sets)
                c2 = list(v.counts)
                c2[t] -= 2
                if S.shanten(c2, v.meld_sets + 1) <= before:
                    return {"action": "peng", "tile": v.last_discard}
        if (phase == "response_chi" and self.g["chi"] and v.last_discard
                and v.meld_chi < 2):          # 吃最多 2 摊（服务端也会拒第 3 次）
            cv = int(self.g["chi"])
            # ★ cv=7 = 最大激进：**只受规则硬上限约束**（最多吃 2 摊），
            #   不再看"已有几组副露"。而 cv=3/4 是按**总副露数**设的上限
            #   （3 → 已有 2 摊就不吃；4 → 已有 1 摊就不吃），与吃的次数无关。
            #   两者不是同一个旋钮：若已有 2 组碰，cv=3/4/7 都会拒绝；
            #   若只有 0~1 组碰，cv=3/4 仍可能拒绝而 cv=7 会接受。
            #   要测"副露量到底该往哪个方向拧"，必须把这两件事分开。
            #   ⚠ 8 必须**排在这个闸之前**：cv>=3 会被下面的速率闸吞掉，
            #     那就变成"1 摊后就不吃"（比 7 更保守）—— 正是 peng=7 当年
            #     踩过的同一个坑（见 on_window 碰分支的注释）。新档位一律前置。
            if cv == 8:
                # ★ chi=8：同 peng=8 的"吃完还没听就不吃"门控（判据同前）。
                #   吃与碰同理：吃也不是自摸（服务端明说 "meld is not self-draw"），
                #   吃掉的是一次摸牌机会，换一个面子。
                if S.shanten(v.counts, v.meld_sets) <= 0:
                    return PASS
                pair0 = chi_tiles(v.counts, v.last_discard, v.meld_sets)
                if not pair0:
                    return PASS
                c8 = list(v.counts)
                for x in pair0:
                    xi = tile_index(x)
                    if xi >= 0:
                        c8[xi] -= 1
                if S.shanten(c8, v.meld_sets + 1) > 0:
                    return PASS
                return {"action": "chi", "tile": v.last_discard, "tiles": pair0}
            if cv != 7 and cv >= 3 and v.meld_sets >= (2 if cv == 3 else 1):
                return PASS                   # 副露速率旋钮
            if cv == 2 and v.counts[WILD] > 0:
                return PASS                   # 同 peng=3：留着白板走爆头
            pair = chi_tiles(v.counts, v.last_discard, v.meld_sets)
            if pair:
                # ★ chi=6：向听门控 —— 吃完必须让向听严格变小才吃。
                # 理由同 peng=6（吃同样消耗手上两张、关闭七对分支）。
                if cv == 6:
                    before = S.shanten(v.counts, v.meld_sets)
                    c2 = list(v.counts)
                    for x in pair:
                        xi = tile_index(x)
                        if xi >= 0:
                            c2[xi] -= 1
                    if S.shanten(c2, v.meld_sets + 1) >= before:
                        return PASS
                return {"action": "chi", "tile": v.last_discard, "tiles": pair}
        return PASS


def build_genome(name):
    """按编码串构造策略；也接受旧的具名策略名。"""
    return Evolved(parse_genome(name))
