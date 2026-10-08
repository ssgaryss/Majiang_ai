#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""杭州麻将 RL 的**动作空间**：候选枚举 + 候选/状态编码（训练与评估共用的**唯一实现**）。

为什么要把它单独钉下来
----------------------
本仓最贵的一类 bug 是"两份实现分叉"（`mj_core.chi_tiles` 的教训）与
"静默退化"（策略返回 gang 被引擎拒掉 → `_fallback_discard` 随机打一张）。
RL 里这两件事的后果比规则策略里更重：**记录下来的 (状态, 动作) 与引擎真正执行的
动作不一致时，策略梯度就是在优化另一个策略**，而且 loss 曲线看不出来。

所以本模块：
  1. 枚举"引擎**确实会执行**的动作"（逐条对齐 `mj_sim.Game` 的判定，见下面每一处注释）；
  2. 把 `Game._fallback_discard` 当作**审计哨兵** —— 训练器那边用 `AuditedGame`
     统计它被触发的次数；只要 >0，就说明本模块的合法集与引擎分叉了（`--selfcheck` 会断言）。

===========================================================================
动作空间（离散动作数 = 1(胡) + ≤3(杠) + ≤14(弃牌)，逐决策点**变长**）
===========================================================================
一次决策点分两类，由引擎分别调用：

A. `on_turn(v)`  —— 轮到自己摸完牌（`phase == "draw"`）时的**回合决策**
   候选：
     · `hu`    : 0 或 1 个。合法 ⇔ `can_win(v.counts, v.meld_sets)` **且 `v.drawn` 非空**。
       ★★ 这里有一条**必须对齐线上**的额外约束（本模块最重要的一个决定）：
       `mj_sim.Game.try_hu` 只检查 `can_win`，**不检查"是否自摸的那张"**；
       而线上服务端明确拒 "meld is not self-draw"（`mj_bot.View.can_win` 的注释，
       有 `reject_dump.jsonl` 证据）。即：吃/碰之后手牌成胡形时，**模拟器允许胡、线上不允许**。
       规则策略因为走 `View.can_win()`（它要求 `drawn` 非空）永远撞不到这个缺口，
       但**一个可学习的策略会** —— 那等于偷偷把奖励调高。
       ⇒ 本模块把 `hu` 限制为 `bool(v.drawn) and can_win(...)`，**与线上一致**。
     · `gang`  : 0~3 个，每个对应一张牌 `t`。合法 ⇔ 引擎 `_try_gang` 会返回 True：
         - `(v.wall or 0) > DEADWALL(=20)`（牌墙最后 10 墩禁止杠）
         - `t != WILD(33)`（财神不能杠）
         - 且 (`counts[t] >= 4` 暗杠) 或 (`counts[t] >= 1` 且已有 `t` 的**碰**出面 —— 补杠)
     · `discard`: 手牌里每一种**还有剩**的牌各一个候选（≤14 张）。
       合法 ⇔ `counts[t] > 0`；**抓打圈**下（`v.restricted`，即圈内且本人不是打财神者）
       只允许打**刚摸的那张**。
   动作总数 = 1 + n_gang + n_distinct_tiles ∈ [1, 15]（实测最大值 15：14 张全不同）。

B. `on_window(v, phase)` —— 别人打出一张牌后的**响应窗口**
   phase 只有两种：`"response_peng"` / `"response_chi"`（`mj_sim._ask_window`）。
   候选**恰好 2 个**：
     · `pass`  : 返回 `None`
     · `meld`  : 返回 `{"action": "peng"}` 或 `{"action": "chi"}`（必须与 phase 同名）
   为什么只有 2 个：
     · 引擎**只在规则允许时才问**（碰：手上 ≥2 张；吃：下家且吃摊<2 且 t<27 且能成搭）
       ⇒ 不需要再判一次合法性；
     · 「吃哪两张」**不是一个可选动作** —— 引擎自己用 `mj_core.chi_tiles` 定，
       策略无从选择（`Game._chi_pair` 的注释就是为了这一点）；
     · 杭州麻将**不能点炮**，窗口里没有"胡"这个选项。

已知的一处**非马尔可夫近似**（照实写下来，别当它不存在）：
`Game._poll_window` 按座位顺序询问，**第一个答应的座位赢得这张牌**。
所以当两家都能碰同一张牌时，靠前的座位可能"抢走"我们的碰 —— 我们记录了决策点
（采样概率也照记），但那一手的动作其实**没有生效**。线上是"1 秒窗口内各家并发提交"，
与"座位序优先"不完全一样。`--window-audit` 会统计"同一张牌还有别家能碰"的频率，
便于判断这条近似的量级。

===========================================================================
编码
===========================================================================
状态 `state_dim = features_full.F(360) + features_full.context(10) = 370`
  · 硬约束（与 `features.py`/`features_full.py` 同一条）：只读 `View.snap`
    —— 也就是线上 `GET /api/games/{id}/state` 的快照，**不许碰模拟器专属属性**。
候选 `turn`：13 维；候选 `window`：8 维（下面的 `_turn_cand_feat` / `_win_cand_feat`）。
所有分量都被压到 ~[0,1]，因此**不需要标准化**（也就不存在"训练侧 mu/sd 没带到
推理侧"那种分叉）。
"""
import os
import sys

sys.path.insert(0, ".")

from mj_core import (NTILE, WILD, can_win, chi_tiles, fan_multiple, is_baotou,
                     tile_code, tile_index)
import shanten_ref as S
import features_full as FF

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- 常量
DEADWALL = 20                     # 与 mj_sim.DEADWALL 同值；下面 `wall > DEADWALL` 是杠的条件
# ★★ 第 550 轮：**对手副露旁路**的显式开关（默认关）。
#   为什么必须是开关而不是直接开：
#     开启后 `STATE_DIM` 从 370 变 376 ⇒ **线上现役的 `rl_bc6`（370 维）会加载不了** ✗
#     ⇒ 默认关 ⇒ 线上与既有脚本**逐字不受影响**；要做"表示"实验时用环境变量显式打开。
#   用法：`$env:MJ_CTX2='1'`（训练进程会把它传给子进程，两侧维度因此一致）。
#   ⚠ 维度必须在**同一进程组内**一致：训练 / 评估 / 线上要用同一个开关值，
#     否则会在加载检查点时报维度不匹配 —— 那是**响亮地失败**，不是静默，可以接受。
USE_CTX2 = bool(int(os.environ.get("MJ_CTX2", "0") or 0))
STATE_DIM = FF.F + FF.CTX + (FF.CTX2 if USE_CTX2 else 0) + (FF.CTX_DEALER if FF.USE_CTX_DEALER else 0) + (FF.CTX_WALL if FF.USE_CTX_WALL else 0)   # 370；+CTX2=376；+庄位/牌墙 各 +1
CAND_TURN_DIM = 14
CAND_WIN_DIM = 8
# 有效牌枚数的**编码上限**（`f[13] = min(ukeire, UK_CAP)/UK_CAP`）。
#   36 = 一个明显够用的上界：一副牌 34 种、每种最多 4 张，理论最大有效牌枚数
#   远小于 36（13 张手牌最多也就能"改善"有限几种牌）。取整数是为了让
#   `f[13]` 在常见区间里有足够分辨率（典型 ukeire 8~24 ⇒ 0.22~0.67）。
UK_CAP = 36.0
MAX_TURN_CAND = 20                # 理论最大 15（1 胡 + 14 弃）；留余量，越界会断言
# 窗口候选的**最大个数**（缓冲区宽度，不足的补零，实际个数由 `len(cands)` 另存）。
#   ★ 第 363 轮：2 → **3** —— 碰窗口在手上 ≥3 张时多一个「明杠」候选。
#   ⚠ 这只是**缓冲区宽度**：`CandNet.head_win` 是**逐候选独立打分**
#     （`Linear(hidden+cand_dim → hidden) → Linear(hidden → 1)`，再按 mask 归一），
#     **不依赖候选个数** ⇒ 改这个常量不会改变网络结构、也不会让旧 checkpoint 失效。
#     （踩过：只改 `window_candidates` 不改这里 ⇒ 子进程
#      `could not broadcast input array from shape (3,8) into shape (2,8)`。）
N_WIN_CAND = 3                    # pass / meld / **明杠**
K_HU, K_GANG, K_DISCARD = "hu", "gang", "discard"
K_PENG, K_CHI, K_PASS = "peng", "chi", "pass"


# ---------------------------------------------------------------- 小工具
def unseen_counts(v):
    """还没露过面的张数（4 − 自己手牌 − 四家弃牌 − 四家副露），长度 34。

    与 `features_full.features` 的"未见牌"段**同口径**（那里是 /4 之后的），
    这里要原始张数供候选特征用。`mj_policy.unseen_from_view` 是另一份手抄实现，
    但本函数只用于**候选特征**，不影响合法性判定，所以不构成"两份实现分叉"的风险点。
    """
    dead = [0] * NTILE
    snap = v.snap or {}
    for row in (snap.get("discards") or []):
        for code in (row or []):
            try:
                dead[tile_index(code)] += 1
            except (ValueError, TypeError, AttributeError):
                pass
    for row in (snap.get("melds") or []):
        for m in (row or []):
            for code in ((m or {}).get("tiles") or []):
                try:
                    dead[tile_index(code)] += 1
                except (ValueError, TypeError, AttributeError):
                    pass
    c = v.counts
    return [max(0, 4 - dead[t] - c[t]) for t in range(NTILE)]


def peng_tile_set(v):
    """本人**碰出**的牌索引集合（补杠只能补碰出来的面子，杠不算）。

    与 `mj_policy.peng_tiles` 同语义；`--selfcheck` 里对拍过两者的集合相等。
    """
    out = set()
    snap = v.snap or {}
    melds = snap.get("melds")
    row = []
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


def _sh(counts, melds):
    try:
        return S.shanten(counts, melds)
    except Exception:                                            # noqa: BLE001
        return 8


def _bt(counts, melds):
    try:
        return 1.0 if is_baotou(counts, melds) else 0.0
    except Exception:                                            # noqa: BLE001
        return 0.0


# ★★★ 第 374 轮新增：**有效牌枚数**（ukeire）—— 规则策略的**第二主键**。
#
# 为什么它必须在候选特征里（这是本轮最重要的发现之一）：
#   `mj_policy.best_discard_shanten` 的裁决顺序是
#       **① 向听最小 → ② 有效牌最多 → ③ 按牌码**（`mj_policy.py:438-510`）。
#   而本轮之前的 13 维候选特征里**只有向听（f[7]）和"这张牌还剩几张"（f[8]）**，
#   **完全没有 ukeire** ⇒ 网络**看不到老师用来分胜负的那个键**。
#   于是 BC 只能学到 71% 一致率，配对评估 **Δμ = −0.634 [−1.172,−0.096]**
#   （BC 连训练器自己的验收标准「|Δμ|<0.15 或 CI 含 0」都没过）。
#   ⇒ 这不是"网络不够大"，是**输入里没有那个信息**。
#
# 口径必须与老师**逐字一致**：老师调 `S.ukeire(c, meld_count, unseen=unseen)[0]`，
#   而 `unseen` 来自 `mj_policy.unseen_from_view`；这里的 `unseen_counts` 是同一算法
#   的另一份实现，`_selftest` 里会**逐个决策点对拍**（不一致就报出来）。
def _uk(counts, melds, unseen):
    try:
        return float(S.ukeire(counts, melds, unseen=unseen)[0])
    except Exception:                                            # noqa: BLE001
        return 0.0


def drawn_index(v):
    d = getattr(v, "drawn", "") or ""
    if not d:
        return -1
    try:
        return tile_index(d)
    except (ValueError, TypeError):
        return -1


def encode_state(v):
    """状态向量：`F(360) + CTX(10) = 370`；开了 `MJ_CTX2` 时再加 `CTX2(6) = 376`。

    ⚠ 维度由 `USE_CTX2` 决定，而它来自环境变量 ⇒ **同一进程组内必须一致**
      （训练会用同一个环境启动子进程）。做成开关而不是直接改维度，是为了
      **不让线上现役的 370 维模型失效**。
    """
    out = list(FF.features(v)) + list(FF.context(v))
    if USE_CTX2:
        out += list(FF.context_extra(v))
    return out


# ---------------------------------------------------------------- A. 回合决策
# ★★★ 第 769 轮：**弃胡门控**（env 门控 `MJ_DECLINE_GATE`，**默认关** ⇒ 线上逐位不变 ✓）
#   动机：模型臂 `missed_hu` = 4.32 次/房 vs 规则臂 2.18 ✗（约 2 倍，§Z.1152）
#   仓内"正确弃胡"已实测正期望：CATE 弃−胡 **+7.765 [+5.655,+9.884]** ✓ ＋ 三道闸 `decline_gate_ok` ✓
#   行为（仅 `MJ_DECLINE_GATE=1`）：可胡且闸门**不成立** ⇒ **只保留 `hu`**（强制胡 ✓）
#   ⚠ 默认关 ⇒ 候选与顺序逐位不变 ⇒ 线上零风险 ✓；夹具对此瞎（safe_piao 0/1200）⇒ 判据只能用 L2 ✓
def _decline_gate_on():
    import os
    return (os.environ.get("MJ_DECLINE_GATE") or "0").strip() == "1"


def _decline_gate_ok(v):
    try:
        from mj_policy import decline_gate_ok
        return bool(decline_gate_ok(v, allow_dealer=True))
    except Exception:                                            # noqa: BLE001
        return False


def turn_candidates(v):
    """★ 弃胡门控包装器（默认关 ⇒ 透传 ✓）。"""
    cands, feats = _turn_candidates_impl(v)
    if not _decline_gate_on():
        return cands, feats
    ki = None
    for i, c in enumerate(cands):
        if c.get("kind") == K_HU:
            ki = i
            break
    if ki is None or _decline_gate_ok(v):
        return cands, feats
    return [cands[ki]], [feats[ki]]

def _turn_candidates_impl(v):
    """返回 `(cands, feats)`：候选动作列表 + 逐候选特征（都已压到 ~[0,1]）。

    `cands[i]` 是 dict：`{"kind": "discard"/"hu"/"gang", "tile": <index or None>}`。
    调用方用 `to_engine_action()` 转成引擎要的 act 字典。
    """
    c = list(v.counts)
    melds = int(v.meld_sets or 0)
    unseen = unseen_counts(v)
    dr = drawn_index(v)
    god = v.god or {}
    chain = int(god.get("chain_count") or 0)
    piao = int(god.get("piao_count") or 0)
    cands, feats = [], []

    # ---- hu ----
    # ★★ 与线上对齐：必须"有刚摸的那张"才允许宣胡（`View.can_win()` 同义）。
    if dr >= 0 and can_win(c, melds):
        c2 = list(c)
        c2[dr] -= 1
        bt_now = _bt(c2, melds)                    # 与 Game.try_hu 的爆头判定同口径
        fan = 0.0
        try:
            fan = float(fan_multiple(c, melds, chain, piao, bool(bt_now)))
        except Exception:                                        # noqa: BLE001
            fan = 0.0
        f = [0.0] * CAND_TURN_DIM
        f[1] = 1.0
        f[4] = c[dr] / 4.0
        f[7] = 0.0                                  # 胡 = 向听 0
        f[10] = 1.0                                 # 胡即听牌
        f[11] = bt_now
        f[12] = min(fan, 16.0) / 16.0               # **这一步胡的真实番数**（含爆头/财飘链）
        cands.append({"kind": K_HU, "tile": dr})
        feats.append(f)

    # ---- gang（暗杠 / 补杠）----
    if (v.wall or 0) > DEADWALL:
        pengs = peng_tile_set(v)
        for t in range(NTILE):
            if t == WILD or c[t] <= 0:
                continue
            if c[t] >= 4 or (c[t] >= 1 and t in pengs):
                after = list(c)
                after[t] -= 4 if c[t] >= 4 else 1
                sh_a = _sh(after, melds + 1)         # 近似：忽略杠后的补牌
                f = [0.0] * CAND_TURN_DIM
                f[2] = 1.0
                f[3] = t / 33.0
                f[4] = c[t] / 4.0
                f[6] = 1.0 if t == dr else 0.0
                f[7] = sh_a / 9.0
                f[8] = unseen[t] / 4.0
                f[9] = 1.0 if t >= 27 else 0.0
                cands.append({"kind": K_GANG, "tile": t})
                feats.append(f)

    # ---- discard ----
    restricted = bool(v.restricted)
    for t in range(NTILE):
        if c[t] <= 0:
            continue
        if restricted and t != dr:                   # 抓打圈：只能打刚摸的那张
            continue
        after = list(c)
        after[t] -= 1
        sh_a = _sh(after, melds)
        f = [0.0] * CAND_TURN_DIM
        f[0] = 1.0
        f[3] = t / 33.0
        f[4] = c[t] / 4.0
        f[5] = 1.0 if t == WILD else 0.0
        f[6] = 1.0 if t == dr else 0.0
        f[7] = sh_a / 9.0
        f[8] = unseen[t] / 4.0
        f[9] = 1.0 if t >= 27 else 0.0
        f[10] = 1.0 if sh_a == 0 else 0.0
        f[11] = _bt(after, melds)
        f[13] = min(_uk(after, melds, unseen), UK_CAP) / UK_CAP   # ★ 有效牌（第 374 轮新增）
        cands.append({"kind": K_DISCARD, "tile": t})
        feats.append(f)

    if not cands:                                     # 理论上不可能（手上总有牌）
        raise RuntimeError("回合决策点没有任何候选动作 —— 引擎状态异常")
    if len(cands) > MAX_TURN_CAND:
        raise RuntimeError("候选项 %d > MAX_TURN_CAND=%d，请调大"
                           % (len(cands), MAX_TURN_CAND))
    return cands, feats


# ---------------------------------------------------------------- B. 窗口决策
def window_candidates(v, phase):
    """返回 `(cands, feats)`：`[{kind:"pass"}, {kind:<peng|chi>, tile:t}]`。"""
    is_chi = (phase == "response_chi")
    kind = K_CHI if is_chi else K_PENG
    c = list(v.counts)
    melds = int(v.meld_sets or 0)
    ld = getattr(v, "last_discard", "") or ""
    try:
        t = tile_index(ld)
    except (ValueError, TypeError):
        t = -1
    sh_now = _sh(c, melds)
    hand_len = sum(c)

    cands = [{"kind": K_PASS, "tile": None}]
    feats = [[1.0, 0.0, (c[t] / 4.0 if t >= 0 else 0.0), 0.0, 0.0,
              melds / 4.0, sh_now / 9.0, hand_len / 14.0]]

    after = list(c)
    if is_chi:
        pair = chi_tiles(c, ld, melds) if t >= 0 else []
        for code in pair:
            after[tile_index(code)] -= 1
        sh_a = _sh(after, melds + 1)
        feats.append([0.0, 1.0, (c[t] / 4.0 if t >= 0 else 0.0),
                      1.0, 0.0, melds / 4.0, sh_a / 9.0, hand_len / 14.0])
        cands.append({"kind": kind, "tile": t})
        return cands, feats

    # 碰窗口：★ 第 363 轮起 **明杠也在候选里**。
    #   为什么必须加：`mj_sim` 从第 363 轮起认碰窗口里的 `gang`（此前静默当 pass），
    #   而**规则策略在线上一直在做明杠**（真值事件流 35,976 局：我方 0.0308 次/局，
    #   对手 0.0252）。若 RL 的候选集里没有它，学出来的策略就**结构上做不到**
    #   一件老师在做的事 —— 那不是"学不会"，是"没有这个动作"。
    #   编码沿用同一套 8 维（`is_peng`/`is_chi` 两列对明杠都给 0，与 pass 一致；
    #   区分靠候选自身的 `kind`，以及 `c[t]/4` 那两列 —— 明杠要求手上 ≥3 张，
    #   而碰只要 ≥2，`c[t]/4` 正好把两者分开）。
    after = list(c)
    if t >= 0:
        after[t] -= 2
    sh_a = _sh(after, melds + 1)
    feats.append([0.0, 1.0, (c[t] / 4.0 if t >= 0 else 0.0),
                  0.0, 1.0, melds / 4.0, sh_a / 9.0, hand_len / 14.0])
    cands.append({"kind": kind, "tile": t})
    if t >= 0 and c[t] >= 3:
        # 明杠：手上 3 张 + 别人打出的第 4 张。门控与 `mj_sim._poll_window`/
        # `window_act` 完全一致（**≥3 张**），否则引擎会把它判为非法。
        after_g = list(c)
        after_g[t] -= 3
        sh_g = _sh(after_g, melds + 1)
        feats.append([0.0, 1.0, (c[t] / 4.0), 0.0, 0.0,
                      melds / 4.0, sh_g / 9.0, hand_len / 14.0])
        cands.append({"kind": K_GANG, "tile": t})
    return cands, feats


# ---------------------------------------------------------------- 引擎动作
def to_engine_action(cand):
    """把候选转成 `mj_sim` 要的 act 字典（pass → None）。"""
    k = cand["kind"]
    if k == K_PASS:
        return None
    if k == K_HU:
        return {"action": "hu"}
    if k == K_GANG:
        return {"action": "gang", "tile": tile_code(cand["tile"])}
    if k == K_DISCARD:
        return {"action": "discard", "tile": tile_code(cand["tile"])}
    return {"action": k}                      # peng / chi


def action_token(cand):
    """紧凑的动作指纹（评估里用来判"两臂是否逐局不同"）。"""
    k = cand["kind"]
    if cand.get("tile") is None:
        return k
    if k == K_HU:
        return "hu"
    return "%s:%s" % (k, tile_code(cand["tile"]))


# ---------------------------------------------------------------- 自测
def _selftest(n_games=40):
    """纯函数级自测：与 `mj_policy.peng_tiles` 对拍 + 特征维度/范围检查。

    ⚠ 这里只测**编码层**的良构性（维度/范围/候选形状）。
      "合法集是否与引擎一致"要靠 `train_selfplay_rl.py --selfcheck`
      （它用 `AuditedGame` 统计 `_fallback_discard` 被触发的次数）。
    """
    from mj_policy import peng_tiles as ref_peng
    # ★ 第 374 轮：有效牌特征的口径必须与老师**逐点一致**，所以把老师那份
    #   `unseen_from_view` 也导进来对拍（`mj_policy.py:249`）。
    from mj_policy import unseen_from_view as ref_unseen
    from mj_bot import View
    from mj_sim import Game
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = (got == want)
        ok = ok and good
        print("  %s %-52s got=%s want=%s" % ("ok  " if good else "FAIL", name, got, want))

    # ---- A) 回合候选：真实对局状态 ----
    from mj_arena import make_strategy
    probe = make_strategy("shanten:1:1:2:5:1:1:0")      # 只用来推进对局，不参与被测逻辑
    n_turn, mx, bad_peng, n_hu, n_gang, n_restricted = 0, 0, 0, 0, 0, 0
    # ★ 第 374 轮：**有效牌特征**的两条检查
    bad_unseen = 0        # 我们的 unseen 与老师的 unseen_from_view 是否逐点一致
    n_uk_gt0 = 0          # f[13] > 0 的候选个数（防"加了但恒为 0"这种假实现）
    for seed in range(n_games):
        g = Game(seed=seed, dealer=seed % 4)
        while g.phase == "draw":
            seat = g.turn
            v = View(g.snapshot(seat), seat)
            cands, feats = turn_candidates(v)
            assert len(cands) == len(feats)
            assert all(len(f) == CAND_TURN_DIM for f in feats)
            assert all(0.0 <= x <= 1.0 for f in feats for x in f), "回合候选特征越界"
            assert sum(1 for cd in cands if cd["kind"] == K_HU) <= 1
            assert len({(cd["kind"], cd["tile"]) for cd in cands}) == len(cands), "候选重复"
            if v.restricted:
                n_restricted += 1
                assert len(cands) == 1 and cands[0]["kind"] == K_DISCARD
            # ★ unseen 口径对拍（我们的 vs 老师用的那一份）
            try:
                if unseen_counts(v) != ref_unseen(v):
                    bad_unseen += 1
            except Exception:                                    # noqa: BLE001
                bad_unseen += 1
            n_uk_gt0 += sum(1 for cd, f in zip(cands, feats)
                            if cd["kind"] == K_DISCARD and f[13] > 0.0)
            n_turn += 1
            n_hu += sum(1 for cd in cands if cd["kind"] == K_HU)
            n_gang += sum(1 for cd in cands if cd["kind"] == K_GANG)
            mx = max(mx, len(cands))
            if peng_tile_set(v) != ref_peng(v):
                bad_peng += 1
            # 用规则策略推进对局（保证能走完）；窗口由 window_policy=None 直接跳过
            a = probe.on_turn(v) or {}
            kind = a.get("action")
            if kind == "hu":
                if not g.try_hu(seat):
                    g._fallback_discard(seat)
            elif kind == "gang":
                if not g._try_gang(seat, a.get("tile")):
                    g._fallback_discard(seat)
            else:
                try:
                    g.discard(seat, a.get("tile", "") or v.drawn or v.hand[0])
                except Exception:                                # noqa: BLE001
                    g._fallback_discard(seat)
            if not g.finished and g.phase != "draw":
                break
        if not g.finished:
            g._settle_draw()

    # ---- B) 窗口候选：合成快照（碰 / 吃 各一）----
    n_win = 0
    for phase, hand, last in (("response_peng", ["1w", "1w", "3b", "4b", "5t"], "1w"),
                              ("response_chi", ["2w", "3w", "7b", "7b", "5t"], "1w")):
        snap = {"phase": phase, "seat": 1, "turn": 0, "my_hand": list(hand),
                "melds": [[], [], [], []], "last_discard": last,
                "wall_remaining": 40, "discards": [["1w"], [], [], []],
                "god": {}, "scores": [0, 0, 0, 0], "dealer": 0}
        v = View(snap, 1)
        wc, wf = window_candidates(v, phase)
        # ⚠ 第 363 轮起**不再是固定 2 个**：碰窗口在手上 ≥3 张时多一个「明杠」。
        #   `N_WIN_CAND` 是**缓冲区宽度（上界）**，不是每点的候选数。
        assert 1 <= len(wc) <= N_WIN_CAND and len(wc) == len(wf), (len(wc), len(wf))
        assert all(len(f) == CAND_WIN_DIM for f in wf)
        assert all(0.0 <= x <= 1.0 for f in wf for x in f), "窗口候选特征越界"
        assert [c["kind"] for c in wc] == [K_PASS, K_CHI if "chi" in phase else K_PENG]
        assert to_engine_action(wc[0]) is None
        assert to_engine_action(wc[1]).get("action") in ("peng", "chi")
        n_win += 1

    # ---- B2) ★ 第 363 轮：碰窗口里手上 **3 张** 时必须多出「明杠」候选 ----
    #   （`mj_sim` 从本轮起认这个动作；真值：线上我方 0.0308 次/局）
    snap3 = {"phase": "response_peng", "seat": 1, "turn": 0,
             "my_hand": ["1w", "1w", "1w", "3b", "4b"], "melds": [[], [], [], []],
             "last_discard": "1w", "wall_remaining": 40,
             "discards": [["1w"], [], [], []], "god": {}, "scores": [0, 0, 0, 0],
             "dealer": 0}
    wc3, wf3 = window_candidates(View(snap3, 1), "response_peng")
    chk("手上 3 张时碰窗口有「明杠」候选",
        [c["kind"] for c in wc3], [K_PASS, K_PENG, K_GANG])
    chk("明杠候选转成引擎动作 `gang`",
        to_engine_action(wc3[2]).get("action"), "gang")
    chk("明杠候选特征维度正确", all(len(f) == CAND_WIN_DIM for f in wf3), True)
    chk("明杠候选特征不越界",
        all(0.0 <= x <= 1.0 for f in wf3 for x in f), True)
    # 只有 2 张时**不能**有明杠（引擎会拒）
    wc2, _wf2 = window_candidates(
        View({**snap3, "my_hand": ["1w", "1w", "3b", "4b", "5t"]}, 1),
        "response_peng")
    chk("手上只有 2 张时**没有**明杠候选", [c["kind"] for c in wc2],
        [K_PASS, K_PENG])

    print("  回合决策点 %d 个（其中能胡 %d、可杠 %d、抓打圈受限 %d），"
          "窗口决策点 %d 个，单点最大候选数 %d（上限 %d）"
          % (n_turn, n_hu, n_gang, n_restricted, n_win, mx, MAX_TURN_CAND))
    chk("peng_tile_set 与 mj_policy.peng_tiles 不一致次数", bad_peng, 0)
    chk("unseen 与 mj_policy.unseen_from_view 不一致次数", bad_unseen, 0)
    chk("有效牌特征 f[13] 真的非零（防空实现）", n_uk_gt0 > 0, True)
    chk("状态维度 STATE_DIM == 370（开 MJ_CTX2 时 376）", STATE_DIM,
        376 if USE_CTX2 else 370)
    chk("单点最大候选数 <= MAX_TURN_CAND", mx <= MAX_TURN_CAND, True)
    chk("回合候选 >= 1", n_turn > 0, True)
    print("  ⇒ %s" % ("全部通过" if ok else "**有失败项**"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)

