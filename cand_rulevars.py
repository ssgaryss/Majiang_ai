#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规则裁决链里那两个**候选级**变量 —— 给排序器补上它现在看不到的东西。

为什么要有这个模块（第 194 轮读代码读出来的，见 `PREREG_RANK_RULEVARS.md`）
--------------------------------------------------------------------------
`mj_policy.best_discard_shanten` 的并列裁决链（`mj_policy.py:674-708`）逐级是：

    ① 向听最小  →  ② 财神保护  →  ③ **自然向听最小**  →  ④ 爆头
    →  ⑤ **unseen 修正 ukeire 最大**  →  ⑥ 牌码最小

而 `features_full.cand_scalars` 只给 6 个标量
`[sh/9, **裸**uk/400, bt, is_wild, is_drawn, tile]`
⇒ **第③步与第⑤步完全没进特征**，而它们**正是"池内选哪张"的主排序键**
（第①步在池内是常数）。

症状（已实测）：`cand_rank_x.pt` 与**规则**的分歧率 **0.726** —— 它没法复现③⑤。

本模块只干一件事：**把③⑤算出来**。两个来源**必须一致**：
* **训练侧**：从 `gen_cand_shard.py` 存下来的 `X`（360 维）**反解**；
* **推理侧**：从 `View v` 现算。

⚠⚠ 本仓纪律「**允许副本，但必须有对拍**」在这里的落地是：
`check_rulevars_identity.py` 会在**同一批真实状态**上把两条路径逐位对齐，
**对拍不过就不许拿去训练/评测**。

★ 安全说明：本模块**目前没有被任何线上路径 import**。
  之所以先单独放一个文件（而不是直接改 `mj_policy.py` / `features_full.py`）：
  **第 191~195 轮 harness 的 shell 坏了、跑不了 `check_syntax_all.py`**，
  而 `mj_policy.py` 里一个语法错会让**线上客户端下次重启时 import 失败**。
  ⇒ 零风险的先落在这里；等 shell 恢复、体检能跑之后，再做那两处两行的接线。

用法（未来接线时）：
    from cand_rulevars import from_view, from_x
    extra = from_view(v, t, branch_b)      # 推理侧
    extra = from_x(X_row, t)               # 训练侧
"""
import numpy as np

import shanten_ref as S
# ⚠ `NPLAY` **不在** `mj_core` 里 —— 它是 `features_full.py:37` 自己定义的常量。
#   第 194 轮写这个模块时我按 `features_full` 的用法想当然写成 `from mj_core import NPLAY`，
#   那会直接 ImportError。**因为 shell 坏着跑不了 `check_imports.py`，只能靠读代码抓到。**
from mj_core import NTILE, WILD

NPLAY = 4                       # 与 `features_full.py:37` 逐字一致

# `features_full.features` 的布局偏移（F = 360）—— 逐字对齐 `features_full.py:103-170`
OFF_HAND = 0                    # 自己手牌 34（×0.25）
OFF_UNSEEN = NTILE              # 未见牌 34（/4）
OFF_DISC = OFF_UNSEEN + NTILE   # 四家弃牌 4×34（×0.25）
OFF_MELD = OFF_DISC + NPLAY * NTILE   # 四家副露 4×34（×0.25）
OFF_MELDCNT = OFF_MELD + NPLAY * NTILE  # 每座副露**组数** 4（/4）
OFF_SEAT = OFF_MELDCNT + NPLAY          # 座位 one-hot 4
F_EXPECT = 360

# 新特征的名字（顺序 = 返回值的顺序），两个都**标准化到 ~[0,1]**
EXTRA_NAMES = ["nat/9", "uk_unseen/400"]
EXTRA_DIM = len(EXTRA_NAMES)


def _uk_unseen(cc, melds, unseen):
    """unseen 修正后的有效牌枚数（规则第⑤步用的那个量）。

    ⚠ 与 `features_full.cand_scalars` 里的 `uk` **不是一回事**：
      那边是 `S.ukeire(c, melds)`（**不传 unseen**，等于假设 4 张全在牌山）。
    """
    try:
        return int(S.ukeire(cc, melds, unseen=unseen)[0])
    except Exception:                                             # noqa: BLE001
        return 0


def _nat(cc, melds):
    """自然向听（规则第③步）：把白板从手牌里清零后再算向听。

    与 `mj_policy.py:675-677` 逐字同口径：
        cn = list(c); cn[WILD] = 0;  nat = S.shanten(cn, melds)
    """
    cn = list(cc)
    cn[WILD] = 0
    try:
        return int(S.shanten(cn, melds))
    except Exception:                                             # noqa: BLE001
        return 9


def pack(nat, uk_unseen):
    """打包成喂给排序器的两个浮点（与 `cand_scalars` 的缩放风格一致）。"""
    return [float(nat) / 9.0, float(min(int(uk_unseen), 400)) / 400.0]


def from_counts(counts, t, melds, unseen):
    """**唯一的布局定义**：从（手牌 14 张, 候选牌, 副露组数, unseen）算这两个量。

    训练侧与推理侧都必须走这里 —— 两端只是**取得输入的途径**不同。
    """
    cc = list(counts)
    cc[t] -= 1
    return pack(_nat(cc, melds), _uk_unseen(cc, melds, unseen))


def from_view(v, t):
    """推理侧：从 `View` 现算。

    ⚠ 依赖 `mj_policy.unseen_from_view`（**延迟 import**，避免本模块被 import 时
      就把整个策略栈拉进来 —— 也让"未被线上引用"这件事成立）。
    """
    from mj_policy import unseen_from_view
    return from_counts(v.counts, t, int(v.meld_sets or 0), unseen_from_view(v))


def from_x(row, t):
    """训练侧：从存下来的 `X`（360 维）**反解**。

    反解依据（`features_full.py:103-170`，逐条对齐）：
        hand   = X[0:34]  / 0.25      （那边是 `c[t] * 0.25`，计数 ≤4 ⇒ 精确可逆）
        unseen = X[34:68] * 4         （那边是 `left / 4.0`）
        me     = argmax(X[344:348])   （座位 one-hot）
        melds  = X[340 + me] * 4      （副露**组数** / 4）
    """
    X = np.asarray(row, dtype=np.float64).ravel()
    if X.size != F_EXPECT:
        raise ValueError("X 长度 %d != %d（布局变了？先更新本模块的偏移）"
                         % (X.size, F_EXPECT))
    hand = np.rint(X[OFF_HAND:OFF_HAND + NTILE] / 0.25).astype(np.int64)
    unseen = np.rint(X[OFF_UNSEEN:OFF_UNSEEN + NTILE] * 4.0).astype(np.int64)
    seat_blk = X[OFF_SEAT:OFF_SEAT + NPLAY]
    me = int(np.argmax(seat_blk))
    melds = int(round(float(X[OFF_MELDCNT + me]) * 4.0))
    return from_counts(hand.tolist(), t, melds, unseen.tolist())


def check_roundtrip(row):
    """自检：`X` 里的量反解回来还是不是整数（浮点缩放的可逆性）。

    返回 (hand_ok, unseen_ok, 说明)。**不通过就别用这份 X。**
    """
    X = np.asarray(row, dtype=np.float64).ravel()
    hand_raw = X[OFF_HAND:OFF_HAND + NTILE] / 0.25
    unseen_raw = X[OFF_UNSEEN:OFF_UNSEEN + NTILE] * 4.0
    hand_ok = bool(np.all(np.abs(hand_raw - np.rint(hand_raw)) < 1e-6))
    unseen_ok = bool(np.all(np.abs(unseen_raw - np.rint(unseen_raw)) < 1e-6))
    msg = "hand 可逆 %s / unseen 可逆 %s" % (hand_ok, unseen_ok)
    return hand_ok, unseen_ok, msg
