#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""决策点特征编码 —— **只读平台快照**，不碰任何模拟器内部对象。

这条约束是硬性的，不是洁癖：
  第一版 features(v, g) 用了 g.dealer 与 g.turns 两个模拟器属性。线上根本没有
  这两个东西（快照里只有 dealer 字段，摸打数要自己从 discards 数出来）。
  用模拟器专属量训出来的网络，线上要么崩、要么行为完全不同 —— 这是这类项目
  最经典的翻车方式，而且在对拍里看不出来（对拍也跑在模拟器上）。

  所以现在全部信息只有一个来源：View.snap，也就是 GET /api/games/{id}/state
  返回的那个 snapshot。同一份代码在模拟器与线上走完全一样的路径。

维度 F = 34(手牌) + 34(未见牌) + 4(座位) + 12(局面标量)
"""
from mj_core import NTILE, can_win, tile_index
import shanten_ref as S

F = 34 + 34 + 4 + 12


def features(v, counts=None):
    """把一次决策编码成定长向量。counts 给定时用它替换手牌（用于评估候选弃牌）。"""
    f = [0.0] * F
    c = list(counts) if counts is not None else v.counts
    snap = v.snap or {}
    k = 0
    for t in range(NTILE):
        f[k + t] = c[t] * 0.25
    k += NTILE

    # 未见牌 = 4 - 自己手牌 - 四家弃牌 - 已副露用掉的
    dead = [0] * NTILE
    for row in (snap.get("discards") or []):
        for code in row:
            _bump(dead, code)
    for row in (snap.get("melds") or []):
        for m in row:
            for code in ((m or {}).get("tiles") or []):
                _bump(dead, code)
    for t in range(NTILE):
        left = 4 - c[t] - dead[t]
        f[k + t] = (left if left > 0 else 0) / 4.0
    k += NTILE

    seat = v.seat if v.seat is not None and 0 <= v.seat < 4 else 0
    f[k + seat] = 1.0
    k += 4

    dealer = snap.get("dealer")
    disc_total = sum(len(r) for r in (snap.get("discards") or []))
    f[k] = 1.0 if dealer is not None and seat == dealer else 0.0
    f[k + 1] = v.meld_sets / 4.0
    f[k + 2] = v.meld_chi / 2.0
    f[k + 3] = (v.wall or 0) / 83.0
    f[k + 4] = disc_total / 64.0          # 用弃牌总数代替 turns，二者等价且线上可得
    f[k + 5] = min(1.0, (v.chain or 0) / 4.0)
    f[k + 6] = 1.0 if v.baotou else 0.0
    f[k + 7] = 1.0 if v.catch_play else 0.0
    f[k + 8] = 1.0 if v.restricted else 0.0
    f[k + 9] = (S.shanten(c, v.meld_sets) + 1) / 9.0
    f[k + 10] = sum(c) / 14.0
    f[k + 11] = 1.0 if can_win(c, v.meld_sets) else 0.0
    return f


def _bump(dead, code):
    try:
        dead[tile_index(code)] += 1
    except (ValueError, AttributeError, TypeError):
        pass


# 特征里「向听」那一维的下标 —— 训练脚本要拿它做「仅向听」基线对照
SHANTEN_COL = 34 + 34 + 4 + 9
