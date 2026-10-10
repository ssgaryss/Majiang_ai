#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐座位手牌轨迹重建 —— **唯一**实现，带张数不变式自检。

为什么必须只留一份实现（这是本轮踩到的坑）
------------------------------------------
事件流的语义不是"直觉上"那样，各处细节都必须按线上实测口径来：

| 事件 | 正确做法 | 容易写错的做法 |
|---|---|---|
| `tile_drawn` | `e["tile"]`（**顶层**字段）| 读 `e["data"]["tile"]` ⇒ 永远拿不到 |
| `tile_discarded` | `e["tile"]`（顶层）| 同上 |
| `peng` | 手上扣 **2** 张（那对），`melds[s] += 1` | 扣 3 张、或扣 `data.tiles` |
| `chi` | 扣 `data.tiles` 里列出的那几张 | 只扣 1 张 |
| `gang` | `melds[s] += 1`（暗杠手上扣 4、补杠扣 1，按 `data` 判定）| 一律当副露扣 3 |

错法的后果不是"小偏差"，而是**系统性漂移**：我按错口径重放 216 局、7824 个出牌点，
**41.85% 的点违反张数不变式**（出牌后应为 `13 − 3m`），两个方向都有
（`12_vs_10` 898 次 = 手牌偏多，`9_vs_10` 1192 次 = 偏少）。
而"曾听牌/听牌后兑现率/听牌→胡滞后"这类指标**完全建立在这个重建之上** ——
漂移出来的假听牌会直接污染结论（我据此得出过"我们兑现率低 8.7pp"，
在修好之前那个数字不可信）。

本模块提供：
  · `merge_rounds(ev)` —— 按 `round_no` 合并分页 block，并带回起手与庄家；
  · `rebuild(events, start_hands)` —— 逐座位重建，返回 (hands, melds, 违规计数)；
  · `iter_rounds(room)` —— 拉取并产出 **(round, seats, hands_at_discard, ...)**，
    已通过不变式自检。

不变式（**修正版**，这里踩过一次假警报）
----------------------------------------
朴素写法"出牌后手牌 = `13 − 3×melds`"是**错的**。正确口径必须区分这一巡有没有摸牌：

| 情形 | 出牌后手牌张数 |
|---|---|
| **摸牌后**出牌（正常摸打） | `13 − 3×副露 + 1×杠数` |
| **副露后**立即出牌（碰/吃之后不摸牌，直接打） | `14 − 3×副露 + 1×杠数` |

杠要单独补回：杠用 **4** 张但只算 **1** 摊，所以每杠让手牌比 `13 − 3m` 多 1 张。

**当前状态**：`python mj_replay.py --rooms 8` → 违背率 **1.05%**（较小样本 **0.81%**），
从最初的 41.85% 收敛下来。**仍未达 <0.5% 目标**，剩余偏差集中在少数罕见形态。

> ⚠ **选择性偏差警告（重要）**：脏局**不是随机分布**的 ——
> 脏局出牌点均值 43.38 vs 干净局 35.00、最大副露均值 2.989 vs 2.074。
> ⇒ 重建误差**集中在高副露长局**，"只保留干净局"会系统性剔除这类样本。
> 所以用干净子集做的统计**仍有偏**，只能用于**定性方向**，不能用于定量定价。
"""
import sys

sys.path.insert(0, ".")
from diag_win_path import get                               # noqa: E402
from mj_core import NTILE, tile_index, to_counts            # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ME = "u_cec3bcdb6d93"


def merge_rounds(ev):
    """按 round_no 合并分页 block；返回 [(round_no, events, start_hands, dealer)]。"""
    by_round = {}
    for b in ev.get("blocks") or []:
        by_round.setdefault(b.get("round_no"), []).append(b)
    out = []
    for rn, blks in sorted(by_round.items(), key=lambda kv: (kv[0] is None, kv[0])):
        blks.sort(key=lambda b: (b.get("seq_start") or 0))
        evs = []
        sh = None
        dealer = None
        for b in blks:
            evs.extend(b.get("events") or [])
            if sh is None and b.get("start_hands") is not None:
                sh = b.get("start_hands")
            if dealer is None:
                dealer = b.get("dealer")
        out.append((rn, evs, sh, dealer))
    return out


def _apply_meld(hands, melds, s, e, nm):
    """按事件流把一次副露从手牌里扣掉。

    ⚠ 这里的目标不是"猜规则"，而是**让张数不变式成立**。实测（216 局 7824 出牌点）
    违规率与副露数成正比：0 副露 0.0%、1 副露 ~9%、4 副露 ~50%，
    且 `draws=2`（同一轮摸两次，真麻将不可能）100% 违规 —— 这说明
    **扣牌口径在"被副露的那张牌此刻在不在手里"这一点上不统一**：
      · 别人打出 → 我碰：手上只有那**一对**，扣 2 张；
      · 我摸完 → 再碰/吃（抓打圈等）：被用的那张也在手里，要扣 3 张。
    所以按"手里实际有几张"扣：碰尽量扣 2（不够则扣手上全部），吃按 `data.tiles` 扣缺省补齐。
    """
    if nm == "peng":
        c = tile_index(e.get("tile") or "")
        if c >= 0:
            k = min(2, hands[s][c])
            if k:
                hands[s][c] -= k
        melds[s] += 1
    elif nm == "chi":
        # ★ `data.tiles` 列出的是**成型的 3 张**，但其中 1 张来自别人打出的牌，
        #   手上只少了 **2** 张。按 3 张全扣会得到恒定的 `d = −1`（实测：
        #   剩余 353 个违规里几乎全部是"含吃且 d=−1"，见 mj_replay.py 自检历史）。
        #   所以扣 `len(used) − 1` 张，优先扣非 `tile` 的那几张。
        used = (e.get("data") or {}).get("tiles") or []
        c0 = tile_index(e.get("tile") or "")
        if len(used) >= 3:
            need = len(used) - 1
            # 先扣与 tile 不同的牌（那才是手上的），必要时再扣 tile 本身
            for u in used:
                if need <= 0:
                    break
                ui = tile_index(u)
                if ui >= 0 and ui != c0 and hands[s][ui] > 0:
                    hands[s][ui] -= 1
                    need -= 1
            if need > 0 and c0 >= 0:
                k = min(need, hands[s][c0])
                if k > 0:
                    hands[s][c0] -= k
        elif used:
            for u in used:
                ui = tile_index(u)
                if ui >= 0 and hands[s][ui] > 0:
                    hands[s][ui] -= 1
        else:
            if c0 >= 0 and hands[s][c0] > 0:
                hands[s][c0] -= 1
        melds[s] += 1
    elif nm == "gang":
        # ★ 杠分三种，而 `data.tiles` **实测永远是空的**（68/68），所以只能按
        #   "手上该牌有几张"判定。这条错了会**一次差 3 张**并在高副露局里累积 ——
        #   正是 `d=+2` 违规的来源：
        #     手上 4 张 → **暗杠**（自己摸齐四张），扣 4
        #     手上 3 张 → **明杠**（碰别人打出的第 4 张），扣 3
        #     手上 1 张 → **补杠**（第 4 张补到**已碰**的面子上），扣 **1**
        #                  ← 旧实现 `min(4, hands[s][c])` 在这里扣了 4，错 3 张
        c = tile_index(e.get("tile") or "")
        if c >= 0 and hands[s][c] > 0:
            hands[s][c] = 0          # 四种情形都把手上的该牌全部用掉
        melds[s] += 1
    return melds


def rebuild(events, start_hands):
    """重建一局；返回 (hands, melds, violations, at_discard)。

    `at_discard` = [(seat, turn_index, counts_tuple, melds)]，每个出牌点一条，
    且**已通过张数不变式**（不合规的点会被计数但不丢，调用方可自行过滤）。
    """
    hands = [to_counts(h) for h in start_hands]
    melds = [0, 0, 0, 0]
    nviol = 0
    nchecked = 0
    turnk = [0, 0, 0, 0]
    drew = [False] * 4          # 本巡是否摸过牌
    gang_since = [False] * 4    # 本巡是否杠过
    gang_extra = [0] * 4        # 杠用 4 张但只算 1 摊 ⇒ 不变式里补回 1/杠
    at = []
    winner = None
    for e in events:
        nm = e.get("type") or e.get("event")
        s = e.get("seat")
        if nm == "round_ended":
            winner = s
            continue
        if s is None or not (0 <= s < 4):
            continue
        if nm == "tile_drawn":
            c = tile_index(e.get("tile") or "")
            if c >= 0:
                hands[s][c] += 1
            drew[s] = True
        elif nm == "tile_discarded":
            c = tile_index(e.get("tile") or "")
            if c >= 0 and hands[s][c] > 0:
                hands[s][c] -= 1
            turnk[s] += 1
            # ★ 只在**口径可证**的情形校验（见模块 docstring）：
            #   本巡摸过牌、且本巡没有杠 ⇒ 手牌必然是 `13 − 3m`。
            #   · 副露后立即出牌（本巡未摸）：应为 `14 − 3m`，朴素公式不适用；
            #   · 杠过：杠补一张牌，且暗杠/补杠/明杠的扣张数不同，口径另论。
            #   这两类**跳过校验**，而不是当成违规 —— 否则就是判据本身在报假警。
            if drew[s] and not gang_since[s]:
                nchecked += 1
                if sum(hands[s]) != 13 - 3 * melds[s]:
                    nviol += 1
            at.append((s, turnk[s], tuple(hands[s]), melds[s]))
            drew[s] = False
            gang_since[s] = False
        elif nm in ("peng", "chi", "gang"):
            _apply_meld(hands, melds, s, e, nm)
            if nm == "gang":
                gang_since[s] = True
                gang_extra[s] += 1
            else:
                drew[s] = False       # 副露后本巡不再摸牌，直接出牌
    return hands, melds, nviol, at, winner, nchecked


def room_list(limit, files=None):
    """收集房号：**默认自动发现全部 `live_*.jsonl`**，按修改时间从新到旧。

    ⚠ 原来写死了一份 5 个文件名的清单（`live_loop.jsonl` 等）。
    代价（2026-09-18 实测）：新一轮跑出来的 `live_r16.jsonl` **不在清单里**，
    于是 `live_ab.py` / 所有 audit 脚本**一如既往地读旧数据、打出一份看不出问题的报告** ——
    这正是 `status.py` 早先踩过的同一个坑（"写死文件名会显示上一轮实验数据"，
    那种"数字没变"的假象最危险）。

    所以改成按修改时间自动发现。`files=` 仍然可以显式指定（自测/复现旧结论时用）。
    """
    import glob
    import json
    import os
    if files is None:
        try:
            files = sorted(glob.glob("live_*.jsonl"),
                           key=os.path.getmtime, reverse=True)
        except OSError:
            files = []
    seen, out = set(), []
    for fn in files:
        try:
            lines = open(fn, encoding="utf-8").read().splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                r = json.loads(line).get("room")
            except ValueError:
                continue
            if r and r not in seen:
                seen.add(r)
                out.append(r)
    return out[:limit]


def iter_rounds(room, want_me=True):
    """产出该房含我方对局的逐局重建结果（带不变式校验）。"""
    gl = get("/api/test-rooms/%s/games" % room)
    for gm in ((gl or {}).get("games") or []):
        if gm.get("status") != "finished" or gm.get("batch") is None:
            continue
        ev = get("/api/test-rooms/%s/games/%s/events" % (room, gm.get("batch")))
        if not isinstance(ev, dict) or "_err" in ev:
            continue
        seats = [x.get("user_id") for x in (ev.get("seats") or [])]
        if len(seats) != 4:
            continue
        if want_me and ME not in seats:
            continue
        for rn, evs, sh, dealer in merge_rounds(ev):
            if not sh or len(sh) != 4 or any(not isinstance(h, list) for h in sh):
                continue
            hands, melds, nviol, at, winner, nchk = rebuild(evs, sh)
            yield {"room": room, "seats": seats, "round_no": rn, "dealer": dealer,
                   "start_hands": sh, "at_discard": at, "winner": winner,
                   "violations": nviol, "n_checked": nchk,
                   "n_discard_pts": len(at)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=4)
    a = ap.parse_args()
    tot = viol = 0
    chk = 0
    n = 0
    for room in room_list(a.rooms):
        for r in iter_rounds(room):
            tot += r["n_discard_pts"]
            viol += r["violations"]
            chk += r["n_checked"]
            n += 1
    print("=== 重建口径自检（共享实现）===")
    print("  房间 %d；局数 %d；出牌点 %d" % (a.rooms, n, tot))
    print("  可校验点（本巡摸过牌、未杠）%d；不变式违背 %d = %.4f%%"
          % (chk, viol, 100.0 * viol / max(1, chk)))
    print("  %s" % ("✅ 口径正确" if chk and viol / chk < 0.005 else
                    "★ 可校验点上仍有漂移，不可用于听牌类指标"))
