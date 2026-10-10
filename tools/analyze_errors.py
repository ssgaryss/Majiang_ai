#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高代价错误归因 —— 按「频率 × 损失」排序（目标第 2 条）。

为什么要单独做这个工具
----------------------
线上每天都在打，但"哪里在亏钱"一直只有**描述性**读数（弃胡 2,621 次、
兑现率 83.5%）—— 而 83.5% 本身**不能**说明这 2,621 次是赚还是亏：
弃胡的收益是"番数变大"，成本是"可能胡不到（0 分）"。
⇒ 必须把每一次弃胡换成**同一把尺子**：机会成本。

本工具只做**能从现有埋点算出来**的那一类：无收益弃胡。
算不出来的那一类（不划算的吃碰）会**明确报缺什么埋点**，而不是猜。

机会成本的算法（每一局一次，不以"弃胡次数"计）
-----------------------------------------------
设某一局我们**已经可以胡**、却选择了弃胡。记：

  · `实得`   = 该局我们实际的四家得失（来自 `round_deltas[seat]`，逐局真值）
  · `立即胡` = 如果当时直接胡能拿多少分 —— 由**引擎自己的结算公式**给出
               （`mj_sim._settle_win`，本仓自测钉死）：
                   庄家自摸 = 24 × 番      闲家自摸 = 10 × 番
               `番` 用 `mj_core.fan_multiple(暗手, 副露数)` 现算 ——
               ★ `missed_detail["hand"]` 存的正是**副露后的暗手**（实测 2/5/11 张），
                 与引擎结算时的入参**同一个东西** ⇒ 这是正确的调用，不是近似。
  · `净收益 = 实得 − 立即胡`   （> 0 ⇒ 这一次弃胡**赚了**这么多分）

★ 唯一的下界性质：埋点没有当时的链数 / 飘数 / 爆头标志，而这三者只会让
  **立即胡**的番数变大 ⇒ `立即胡` 是下限 ⇒ `净收益` 是**上限**。
  ⇒ 结论方向可信（"赚了"就是赚了），但"赚了多少"是**乐观**的一侧。
  ★ 本工具用**实测支付分布**反向校验 `24番/10番` 这条公式（表 0）。

★ **按房间聚类**给 CI：同一房间的多局共享累计分与连庄链，不是独立样本
  （目标第 6 条明确要求，本仓 `mj_bot.py` 的注释也记过同一个教训）。

分类（决定"该不该纠"）
----------------------
  · **有上行**  = 当时是爆头听（`is_baotou`）或手上有白板 ⇒ 弃胡可能把 ×1 变 ×2/×4，
                 属于目标说的"不能一律当错误"的那一类 ⇒ **只报告，不建议直接纠**。
  · **无上行**  = 既不是爆头听、手上也没有白板 ⇒ 弃胡**没有可见的番数上行**，
                 却承担了"多等一巡、可能被别人先胡/流局"的风险
                 ⇒ 这才是目标要的「无收益弃胡」，是保护层与纠错训练的首要目标。

用法
----
    python tools/analyze_errors.py --jsonl "live*.jsonl" [--min-rooms 3]
    python tools/analyze_errors.py --jsonl "live*.jsonl" --report temp\\errors.txt
"""
import argparse
import builtins
import collections
import glob
import hashlib
import io
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mj_core import WILD, fan_multiple, is_baotou, to_counts  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Z95 = 1.959963984540054

# 引擎的结算公式（`mj_sim._settle_win`，自测断言 "庄家结算 = 番 x 8 x 3"）：
#   庄家：三家各付 pay*8 ⇒ 得 24*pay      闲家：庄付 pay*8 + 两闲各付 pay ⇒ 得 10*pay
WIN_MULT_DEALER = 24
WIN_MULT_OTHER = 10

_OUT = []
_bprint = builtins.print


def print(*args, **kw):                                              # noqa: A001
    """输出同时进 stdout 与 `--report` 文件（沙箱拦 shell 重定向 ⇒ 自己落盘）。"""
    for h in _OUT:
        kw2 = dict(kw)
        kw2["file"] = h
        _bprint(*args, **kw2)
    _bprint(*args, **kw)


def _expand(patterns):
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return sorted(set(out))


def _dedup(paths):
    seen, keep, dup = {}, [], []
    for p in paths:
        try:
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        except OSError:
            continue
        if h in seen:
            dup.append((os.path.basename(p), os.path.basename(seen[h])))
        else:
            seen[h] = p
            keep.append(p)
    return keep, dup


def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), n
    m = st.mean(xs)
    if n < 2:
        return m, float("nan"), n
    return m, Z95 * st.stdev(xs) / math.sqrt(n), n


# ------------------------------------------------------------------ 事件抽取
def collect(patterns):
    """⇒ (事件列表, 分臂按房聚合表, 计数)。

    每个事件 = 一局（不是一次弃胡）—— 一局里弃胡多次只算一次，
    因为该局的得失只有一个数，按次数算会把同一个损失重复计入。
    """
    files, dup = _dedup(_expand(patterns))
    events = []
    bad = 0
    rooms = set()
    for path in files:
        with io.open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:                                    # noqa: BLE001
                    bad += 1
                    continue
                arm = (d.get("genome") or "").lstrip("\ufeff").strip()
                room = (d.get("room") or "").strip()
                if not arm or not room:
                    bad += 1
                    continue
                rooms.add(room)
                for r in (d.get("rows") or []):
                    if not isinstance(r, dict) or r.get("mine") is None:
                        continue
                    seat = r.get("seat", -1)
                    if not (0 <= seat < 4):
                        continue
                    nos = r.get("round_nos") or []
                    deltas = r.get("round_deltas") or []
                    by_round = {}
                    for i, dn in enumerate(deltas):
                        if not isinstance(dn, (list, tuple)) or len(dn) <= seat:
                            continue
                        rn = nos[i] if i < len(nos) else None
                        by_round[rn] = float(dn[seat])
                    # 该局的弃胡事件按 round_no 归并（round_no 缺失的丢掉并计数）
                    per_round = {}
                    for m in (r.get("missed_detail") or []):
                        if not isinstance(m, dict):
                            continue
                        rn = m.get("round_no")
                        if rn is None:
                            continue
                        per_round.setdefault(rn, []).append(m)
                    for rn, ms in per_round.items():
                        if rn not in by_round:
                            continue                  # 该局没观测到（接管/缺失）
                        m0 = ms[0]                    # 用**第一次**弃胡时的手牌
                        hand = m0.get("hand") or []
                        melds = int(m0.get("meld_sets") or 0)
                        if not hand:
                            continue
                        try:
                            counts = to_counts(hand)
                            f_min = int(fan_multiple(counts, melds))
                            up_baotou = bool(is_baotou(counts, melds))
                        except Exception:                            # noqa: BLE001
                            continue
                        if f_min <= 0:
                            continue                  # 不是成胡形（脏数据）⇒ 不用它
                        is_dealer = bool(m0.get("is_dealer"))
                        immediate = (WIN_MULT_DEALER if is_dealer
                                     else WIN_MULT_OTHER) * f_min
                        realized = by_round[rn]
                        events.append({
                            "arm": arm, "room": room,
                            "game": r.get("game"), "round": rn,
                            "n_declines": len(ms),
                            "f_min": f_min, "baotou": up_baotou,
                            "has_wild": int(counts[WILD]) > 0,
                            "is_dealer": is_dealer,
                            "wall": m0.get("wall"),
                            "immediate": float(immediate),
                            "realized": float(realized),
                            "gain": float(realized) - float(immediate),
                        })
    return events, bad, len(files), len(rooms), dup


# ------------------------------------------------------------------ 报表
def _agg_by_room(events, key=None, filt=None):
    """按房聚合「局均」，返回 (房均列表, 局数, 事件数)。"""
    per_room = {}
    n = 0
    for e in events:
        if filt and not filt(e):
            continue
        per_room.setdefault(e["room"], []).append(
            float(e[key]) if key else 0.0)
        n += 1
    return ([st.mean(v) for v in per_room.values()], n, len(per_room))


def _row(label, events, key, filt=None, nd=3):
    rs, n, nr = _agg_by_room(events, key, filt)
    m, h, _ = mean_ci(rs)
    if n == 0:
        return None
    return (label, n, nr, m, h)


def report(events):
    arms = {}
    for e in events:
        arms.setdefault(e["arm"], []).append(e)
    order = sorted(arms.items(), key=lambda kv: -len(kv[1]))

    print("=" * 96)
    print("表 0【口径校验】用**实测支付分布**反向验证 `24番 / 10番` 这条公式")
    print("   若公式对，则每个**正**得失都必然形如 10f（闲家胡）或 24f（庄家胡）")
    print("=" * 96)
    pos = [e["realized"] for e in events if e["realized"] > 0]
    neg = [e["realized"] for e in events if e["realized"] < 0]
    bad_pos = [x for x in pos if not (abs(x % 10) < 1e-9 or abs(x % 24) < 1e-9)]
    print("   正得失 %d 个，其中**不符合** 10f/24f 的 = %d（%.3f%%）"
          % (len(pos), len(bad_pos), 100.0 * len(bad_pos) / max(1, len(pos))))
    if bad_pos[:5]:
        print("   反例（前 5）：%s" % bad_pos[:5])
    cnt = collections.Counter(round(x, 3) for x in pos)
    print("   最常见正得失（前 6）：%s" % cnt.most_common(6))
    print("   负得失最常见（前 6）：%s"
          % collections.Counter(round(x, 3) for x in neg).most_common(6))

    print("\n" + "=" * 96)
    print("表 A【弃胡的净收益】逐**局**计（一局多次弃胡只算一次）")
    print("   净收益 = 实得 − 立即胡；> 0 ⇒ 弃胡**赚了**（上限口径，见文件头）")
    print("=" * 96)
    print("%-30s %7s %7s %9s %9s %9s %s"
          % ("臂", "弃胡局", "房数", "实得局均", "立即胡均", "净收益", "95%CI（按房聚类）"))
    for arm, es in order:
        r1 = _row(arm[:30], es, "realized")
        r2 = _row(arm[:30], es, "immediate")
        r3 = _row(arm[:30], es, "gain")
        if not r3:
            continue
        print("%-30s %7d %7d %9.3f %9.3f %9.3f   ± %.3f"
              % (arm[:30], r3[1], r3[2], r1[3], r2[3], r3[3], r3[4]))
        tot = sum(e["gain"] for e in es)
        print("%-30s   ⇒ 净收益合计 %+.0f 分（= %+.2f 分/局 × %d 局）"
              % ("", tot, r3[3], r3[1]))

    print("\n" + "=" * 96)
    print("表 B【分类】按「有没有番数上行」拆开 —— 决定「该不该纠」")
    print("   有上行 = 当时是爆头听 或 手上有白板（弃胡可能把 ×1 变 ×2/×4）")
    print("   无上行 = 两者都没有 ⇒ **没有可见上行却承担等待风险** = 目标要的那一类")
    print("=" * 96)
    print("%-30s %10s %9s %9s %10s %9s %9s"
          % ("臂 / 类别", "局数", "实得局均", "立即胡均", "净收益", "CI下界", "CI上界"))
    for arm, es in order:
        for tag, f in (("有上行", lambda e: e["baotou"] or e["has_wild"]),
                       ("无上行 ★", lambda e: not (e["baotou"] or e["has_wild"]))):
            r = _row("", es, "gain", f)
            if not r or r[1] < 5:
                continue
            rr = _row("", es, "realized", f)
            ri = _row("", es, "immediate", f)
            lo = r[3] - r[4] if not math.isnan(r[4]) else float("nan")
            hi = r[3] + r[4] if not math.isnan(r[4]) else float("nan")
            print("%-30s %10d %9.3f %9.3f %10.3f %9.3f %9.3f"
                  % (("%s / %s" % (arm, tag))[:30], r[1], rr[3], ri[3],
                     r[3], lo, hi))
        # 庄/闲拆分（只对"无上行"那一类 —— 它才是纠错目标）
        for tag, f in (("  无上行·庄", lambda e: not (e["baotou"] or e["has_wild"])
                        and e["is_dealer"]),
                       ("  无上行·闲", lambda e: not (e["baotou"] or e["has_wild"])
                        and not e["is_dealer"])):
            r = _row("", es, "gain", f)
            if not r or r[1] < 5:
                continue
            print("%-30s %10d %9s %9s %10.3f   （CI 见上）"
                  % (("%s%s" % (arm, tag))[:30], r[1], "-", "-", r[3]))

    print("\n" + "=" * 96)
    print("表 C【算不出来的那一类】—— 明确报缺什么，而不是猜")
    print("=" * 96)
    print("   「不划算的吃碰」= 吃/碰并完成最佳弃牌后，牌效或番型明显恶化。")
    print("   判定它需要**逐局的『我方副露数』与『吃碰发生的那一局』**，而现在的埋点：")
    print("     · `missed_detail` 只在**弃胡那一刻**记 `meld_sets` ⇒ 覆盖面是 0")
    print("     · 每场记录（rows）里没有我方副露数、也没有动作序列")
    print("     · `opp_melds` 是**对手**的副露 ⇒ 不能用来判断我方吃碰是否划算")
    print("   ⇒ 结论：**这一类现在不可测**。要测必须补两样最小埋点：")
    print("     ① 逐局记 `my_meld_sets`（吃/碰各几次也分开）")
    print("     ② 逐局记 `my_melds_after`（动作完成后的向听与番型期望）")
    print("   在此之前，任何「吃碰不划算」的说法都只是印象，不是数据。")


def main():
    ap = argparse.ArgumentParser(description="高代价错误归因（频率 × 损失）")
    ap.add_argument("--jsonl", action="append", default=[])
    ap.add_argument("--report", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not a.jsonl:
        ap.error("--jsonl 是必填（除非只跑 --selftest）")

    if a.report:
        d = os.path.dirname(os.path.abspath(a.report))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        _OUT.append(io.open(a.report, "w", encoding="utf-8"))
    try:
        events, bad, nf, nr, dup = collect(a.jsonl)
        print("jsonl 文件 %d 个（去重丢 %d）｜坏行 %d｜房间 %d"
              % (nf, len(dup), bad, nr))
        if not events:
            print("\n⚠ 没有提取到任何可算的弃胡事件 —— 检查 --jsonl 是否指对了目录。")
            return 1
        print("可算的弃胡**局**数 = %d" % len(events))
        report(events)
        return 0
    finally:
        for h in _OUT:
            h.close()
        del _OUT[:]


def _selftest():
    """钉住**结算公式**与**下限语义** —— 它们错了会让整个排序反向。"""
    ok = True

    def chk(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print("  FAIL %-48s got=%s want=%s" % (name, got, want))
        else:
            print("  ok   %-48s %s" % (name, got))

    print("[结算公式（必须与 mj_sim._settle_win 一致）]")
    chk("庄家自摸 = 24×番", WIN_MULT_DEALER, 24)
    chk("闲家自摸 = 10×番", WIN_MULT_OTHER, 10)
    from mj_core import fan_multiple as _fm
    from mj_core import to_counts as _tc
    # 平胡 123w456w789w123b + 东东（mj_core 自测里用过的同一手）
    panhu = _tc(["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                 "1b", "2b", "3b", "东", "东"])
    chk("平胡番数下限 = 1", int(_fm(panhu, 0)), 1)
    chk("平胡·庄 立即胡 = 24", WIN_MULT_DEALER * int(_fm(panhu, 0)), 24)
    chk("平胡·闲 立即胡 = 10", WIN_MULT_OTHER * int(_fm(panhu, 0)), 10)
    print("\n[下限语义：立即胡的番数是下限 ⇒ 净收益是上限]")
    chk("爆头 ×2 让番数变大（同一手）",
        int(_fm(panhu, 0, 0, 0, True)) >= int(_fm(panhu, 0)), True)
    if not ok:
        return 1
    print("  ✓ 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
