#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""线上匹配战绩判据（**L2：唯一有权宣布"模型更强"的一层**）。

规范来源：`GOAL_STAGE2.md` §0.1 / §4.4。

它回答四个问题（全部只用 `live_r16.jsonl`，零网络、零 GPU ✓）：
  1) 每条臂的**房均/局均**是多少？95%CI 多宽？（单位：分/局，按【房】聚合）
  2) 「模型臂 − 规则臂」差多少？CI 含 0 吗？（Welch；若存在同房样本则再加**房间配对**版）
  3) 线上吞吐是多少（房/小时、局/小时）？丢胡牌（`missed_hu`）与节流各多少？
  4) 按当前 SD 与吞吐，**要分辨 +2/+1/+0.5 分/局 各需多少房、多少天**？（M1 的度量）

用法：
    python analyze_live_paired.py                       # 全部历史
    python analyze_live_paired.py --window-hours 24     # 只看最近 24 小时（推荐：口径变了就换窗口）
    python analyze_live_paired.py --model-arm ckpt:rl_bc6.pt --rule-arm shanten:1:1:2:5:1:1:0
    python analyze_live_paired.py --json _tmp/l2.json

⚠ 纪律（写进 `GOAL_STAGE2.md` §5）：
  · 只有 **CI 下界 > 0** 才算"模型真的更强" ✓
  · 不许用 μ / 胡牌率 / 番数 / train 曲线 / **离线 CE** 替代本脚本的结论 ✗
  · 小样本必须把 **样本量 + 保留** 写在结论旁边 ✓
"""
from __future__ import annotations

import argparse

def _default_model_arm():
    """★ 当前模型臂 = `live_ab_arms.txt` 里第一个 `ckpt:` 项（**单一真相源** ✓）。

    写在文件里而不是写死，是因为臂会**热切换**（`run_live.py` 每房间重读 ✓），
    而一个写死的默认值会让分析**静默地看错臂** ✗（2026-09-30 00:36 已发生过一次 ✓）。
    """
    try:
        for g in io.open("live_ab_arms.txt", encoding="utf-8").read().strip().split(","):
            g = g.strip().lstrip("\ufeff")
            if g.startswith("ckpt:"):
                return g
    except Exception:                                                # noqa: BLE001
        pass
    return "ckpt:rl_bc6.pt"

import collections
import datetime as _dt
import io
import json
import math
import statistics as st
import sys

Z95 = 1.959964

# ★★ 2026-10-03（§Z.1373）：小样本必须用 **t 乘子**，不能用 z=1.96 ✗
#   为什么：目标规定「只有 CI 下界 > 0 才算模型更强」⇒ 用 z 会**在 n 很小时虚假转正** ✗✗
#   实测（`rl_v2`，n=2 房）：z 给 [−1.10, +38.40]（**看着接近显著** ✗）而 t(1) 给 [−42.67, +79.97] ✓
#   ⇒ ★ 同一份数据，一个说"快了"，一个说"什么都说明不了" ⇒ 必须用 t ✓
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
       19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
       25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t95(df):
    """df 自由度下的 95% 双侧 t 乘子（df > 30 ⇒ 退化为 z ✓）。"""
    try:
        df = int(df)
    except Exception:
        return Z95
    return T95.get(df, Z95) if df >= 1 else Z95


# --------------------------------------------------------------------- 读数据
def load_rows(path):
    """读 `live_r16.jsonl` ⇒ `[(房间, 臂, [完整观测的逐局分], missed, thr, ended, started, n_all)]`。

    ★ 去重：同一 (房间, 臂) 只保留 `ended` 最大的一条（重跑/续跑会写多条 ✓）
    ★ 容错：坏行、缺字段一律跳过并计数（**不怕崩，怕静默错** —— 所以计数要打印 ✓）
    ★★★★★ **只保留"完整观测"的局**（2026-09-29 实测教训）：
        `deltas_base` 非零 ⇒ 该局有几小局是**我们未参与时打掉的** ✗
        实测：规则臂非零率 0.2~2.3% ✓，而**模型臂 37.2%** ✗✗
        ⇒ 那些局的 `mine` 含"我们缺席时"的分 ⇒ **不可比**，且会系统性拉低该臂 ✗
        ⇒ 默认剔除（`--keep-partial` 可关掉该过滤，仅用于诊断 ✓）
    """
    best, bad, n_partial = {}, 0, collections.Counter()
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:                                       # noqa: BLE001
                bad += 1
                continue
            # ★ 2026-10-05（§Z.1430）：**在源头去掉 BOM** ✗ —— `str.strip()` **不会**去掉 `\ufeff`
            #   （它不属于空白符 ✓）⇒ 之前所有按键查表（`stats[A]` ✓）都因 BOM 而失败 ✗
            #   ⇒ ★ 主口径 β 被**静默跳过** ✓；此处根治后，键/参数/表格三者一致 ✓
            arm = (d.get("genome") or "").lstrip("\ufeff").strip()
            room = (d.get("room") or "").strip()
            if not arm or not room:
                bad += 1
                continue
            allsc, fullsc = [], []
            for r in (d.get("rows") or []):
                if not isinstance(r, dict) or r.get("mine") is None:
                    continue
                v = float(r["mine"])
                allsc.append(v)
                base = r.get("deltas_base") or []
                if sum(abs(float(x)) for x in base) == 0:
                    fullsc.append(v)
            if not allsc:
                bad += 1
                continue
            n_partial[arm] += (len(allsc) - len(fullsc))
            if not fullsc:
                # ★ 该房的局**全部**是部分观测 ⇒ 无法参与可比判据 ⇒ 丢掉并计入剔除 ✓
                bad += 1
                continue
            key = (room, arm)
            ended = float(d.get("ended") or d.get("started") or 0.0)
            if key not in best or ended >= best[key][6]:
                best[key] = (room, arm, fullsc, allsc,
                             int(d.get("missed_hu") or 0), int(d.get("throttled") or 0),
                             ended, float(d.get("started") or ended))
    recs = [v for v in best.values()]
    recs.sort(key=lambda r: r[7])
    return recs, bad, n_partial


def room_mean(rec):
    """一房的代表值 = 该房内**完整观测**逐局分的均值（分/局，按房聚合 ✓）。"""
    return st.mean(rec[2])


# ----------------------------------------------------------------- 统计小工具
def mean_ci(xs):
    """均值 + 95%CI（正态近似）；n<2 时 CI 为 nan。"""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = st.mean(xs)
    if n < 2:
        return m, float("nan"), float("nan"), n
    # ★★ §Z.1373：**样本** SD（÷(n−1)）—— 原来用 pstdev（÷n）会**低估**小样本的不确定性 ✗
    sd = st.stdev(xs)
    se = sd / math.sqrt(n)
    return m, sd, se, n


def welch(a, b):
    """两独立样本均值差 + 95%CI（Welch / 正态近似）。返回 (diff, se, lo, hi)。"""
    ma, sda, sea, na = mean_ci(a)
    mb, sdb, seb, nb = mean_ci(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diff = ma - mb
    se = math.sqrt(sea ** 2 + seb ** 2) if (na > 1 and nb > 1) else float("nan")
    if math.isnan(se):
        return diff, se, float("nan"), float("nan")
    return diff, se, diff - Z95 * se, diff + Z95 * se


def need_rooms(sda, sdb, target):
    """要"分辨 target 分/局"所需的**每臂房数**：1.96²(σa²+σb²)/target² ✓"""
    return (Z95 ** 2) * (sda ** 2 + sdb ** 2) / (target ** 2)


# --------------------------------------------------------------------- 主逻辑
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="live_r16.jsonl")
    ap.add_argument("--window-hours", type=float, default=0.0,
                    help="只看最近 N 小时（>0 生效）。★ 口径/限速/并发变了就换窗口 ✓")
    ap.add_argument("--model-arm", default=_default_model_arm(),
                    help="★ 默认从 live_ab_arms.txt 自动取（单一真相源 ✓）")
    ap.add_argument("--rule-arm", default="shanten:1:1:2:5:1:1:0")
    # ★ 2026-10-05（§Z.1432）：桶宽可调 ✓ —— 用来做 β 的**稳健性检验**（符号是否随桶宽翻转 ✓）
    #   动机：桶数只有 13 ⇒ 主口径不稳 ✗；加宽桶能让"两臂同在场"的桶数变多 ✓（但时段控制变弱 ✗）
    ap.add_argument("--bucket-hours", type=float, default=1.0,
                    help="配对桶宽（小时；默认 1.0 ✓）。>1 会提高同场桶数但削弱时段控制 ✓")
    ap.add_argument("--share", type=float, default=0.5,
                    help="投影时假设「模型臂占场上总吞吐的比例」（A/B 轮换 ⇒ 0.5 ✓）")
    ap.add_argument("--json", default="", help="把汇总写成 JSON（供台账引用 ✓）")
    ap.add_argument("--no-overlap", action="store_true",
                    help="不做重叠期限制（默认会限制到两臂都在场的天数跨度 ✓）")
    ap.add_argument("--keep-partial", action="store_true",
                    help="★ 保留「部分观测」的局（deltas_base 非零 ⇒ 我们缺席过小局）；默认剔除 ✓")
    a = ap.parse_args()

    recs, bad, n_partial = load_rows(a.jsonl)
    if a.keep_partial:                    # 只用于诊断：把"全部局"当"完整局"用
        recs = [(r[0], r[1], r[3], r[3], r[4], r[5], r[6], r[7]) for r in recs]
        n_partial = collections.Counter()
    if not recs:
        print("  ✗ 没有可用记录（文件=%s）" % a.jsonl)
        return 2

    if sum(n_partial.values()):
        print("\n  ★★ 已剔除「部分观测」的局（deltas_base 非零 ⇒ 有几小局是我们未参与时打掉的）")
        print("     理由：那些局的 mine 含「我们缺席时」的分 ⇒ **不可比**，且会系统性拉低该臂 ✗")
        for arm, k in sorted(n_partial.items(), key=lambda kv: -kv[1]):
            print("       %-24s 剔除 %4d 局" % (arm, k))
        print("     （--keep-partial 可关掉该过滤，仅用于诊断 ✓）")

    end = max(r[6] for r in recs)
    if a.window_hours > 0:
        cut = end - a.window_hours * 3600.0
        recs = [r for r in recs if r[6] >= cut]
    span_h = max(1e-9, (end - min(r[7] for r in recs)) / 3600.0)

    print("=" * 78)
    print("线上匹配战绩判据（L2）—— 唯一有权宣布「模型更强」的一层")
    print("=" * 78)
    print("  数据：%s ｜ 房间×臂 记录 %d 条（坏行 %d）" % (a.jsonl, len(recs), bad))
    print("     ⚠ 坏行里绝多数是【测试房】`t_…` 的反复写入 ✗（非产出损失 ✓）；"
          "正式房是 `a_…` ✓ —— 详见 `GOAL_STATUS.md §Z.882`")
    print("  窗口：%s ～ %s（%.2f 小时）"
          % (_dt.datetime.fromtimestamp(min(r[7] for r in recs)).strftime("%m-%d %H:%M"),
             _dt.datetime.fromtimestamp(end).strftime("%m-%d %H:%M"), span_h))

    # ★ 2026-10-05（§Z.1429）：**BOM 规范化** ✗ —— JSONL 里的 genome 可能带 `\ufeff`（写入端偶发 ✓）
    #   而命令行传进来的臂名不带 ⇒ `r[1] == arm` **永远为假** ⇒ 报"两臂没有任何同一天在场" ✗✗
    #   ⇒ 这会让**主口径（小时固定效应 β）静默失效** ✗ —— 而那正是 goal §0 指定的判据 ✓
    A, B = a.model_arm.lstrip("\ufeff").strip(), a.rule_arm.lstrip("\ufeff").strip()
    # ★★★★★ 重叠期限制（**判据正确性的必需项**，2026-09-29 实测教训）：
    #   两臂若不在同一时期跑，Δ 就被"时段效应"混淆 ✗
    #   实测：模型臂只上场 4 天（09-26 起），规则臂 9 天 ⇒ 历史 Δ 混了两个时代 ✗
    #   ⇒ 默认：只保留【两臂都在场】的日期跨度内的记录 ✓
    if not a.no_overlap:
        def _days(arm, _rs=None):
            return {int(r[6] // 86400) for r in (_rs if _rs is not None else recs)
                    if r[1].lstrip("\ufeff").strip() == arm}
        da, db = _days(A), _days(B)
        both = sorted(da & db)
        if both:
            lo_d, hi_d = both[0] * 86400, both[-1] * 86400 + 86399
            keep = [r for r in recs if lo_d <= r[6] <= hi_d]
            nA = sum(1 for r in keep if r[1] == A)
            nB = sum(1 for r in keep if r[1] == B)
            if len(both) < len(da | db):
                print("\n  ★★ 重叠期限制：两臂同时在场仅 **%d** 天（模型 %d 天 / 规则 %d 天）✗"
                      % (len(both), len(da), len(db)))
                print("     ⇒ 只保留 %s ～ %s 的记录（模型 %d 房 / 规则 %d 房）✓"
                      % (_dt.datetime.fromtimestamp(lo_d).strftime("%m-%d"),
                         _dt.datetime.fromtimestamp(hi_d).strftime("%m-%d"), nA, nB))
                print("     ⚠ 若两臂在场天数差别大 ⇒ 非重叠期的 Δ **不可比**（时段混淆 ✗）")
                recs = keep
                span_h = max(1e-9, (max(r[6] for r in recs) - min(r[7] for r in recs)) / 3600.0)
        else:
            print("\n  ⚠ 两臂**没有任何同一天在场** ⇒ 无法做可比裁决 ✗（先让两臂同时跑 ✓）")

    # 重新按臂聚合（重叠期限制后 ✓）
    by_arm = collections.defaultdict(list)
    for r in recs:
        by_arm[r[1]].append(r)

    # --- 按臂聚合
    stats = {}
    print("\n  --- 每条臂（单位：分/局；按【房】聚合）---")
    for arm, rs in sorted(by_arm.items(), key=lambda kv: -len(kv[1])):
        rms = [room_mean(r) for r in rs]
        m, sd, se, n = mean_ci(rms)
        ngames = sum(len(r[2]) for r in rs)
        gm = st.mean([g for r in rs for g in r[2]])
        miss = sum(r[4] for r in rs)
        thr = sum(r[5] for r in rs)
        tput = len(rs) / span_h
        stats[arm] = dict(rooms=n, games=ngames, room_mean=m, room_sd=sd, se=se,
                          lo=m - _t95(n - 1) * se if n > 1 else float("nan"),
                          hi=m + _t95(n - 1) * se if n > 1 else float("nan"),
                          game_mean=gm, missed_hu=miss, throttled=thr,
                          rooms_per_h=tput)
        ci = ("[%+.2f, %+.2f]" % (m - _t95(n - 1) * se, m + _t95(n - 1) * se)) if n > 1 else "(n<2)"
        print("    %-24s 房=%3d 局=%4d ｜ 房均 %+.3f ｜ SD %5.2f ｜ SE %.3f ｜ 95%%CI %s"
              % (arm, n, ngames, m, sd if n > 1 else float("nan"), se if n > 1 else float("nan"), ci))
        print("      %-22s 局均 %+.3f ｜ 丢胡牌 %d ｜ 节流 %d ｜ %.2f 房/小时"
              % ("", gm, miss, thr, tput))

    # --- 两臂差（Welch；若有同房样本再加配对版）
    # ★ 2026-10-05（§Z.1429）：**BOM 规范化** ✗ —— JSONL 里的 genome 可能带 `\ufeff`（写入端偶发 ✓）
    #   而命令行传进来的臂名不带 ⇒ `r[1] == arm` **永远为假** ⇒ 报"两臂没有任何同一天在场" ✗✗
    #   ⇒ 这会让**主口径（小时固定效应 β）静默失效** ✗ —— 而那正是 goal §0 指定的判据 ✓
    A, B = a.model_arm.lstrip("\ufeff").strip(), a.rule_arm.lstrip("\ufeff").strip()
    print("\n  --- 主口径：模型臂 − 规则臂 ---")
    result = {}
    if A in stats and B in stats:
        ra = [room_mean(r) for r in by_arm[A]]
        rb = [room_mean(r) for r in by_arm[B]]
        diff, se, lo, hi = welch(ra, rb)
        verdict = ("★ 模型**更强**（CI 下界 > 0）" if (lo == lo and lo > 0)
                   else ("★ 模型**更差**（CI 上界 < 0）" if (hi == hi and hi < 0)
                         else "⇒ **未分辨**（CI 含 0）"))
        print("    对照（Welch 非配对）：Δ = %+.3f ｜ SE %.3f ｜ 95%%CI [%+.3f, %+.3f]  %s"
              % (diff, se, lo, hi, verdict))
        if lo == lo and lo > 0:
            print("    ⇒ 按 GOAL_STAGE2 §0.1：**CI 下界 > 0 ⇒ 可以宣布模型更强** ✓")
        else:
            n_need = need_rooms(stats[A]["room_sd"], stats[B]["room_sd"], abs(diff)) if diff else float("nan")
            if n_need == n_need:
                print("    ⇒ 若真实差就是 %+.3f，要把它分辨出来需**每臂约 %.0f 房**"
                      % (diff, n_need))
        result["welch"] = dict(diff=diff, se=se, lo=lo, hi=hi)

        # --- ★★★★★ 2026-09-30：**小时固定效应（组内）估计量**（`§Z.873`）---
        #   动机：现有"桶均值的均值"**无视桶内房数** ✗（1 房与 3 房同等对待 ✓）
        #     ⇒ 桶级 SD 在 20 分钟内从 8.27 跳到 14.17 ✗ ⇒ 极不稳 ✓
        #   β = Σ_h w_h·(ȳ_模型,h − ȳ_规则,h) / Σ_h w_h，w_h = n_m·n_r/(n_m+n_r) ✓
        #   ⚠ β 只由"同小时内两臂都有房"的变异识别 ✓；同方差假设 ⇒ SE 可能略乐观 ✓
        #   ⚠ 纯 Python 实现（该分析器不 import 训练模块 ⇒ numpy 未必可用 ✗）
        _hm = collections.defaultdict(list)
        _hr2 = collections.defaultdict(list)
        _bw = max(1.0, a.bucket_hours * 3600)      # ★ 主口径 β 的桶宽（§Z.1432 ✓）
        for r in recs:
            if r[1] == A:
                _hm[int(r[6] // _bw)].append(room_mean(r))
            elif r[1] == B:
                _hr2[int(r[6] // _bw)].append(room_mean(r))
        _both_h = [h for h in _hm if h in _hr2]
        if len(_both_h) >= 3:
            _num = _den = 0.0
            _ss = 0.0
            _cnt = 0
            for h in _both_h:
                _a, _b2 = _hm[h], _hr2[h]
                _na, _nb = len(_a), len(_b2)
                _w = _na * _nb / float(_na + _nb)
                _ma, _mb = st.mean(_a), st.mean(_b2)
                _num += _w * (_ma - _mb)
                _den += _w
                _ss += sum((x - _ma) ** 2 for x in _a) + sum((x - _mb) ** 2 for x in _b2)
                _cnt += _na + _nb
            _beta = _num / _den
            _dof = max(1, _cnt - len(_both_h) - 1)
            _se3 = ((_ss / _dof) / _den) ** 0.5
            _lo3 = _beta - _t95(_dof) * _se3
            _hi3 = _beta + _t95(_dof) * _se3
            # ★★ 2026-10-05（§Z.1439）：**退化保护** ✗ —— `_ss` 只计桶**内**方差 ✓
            #   若每桶各臂仅 1 房 ⇒ `_ss = 0` ⇒ **SE = 0** ⇒ CI 塌缩成一点 ⇒ 报出**假显著** ✗✗
            #   实测（`--bucket-hours 0.25`）：β = −11.133、SE = 0.000、判"模型更差" ✗ —— **这是假的** ✓
            #   ⇒ ★ **退化时拒绝给判定** ✗；桶数不足时也不给判定 ✓（只报数 ✓）
            _degen = (_se3 <= 0.0) or (_ss <= 0.0)
            if _degen:
                _v3 = "⚠ **口径失效（桶内方差为 0 ⇒ 每桶各臂仅 1 房 ✗）⇒ 不给判定** ✗"
            elif len(_both_h) < 5:
                _v3 = "⚠ **桶数 < 5 ⇒ 不给判定** ✗（加宽桶或等更多数据 ✓）"
            else:
                _v3 = ("★ 模型**更强**（CI 下界 > 0）" if _lo3 > 0
                       else ("★ 模型**更差**（CI 上界 < 0）" if _hi3 < 0 else "⇒ **未分辨**（CI 含 0）"))
            print("    ★★★ **主口径（小时固定效应 / 组内）**：β = %+.3f ｜ SE %.3f ｜ 95%%CI [%+.3f, %+.3f]  %s"
                  % (_beta, _se3, _lo3, _hi3, _v3))
            if (not _degen) and len(_both_h) < 30:
                print("       ⚠ 桶数 %d < 30 ⇒ **桶级不稳** ✗（`§Z.868`：桶数够多后此表才可靠 ✓）"
                      % len(_both_h))
            print("       （n=%d 房 ｜ %d 个两臂同在场的小时 ｜ dof=%d；权重 w_h = n_m*n_r/(n_m+n_r) "
                  "⇒ **按房数正确加权** ✓，修掉「桶均值的均值」的毛病 ✓）"
                  % (_cnt, len(_both_h), _dof))
            result["hour_fe"] = dict(beta=_beta, se=_se3, lo=_lo3, hi=_hi3,
                                     n_rooms=_cnt, n_hours=len(_both_h))
            print("       ★ **这是统计学上正确的配对口径** ✓（含桶内房间噪声 ✓）；⚠ 但其识别力受「两臂同在场的小时数」限制 ✗")
        else:
            print("    ⚠ 小时固定效应：两臂同在场的小时 = %d（< 3）⇒ 略过 ✓" % len(_both_h))

        # --- ★ 时间配对（不需要改客户端：两臂虽不同房，但同一小时面对同一批玩家/口径/负载）---
        buckets = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in recs:
            b = int(r[6] // max(1.0, a.bucket_hours * 3600))   # ★ 桶宽可调（§Z.1432 ✓）
            buckets[b][r[1]].append(room_mean(r))
        bd = []
        for b, d in sorted(buckets.items()):
            if A in d and B in d:
                bd.append(st.mean(d[A]) - st.mean(d[B]))
        result["time_paired_buckets"] = len(bd)
        if len(bd) >= 3:
            m2, sd2, se2, n2 = mean_ci(bd)
            lo2 = m2 - _t95(n2 - 1) * se2
            hi2 = m2 + _t95(n2 - 1) * se2
            v = ("★ 模型**更强**（CI 下界 > 0）" if lo2 > 0
                 else ("★ 模型**更差**（CI 上界 < 0）" if hi2 < 0 else "⇒ **未分辨**（CI 含 0）"))
            print("    ★★ 诊断口径（桶均值的均值 —— **SE 偏乐观 ✗**）：n=%d 桶 ｜ Δ = %+.3f ｜ SE %.3f ｜ "
                  "95%%CI [%+.3f, %+.3f]  %s" % (n2, m2, se2, lo2, hi2, v))
            print("       （依据 `§Z.868`：同一次分析里小时配对把 SE 9.18→4.24 ⇒ "
                  "**省样本 4.70 倍**；它控制了「时期」这一最大混淆 ⇒ 主用它 ✓；"
                  "桶数 < 3 时回落到上面的非配对对照 ✓）")
            if se > 0 and se2 > 0:
                print("       ⇒ 与非配对相比：SE %.3f → **%.3f** ⇒ 等效**省样本 %.2f 倍** ✓"
                      % (se, se2, (se / se2) ** 2))
                print("       （这是**无需改动客户端**就能拿到的方差下降 ✓ —— 见 GOAL_STAGE2 §4.2）")
            print("       ⚠⚠ **但本口径的 SE 偏乐观** ✗：它把每个桶的均值当成一个「干净观测」✗，\n            **无视了「桶均值自身的房间噪声」** ⇒ 请看上一行的**组内估计量**（那才是统计学正确的 ✓）")
        else:
            print("    ⚠ 时间配对可用桶数 = %d（两臂同时在场的整点太少）⇒ 先用非配对" % len(bd))


        per_room = collections.defaultdict(dict)
        for r in recs:
            per_room[r[0]][r[1]] = room_mean(r)
        pairs = [(v[A], v[B]) for v in per_room.values() if A in v and B in v]
        result["paired_rooms"] = len(pairs)
        if len(pairs) >= 3:
            ds = [x - y for x, y in pairs]
            m2, sd2, se2, n2 = mean_ci(ds)
            print("    ★ 房间配对：n=%d 房 ｜ Δ = %+.3f ｜ SE %.3f ｜ 95%%CI [%+.3f, %+.3f]"
                  % (n2, m2, se2, m2 - _t95(n2 - 1) * se2, m2 + _t95(n2 - 1) * se2))
            print("       （配对后 SE 通常远小于非配对 ⇒ 这是 §4.2 要争取的协议 ✓）")
        else:
            print("    ⚠ 房间配对样本 = %d ⇒ **当前是非配对比较** ✗" % len(pairs))
            print("       ⇒ 按 GOAL_STAGE2 §4.2：应让两臂进【同一房】（最好【同一桌】）以降方差 ✓")

    # --- 吞吐
    total_rooms = len(recs)
    total_games = sum(len(r[2]) for r in recs)
    print("\n  --- 吞吐（决定 M1 能不能达标）---")
    print("    合计 %d 房 / %d 局 / %.2f 小时 ⇒ **%.2f 房/小时 ｜ %.1f 局/小时**"
          % (total_rooms, total_games, span_h, total_rooms / span_h, total_games / span_h))

    # --- 方差分解：房间间 vs 房间内（决定 §4.2「配对」能省多少样本 ✓）
    print("\n  --- 方差分解（判断 §4.2「两臂同房/同桌」能省多少样本）---")
    print("    模型：σ²_房间 = σ²_房间间 + σ²_房间内/n_games")
    decomp = {}
    for arm, rs in sorted(by_arm.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 5:
            continue
        rms = [room_mean(r) for r in rs]
        s_room = st.pstdev(rms)
        n_g = st.mean([len(r[2]) for r in rs])
        # 房间内方差（对每房取总体方差后取均值）
        win = st.mean([st.pvariance(r[2]) if len(r[2]) > 1 else 0.0 for r in rs])
        within_contrib = win / n_g
        between_var = max(0.0, s_room ** 2 - within_contrib)
        decomp[arm] = dict(s_room=s_room, s_within=math.sqrt(win), n_games=n_g,
                           within_contrib=math.sqrt(within_contrib),
                           s_between=math.sqrt(between_var))
        print("    %-24s σ_房间 %5.2f（%d 房，均 %.1f 局/房）"
              % (arm, s_room, len(rs), n_g))
        print("      %-22s 房间内 σ_逐局 %5.2f ⇒ 贡献 %5.2f ｜ **房间间 σ %5.2f**"
              % ("", math.sqrt(win), math.sqrt(within_contrib), math.sqrt(between_var)))
    if (A in decomp) and (B in decomp):
        # ★ 正确模型：Score_臂,房 = μ_臂 + R_房 + ε（R = 该房的共有冲击，两臂若同房则抵消）
        #   非配对：Var(Δ̂) = (σ²A,房间 + σ²B,房间)/n
        #   配对  ：Var(Δ̂) = (σ²A,内  + σ²B,内 )/n     ← R 抵消
        #   ⚠ 但两臂各自反推出的 σ_R 不一致（14.11 vs 8.13）⇒ 取**较小者**（保守）✓
        sR2_a = max(0.0, decomp[A]["s_room"] ** 2 - decomp[A]["within_contrib"] ** 2)
        sR2_b = max(0.0, decomp[B]["s_room"] ** 2 - decomp[B]["within_contrib"] ** 2)
        sR2 = min(sR2_a, sR2_b)                      # 保守：共有成分取小
        pair_var = decomp[A]["within_contrib"] ** 2 + decomp[B]["within_contrib"] ** 2
        unpr_var = decomp[A]["s_room"] ** 2 + decomp[B]["s_room"] ** 2
        print("    ⇒ 共有（房间间）成分 σ_R：模型反推 %.2f ｜ 规则反推 %.2f ⇒ **保守取 %.2f**"
              % (math.sqrt(sR2_a), math.sqrt(sR2_b), math.sqrt(sR2)))
        print("    ⇒ 非配对 σ_Δ = %.2f ｜ **配对后 σ_Δ = %.2f** ⇒ 省样本 **%.2f 倍**"
              % (math.sqrt(unpr_var), math.sqrt(pair_var),
                 (unpr_var / pair_var) if pair_var > 0 else float("nan")))
        print("       ⚠ σ_R 取小 ⇒ 这是**保守**估计；真实收益必须等协议改造后**实测** ✓")
        if total_rooms > 0 and span_h > 0:
            tput_arm = (total_rooms / span_h) * a.share
            for tgt in (1.0,):
                n_now = (Z95 ** 2) * unpr_var / (tgt ** 2)
                n_pr = (Z95 ** 2) * pair_var / (tgt ** 2)
                print("      分辨 %+.2f 分/局：非配对 %5.0f 房 ⇒ **%.1f 天** ｜ "
                      "配对 %5.0f 房 ⇒ **%.1f 天**"
                      % (tgt, n_now, n_now / tput_arm / 24.0 if tput_arm > 0 else float("inf"),
                         n_pr, n_pr / tput_arm / 24.0 if tput_arm > 0 else float("inf")))

    # --- M1 分辨力（M1 的度量）
    print("\n  --- M1 分辨力：按当前 SD 与吞吐，要「分辨 Δ 分/局」需要多久 ---")
    if A in stats and B in stats:
        sda, sdb = stats[A]["room_sd"], stats[B]["room_sd"]
        tput_arm = (total_rooms / span_h) * a.share
        print("    σ_模型 %.2f ｜ σ_规则 %.2f ｜ 假设模型臂吞吐 = 总吞吐 × %.2f = **%.2f 房/小时**"
              % (sda, sdb, a.share, tput_arm))
        for tgt in (2.0, 1.0, 0.5, 0.25):
            n = need_rooms(sda, sdb, tgt)
            hrs = n / tput_arm if tput_arm > 0 else float("inf")
            flag = "✅ 达标" if (tgt >= 1.0 and hrs / 24.0 <= 14.0) else ""
            print("      分辨 %+.2f 分/局 ⇒ 每臂 %6.0f 房 ⇒ **%.1f 天** %s"
                  % (tgt, n, hrs / 24.0, flag))
        print("    ⇒ ★ M1 通过线：**分辨 1 分/局 ≤ 14 天**（现状见上）")
        # ★★★★★ 2026-09-30：**同时给出「小时配对」口径的 M1 时间线**（`§Z.872`）
        #   为什么必须加：上表用的是**非配对房间级 SD** ⇒ 给出 ~56 天 ✗，
        #   而**小时配对实测桶级 SD = 8.27 分/局** ⇒ ~262 桶 ⇒ **~10.9 天** ✓
        #   ⇒ 只看上表会误判"M1-目标 不可达" ✗
        #   ⚠ 桶数 < 30 时明确标注"不稳" ✓（否则会被当成"已达标" ✗）
        #   ⚠ 桶速率取 **24/天**（上限口径：两臂每轮交替 ⇒ 每小时都该有模型臂 ✓）
        if len(bd) >= 3:
            _sd_b = st.stdev(bd)
            _flag = "✓" if len(bd) >= 30 else " ⚠ **桶数 <30 ⇒ 不稳** ✗"
            print("    ★★ **小时配对口径**（桶级 SD = **%.2f 分/局** ｜ 桶数 %d%s）"
                  % (_sd_b, len(bd), _flag))
            for tgt in (2.0, 1.0, 0.5):
                _n = (_sd_b / (tgt / Z95)) ** 2
                _d = _n / 24.0
                _ok = "✅ 达标" if (tgt >= 1.0 and _d <= 14.0) else ""
                print("      分辨 %+.2f 分/局 ⇒ 需 %5.0f 桶 ⇒ **%.1f 天** %s"
                      % (tgt, _n, _d, _ok))
            print("      （桶速率按上限 24/天 ✓；桶数够多后此表才是可靠判据 ✓）")

    # --- 台账用 JSON
    if a.json:
        out = dict(generated=end, span_hours=span_h, total_rooms=total_rooms,
                   total_games=total_games, bad_lines=bad,
                   rooms_per_hour=total_rooms / span_h, arms=stats, compare=result)
        with io.open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\n  ✓ 已写 %s（供 GOAL_STATUS.md 引用）" % a.json)

    print("\n  判读：**只看两臂差的 CI**。CI 下界 > 0 ⇒ 模型更强（这才是部署依据）；")
    print("      含 0 ⇒ 如实说未分辨，**不许**拿 μ / 胡牌率 / 番数 / CE 替它下结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
