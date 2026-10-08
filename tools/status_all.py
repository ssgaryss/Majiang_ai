# -*- coding: utf-8 -*-
"""tools/status_all.py -- 一条命令看全况（§6 的"每轮必产出"清单自动化）

读**本地文件**即可（不联网、不加负载 ✓ 硬规则⑥⑪ ✓）：
  ① 进程与新鲜度：python 进程、线上日志 age、各训练日志 age
  ② 线上：当前臂（读 `live_ab_arms.txt`）、最新"轮"横幅、**该臂**的房均 + 诚实 95%CI
  ③ ③ 的 A/B：两臂逐轮 L1 列表 + 预注册判定（调 `tools/ab_value_head.py` 的逻辑）
  ④ 里程碑一行（写死在下方 §M，改数字要连着台账一起改 ✓）

用法：python tools/status_all.py [--hours 3]
⚠ 诚实性 ✓：CI 用**样本** SD（÷(n−1)）＋ **t 乘子**（§Z.1373 ✓）；n<2 不给 CI ✓
⚠ 沙箱 ✓：本脚本只读文件；若要落日志，用 `tools/run_train.py`（进程内重定向 ✓）
"""
import argparse
import collections
import io
import json
import math
import os
import re
import statistics as st
import sys
import time

RE_ROUND = re.compile(r"第\s*(\d+)\s*/\s*(\d+)\s*轮（策略\s*([^\s）]+)）")
RE_EVAL = re.compile(r"\[配对评估\s*迭代(\d+)\].*?Δμ\(分/局\)\s*=\s*([+-]?\d+\.?\d*)\s*"
                     r"95%CI\s*\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]")
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
       14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 25: 2.060, 30: 2.042}


def t95(df):
    return T95.get(int(df), 1.959964) if df and int(df) >= 1 else 1.959964


def age_min(path):
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 60.0


def tail_lines(path, n):
    if not os.path.exists(path):
        return []
    out = collections.deque(maxlen=n)
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            out.append(ln.rstrip("\n"))
    return list(out)


def read_arms(path="live_ab_arms.txt"):
    try:
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            return [x.strip() for x in fh.read().replace("\ufeff", "").split(",") if x.strip()]
    except Exception:
        return []


def all_arm_rooms(jsonl, arms, hours):
    """★ 单次扫描 ✓ 返回 {臂: (房均列表, missed, throttled, 局数)}
    （原来对每条臂各扫一遍 40MB 的 JSONL ✗ ⇒ 每轮要好几分钟 ✗ ⇒ 改成一次扫描 ✓ §Z.1382）"""
    want = set(arms)
    cut = time.time() - hours * 3600.0
    out = {a: ([], 0, 0, 0) for a in arms}
    if not os.path.exists(jsonl):
        return out
    with io.open(jsonl, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            if '"a_' not in ln:          # 快速跳过测试房 t_… ✓
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if not str(d.get("room") or "").startswith("a_"):
                continue
            if (d.get("ended") or 0) < cut:
                continue
            g = (d.get("genome") or "").lstrip("\ufeff").strip()
            if g not in want:
                continue
            rows = []
            for r in (d.get("rows") or []):
                base = r.get("deltas_base") or []
                if sum(abs(float(x)) for x in base) == 0:
                    rows.append(r)
            if not rows:
                continue
            mine = [r.get("mine") for r in rows if r.get("mine") is not None]
            if not mine:
                continue
            means, missed, thr, games = out[g]
            means.append(sum(mine) / float(len(mine)))
            out[g] = (means, missed + int(d.get("missed_hu") or 0),
                      thr + int(d.get("throttled") or 0), games + len(mine))
    return out


def _arm_rooms_unused(jsonl, arm, hours):
    """（保留旧实现以备对照；主流程已改用 all_arm_rooms ✓）"""
    cut = time.time() - hours * 3600.0
    means, missed, thr, games = [], 0, 0, 0
    if not os.path.exists(jsonl):
        return means, missed, thr, games
    with io.open(jsonl, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            room = str(d.get("room") or "")
            if not room.startswith("a_"):
                continue
            if (d.get("ended") or 0) < cut:
                continue
            g = (d.get("genome") or "").lstrip("\ufeff").strip()
            if g != arm:
                continue
            rows = []
            for r in (d.get("rows") or []):
                base = r.get("deltas_base") or []
                if sum(abs(float(x)) for x in base) == 0:
                    rows.append(r)
            if not rows:
                continue
            mine = [r.get("mine") for r in rows if r.get("mine") is not None]
            if not mine:
                continue
            means.append(sum(mine) / float(len(mine)))
            games += len(mine)
            missed += int(d.get("missed_hu") or 0)
            thr += int(d.get("throttled") or 0)
    return means, missed, thr, games


def l1s(path):
    out = {}
    if not os.path.exists(path):
        return out
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = RE_EVAL.search(ln)
            if m:
                out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    a = ap.parse_args(argv)

    print("=" * 78)
    print("全况速览（§6 清单）｜ %s ｜ 窗口 %.1f 小时" % (time.strftime("%m-%d %H:%M:%S"), a.hours))
    print("=" * 78)

    # ① 新鲜度
    for f in ("live_r16.log", "live_r16.jsonl", "rl_vA.log", "rl_vB.log", "rl_dl2c.log"):
        ag = age_min(f)
        print("  %-18s %s" % (f, ("%.1f 分钟前" % ag) if ag is not None else "（不存在）"))

    # ② 线上
    arms = read_arms()
    print("\n  --- 线上 ---")
    print("  臂文件 = %s" % (", ".join(arms) if arms else "(读不到)"))
    cur = None
    for ln in reversed(tail_lines("live_r16.log", 400)):
        m = RE_ROUND.search(ln.lstrip("\ufeff"))
        if m:
            cur = (m.group(1), m.group(2), m.group(3).lstrip("\ufeff").strip())
            break
    print("  最新轮 = %s" % (("第 %s / %s 轮（%s）" % cur) if cur else "（未找到）"))
    # ★★★ 2026-10-07（§Z.1757）：★ **不能只读 `live_r16.jsonl`** ✗✗
    #   · ★ 客户端每换一次日志名（`e`/`f`/`g` … ✓ §Z.1452 ✓）⇒ **新房在别的文件里** ✗
    #   · ★ 旧写法 `all_arm_rooms("live_r16.jsonl", ...)` ⇒ ★★ **静默漏房** ✗
    #     （★ 实测：漏了 14 房 ⇒ 见 §Z.1755 ✓）
    #   ⇒ ✓ 现在：★ **扫所有 `live_r16*.jsonl` 并合并** ✓
    import glob as _glob
    _files = sorted(_glob.glob("live_r16*.jsonl"))
    rooms = {}
    for _p in _files:
        for _arm, _v in all_arm_rooms(_p, arms, a.hours).items():
            _means, _missed, _thr, _games = _v
            if _arm in rooms:
                _m0, _ms0, _t0, _g0 = rooms[_arm]
                rooms[_arm] = (_m0 + list(_means), _ms0 + _missed, _t0 + _thr, _g0 + _games)
            else:
                rooms[_arm] = (list(_means), _missed, _thr, _games)
    print("  （★ 已扫 %d 个 `live_r16*.jsonl` ✓）" % len(_files))
    for arm in arms:
        if not arm:
            continue
        means, missed, thr, games = rooms.get(arm, ([], 0, 0, 0))
        n = len(means)
        if n == 0:
            print("  %-24s 房=0（本窗口无完整房）" % arm)
            continue
        m = st.mean(means)
        if n < 2:
            print("  %-24s 房=%d ｜ 房均 %+.2f ｜ **n<2 ⇒ 不给 CI** ✓ ｜ 局=%d ｜ 丢胡牌 %d ｜ 节流 %d"
                  % (arm, n, m, games, missed, thr))
            continue
        sd = st.stdev(means)
        se = sd / math.sqrt(n)
        t = t95(n - 1)
        print("  %-24s 房=%d ｜ 房均 %+.3f ｜ SD %.2f ｜ SE %.3f ｜ 95%%CI [%+.2f, %+.2f] ｜ 局=%d ｜ 丢胡牌 %d ｜ 节流 %d"
              % (arm, n, m, sd, se, m - t * se, m + t * se, games, missed, thr))

    # ③ A/B
    la, lb = l1s("rl_vA.log"), l1s("rl_vB.log")
    print("\n  --- ③ 价值头 A/B（arm A 对照 ／ arm B 唯一变量 --value-init）---")
    print("  arm A L1 条数 = %d ｜ arm B L1 条数 = %d" % (len(la), len(lb)))
    for tag, d in (("A", la), ("B", lb)):
        for i in sorted(d):
            print("    %s 迭代%-2d  Δμ %+8.4f  CI [%+.4f, %+.4f]" % (tag, i, d[i][0], d[i][1], d[i][2]))
    common = sorted(set(la) & set(lb))
    if len(common) >= 1:
        ds = [lb[i][0] - la[i][0] for i in common]
        n = len(ds)
        m = st.mean(ds)
        if n >= 2:
            sd = st.stdev(ds)
            se = sd / math.sqrt(n)
            t = t95(n - 1)
            lo, hi = m - t * se, m + t * se
        else:
            lo = hi = float("nan")
        print("  ★ 主读数 mean(D) = %+.4f ｜ n = %d ｜ 95%%CI [%s]" % (m, n, ("%+.4f, %+.4f" % (lo, hi)) if n >= 2 else "n<2"))
        if n < 8:
            print("  ★ 判定 = 未完成（n=%d/8）⇒ **不出判定** ✓（部分均值仅供参考 ✗）" % n)
        elif m >= 0.5 and lo > 0:
            print("  ★ 判定 = ③ 有效 ⇒ 换臂上线 ⇒ 记 L2 ✓")
        elif m <= -0.5 and hi < 0:
            print("  ★ 判定 = ③ 有害 ⇒ 换队列② rl_pool ✓")
        else:
            print("  ★ 判定 = 未分辨 ⇒ 按 goal §3 熔断换类 ✓")
    else:
        print("  ⚠ 还没有可配对的轮次（arm B 尚未开始 ✓）⇒ 什么都不判 ✓")

    # ④ 里程碑（改这里必须连着台账一起改 ✓）
    print("\n  --- 里程碑 ---")
    _ma = next((x for x in arms if x.startswith("ckpt:")), "(未知)")
    print("  M4 ✅（线上模型臂 %s 来自 PPO/自对弈 ✓ ｜ ★ 换臂后请看臂文件的当前值 ✓）" % _ma)
    print("  M1-实用 ✅ 96h：+2 分/局 ⇒ 13.4 天 ｜ +1 分/局 ⇒ 53.4 天 ✗")
    print("  M1-目标 ✗ ｜ M2 ✗ ｜ M3 ✗（β +0.108，CI 含 0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
