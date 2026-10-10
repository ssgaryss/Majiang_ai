#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""线上可信基线 —— 原模型与新模型必须用同一口径。

为什么需要它（问题陈述）
------------------------
线上 `live*.jsonl` 里每一场的 `mine` 是**最终累计分**。客户端中途接管一场
进行中的对局时，那条记录里包含**接管之前**的得失 —— 那是别人打的，不能算在
当前策略头上。`analyze_live_paired.py` 已经用 `deltas_base` 剔除了这类局，
但那只解决了一半问题：它仍以**客户端视角**的 `mine` 为准，而且没有给出
"接管后到底打了多少分"这个单独的数。

本工具建立三份**互不混淆**的账：

  1. 【主指标】每局净得分（分/局）
     来源 = `reviews/*.json`（服务端房间复盘，逐局含 winner / fan / 四家得失）。
     ★ 关键：复盘派生自**官方房间日志**，因此**天然不含"接管前分数"偏差** ——
       它是按座位解析好的逐局结果，不是客户端累计分的差分。
     口径 = 该臂所有**全程无接管**房间的全部局，按**房间聚类**求 95% CI
            （同房间的局共享累计分与连庄链，不是独立样本）。
     为兼容既有台账，同时给出 `live*.jsonl` 的客户端口径（完整观测局）作对照。

  2. 【接管账】接管后的得分变化
     口径 = `mine − sum(deltas_base[seat])`，按房单独统计。
     ★ 它**不参与**主指标，只用于回答"接管之后我们打得好不好"。

  3. 【执行账】提交 / 放弃 / 陈旧 / 窗口错过 / 409 / 杠被拒
     来源 = 客户端日志的每场汇总行，按该场所属**策略臂**归属
     （臂从日志里的 `########## 第 i / n 轮（策略 X）##########` 读出）。

用法
----
    python tools/live_baseline.py \
        --jsonl "live*.jsonl" \
        --reviews "reviews/*.json" \
        --logs "live*.log" \
        --model-arm ckpt:rl_aux_0004.pt --rule-arm shanten:1:1:2:5:1:1:0

    # 路径按**当前工作目录**解析；要读别处的数据就自己把 glob 指过去，
    # 本脚本不写死任何绝对路径（曾经在示例里写死过开发仓路径，已改）。

设计纪律（沿用本仓既有判据）
--------------------------
· 只有 CI 下界 > 0 才算"真的更强"，否则记「未分辨」。
· 不挑局：所有局都进统计，输的局一并不剔除。
· 口径与臂一起打印；任何被剔除的数据都计入并显示（不怕崩，怕静默错）。
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
import re
import statistics as st
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Z95 = 1.959963984540054

# ★ 沙箱事实：本环境里 `& python ... > file` 会**挂住**（stdout 重定向被拦 ⚠）。
#   ⇒ 报告由**进程自己**落盘（`--report`），不依赖 shell 重定向。
_OUT = []
_bprint = builtins.print


def print(*args, **kw):                                              # noqa: A001
    """所有输出同时进 stdout 与 `--report` 文件。"""
    for h in _OUT:
        kw2 = dict(kw)
        kw2["file"] = h
        _bprint(*args, **kw2)
    _bprint(*args, **kw)

# 日志里的每场汇总行：
#   场 <gid> 结束：N 行动 / M 杠(被拒K) / P 窗口错过 / Q 次409 / R 推送
#       放弃a 陈旧b 提交c 过d 延迟 中位e/p99f/最大g ms ...
RE_SUMMARY = re.compile(
    r"结束：(\d+) 行动 / (\d+) 杠\(被拒(\d+)\) / (\d+) 窗口错过 / (\d+) 次409"
    r" / (\d+) 推送 放弃(\d+) 陈旧(\d+) 提交(\d+) 过(\d+)"
    r" 延迟 中位(\d+)/p99(\d+)/最大(\d+)"
    # ★ 2026-10-09 新增的两个尾部计数（旧日志没有 ⇒ 必须可选 ✗ 否则解析器全废）
    #   · 网络     = 被网络抖动打断、但重取状态后救回来的次数
    #   · 提交超时 = 其中发生在动作提交上的子集
    r"(?:.*?网络(\d+) 提交超时(\d+))?")
RE_ROUND_HDR = re.compile(r"#+ 第 \d+ / \d+ 轮（策略 (.+?)）")
RE_ARM_PREFIX = re.compile(r"\[主\|([^\]]+)\]")


# ============================================================ 通用
def dedup_paths(paths):
    """按 SHA256 去掉内容完全相同的副本（`*.log` 与 `*.stdout.log` 是同一份）。"""
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


def expand(patterns):
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return sorted(set(out))


def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), n
    m = st.mean(xs)
    if n < 2:
        return m, float("nan"), n
    se = st.stdev(xs) / math.sqrt(n)
    return m, Z95 * se, n


def fmt_ci(m, h, n, nd=3):
    if n == 0:
        return "     —      (n=0)"
    if math.isnan(h):
        return "%+9.*f      —      (n=%d)" % (nd, m, n)
    return "%+9.*f ± %-8.*f (n=%d)" % (nd, m, nd, h, n)


# ============================================================ 读 jsonl
def load_jsonl(patterns):
    """⇒ (room→臂, room→统计, 计数)。

    ★ 去重：同一 (房间, 臂) 只保留 `ended` 最大的一条（重跑/续跑会写多条）。
    """
    files = expand(patterns)
    files, dup = dedup_paths(files)
    best = {}
    bad = 0
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
                ended = float(d.get("ended") or d.get("started") or 0.0)
                key = (room, arm)
                if key not in best or ended >= best[key]["ended"]:
                    best[key] = {"ended": ended, "arm": arm, "room": room,
                                 "rows": d.get("rows") or [],
                                 "started": float(d.get("started") or ended),
                                 "missed_hu": int(d.get("missed_hu") or 0),
                                 "throttled": int(d.get("throttled") or 0)}
    room_arm, room_stat = {}, {}
    for (room, arm), rec in best.items():
        rows = [r for r in rec["rows"] if isinstance(r, dict) and r.get("mine") is not None]
        if not rows:
            continue
        n_full = n_take = 0
        full_rounds, take_rounds = [], []
        post_take = []
        decline_rounds = collections.Counter()
        decline_seen = collections.Counter()
        for r in rows:
            seat = r.get("seat", -1)
            base = [float(x) for x in (r.get("deltas_base") or [])]
            sbase = sum(abs(x) for x in base)
            deltas = r.get("round_deltas") or []
            mine_rounds = [float(d[seat]) for d in deltas
                           if isinstance(d, (list, tuple)) and 0 <= seat < len(d)]
            if sbase == 0:
                n_full += 1
                full_rounds.extend(mine_rounds)
            else:
                n_take += 1
                take_rounds.extend(mine_rounds)
                b = base[seat] if 0 <= seat < len(base) else 0.0
                post_take.append((float(r["mine"]) - b, len(mine_rounds)))
            # 弃胡：按局号归并；round_no 缺失的记为 None 桶（保守：不算兑现机会）
            for m in (r.get("missed_detail") or []):
                if not isinstance(m, dict):
                    continue
                decline_rounds[(r.get("game"), m.get("round_no"))] += 1
            nos = r.get("round_nos") or []
            if 0 <= seat < 4:
                for i, dn in enumerate(deltas):
                    if not isinstance(dn, (list, tuple)) or len(dn) <= seat:
                        continue
                    rn = nos[i] if i < len(nos) else None
                    decline_seen[(r.get("game"), rn)] = float(dn[seat])
        room_arm[room] = arm
        room_stat[room] = {
            "arm": arm, "n_rows": len(rows), "n_full": n_full, "n_take": n_take,
            "full_rounds": full_rounds, "take_rounds": take_rounds,
            "post_take": post_take,
            "decline_events": sum(decline_rounds.values()),
            "decline_rounds": decline_rounds, "round_delta": decline_seen,
        }
    return room_arm, room_stat, bad, len(files), dup


# ============================================================ 读 reviews
def load_reviews(patterns, account_filter=None):
    """⇒ room→逐局序列。★ 同房间只保留 `at` 最大的一份（复盘会被重复生成）。"""
    files = expand(patterns)
    best = {}
    bad = 0
    for path in files:
        try:
            with io.open(path, encoding="utf-8", errors="replace") as f:
                d = json.load(f)
        except Exception:                                            # noqa: BLE001
            bad += 1
            continue
        room = (d.get("room") or "").strip()
        if not room:
            bad += 1
            continue
        acct = (d.get("account") or "").strip()
        if account_filter and acct != account_filter:
            continue
        at = str(d.get("at") or "")
        if room in best and at <= best[room]["at"]:
            continue
        rounds = []
        for g in (d.get("per_game") or []):
            for r in (g.get("rounds") or []):
                rounds.append(r)
        if not rounds:
            continue
        best[room] = {"at": at, "account": acct, "rounds": rounds,
                      "seats": d.get("seats") or [],
                      "n_rounds": d.get("n_rounds"), "mu": d.get("mu"),
                      "n_seat_rounds": d.get("n_seat_rounds"),
                      "wins": d.get("wins"),
                      "self_draw_rate": d.get("self_draw_rate"),
                      "mine_fan_avg": d.get("mine_fan_avg"),
                      "mine_baotou": d.get("mine_baotou"),
                      "mine_chiitoi": d.get("mine_chiitoi"),
                      "draws": d.get("draws"),
                      "identity_bad": d.get("identity_bad"),
                      "succession": d.get("succession")}
    return best, bad, len(files)


# ============================================================ 读日志
def load_log_exec(patterns):
    """⇒ 臂→执行计数。臂从日志的轮次头读取；读不到就归到 '(未知' 桶。"""
    files = expand(patterns)
    files, dup = dedup_paths(files)
    agg = collections.defaultdict(lambda: collections.Counter())
    n_lines = 0
    for path in files:
        cur = "(未知)"
        with io.open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                mh = RE_ROUND_HDR.search(line)
                if mh:
                    cur = mh.group(1).strip().lstrip("\ufeff")
                    continue
                mp = RE_ARM_PREFIX.search(line)
                if mp and not mh:
                    # `[主|rl:xxx] 场 ...` 也带臂名；只在轮次头缺失时用作兜底
                    if cur == "(未知)":
                        cur = mp.group(1).strip()
                ms = RE_SUMMARY.search(line)
                if not ms:
                    continue
                n_lines += 1
                c = agg[cur]
                c["games"] += 1
                c["act"] += int(ms.group(1))
                c["gang"] += int(ms.group(2))
                c["gang_rej"] += int(ms.group(3))
                c["miss"] += int(ms.group(4))
                c["c409"] += int(ms.group(5))
                c["push"] += int(ms.group(6))
                c["abandon"] += int(ms.group(7))
                c["stale"] += int(ms.group(8))
                c["submit"] += int(ms.group(9))
                c["skip"] += int(ms.group(10))
                c["lat_med"] += int(ms.group(11))
                c["lat_p99"] += int(ms.group(12))
                # ⚠ 组号必须与 RE_SUMMARY 逐个数：`最大` 是 **13**，
                #   所以新增的两个尾部字段是 14 / 15 —— 不是 13 / 14。
                #   本文件刚因为数错一位而把"网络"读成"最大延迟的累加"（3,809,926），
                #   而且**不报错**。`_selftest` 就是钉这个的。
                c["net"] += int(ms.group(14) or 0)
                c["net_sub"] += int(ms.group(15) or 0)
    return agg, n_lines, len(files), dup


# ============================================================ 主逻辑
def _selftest():
    """钉住汇总行的**字段序号**。

    为什么必须有：新增一个尾部字段时，`ms.group(i)` 的 i 极易数错一位，
    而数错**不会报错** —— 只会把读数变成荒谬的大数。本文件刚踩过一次：
    `最大` 是第 13 组，我却用 13 去取"网络"，于是"网络"变成了所有最大延迟之和
    （3,809,926），而表里看起来只是"一个大数"。

    同时钉住**向后兼容**：旧日志没有尾部两字段，必须仍能解析且新字段为 None。
    """
    line = ("[x] [主] 场 a_x_r1_b0_t0 结束：20 行动 / 1 杠(被拒2) / 3 窗口错过 / "
            "4 次409 / 5 推送 放弃6 陈旧7 提交8 过9 "
            "延迟 中位10/p9911/最大12 ms 被拒13次 | 最慢5%分段: gap0 令牌0 "
            "GET0 决策0 提交0 网络14 提交超时15")
    want = ["20", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12", "14", "15"]
    m = RE_SUMMARY.search(line)
    got = list(m.groups()) if m else []
    ok1 = got == want
    print("  汇总行字段序号 = %s" % ("ok" if ok1 else "FAIL got=%s want=%s" % (got, want)))
    old = line.split(" 网络")[0]
    mo = RE_SUMMARY.search(old)
    ok2 = bool(mo) and mo.group(14) is None and mo.group(15) is None
    print("  旧格式向后兼容 = %s" % ("ok" if ok2 else "FAIL"))
    if not (ok1 and ok2):
        return 1
    print("  ✓ 通过（新增尾部字段不会打坏旧日志的解析）")
    return 0


def main():
    ap = argparse.ArgumentParser(description="线上可信基线（主指标/接管账/执行账）")
    ap.add_argument("--jsonl", action="append", default=[],
                    help="live*.jsonl 的 glob（可重复给）")
    ap.add_argument("--reviews", action="append", default=[],
                    help="reviews/*.json 的 glob（可重复给）")
    ap.add_argument("--logs", action="append", default=[],
                    help="live*.log 的 glob（可重复给）")
    ap.add_argument("--account", default="", help="只认这个 account 的复盘")
    ap.add_argument("--model-arm", default="ckpt:rl_aux_0004.pt")
    ap.add_argument("--rule-arm", default="shanten:1:1:2:5:1:1:0")
    ap.add_argument("--min-rooms", type=int, default=3,
                    help="少于这么多房的臂不进主表（样本太小没有判据价值）")
    ap.add_argument("--json", default="", help="把汇总写成 JSON")
    ap.add_argument("--report", default="",
                    help="把整份报告同时写进这个文件（沙箱不允许 shell 重定向 ✓）")
    ap.add_argument("--selftest", action="store_true",
                    help="只跑字段序号自测后退出（新增尾部字段后必跑 ✓）")
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
        return _run(a)
    finally:
        for h in _OUT:
            h.close()
        del _OUT[:]


def _run(a):
    print("=" * 78)
    print("线上可信基线")
    print("=" * 78)

    room_arm, room_stat, bad_j, n_jf, dup_j = load_jsonl(a.jsonl)
    print("\n[jsonl] 文件 %d 个（去重丢 %d）｜坏行 %d｜房间 %d"
          % (n_jf, len(dup_j), bad_j, len(room_stat)))
    for x, y in dup_j:
        print("        重复丢弃 %s (= %s)" % (x, y))

    reviews, bad_r, n_rf = {}, 0, 0
    if a.reviews:
        reviews, bad_r, n_rf = load_reviews(a.reviews, a.account or None)
        print("[reviews] 文件 %d 个｜坏 %d｜可用房间 %d"
              % (n_rf, bad_r, len(reviews)))
        joined = sum(1 for r in reviews if r in room_arm)
        print("          其中能与 jsonl 的「房间→臂」对上的 = %d（%.1f%%）"
              % (joined, 100.0 * joined / max(1, len(reviews))))

    exec_agg, n_sum, n_lf, dup_l = (collections.defaultdict(collections.Counter),
                                    0, 0, [])
    if a.logs:
        exec_agg, n_sum, n_lf, dup_l = load_log_exec(a.logs)
        print("[logs]    文件 %d 个（去重丢 %d）｜场汇总行 %d"
              % (n_lf, len(dup_l), n_sum))
        for x, y in dup_l:
            print("        重复丢弃 %s (= %s)" % (x, y))

    # ---------------------------------------------------------- 建臂
    arm_rooms = collections.defaultdict(list)
    for room, s in room_stat.items():
        arm_rooms[s["arm"]].append(room)

    # ★ 口径自动降级：本包自包含，`review_room.py` 不在包内 ⇒ 复盘可能没有。
    #   此时主指标退回**客户端 jsonl 的逐局得失**（`round_deltas`，同样是逐局，
    #   且用 `deltas_base` 剔除接管房）—— 只有番型相关指标拿不到。
    use_rev = bool(reviews)
    if not use_rev:
        print("\n⚠ 没有可用复盘 ⇒ 主指标降级为 **jsonl 客户端口径**"
              "（逐局 `round_deltas`，接管房已剔除）。番型/爆头/七对指标不可得。")

    # ---------------------------------------------------------- 表 1：主指标
    print("\n" + "=" * 78)
    print("表 1【主指标】每局净得分（分/局）—— %s口径，按房间聚类 95%% CI"
          % ("服务端复盘" if use_rev else "客户端 jsonl"))
    if use_rev:
        print("   ★ 复盘派生自官方房间日志 ⇒ 天然不含「接管前分数」偏差")
    else:
        print("   ★ 无复盘 ⇒ 用逐局得失；接管房按 deltas_base 整房剔除")
    print("   ★ 只统计**全程无接管**的房间（接管房见 表 2）")
    print("=" * 78)
    print("%-34s %5s %6s %6s  %s" % ("臂", "房", "局", "自摸率", "局均净得分 ± 95%CI"))
    main_rows = []
    for arm, rooms in sorted(arm_rooms.items(), key=lambda kv: -len(kv[1])):
        if len(rooms) < a.min_rooms:
            continue
        per_room, n_rounds, n_win = [], 0, 0
        for room in rooms:
            s = room_stat[room]
            if s["n_take"] > 0:
                continue                       # 接管房：整房排除（臂不稳定）
            if use_rev:
                rv = reviews.get(room)
                if not rv:
                    continue
                ms = [float(r["mine"]) for r in rv["rounds"]
                      if r.get("mine") is not None]
            else:
                ms = [float(x) for x in s["full_rounds"]]
            if not ms:
                continue
            per_room.append(st.mean(ms))
            n_rounds += len(ms)
            n_win += sum(1 for x in ms if x > 0)
        m, h, n = mean_ci(per_room)
        if n == 0:
            continue
        rate = 100.0 * n_win / n_rounds if n_rounds else float("nan")
        print("%-34s %5d %6d %5.1f%%  %s"
              % (arm[:34], n, n_rounds, rate, fmt_ci(m, h, n)))
        main_rows.append({"arm": arm, "rooms": n, "rounds": n_rounds,
                          "mu": m, "ci": h, "self_draw_rate": rate,
                          "per_room": per_room})

    # ---------------------------------------------------------- 表 2：接管账
    print("\n" + "=" * 78)
    print("表 2【接管账】中途接管的房间：接管后得分变化（不参与 表 1）")
    print("   口径 = mine − sum(deltas_base[seat])，按观测到的局数折算局均")
    print("=" * 78)
    print("%-34s %6s %8s %10s  %s" % ("臂", "接管房", "接管局", "局均", "接管后仍为负的场数"))
    for arm, rooms in sorted(arm_rooms.items(), key=lambda kv: -len(kv[1])):
        tk = [room_stat[r] for r in rooms if room_stat[r]["n_take"] > 0]
        if not tk:
            continue
        posts = [p for s in tk for p, _ in s["post_take"]]
        tot_rounds = sum(n for s in tk for _, n in s["post_take"])
        behind = sum(1 for s in tk for p, _ in s["post_take"] if p < 0)
        m, h, n = mean_ci(posts)
        print("%-34s %6d %8d %10.2f  %d / %d"
              % (arm[:34], len(tk), tot_rounds, m if n else float("nan"),
                 behind, len(posts)))

    # ---------------------------------------------------------- 表 3：决策结果
    print("\n" + "=" * 78)
    print("表 3【决策结果】自摸率 / 平均番 / 爆头率 / 七对率 / 流局率 / 平均顺位")
    print("   全部来自服务端复盘，逐局逐座位解析（无接管偏差）")
    print("=" * 78)
    print("%-34s %6s %7s %7s %7s %7s %7s %7s"
          % ("臂", "座位局", "自摸率", "平均番", "爆头率", "七对率", "流局率", "均顺位"))
    for arm, rooms in sorted(arm_rooms.items(), key=lambda kv: -len(kv[1])):
        if len(rooms) < a.min_rooms:
            continue
        n_sr = win = fan_sum = fan_n = 0
        bao = chi = draw = 0
        for room in rooms:
            s = room_stat[room]
            if s["n_take"] > 0:
                continue
            if not use_rev:
                # 只有逐局得分可用 ⇒ 只能给自摸率，番型留空
                ms = s["full_rounds"]
                n_sr += len(ms)
                win += sum(1 for x in ms if x > 0)
                continue
            rv = reviews.get(room)
            if not rv:
                continue
            for r in rv["rounds"]:
                n_sr += 1
                mn = r.get("mine")
                if mn is None:
                    continue
                if r.get("winner") == -1 or (r.get("winner") is None and (mn or 0) == 0
                                             and (r.get("fan") or 0) == 0):
                    draw += 1
                if (mn or 0) > 0:
                    win += 1
                    f = r.get("fan")
                    if isinstance(f, (int, float)) and f > 0:
                        fan_sum += float(f)
                        fan_n += 1
                    det = " ".join(r.get("detail") or [])
                    if "爆头" in det:
                        bao += 1
                    if "七对" in det:
                        chi += 1
        if not n_sr:
            continue
        print("%-34s %6d %6.1f%% %7s %7s %7s %7s %7s"
              % (arm[:34], n_sr, 100.0 * win / n_sr,
                 ("%.2f" % (fan_sum / fan_n)) if fan_n else "—",
                 ("%.1f%%" % (100.0 * bao / win)) if (use_rev and win) else "—",
                 ("%.1f%%" % (100.0 * chi / win)) if (use_rev and win) else "—",
                 ("%.1f%%" % (100.0 * draw / n_sr)) if use_rev else "—", "—"))

    # ---------------------------------------------------------- 表 4：弃胡
    print("\n" + "=" * 78)
    print("表 4【弃胡】次数与兑现率（可胡未胡的**客户端当日埋点**，唯一来源）")
    print("   兑现 = 该局最终由我们胡到（用逐局得分 >0 判定）")
    print("=" * 78)
    print("%-34s %8s %8s %8s %9s" % ("臂", "弃胡次", "机会局", "兑现局", "兑现率"))
    for arm, rooms in sorted(arm_rooms.items(), key=lambda kv: -len(kv[1])):
        if len(rooms) < a.min_rooms:
            continue
        ev = 0
        chance = cash = 0
        for room in rooms:
            s = room_stat[room]
            ev += s["decline_events"]
            for key, cnt in s["decline_rounds"].items():
                if key[1] is None:
                    continue
                chance += 1
                if s["round_delta"].get(key, 0.0) > 0:
                    cash += 1
        if not ev:
            continue
        print("%-34s %8d %8d %8d %8.1f%%"
              % (arm[:34], ev, chance, cash,
                 100.0 * cash / max(1, chance)))

    # ---------------------------------------------------------- 表 5：执行
    if exec_agg:
        print("\n" + "=" * 78)
        print("表 5【执行账】动作执行质量（客户端日志，按场所属臂归属）")
        print("=" * 78)
        print("%-34s %6s %10s %8s %8s %7s %8s %8s %8s"
              % ("臂", "场", "提交", "放弃", "陈旧", "窗口错过", "409", "杠被拒", "延迟中位"))
        for arm, c in sorted(exec_agg.items(), key=lambda kv: -kv[1]["games"]):
            g = max(1, c["games"])
            print("%-34s %6d %10d %8d %8d %8d %8d %8d %8.1f ms"
                  % (arm[:34], c["games"], c["submit"], c["abandon"], c["stale"],
                     c["miss"], c["c409"], c["gang_rej"], c["lat_med"] / g))
            print("%-34s %6s %9.3f%% %7.3f%% %7.3f%% %7.3f%% %7.3f%%   网络%d 提交超时%d"
                  % ("  占提交动作比", "", 100.0 * c["abandon"] / max(1, c["submit"]),
                     100.0 * c["stale"] / max(1, c["submit"]),
                     100.0 * c["miss"] / max(1, c["submit"]),
                     100.0 * c["c409"] / max(1, c["submit"]),
                     100.0 * c["gang_rej"] / max(1, c["submit"]),
                     c["net"], c["net_sub"]))

    # ---------------------------------------------------------- 表 6：两臂对比
    if a.model_arm and a.rule_arm:
        print("\n" + "=" * 78)
        print("表 6【两臂对比】主指标差（按房间配对；未配对则退化为 Welch）")
        print("=" * 78)
        mr = next((r for r in main_rows if r["arm"] == a.model_arm), None)
        rr = next((r for r in main_rows if r["arm"] == a.rule_arm), None)
        if not mr or not rr:
            print("   ! 两臂中至少一个不在 表 1（房数不足或被全部剔除）")
        else:
            # 配对：同一房间不可能同时属于两臂；退化为按房间的 Welch
            diff = mr["mu"] - rr["mu"]
            sea = (mr["ci"] / Z95) if not math.isnan(mr["ci"]) else float("nan")
            seb = (rr["ci"] / Z95) if not math.isnan(rr["ci"]) else float("nan")
            se = math.sqrt(sea ** 2 + seb ** 2)
            lo, hi = diff - Z95 * se, diff + Z95 * se
            print("  模型臂 %s" % a.model_arm)
            print("    房 %d ｜ 局 %d ｜ 局均 %+.3f" % (mr["rooms"], mr["rounds"], mr["mu"]))
            print("  规则臂 %s" % a.rule_arm)
            print("    房 %d ｜ 局 %d ｜ 局均 %+.3f" % (rr["rooms"], rr["rounds"], rr["mu"]))
            print("  ── 差（模型 − 规则）= %+.3f ｜ 95%%CI [%+.3f, %+.3f] ｜ SE %.3f"
                  % (diff, lo, hi, se))
            # ★ 功效：两独立样本、房间级 SD。n/臂 = (Z_{α/2}+Z_β)²(σa²+σb²)/Δ²
            Z80 = 0.8416212335729143
            sda = st.stdev(mr["per_room"]) if len(mr["per_room"]) > 1 else float("nan")
            sdb = st.stdev(rr["per_room"]) if len(rr["per_room"]) > 1 else float("nan")
            print("  房间级 SD：模型 %.3f（%d 房）｜规则 %.3f（%d 房）"
                  % (sda, mr["rooms"], sdb, rr["rooms"]))
            print("  判定：%s" % ("★ 模型更强（CI 下界 > 0）" if lo > 0 else
                                 ("规则更强（CI 上界 < 0）" if hi < 0 else
                                  "未分辨（CI 含 0）—— 按纪律不宣称更强")))
            for tgt in (0.20, 0.30, 0.50):
                nreq = ((Z95 + Z80) ** 2) * (sda ** 2 + sdb ** 2) / (tgt ** 2)
                print("  要分辨 %+.2f 分/局（80%% 功效，双侧 95%%）：每臂约需 %.0f 房"
                      % (tgt, math.ceil(nreq)))

    if a.json:
        with io.open(a.json, "w", encoding="utf-8") as f:
            json.dump({"main": [{k: v for k, v in r.items() if k != "per_room"}
                                for r in main_rows]}, f, ensure_ascii=False, indent=1)
        print("\n已写出 %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
