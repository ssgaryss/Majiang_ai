#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分臂损失归集器 —— 把线上「可见损失 / 长尾分段」按【臂】汇总，回答 M3 的归因问题。

背景（GOAL_STATUS §Z.1289 ~ §Z.1300）
------------------------------------
线上实测：`ckpt:`（神经网络）臂所在房 **100% 是「脏房」**（该房有 409 或窗口错过），
而 `shanten:`（规则）臂既有干净房也有脏房；且模型臂每场被拒 **28×**（0.69 vs 0.024）。

长尾分解（场结束行 `最慢5%分段`）：**令牌排队 = 0 ／ GET 4~13 ms ／ 本地计算 17 ms**，
主项是 `gap`。注意 §Z.1300 已证：实测 `gap` 91~114 ms **远低于** `--fetch-gap-max`（线上 1.2 s）
⇒ **该旋钮未 binding**，改它不会改变实测 `gap`。所以本脚本按**臂**看 `gap`，用数据而不是猜。

两张表
------
① **最慢 5% 分段**（`gap` / `令牌` / `GET`，毫秒）：字段**早已存在** ⇒ **现在就有数据，不需重启**。
② **可见损失**（`放弃` / `陈旧` / `409` / `窗口错过`，每场）：`放弃/陈旧` 需要**客户端重启**
   让插桩（`mj_bot.py:1071`，第 922 轮）生效；未生效时本脚本**如实说无话可说**，不编造。

用法
----
    python tools/live_loss_by_arm.py [--log live_r16.log] [--jsonl live_r16.jsonl]
                                     [--since-hours 48] [--min-games 3]
    python tools/live_loss_by_arm.py --selftest      # 用合成行自检解析（不依赖线上日志）
"""

import argparse
import collections
import datetime
import json
import os
import re
import sys
import time

# 场结束行（`mj_bot.py:1071`）。两代日志都兼容：
#   · 插桩前没有 `放弃%d 陈旧%d`
#   · `最慢5%分段` 用 `.*?`（行中还有 `超窗[...]` 等字段）
RE_GAME = re.compile(
    r"\[(?P<mo>\d\d)-(?P<dy>\d\d) (?P<hh>\d\d):(?P<mm>\d\d):(?P<ss>\d\d)\].*?"
    r"场\s+(?P<room>a_[0-9a-f]+)\S*\s+结束：\s*(?P<acts>\d+)\s*行动\s*/\s*\d+\s*杠\(被拒(?P<gang>\d+)\)"
    r"\s*/\s*(?P<miss>\d+)\s*窗口错过\s*/\s*(?P<c409>\d+)\s*次409\s*/\s*(?P<push>\d+)\s*推送"
    r"(?:\s+放弃(?P<giveup>\d+)\s+陈旧(?P<stale>\d+)"
    r"(?:\s+提交(?P<submits>\d+)\s+过(?P<passes>\d+))?)?"
    r"(?:.*?\|\s*最慢5%分段:\s*gap(?P<gap>\d+)\s*令牌(?P<tok>\d+)\s*GET(?P<get>\d+))?"
)


def load_room_to_arm(jsonl_path):
    """房间号 → 臂名。`live_r16.jsonl` 每房一条，含 `room` 与 `genome`。"""
    m = {}
    if not os.path.exists(jsonl_path):
        return m
    with open(jsonl_path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            room = (d.get("room") or "").strip()
            arm = (d.get("genome") or "").strip()
            if room and arm:
                m[room] = arm
    return m


def load_round_spans(log_path):
    """从 `########## 第 N / 400 轮（策略 X）##########` 横幅取 (起始时间戳, 臂) ✓

    ★ 用途（2026-10-03 / §Z.1334）：房可能由**上一轮**创建（match API 会复用开放房 ✓），
      而 `live_r16.jsonl` 只在**房结束**时才写映射 ⇒ 被重启中断的房**永久无映射** ✗
      （实测 §Z.1333：9 场插桩数据因此被丢掉 ✗）
      ⇒ 回落到"按时间戳取当时生效的轮次臂" ✓ —— ⚠ 这是**推断**，故单独计数 ✓
    """
    out = []
    if not os.path.exists(log_path):
        return out
    pat = re.compile(
        r"\[(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\]\s*#{3,}\s*第 \d+ / \d+ 轮（策略 (.+?)）")
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = pat.search(ln)
            if not m:
                continue
            try:
                ts = datetime.datetime(
                    2026, int(m.group(1)), int(m.group(2)),
                    int(m.group(3)), int(m.group(4)), int(m.group(5))).timestamp()
            except ValueError:
                continue
            out.append((ts, m.group(6).strip()))
    out.sort()
    return out


def arm_of_ts(spans, ts):
    """取"在 `ts` 之前开始的最后一轮"的臂 ✓（= 当时生效的策略 ✓）。"""
    arm = None
    for s_ts, s_arm in spans:
        if s_ts <= ts:
            arm = s_arm
        else:
            break
    return arm


def collect(log_path, room_to_arm, since_hours):
    """按臂累加。返回 (per_arm, stats)。"""
    per = collections.defaultdict(lambda: collections.Counter())
    cut = time.time() - since_hours * 3600.0
    instrumented = 0
    total = 0
    unmapped = 0
    inferred = 0
    round_spans = load_round_spans(log_path)
    if not os.path.exists(log_path):
        return per, dict(instrumented=0, total=0, unmapped=0, note="日志不存在")
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = RE_GAME.search(ln)
            if not m:
                continue
            total += 1
            try:
                ts = datetime.datetime(
                    2026,
                    int(m.group("mo")),
                    int(m.group("dy")),
                    int(m.group("hh")),
                    int(m.group("mm")),
                    int(m.group("ss")),
                ).timestamp()
            except ValueError:
                continue
            if ts < cut:
                continue
            arm = room_to_arm.get(m.group("room"))
            if not arm:
                # ★ 2026-10-03（§Z.1334）：回落到"按时间戳取当时生效的轮次臂" ✓
                #   ⚠ 这是**推断**归属（非精确映射 ⇒ JSONL 里没有这个房 ✓）⇒ 单独计数 ✓
                arm = arm_of_ts(round_spans, ts)
                if not arm:
                    unmapped += 1
                    continue
                inferred += 1
            c = per[arm]
            c["n"] += 1
            c["giveup"] += int(m.group("giveup") or 0)
            c["stale"] += int(m.group("stale") or 0)
            c["c409"] += int(m.group("c409"))
            c["miss"] += int(m.group("miss"))
            c["acts"] += int(m.group("acts"))
            if m.group("giveup") is not None:
                instrumented += 1
            # ★ 2026-10-03（§Z.1326）：`提交`/`过` 是 `行动` 的两个成分 ✓
            #   （`行动` 有 **3 个**自增点 ✗ ⇒ 恒等式自检能暴露第三个成分的大小 ✓）
            if m.group("submits") is not None:
                c["submits"] += int(m.group("submits"))
                c["passes"] += int(m.group("passes"))
                # ★ 2026-10-03（§Z.1334）：拆分字段**必须有自己的分母** ✗
                #   否则 `提交/场` 会用"所有场"去除（含旧格式场 ✗）⇒ 恒等式假性不成立 ✗
                #   实测教训 ✓：20 场里只有 ~11 场带拆分 ⇒ 一度显示 `72.7 ≠ 30.35 + 0` ✗
                c["sub_n"] += 1
                c["acts_sub"] += int(m.group("acts"))   # 只为"同一子集"上的恒等式检验 ✓
            # 最慢 5% 的分段（`mj_bot.py:1060-1070`）：gap / 令牌 / GET，毫秒。
            if m.group("gap") is not None:
                c["seg_n"] += 1
                c["gap"] += int(m.group("gap"))
                c["tok"] += int(m.group("tok"))
                c["get"] += int(m.group("get"))
    return per, dict(instrumented=instrumented, total=total, unmapped=unmapped,
                     inferred=inferred, note="")


def print_segments(rows):
    """表①：最慢 5% 分段（无需重启即有数据）。"""
    segs = [(a, c) for a, c in rows if c["seg_n"] > 0]
    if not segs:
        return False
    print()
    print("  ① 最慢 5% 分段（毫秒；字段早已存在 ⇒ **现在就有数据**）")
    print("  %-26s %7s %8s %8s %8s" % ("臂", "有分段场", "gap", "令牌", "GET"))
    print("  " + "-" * 64)
    for arm, c in sorted(segs, key=lambda kv: -kv[1]["seg_n"]):
        k = float(c["seg_n"])
        print("  %-26s %7d %8.0f %8.0f %8.0f"
              % (arm[:26], c["seg_n"], c["gap"] / k, c["tok"] / k, c["get"] / k))
    print("  判读：`gap` 大 ⇒ 卡在我们自己的取快照节流；`令牌` > 0 ⇒ 额度真不够；`GET` 大 ⇒ 网络/服务端慢")
    return True


def print_losses(rows, have_instr):
    """表②：可见损失。`409/错过/行动` **现在就有**；`放弃/陈旧` 需插桩生效（否则显 n/a）。"""
    print()
    print("  ② 可见损失（每场）%s"
          % ("" if have_instr else "  ← 「放弃/陈旧」两列需客户端重启（插桩）后才可用"))
    print("  %-26s %5s %9s %9s %8s %9s %9s %9s %9s"
          % ("臂", "场", "放弃/场", "陈旧/场", "409/场", "错过/场", "行动/场", "提交/场", "过/场"))
    print("  " + "-" * 96)
    for arm, c in sorted(rows, key=lambda kv: -kv[1]["n"]):
        n = float(c["n"])
        if have_instr:
            gs = "%9.3f %9.3f" % (c["giveup"] / n, c["stale"] / n)
            sub = ("%9.2f %9.2f" % (c["submits"] / float(c["sub_n"]),
                                    c["passes"] / float(c["sub_n"]))
                   if c["sub_n"] else "%9s %9s" % ("n/a", "n/a"))
        else:
            gs = "%9s %9s" % ("n/a", "n/a")
            sub = "%9s %9s" % ("n/a", "n/a")
        print("  %-26s %5d %s %8.3f %9.3f %9.1f %s"
              % (arm[:26], c["n"], gs, c["c409"] / n, c["miss"] / n, c["acts"] / n, sub))
        if c["sub_n"]:
            # ★ 恒等式只在**带拆分的子集**上检验 ✓（否则分母不同 ⇒ 假性不成立 ✗）
            d = c["acts_sub"] - (c["submits"] + c["passes"])
            print("      ↳ 拆分场 %d 场：行动 %d ＝ 提交 %d ＋ 过 %d ｜ 差 = **%d**%s"
                  % (c["sub_n"], c["acts_sub"], c["submits"], c["passes"], d,
                     "  ✓ 恒等式成立" if d == 0 else "  ✗ 还有别的自增点"))
    print("  ★ 恒等式自检：`行动 −（提交 + 过）` 应在**带拆分的子集**上为 **0** ✓"
          "（不为 0 ⇒ `行动` 还有别的自增点 ✗；源码里有 3 处 `self.actions += 1` ✓）")
    if have_instr:
        print("  判读（预先写死）：模型臂的 放弃/场 或 陈旧/场 远高于规则臂 ⇒ 丢分归因完成 ⇒ 修它（与训练无关）")
        print("                    两臂相当 ⇒ 该假设被否 ⇒ 回到「等待/重试」方向继续查")
    else:
        print("  判读（现在就能读的三列）：`409/场` 与 `错过/场` 的臂间比 = 基础设施劣势的直接量级")


def main(argv=None):
    ap = argparse.ArgumentParser(description="按【臂】汇总线上可见损失与长尾分段")
    ap.add_argument("--log", default="live_r16.log")
    ap.add_argument("--jsonl", default="live_r16.jsonl")
    ap.add_argument("--since-hours", type=float, default=48.0)
    ap.add_argument("--min-games", type=int, default=3)
    a = ap.parse_args(argv)

    room_to_arm = load_room_to_arm(a.jsonl)
    per, s = collect(a.log, room_to_arm, a.since_hours)

    print("=" * 78)
    print("分臂归集（窗口 = 最近 %.1f 小时）" % a.since_hours)
    print("=" * 78)
    print("  房间→臂 映射 = %d 条" % len(room_to_arm))
    print("  日志里的场结束行 = %d ｜ 已插桩（含 放弃/陈旧）= %d ｜ 无法映射 = %d ｜ 按轮次推断 = %d"
          % (s["total"], s["instrumented"], s["unmapped"], s.get("inferred", 0)))

    if not per:
        print()
        print("  ⚠ 该窗口内没有可用样本（试试 --since-hours 更大）")
        return 0

    rows = [kv for kv in per.items() if kv[1]["n"] >= a.min_games]

    # ★ 表①先行：它**不需要重启**就有数据（这正是本脚本存在的理由之一）。
    print_segments(rows)

    # ★ 表②也总是打：`409/错过/行动` 现在就有；只有 `放弃/陈旧` 需插桩。
    print_losses(rows, s["instrumented"] > 0)

    if s["instrumented"] == 0:
        print()
        print("  ⚠ 已插桩的场 = 0 ⇒ 表②的「放弃/陈旧」两列暂为 n/a（**不是「两臂一样」！**）")
        print("     原因：插桩（mj_bot.py:1071，第 922 轮）只在**客户端重启后**生效。")
        print("     行动：重启客户端 ⇒ 等 ≥ 1 小时 ⇒ 重跑本脚本。")
        return 0

    return 0


def _selftest():
    """自检：用一条**合成的插桩行**验证解析（不依赖线上日志）。"""
    line = ("[10-03 13:30:04] [主] 场 a_d56eb414021b_r1_b3_t0 结束：84 行动 / 1 杠(被拒0) / "
            "4 窗口错过 / 4 次409 / 1256 推送 放弃7 陈旧8 超窗[出牌0/窗口0] | 最慢5%分段: gap91 令牌0 GET6")
    m = RE_GAME.search(line)
    assert m, "插桩行解析失败"
    assert (int(m.group("acts")), int(m.group("miss")), int(m.group("c409")),
            int(m.group("giveup")), int(m.group("stale"))) == (84, 4, 4, 7, 8), "损失字段不符"
    assert (int(m.group("gap")), int(m.group("tok")), int(m.group("get"))) == (91, 0, 6), "分段字段不符"
    old = ("[10-03 13:30:45] [主] 场 a_d56eb414021b_r1_b6_t0 结束：98 行动 / 1 杠(被拒0) / "
           "0 窗口错过 / 0 次409 / 1332 推送 超窗[出牌0/窗口0] | 最慢5%分段: gap114 令牌0 GET13")
    m2 = RE_GAME.search(old)
    assert m2 and m2.group("giveup") is None, "旧格式（无插桩）应被兼容且 giveup 为 None"
    assert (int(m2.group("gap")), int(m2.group("tok")), int(m2.group("get"))) == (114, 0, 13), "旧格式分段不符"
    print("  自检通过 ✓ 插桩行 = (行动84, 错过4, 409=4, 放弃7, 陈旧8, gap91, 令牌0, GET6)")
    print("             旧格式兼容 ✓ giveup=None ｜ 分段 (gap114, 令牌0, GET13) ✓")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(main())
