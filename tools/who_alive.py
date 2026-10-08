# -*- coding: utf-8 -*-
"""tools/who_alive.py -- ★★ 只读报告：**哪些线上客户端还在跑** ✓（§Z.1736 的诊断落地 ✓）

★★ 为什么需要它 ✓（★ 这是"409 高"的真根因 ✓ §Z.1736）：
   · ★ **多个客户端共享同一个 14 req/s 的令牌桶** ✗
     ⇒ ★ 超额请求被服务器拒（**409** ✓）｜ ★ 实测两个尖峰（13 时 296 ／ 18 时 260 ✓）
     ⇒ ★ 而 ★ 只有一个客户端的时段 ⇒ **409 = 4/8** ✓
   · ★★ 但 ★ **沙箱里【进程视图是盲的】** ✗（硬规则②：身份不明就不动 ✓）
     ⇒ ★★ **所以用【日志 mtime】当代理** ✓（与 `liveness_monitor.py`／`next_training.py` 同一原则 ✓）

★★ 它做 3 件事 ✓（★ **全部只读** ✓ 绝不 kill ✗）：
   ① ★ 扫 `live_r16*.log` ✓ ⇒ 报每个日志的 **mtime 距今** ✓
   ② ★ 按"距今 ≤ 阈值"判定 **活着 / 已停** ✓（★ **绝不宣称是哪个 PID** ✗）
   ③ ★★ **若发现 ≥2 个同时活着 ⇒ 明确警告** ✓：「抢桶 ⇒ 409 ⇒ 应杀掉多余的」✓
      ⇒ ★ **并给出【你自己的】操作建议** ✓（★ 我不动手 ✓）

用法 ✓：
  .\\_venv\\Scripts\\python.exe tools\\who_alive.py
  .\\_venv\\Scripts\\python.exe tools\\who_alive.py --stale 300
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", type=float, default=300.0,
                    help="mtime 距今 ≤ 此秒数 ⇒ 判定活着（默认 300 = 5 分钟；哨兵每 60 s 一次 ✓）")
    a = ap.parse_args()

    logs = sorted(glob.glob("live_r16*.log"))
    print("=== ① 所有线上客户端日志 ✓（★ 只看 mtime，不碰进程 ✓）===")
    if not logs:
        print("  （没有 live_r16*.log）")
        return 0

    alive, dead = [], []
    for p in logs:
        age = time.time() - os.path.getmtime(p)
        # ★ 先算"最后写入的绝对时刻" ✓（★ 便于你对照 ✓）
        t = time.strftime("%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
        tag = "✓ 活着" if age <= a.stale else "✗ 已停"
        (alive if age <= a.stale else dead).append((p, age))
        print("  %-22s %s ｜ mtime 距今 %7.0f s ｜ 最后写入 %s" % (p, tag, age, t))

    print("=== ② 判定 ✓（★ 代理 = 日志静默 ✓ 不是进程表 ✓）===")
    print("  活着 = **%d** ｜ 已停 = **%d**" % (len(alive), len(dead)))

    print("=== ③ 抢桶检查 ✓（★ 这是 409 的真根因 ✓ §Z.1736）===")
    if len(alive) >= 2:
        print("  ★★★ **警告：≥2 个客户端同时在跑** ✗✗")
        print("      ⇒ ★ 它们**共享 14 req/s 的令牌桶** ⇒ **超额请求被拒（409）** ✗")
        print("      ⇒ ★ 实测：两个客户端时 409 尖峰 **296/260** ✗ ｜ 一个客户端时 **4/8** ✓")
        print("      ⇒ ★★ **建议（★ 由你操作 ✓ 我不动手 ✓ 硬规则②）**：")
        print("         · 保留**最新**那个（★ 日志 mtime 最小 ✓ = 我们刚起的 ✓）")
        for p, age in sorted(alive, key=lambda kv: kv[1]):
            print("           · `%s`（距今 %.0f s）" % (p, age))
        print("         · **杀掉其余**（★ 用你自己确认过身份的方式 ✓ 不要凭本报告盲杀 ✓）")
        print("      ⇒ ★ 杀完后**桶会回到 14.0** ✓ 且 **409 应大幅下降** ✓")
    elif len(alive) == 1:
        print("  ✓ **只有一个客户端在跑** ✓ ⇒ ★ **不会抢桶** ✓ ⇒ 409 应低 ✓")
    else:
        print("  ✗ **没有一个客户端在跑** ✗ ⇒ ★ **线上测量已断** ✗ ⇒ 需**立即重启** ✓（I1 ✓）")

    print("=== ④ 吞吐提示 ✓ ===")
    print("  ★ 两个客户端**不会**带来两倍有效吞吐 ✓ —— ★ 它们只是**分同一个桶** ✗")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
