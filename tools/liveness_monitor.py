# -*- coding: utf-8 -*-
"""tools/liveness_monitor.py -- I1 保护：**只告警，不自动重启** ✓

★★ 为什么【不】自动重启 ✗（这一条比功能更重要 ✓）：
   · ★ 沙箱里**进程视图是盲的** ✗（我无法可靠地判断"训练进程还在不在" ✗）
   · ★ 实测代价 ✓：§Z.1432 那次**双开**（同一个房双写 ✗）就是因为**没确认旧进程已死** ✗
   · ★ 另 ✓：§Z.1448 我**误停了健康的客户端** ✗（因为分不清僵尸与活体 ✗）
   ⇒ ★★ 所以本工具**只回答一个问题** ✓：★★「**日志还在长吗**」✓ —— ★ 这是**唯一可靠**的信号 ✓
     （★ 它对"训练是否在推进"是**必要**条件 ✓；★ 但不是充分条件 ✓ ⇒ 所以只告警 ✓）

★ 判据 ✓（与 §Z.1452 的"两次采样"原则一致 ✓）：
   · 读**两次**间隔 `--gap` 秒 ✓ ⇒ 比较 `mtime` 与**行数** ✓
   · `mtime` 变新 **或** 行数变多 ⇒ ★ **活** ✓
   · 两次**都不动** 且 `mtime` 已旧于 `--stale` 秒 ⇒ ★★ **告警** ✗（★ 打印处置命令 ✓ 不自己动手 ✗）

★ 用法 ✓：
   .\\_venv\\Scripts\\python.exe tools\\liveness_monitor.py                      # 默认看 rl_aux.log
   .\\_venv\\Scripts\\python.exe tools\\liveness_monitor.py --log live_r16d.log --stale 600
   ★ 循环看护 ✓（★ 建议交给**任务计划程序** ✗，不要靠前台窗口 ✓）：
   .\\_venv\\Scripts\\python.exe tools\\liveness_monitor.py --watch --interval 300
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time


def _stat(path):
    """返回 (mtime, 行数) ✓；文件不存在 ⇒ (None, None) ✓"""
    if not os.path.exists(path):
        return None, None
    mt = os.path.getmtime(path)
    try:
        n = sum(1 for _ in io.open(path, encoding="utf-8", errors="replace"))
    except Exception:                                          # noqa: BLE001
        n = -1
    return mt, n


def check(path, stale, gap):
    t0 = time.time()
    m1, n1 = _stat(path)
    if m1 is None:
        print("  ✗ %s **不存在** ✗ ⇒ 无法判断（★ 若是训练：说明从未启动或路径写错 ✓）" % path)
        return 2
    print("  采样① t=%.0f ｜ mtime=%.0f ｜ 行数=%s" % (t0, m1, n1))
    time.sleep(max(1.0, gap))
    t1 = time.time()
    m2, n2 = _stat(path)
    print("  采样② t=%.0f ｜ mtime=%.0f ｜ 行数=%s  （间隔 %.0fs）" % (t1, m2, n2, gap))
    age = t1 - (m2 if m2 is not None else 0)
    grew = (n2 is not None and n1 is not None and n2 > n1) or (m2 is not None and m1 is not None and m2 > m1)
    if grew:
        print("  ✓ **活**：日志在长（mtime 变新 或 行数变多）｜ 距今 %.0fs" % age)
        return 0
    if age <= stale:
        print("  ⚠ 两次都没动，但**距今只有 %.0fs ≤ %ds** ⇒ ★ **还不能判死** ✓（§Z.1484 ✓：先定时点）"
              % (age, stale))
        return 0
    print("  ✗✗ **告警**：日志已 **%.0fs（≈%.1f 分钟）没有变化** ✗（阈值 %ds）"
          % (age, age / 60.0, stale))
    print("  ★★ **本工具【不】自动重启** ✗（★ 沙箱进程视图是盲的 ⇒ 自动重启会造成**双开** ✗ §Z.1432 ✓）")
    print("  ★ 请人工确认后再决定 ✓：")
    print("     ① 先看**是否真的停**：再看一次本工具 ✓")
    print("     ② 若确已停 ⇒ **用新的日志路径**重启 ✓（★ 旧进程可能仍持有旧文件 ✗ §Z.1452 ✓）")
    print("     ③ ★ **绝不用 Stop-Process** ✗（硬规则② ✓）")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="rl_aux.log")
    ap.add_argument("--stale", type=float, default=900.0,
                    help="多久没变化就告警（秒，默认 900 = 15 分钟；I1 要求空转 ≤ 10 分钟 ⇒ 15 分是宽松余量 ✓）")
    ap.add_argument("--gap", type=float, default=60.0, help="两次采样间隔（秒，默认 60）")
    ap.add_argument("--watch", action="store_true", help="循环看护（★ 建议交给任务计划程序 ✓）")
    ap.add_argument("--interval", type=float, default=300.0, help="--watch 的周期（秒）")
    a = ap.parse_args()
    if not a.watch:
        return check(a.log, a.stale, a.gap)
    print("=== 循环看护 %s（每 %.0fs ✓；★ 只告警不重启 ✓）===" % (a.log, a.interval))
    while True:
        print("--- %s ---" % time.strftime("%m-%d %H:%M:%S"))
        check(a.log, a.stale, a.gap)
        time.sleep(max(5.0, a.interval))
    return 0


if __name__ == "__main__":
    sys.exit(main())
