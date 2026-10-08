# -*- coding: utf-8 -*-
"""tools/next_training.py -- ★★ I1 保护：**上一轮训练结束后立刻起下一轮** ✓

★★ 为什么需要它 ✓（I1 是硬不变式 ✓：「训练不停（空转 ≤ 10 分钟）」✗）：
   · ★ `rl_aux` 跑完 **iter 4** 就结束了 ✗ ⇒ ★★ **必须立刻有下一轮** ✓
   · ★ 否则**空转** ✗（★ 实测代价 ✓：训练曾空转 **39.6 h** ✗ —— 见台账 ✓）
   · ★ 而"下一根杠杆是谁"✓ 已由 **§Z.1600／§Z.1639** 预先决定 ✓ ⇒ ★★ **不该靠临场决定** ✗

★ ★★ 它【不】自己判断"该起哪个" ✗（★ 刻意的 ✓）：
   · ★ **杠杆的选择是研究决策** ✓ ⇒ ★ 由**台账／人**决定 ✓
   · ★ 本脚本只负责**机械动作** ✓：★ 确认旧训练已停 ✓ ＋ 起新的 ✓ ＋ 用**新日志名** ✓

★ ★★ 为什么要"确认旧训练已停" ✗（★ 这是**付过代价**的 ✓）：
   · §Z.1432 ✗：★ **双开**（同一个房双写 ✗）就是因为**没确认旧进程已死** ✗
   · ★ 但沙箱里**进程视图是盲的** ✗ ⇒ ★★ 用**日志静默**当代理 ✓（★ 与 `liveness_monitor.py` 同一原则 ✓）
   · ★★ 且 ★ **绝不用 `Stop-Process`** ✗（硬规则② ✓）⇒ ★ 本脚本**只报告，不动手** ✓

★ 用法 ✓：
   # 看当前状态（默认 ✓ 不起任何东西）
   .\\_venv\\Scripts\\python.exe tools\\next_training.py --check rl_aux.log
   # 旧训练已静默 ⇒ 起 ⑦ 自模仿（★ 零代码 ✓ 已实现＋冒烟 ✓）
   .\\_venv\\Scripts\\python.exe tools\\next_training.py --check rl_aux.log --launch self-imit --yes
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time

# ★★ 预注册的下一根杠杆 ✓（★ 只列**已就绪**的 ✓ ⇒ 每条都能立刻跑 ✓）
LEVERS = {
    # ⑦ 自模仿（§Z.1554 已实现＋冒烟 ✓ ⇒ ★ **零代码** ✓）
    "self-imit": dict(
        log="rl_sil.log", hist="rl_sil_history.jsonl", prefix="rl_sil_",
        args=["--init", "rl_vA_latest.pt", "--iters", "4", "--games", "250",
              "--nproc", "3", "--epochs", "20", "--per-hand-return", "1",
              "--nstep", "5", "--val-n", "300", "--ent-coef", "0.1",
              "--ret-norm", "running", "--live-seats", "1",
              "--self-imit", "0.5"],
        note="⑦ 自模仿：只用【结果】当老师（不是规则）⇒ 监督学习 = 本仓唯一被证明能 work 的机制",
    ),
    # ⑤ 的加深（★ 若 iter4 的 L1 仍为正 ⇒ 值得再跑一组更大 β ✓）
    "aux-strong": dict(
        log="rl_aux2.log", hist="rl_aux2_history.jsonl", prefix="rl_aux2_",
        args=["--init", "rl_vA_latest.pt", "--iters", "4", "--games", "250",
              "--nproc", "3", "--epochs", "20", "--per-hand-return", "1",
              "--nstep", "5", "--val-n", "300", "--ent-coef", "0.1",
              "--ret-norm", "running", "--live-seats", "1",
              "--aux", "--aux-scale", "0.3"],
        note="⑤ 加深：与 rl_aux **同配置**（★ 只是换 seed／续跑 ⇒ ★ 但严格说这【不是】单变量 ✗ ⇒ 慎用 ✓）",
    ),
}


def _tail_lines(path, n=400):
    try:
        with io.open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except Exception:                                          # noqa: BLE001
        return []


def check(log, stale):
    """★ 用**日志静默**判"还在不在" ✓（★ 不碰进程 ✓）"""
    if not os.path.exists(log):
        print("  ① %s **不存在** ⇒ ★ 视为**未在跑** ✓" % log)
        return True
    mt = os.path.getmtime(log)
    age = time.time() - mt
    print("  ① %s：mtime 距今 **%.0fs**（阈值 %.0fs）" % (log, age, stale))
    if age > stale:
        print("     ⇒ ★ **静默超过阈值 ⇒ 大概率已结束** ✓（★ 但请再确认一次 ✓ §Z.1432 双开教训 ✗）")
        return True
    print("     ⇒ ★★ **仍在写日志 ⇒ 判定【还在跑】** ✓ ⇒ ★ **本脚本拒绝起新的** ✗（防双开 ✓）")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="旧训练的日志路径（如 rl_aux.log）")
    ap.add_argument("--stale", type=float, default=1800.0, help="静默多久算结束（秒，默认 30 分）")
    ap.add_argument("--launch", choices=sorted(LEVERS), default=None)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    print("=== ① 确认旧训练状态 ✓（★ 只看日志，不碰进程 ✓）===")
    done = check(a.check, a.stale)

    print("=== ② 下一根杠杆（预注册 ✓）===")
    for k, v in sorted(LEVERS.items()):
        print("  %-12s ⇒ %s" % (k, v["note"]))

    if a.launch is None:
        print("  ⇒ ★ 未指定 `--launch` ⇒ **只检查，不起任何东西** ✓")
        return 0 if done else 1

    if not done:
        print("  ⇒ ✗ **旧训练仍活 ⇒ 拒绝起新的** ✓（★ 防双开 ✗ §Z.1432 ✓）")
        return 1

    v = LEVERS[a.launch]
    cmd = [sys.executable, "tools/run_train.py", v["log"], "train_selfplay_rl.py"] + v["args"] + [
        "--out-prefix", v["prefix"], "--log", v["log"], "--history", v["hist"]]
    print("=== ③ 启动命令 ✓ ===")
    print("  " + " ".join(cmd))
    if not a.yes:
        print("  ⇒ ★ **干跑**（★ 加 `--yes` 才真起 ✓）")
        return 0

    print("=== ④ 启动 ✓（★ 独立进程 ✓ 用新日志名 ✓）===")
    # ★ 用 `--yes` 才走到这里 ✓；★ 起法照抄既有可用命令 ✓（`tools/run_train.py` ✓）
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  ✓ 已提交 ✓ ⇒ ★ **请下一轮用 `tools/liveness_monitor.py --log %s` 确认它活了** ✓"
          % v["log"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
