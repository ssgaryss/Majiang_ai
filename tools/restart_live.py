# -*- coding: utf-8 -*-
"""tools/restart_live.py -- ★★ 线上客户端的**安全续跑** ✓（§Z.1686 的机械化 ✓）

★★ 为什么需要它 ✓（★ 这次是**硬倒计时** ✗）：
   · ★ 客户端带 `--loop-hours` ✓ ⇒ **到时自停** ✗（实测：`live_r16d` 上限 24 h ✓）
   · ★ 一停 ⇒ 模型臂的新房**停增** ✗ ⇒ **L2 断档** ✗ ⇒ ★ **校准台账无法回填** ✗
     ⇒ ★★ **goal §2 的校准条款继续卡住** ✗ ⇒ **M1 恶化** ✗
   ⇒ ★★ 所以续跑**不是可选项** ✓ ⇒ ★ 而手工续跑**风险高** ✗：
     · §Z.1432 ✗：★ **双开**（旧客户端没停就起新的 ⇒ 抢桶 ⇒ 429 ✗）
     · §Z.1452 ✗：★ **复用旧日志名** ✗ ⇒ 两个进程写同一文件 ✗
     · §Z.1543 ✗：★ 臂文件格式错 ⇒ 臂名非法 ✗

★★ 它做 5 件事 ✓（★ 全部只做机械动作 ✓）：
   ① ★ **核验臂文件** ✓（复用 `tools/check_arms_file.py` ✓ 不重复实现 ✓）
   ② ★★ **等旧客户端【静默】** ✓ —— ★ 用日志 mtime 当代理 ✓
      （★ 与 `liveness_monitor.py`／`next_training.py` **同一原则** ✓；
       ★★ 静默不够就**拒绝** ✗ ⇒ **绝不双开** ✓）
   ③ ★ **自动取下一个空闲日志名** ✓（`live_r16?.log` ⇒ 找第一个不存在的 ✓）
   ④ ★ **起** ✓（★ 臂**不写死** ✓ ⇒ 自动跟随 `live_ab_arms.txt` ✓ 这才是热切换的设计 ✓）
   ⑤ ★ **验证它真的活了** ✓（★ 等 N 秒后看新日志是否在长 ✓）

★ ★★ **它【不】做** ✗（★ 刻意 ✓）：★ **不 kill 旧进程** ✗（硬规则② ✓：身份不明就不动 ✓）

★ 用法 ✓：
   # 干跑（默认 ✓ 只检查）
   .\\_venv\\Scripts\\python.exe tools\\restart_live.py --old-log live_r16d.log
   # 真起（★ 确认旧客户端已静默后 ✓）
   .\\_venv\\Scripts\\python.exe tools\\restart_live.py --old-log live_r16d.log --yes
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time


def _mtime_age(path):
    if not os.path.exists(path):
        return None
    return time.time() - os.path.getmtime(path)


def _next_free_log():
    """★ 找第一个不存在的 `live_r16<letter>.log` ✓（★ 从 e 开始 ✓）"""
    for ch in "efghijklmnopqrstuvwxyz":
        cand_log = "live_r16%s.log" % ch
        cand_json = "live_r16%s.jsonl" % ch
        if not os.path.exists(cand_log) and not os.path.exists(cand_json):
            return cand_log, cand_json
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-log", required=True, help="旧客户端的日志（判它是否已停）")
    ap.add_argument("--stale", type=float, default=420.0,
                    help="旧日志静默多久算已停（秒，默认 420 = 7 分钟；> 一局的间隔 ✓）")
    ap.add_argument("--loop-hours", type=float, default=72.0)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=16)
    # ★★ 以下 5 项**必须与旧客户端逐字一致** ✗（★ 否则语义变了 ✓ §Z.1699）：
    #   ★ `--token-file` 的 **argparse 默认是 `tournament_token.txt`** ✗
    #     ⇒ ★ 旧客户端用的是 **`global_token.txt`** ✓ ⇒ ★ **不传就会用错令牌** ✗✗
    #   ★ `--minutes` 默认 300 ✓ 而旧命令给 90 ✓（「比赛段时长上限」✓ `run_live.py:581`）
    #   ★ `--fetch-gap-max` 默认 0.5 ✓ 而旧命令给 1.2 ✓（★ 放宽取快照间隔 ✓）
    #   ★ `--exit-after-games 10` **不是**退出触发器 ✓ —— ★ 它是「自动房设 10（M=10）」
    #     （★ `run_live.py:450` 注释 ＋ `:598` 传参 ✓）
    #   ★ `--wait-rooms 8` 与默认相同 ✓
    ap.add_argument("--token-file", default="global_token.txt",
                    help="★ 必须与旧客户端一致（默认 tournament_token.txt ✗）")
    ap.add_argument("--minutes", type=int, default=90)
    ap.add_argument("--fetch-gap-max", type=float, default=1.2)
    ap.add_argument("--exit-after-games", type=int, default=10)
    ap.add_argument("--wait-rooms", type=float, default=8.0)
    ap.add_argument("--wait-verify", type=float, default=90.0, help="起后等多久验证（秒）")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    ok = True

    # ── ① 核验臂文件 ✓ ─────────────────────────────────────────────────
    print("=== ① 核验臂文件 ✓（★ 复用既有检查器 ✓）===")
    print("  当前臂 = %r" % io.open("live_ab_arms.txt", encoding="utf-8-sig").read().strip())
    rc = subprocess.call([sys.executable, os.path.join("tools", "check_arms_file.py"),
                          "live_ab_arms.txt"])
    print("  check_arms_file rc = %d" % rc)
    if rc != 0:
        ok = False

    # ── ② 等旧客户端静默 ✓（★ 绝不双开 ✓）────────────────────────────────
    print("=== ② 旧客户端是否已停 ✓（★ 只看日志，不碰进程 ✓）===")
    age = _mtime_age(a.old_log)
    if age is None:
        print("  %s 不存在 ⇒ ★ 视为**未在跑** ✓" % a.old_log)
    else:
        print("  %s：mtime 距今 **%.0fs**（阈值 %.0fs）" % (a.old_log, age, a.stale))
        if age <= a.stale:
            print("  ⇒ ✗ **仍在写日志 ⇒ 判定【还在跑】⇒ 拒绝起新的** ✓（★ 防双开 ✗ §Z.1432）")
            ok = False
        else:
            print("  ⇒ ✓ **静默已超阈值 ⇒ 判定已停** ✓")

    # ── ③ 取新日志名 ✓ ─────────────────────────────────────────────────
    new_log, new_json = _next_free_log()
    print("=== ③ 新日志名 ✓ ===")
    if new_log is None:
        print("  ✗ 找不到空闲的 live_r16?.log ⇒ 需人工清理 ✓")
        ok = False
    else:
        print("  新日志 = %s ／ %s ✓（★ §Z.1452：绝不复用旧日志名 ✓）" % (new_log, new_json))

    if not ok:
        print("=> ✗ **有前置问题 ⇒ 未起任何东西** ✓")
        return 1

    cmd = [sys.executable, "run_live.py",
           "--token-file", a.token_file, "--match",
           "--loop", "400", "--loop-hours", str(a.loop_hours),
           "--wait-rooms", str(a.wait_rooms), "--minutes", str(a.minutes),
           "--rounds", str(a.rounds), "--m", str(a.m),
           "--exit-after-games", str(a.exit_after_games),
           "--fetch-gap-max", str(a.fetch_gap_max),
           "--log", new_log, "--result-json", new_json]
    print("=== ④ 启动命令 ✓（★ 臂不写死 ⇒ 自动跟随 live_ab_arms.txt ✓）===")
    print("  " + " ".join(cmd))
    if not a.yes:
        print("  ⇒ ★ **干跑**（★ 加 `--yes` 才真起 ✓）")
        return 0

    print("=== ⑤ 启动 ✓（★★ 必须由【你】在 `pwsh` 后台作业里直接跑 ✗ §Z.1701）===")
    print("  ★★★ 血的教训 ✓：★ 本脚本【不能】自己起客户端 ✗ —— ")
    print("    它用 `subprocess.Popen` 起的子进程，会在**本脚本退出时被孤立/杀掉** ✗")
    print("    实测（§Z.1701）：★ 这样起的 `live_r16e` 只活了 **≈2 分钟** ✗ 然后静默挂死 ✗")
    print("    而活了 **24 h** 的旧客户端，是 **`pwsh` 后台作业【直接】跑 `run_live.py`** ✓")
    print("  ⇒ ★★ **请把下面这行【照抄】进一个 `pwsh` 后台作业** ✓：")
    print("")
    print("  " + " ".join(cmd))
    print("")
    print("  （★ 即：用 `run_in_background: true` 的 pwsh 调用，命令就是上面那一行 ✓）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
