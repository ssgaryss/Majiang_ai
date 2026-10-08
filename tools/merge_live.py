# -*- coding: utf-8 -*-
"""tools/merge_live.py -- 把多段线上 JSONL 合成【干净】的分析文件 ✓

★ 为什么需要 ✓：客户端换过日志路径（`live_r16` → `live_r16c` → `live_r16d` ✓，
   原因是"旧进程仍持有旧文件句柄" ✗），而所有分析工具默认读 `live_r16.jsonl` ✗
   ⇒ 新房不会被算进去 ✗ ⇒ **判据会悄悄失真** ✓

★★ 已知污染窗口（`GOAL_STATUS.md` §Z.1446 / §Z.1448 / §Z.1452 ✓）：
   · **15:44–16:21** ✗：双开（`live_r16b` 与 `live_r16` 同房双写 ✓ 两条不同的臂 ✓）⇒ **必须剔除** ✓
   · **16:16–16:21** ✗：无客户端 ✓（含在上面的窗口内 ✓）
   · **16:43–16:44** ✗：旧客户端短暂苏醒 ✓（`live_r16` 又写了一点 ✓）⇒ 一并剔除 ✓
   ⇒ ★ 其余时段**单客户端** ✓ ⇒ 干净 ✓

用法 ✓：  .\\_venv\\Scripts\\python.exe tools\\merge_live.py
产物 ✓：  `_tmp/live_clean.jsonl`（供 `analyze_live_paired.py --jsonl` 使用 ✓）
"""
import glob
import io
import json
import os
import sys
import time

# ★★★ 2026-10-07（§Z.1755）：★ **改成【自动发现】** ✗✗
#   ✗ 旧写法：`SRC = ["live_r16.jsonl", "live_r16c.jsonl", "live_r16d.jsonl"]` ✗
#     ⇒ ★ **硬编码** ✗ ⇒ ★ 客户端每换一次日志名（`live_r16e` / `f` / `g` … ✓ §Z.1452 ✓）
#       ⇒ ★★ **新房【静默漏掉】** ✗✗ —— ★ 实测：跨文件原始 **36 房** 而干净集只 **10 房** ✗
#     ⇒ ★★ **而且 `--glob` 参数【根本没被用】** ✗ ⇒ ★ 传了也没用 ⇒ **静默失效** ✗✗
#   ✓ 现在：★ **自动发现所有 `live_r16*.jsonl`** ✓（★ 按文件名排序 ✓ 保证可复现 ✓）
#     ⇒ ★ **再也不会因为换了日志名而漏房** ✓
SRC = sorted(glob.glob("live_r16*.jsonl"))
OUT = os.path.join("_tmp", "live_clean.jsonl")

# ★ 污染窗口（本仓实测 ✓ 秒级时间戳 ✓）
BAD = [
    (time.mktime(time.strptime("2026-10-05 15:44:00", "%Y-%m-%d %H:%M:%S")),
     time.mktime(time.strptime("2026-10-05 16:21:00", "%Y-%m-%d %H:%M:%S"))),
    (time.mktime(time.strptime("2026-10-05 16:43:00", "%Y-%m-%d %H:%M:%S")),
     time.mktime(time.strptime("2026-10-05 16:45:00", "%Y-%m-%d %H:%M:%S"))),
]


def contaminated(ts):
    return any(lo <= ts < hi for lo, hi in BAD)


def main():
    os.makedirs("_tmp", exist_ok=True)
    seen = set()          # (room, genome) 去重 ✓（跨文件重复写入 ✓）
    kept, dropped_bad, dropped_dup, bad_lines = 0, 0, 0, 0
    per_arm = {}
    per_arm_rooms = {}
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fo:
        for fn in SRC:
            if not os.path.exists(fn):
                print("  （跳过，不存在）%s" % fn)
                continue
            n_kept = 0
            for ln in io.open(fn, encoding="utf-8", errors="replace"):
                ln = ln.strip()
                if not ln or '"a_' not in ln:
                    continue
                try:
                    d = json.loads(ln)
                except Exception:
                    bad_lines += 1
                    continue
                room = str(d.get("room") or "")
                if not room.startswith("a_"):
                    continue
                arm = (d.get("genome") or "").lstrip("\ufeff").strip()
                ts = float(d.get("ended") or 0)
                if contaminated(ts):
                    dropped_bad += 1
                    continue
                key = (room, arm)
                if key in seen:
                    dropped_dup += 1
                    continue
                seen.add(key)
                fo.write(ln + "\n")
                kept += 1
                n_kept += 1
                # 统计 ✓（只算**完整观测**的局 ✓）
                mine = [r["mine"] for r in (d.get("rows") or [])
                        if not (r.get("deltas_base") and any(r["deltas_base"]))
                        and r.get("mine") is not None]
                if mine:
                    a = per_arm.setdefault(arm, [0, 0.0])
                    a[0] += 1
                    a[1] += sum(mine) / float(len(mine))
                    per_arm_rooms.setdefault(arm, []).append(sum(mine) / float(len(mine)))
            print("  %-20s 收入 %d 条 ✓" % (fn, n_kept))
    print("\n  ⇒ 写入 %s ｜ 保留 %d ｜ 剔除(污染) %d ｜ 去重 %d ｜ 坏行 %d"
          % (OUT, kept, dropped_bad, dropped_dup, bad_lines))
    print("  --- 各臂（仅完整观测局 ✓ 房均单位：分/局）---")
    for arm in sorted(per_arm, key=lambda k: -per_arm[k][0]):
        n, s = per_arm[arm]
        print("    %-26s 房=%3d ｜ 房均 %+7.3f" % (arm[:26], n, s / n if n else 0.0))


if __name__ == "__main__":
    main()
