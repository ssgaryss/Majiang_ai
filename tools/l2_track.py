# -*- coding: utf-8 -*-
"""tools/l2_track.py -- ★★ 跨文件的 L2 追踪器 ✓（§Z.1704 的新口径 ✓）

★★ 为什么需要它 ✓：
   · ★ 客户端**换过日志名** ✓（`live_r16` → `d` → `f` ✓ §Z.1452 ✓）
     ⇒ ★★ **单个 `live_r16*.jsonl` 会严重低估** ✗
       （★ 实测：只看 `live_r16f` ⇒ **1 房** ✗ 而实际 **10 房** ✓ ⇒ **低估 10 倍** ✗✗）
   · ★ goal §0 的终局判据是**按房聚合** ✓ ⇒ ★ 房的**归属**（哪个臂 ✓）与**时间**都要看 ✓

★★ 它做 3 件事 ✓：
   ① ★ **跨所有 `live_r16*.jsonl`** 按臂统计房数 ✓（★ 房间级 ✓ 不重复计 ✓ 按 `room` 去重 ✓）
   ② ★ **算"房/小时"** ✓（★ 长时间窗 ✓ 避免"突发式给房"造成的误导 ✓）
   ③ ★ **算主口径** ✓：★ 模型臂房均 − 规则臂房均 ＋ 95 %CI ✓（★ 房间级 Welch ✓）
      ⇒ ★★ **只有 CI 下界 > 0 才算模型更强** ✓（★ goal §0 ✓）

★ ★★ 它**不说**"模型变强了" ✗ —— ★ 它只把**可引用的数字**摆出来 ✓
  （★ 依 goal §2：★ **唯一有权宣布的是 L2** ✓ 而本工具只是**读 L2** ✓）

用法 ✓：
  .\\_venv\\Scripts\\python.exe tools\\l2_track.py --model-arm ckpt:rl_aux_0004.pt
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import time


def load_rooms(paths):
    """★ 读所有 jsonl ✓ 按 `room` 去重 ✓（★ 同一个房可能被多个客户端记 ✓ 取最后一次 ✓）"""
    rooms = {}
    for p in paths:
        try:
            with io.open(p, encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    # ★ header 行没有 room ✓ 跳过 ✓
                    rid = o.get("room")
                    if not rid:
                        continue
                    o["_src"] = os.path.basename(p)
                    rooms[rid] = o
        except Exception as e:
            print("  (跳过 %s: %s)" % (p, e))
    return rooms


def welch(a, b):
    """★ 房间级 Welch ✓ 返回 (diff, se, lo, hi) ✓"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    d = ma - mb
    return d, se, d - 1.96 * se, d + 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-arm", required=True)
    ap.add_argument("--rule-arm", default="shanten:1:1:2:5:1:1:0")
    ap.add_argument("--hours", type=float, default=0.0, help="只用最近 N 小时（0 = 全用）")
    a = ap.parse_args()

    paths = sorted(glob.glob("live_r16*.jsonl"))
    print("=== ① 读入（★ 跨文件 ✓ 新口径 ✓）===")
    for p in paths:
        print("  %s" % p)
    rooms = load_rooms(paths)
    print("  ★ 去重后总房数 = **%d**" % len(rooms))

    if a.hours > 0:
        cut = time.time() - a.hours * 3600
        rooms = {k: v for k, v in rooms.items() if (v.get("ended") or 0) >= cut}
        print("  ★ 最近 %.1f h 内 = **%d 房**" % (a.hours, len(rooms)))

    # ── ② 按臂分组 ✓ ────────────────────────────────────────────────────
    by_arm = {}
    for rid, o in rooms.items():
        by_arm.setdefault(o.get("genome", "?"), []).append(o)

    print("=== ② 各臂（★ 房间级 ✓）===")
    print("  %-38s %6s %10s %10s" % ("臂", "房数", "房均", "局均"))
    for arm, rs in sorted(by_arm.items(), key=lambda kv: -len(kv[1])):
        tot = [r.get("total", 0) for r in rs]
        gm = [r.get("games", 0) for r in rs]
        avg = sum(tot) / len(tot)
        per = sum(tot) / max(1, sum(gm))
        print("  %-38s %6d %10.3f %10.4f" % (arm, len(rs), avg, per))

    # ── ③ 吞吐（★ 房/小时 ✓ 用真实时间跨度 ✓）──────────────────────────
    ends = [r.get("ended") for r in rooms.values() if r.get("ended")]
    if len(ends) >= 2:
        span_h = (max(ends) - min(ends)) / 3600.0
        print("=== ③ 吞吐 ✓（★ 用真实时间跨度 ✓ 不用快照 ✓）===")
        t0 = time.strftime("%H:%M:%S", time.localtime(min(ends)))
        t1 = time.strftime("%H:%M:%S", time.localtime(max(ends)))
        print("  跨度 = %s → %s（%.2f h）⇒ **%.2f 房/小时**" % (t0, t1, span_h,
                                                              len(rooms) / max(1e-9, span_h)))

    # ── ④ 主口径 β ✓ ───────────────────────────────────────────────────
    m = [r.get("total", 0) for r in by_arm.get(a.model_arm, [])]
    rl = [r.get("total", 0) for r in by_arm.get(a.rule_arm, [])]
    print("=== ④ 主口径 ✓（★ 模型臂 − 规则臂 ✓ 房间级 Welch ✓）===")
    print("  模型臂 %s ⇒ **%d 房**" % (a.model_arm, len(m)))
    print("  规则臂 %s ⇒ **%d 房**" % (a.rule_arm, len(rl)))
    w = welch(m, rl)
    if w is None:
        print("  ★ 房数不足（各需 ≥2）⇒ **还不能算 β** ✗")
    else:
        d, se, lo, hi = w
        print("  ★★ β = **%+.3f 分/局** ｜ SE %.3f ｜ 95%%CI **[%+.3f, %+.3f]**" % (d, se, lo, hi))
        if lo > 0:
            print("  ⇒ ★★★ **CI 下界 > 0 ⇒ 可以宣布模型更强** ✓（★ goal §0 ✓）")
        elif hi < 0:
            print("  ⇒ ✗ **CI 上界 < 0 ⇒ 模型【显著更差】** ✗")
        else:
            print("  ⇒ ★ **CI 含 0 ⇒ 未分辨** ✗（★ 既不宣布也不淘汰 ✓ §2 ✓）")
        if len(m) < 40:
            print("  ⚠ ★ 模型臂只 **%d 房** < 40 ⇒ ★ **小样本，带保留** ✓（★ 硬规则④ ✓）" % len(m))
        print("")
        print("  ★★★ **本数字【不可引用】作为终局判据** ✗✗（§Z.1571 ｜ §Z.1706 ✓）")
        print("      ★ 原因 ✓：★ **这是裸 Welch ⇒【期间混淆】** ✗ —— ★ 实测：")
        print("        · 最近 6 h 内 β = **+27.750** ✓ ｜ 最近 12 h 内 β = **−5.375** ✗ ｜ 全窗 β = **−29.061** ✗")
        print("        ⇒ ★★ **同一对臂，换个窗口就【换符号】** ✗✗ ⇒ ★ **裸 Welch 毫无分辨力** ✓")
        print("      ★★ **唯一可引用的口径** ✓：★ `analyze_live_paired.py` 的 **小时固定效应 β** ✓")
        print("        （★ 它把「小时」当固定效应 ⇒ **把「环境变难」与「模型变强」分开** ✓ goal §0 ✓）")
        print("      ⇒ ★ 本工具只用于 ✓：**看房数、看吞吐、看各臂的粗读数** ✓（★ 不当判据 ✓）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
