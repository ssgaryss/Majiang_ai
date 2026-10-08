# -*- coding: utf-8 -*-
"""tools/calibration_ledger.py -- goal §2「校准条款」的登记表与一致率 ✓

★★ 为什么需要它 ✓（goal 强制要求，而**本仓此前没有维护** ✗）：
   goal §2 原文：
     「★★ 校准条款：每个候选记录「**L1 预测方向 vs L2 实测方向**」；
       累计 **≥10 组**算一致率：**≥80% ⇒ L1 可继续当筛子**
       ｜ **<80% ⇒ L1 夹具作废重建，重建前不得用 L1 排序候选**」
   ⇒ ★ 这是一条**门控** ✓：★ 一致率达标前，**离线只能否决，不能宣布成功** ✓（§5 硬规则⑬ ✓）

★ 数据从哪来 ✓：
   · **L1**（离线）✓ = 训练日志／history 里的**配对评估 Δμ**（`[配对评估 迭代N] Δμ(分/局)` ✓）
   · **L2**（线上）✓ = `analyze_live_paired.py` 的**主口径 β**（★ 唯一有权宣布"更强"的一层 ✓）
   · **对应关系** ✗ = 哪个 run 的权重最后上了线上、以哪个臂名 ✓
     ⇒ ★★ 这层**无法自动推断** ✗ ⇒ ★ 用本目录下的 **`calibration_pairs.jsonl`** 手工登记 ✓
       （★ 每部署一个候选就 append 一行 ✓ —— ★ 这本身就是"预注册"的精神 ✓）

★ 用法 ✓：
   .\\_venv\\Scripts\\python.exe tools\\calibration_ledger.py                 # 打印表 + 一致率 + 门槛判定
   .\\_venv\\Scripts\\python.exe tools\\calibration_ledger.py --add "run,arm,l1,l2,note"

★ 记录格式（JSONL ✓ 每行一个候选 ✓）：
   {"run": "rl_aux", "arm": "ckpt:rl_aux_0004.pt", "l1_dir": +1, "l1_val": 0.12,
    "l2_dir": -1, "l2_val": -3.39, "l2_ci": [-6.25, -0.52], "note": "主口径 β@0.5h"}
   · `l1_dir`／`l2_dir` ∈ {+1, 0, -1} ✓（★ +1 = 预测"模型更好"／0 = 未分辨／-1 = 更差 ✓）
   · ★ **方向一致**的定义 ✓：★★ 两者**同为 +1** 或 **同为 -1** ⇒ 一致 ✓；
     任一为 **0（未分辨）** ⇒ ★ **不计入一致率** ✗（★ 但**计入"已用掉"** ✓）
"""
from __future__ import annotations

import io
import json
import os
import sys

LEDGER = os.path.join("tools", "calibration_pairs.jsonl")
MIN_PAIRS = 10          # goal §2
MIN_RATE = 0.80         # goal §2


def load():
    if not os.path.exists(LEDGER):
        return []
    out = []
    for ln in io.open(LEDGER, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            out.append(json.loads(ln))
        except Exception as e:                                # noqa: BLE001
            print("  !! 跳过坏行：%r（%r）" % (ln[:60], e))
    return out


def add(spec):
    """`--add "run,arm,l1_val,l2_val[,note]"` ✓（方向由值的符号推断 ✓）"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < 4:
        print("  ✗ --add 需要 run,arm,l1_val,l2_val[,note]")
        return 1
    run, arm = parts[0], parts[1]
    l1v, l2v = float(parts[2]), float(parts[3])
    note = parts[4] if len(parts) > 4 else ""

    def sgn(x, tol=1e-9):
        return 1 if x > tol else (-1 if x < -tol else 0)
    rec = {"run": run, "arm": arm, "l1_val": l1v, "l2_val": l2v,
           "l1_dir": sgn(l1v), "l2_dir": sgn(l2v), "note": note}
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with io.open(LEDGER, "a", encoding="utf-8", newline="\n") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("  ✓ 已登记：%s" % json.dumps(rec, ensure_ascii=False))
    return 0


def main():
    if "--add" in sys.argv:
        i = sys.argv.index("--add")
        return add(sys.argv[i + 1] if i + 1 < len(sys.argv) else "")

    rows = load()
    print("=== goal §2 校准台账（L1 预测方向 vs L2 实测方向）===")
    if not rows:
        print("  （空 ⇒ ★ 一致率【无法计算】✗ ⇒ 按 §5 硬规则⑬：**离线只能否决，不能宣布成功** ✓）")
        print("  ★ 每部署一个候选请 append：")
        print("    .\\_venv\\Scripts\\python.exe tools\\calibration_ledger.py "
              "--add \"run,arm,l1_val,l2_val,note\"")
        return 0

    n_tot = len(rows)
    n_dec = n_agree = 0
    for r in rows:
        a, b = int(r.get("l1_dir", 0)), int(r.get("l2_dir", 0))
        ok = "-"
        if a != 0 and b != 0:
            n_dec += 1
            if a == b:
                n_agree += 1
                ok = "一致 ✓"
            else:
                ok = "**相反** ✗"
        print("  %-10s %-26s L1 %+7.3f(%+d) ｜ L2 %+7.3f(%+d) ｜ %s  %s"
              % (r.get("run", "?"), str(r.get("arm", "?"))[:26],
                 float(r.get("l1_val", 0)), a, float(r.get("l2_val", 0)), b, ok,
                 r.get("note", "")))

    print("  --- 汇总 ---")
    print("  已记录候选 = %d ｜ **可判定（双方都非 0）** = %d ｜ 一致 = %d"
          % (n_tot, n_dec, n_agree))
    if n_dec == 0:
        print("  ⇒ ★ 无可判定组 ⇒ 一致率无法计算 ✗ ⇒ **离线只能否决** ✓（§5⑬ ✓）")
        return 0
    rate = n_agree / float(n_dec)
    print("  一致率 = %.3f（%d/%d）" % (rate, n_agree, n_dec))
    if n_dec < MIN_PAIRS:
        print("  ⇒ ★ **组数 %d < %d** ✗ ⇒ 按 §2 **尚未达标** ⇒ "
              "**L1 不能用于排序候选**（只能否决 ✓）" % (n_dec, MIN_PAIRS))
        return 0
    if rate >= MIN_RATE:
        print("  ⇒ ★★ **≥%d 组且一致率 %.1f%% ≥ %.0f%%** ✓ ⇒ **L1 可继续当筛子** ✓"
              % (MIN_PAIRS, 100 * rate, 100 * MIN_RATE))
    else:
        print("  ⇒ ★★ **一致率 %.1f%% < %.0f%%** ✗✗ ⇒ 按 §2 **L1 夹具作废重建** ✗；"
              "★ 重建前**不得用 L1 排序候选** ✓" % (100 * rate, 100 * MIN_RATE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
