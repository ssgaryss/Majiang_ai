#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""jsonl_to_value_npz.py - 把 _tmp/value_data.jsonl 转成 train_value_torch.py 要的 npz。

npz 键（照 train_value_torch.py:45-56 的 load()）：X / y_score / y_win / y_draw / a
  · y_draw 与 a 在训练器里**不参与损失**（仅 y_win 打印分布）⇒ 这里填 0 占位 ✓
  · 形状/类型：X float32 ｜ y_* float32 ｜ a int64 ✓
用法: python tools/jsonl_to_value_npz.py [max_rows] [out.npz]
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rl_boot                                                        # noqa: E402
rl_boot.bootstrap(verbose=False)      # ★ numpy 不在 _pkgs ⇒ 必须靠 rl_boot 修好（已实测 ✓）
import numpy as np                                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "_tmp", "value_data.npz")

X, ys, yw = [], [], []
with open(os.path.join(ROOT, "_tmp", "value_data.jsonl"), encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        X.append(d["s"]); ys.append(d["y"]); yw.append(d.get("y_win", 0.0))
        if len(X) >= CAP:
            break

X = np.asarray(X, dtype=np.float32)
ys = np.asarray(ys, dtype=np.float32)
yw = np.asarray(yw, dtype=np.float32)
n = X.shape[0]
np.savez(OUT, X=X, y_score=ys, y_win=yw,
         y_draw=np.zeros(n, dtype=np.float32), a=np.zeros(n, dtype=np.int64))
print("=== jsonl -> npz ===")
print("  样本 = %d ｜ X %s ｜ y_score %s ｜ y_win 均值 %.4f"
      % (n, X.shape, ys.shape, float(yw.mean()) if n else 0.0))
print("  输出 = %s ｜ %.1f MB" % (OUT, os.path.getsize(OUT) / 1e6))
d = np.load(OUT)
print("  校验：键 =", sorted(d.files), "｜ X dtype =", d["X"].dtype, "｜ a dtype =", d["a"].dtype)