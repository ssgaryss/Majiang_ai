#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""value_head_score.py - 用**我的仪器**复核价值头：按连续块切分的样本外 Spearman + CI + 分桶。

与 train_value_torch.py 自带指标的区别：那个报的是"相关"，我报 **Spearman + bootstrap CI**，
并与**门槛 0.387** 直接对比（§Z.1212 的无泄漏门槛）。
用法: python tools/value_head_score.py <npz> <ckpt> [val_frac]
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rl_boot                                                        # noqa: E402
TORCH = rl_boot.bootstrap(verbose=False)
import numpy as np                                                    # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "tools"))
import value_head_eval as V                                           # noqa: E402
import train_value_torch as TV                                        # noqa: E402

npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "_tmp", "value_data100k.npz")
ck = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "_tmp", "value_v2.pt")
vf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

d = np.load(npz)
X, ys = d["X"], d["y_score"]
n = len(X)
cut = int(n * (1 - vf))
Xv, yv = X[cut:], ys[cut:]                      # ★ 连续块切分（同一小局不跨折）
print("=== 我的仪器复核（连续块切分 ✓ n_train=%d n_val=%d）===" % (cut, n - cut))

ckd = TORCH.load(ck, map_location="cpu", weights_only=False)   # ★ 本地可信文件 ✓
print("  检查点键 =", sorted(ckd.keys()) if isinstance(ckd, dict) else type(ckd).__name__)
sd = None
for k in ("state", "sd", "state_dict", "model"):
    if isinstance(ckd, dict) and k in ckd:
        sd = ckd[k]; break
if sd is None:
    sd = ckd
net = TV.VNet(int(ckd["din"]), int(ckd["hidden"]), int(ckd["depth"]), float(ckd["drop"]))
net.load_state_dict(sd)
net.eval()
# ★ 关键 ✓：训练器在**标准化后**的特征上训练 ⇒ 评估也必须同口径 ✗（否则尺度不匹配 ✓）
mu, sdv = np.asarray(ckd["mu"], dtype=np.float32), np.asarray(ckd["sd"], dtype=np.float32)
Xv_std = (Xv - mu) / sdv
with TORCH.no_grad():
    pred = net(TORCH.tensor(Xv_std, dtype=TORCH.float32))[1].numpy()
rho = V.spearman(list(pred), list(yv))
lo, hi = V.boot_ci(list(pred), list(yv))
print("  样本外 Spearman = %+.4f ｜ 95%%CI [%+.4f, %+.4f]" % (rho, lo, hi))
print("  ★ 门槛（线性基线，无泄漏）= 0.387 ⇒ %s" % ("**超过 ✓**" if lo > 0.387 else ("超过点估计但 CI 跨门槛" if rho > 0.387 else "**未超过 ✗**")))
for i, (cnt, pm, ym) in enumerate(V.buckets(list(pred), list(yv), 5)):
    print("     桶%d n=%-5d 预测均 %+8.3f ｜ 实际均分 %+8.3f" % (i + 1, cnt, pm, ym))