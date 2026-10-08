#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""value_head_preheat_probe.py - 只读探针：冻结 CandNet 的 trunk 与打分头，只训 self.value，
在验证折上量 Spearman。回答一个问题：**CandNet 自带的 trunk 能不能支撑价值预测** ✓
（若能 ⇒ §Z.1230 的"只预热 value 头"方案成立 ✓；若不能 ⇒ ③ 需要换方案 ✗）
不写任何生产文件（输出仅到 _tmp/）✓
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rl_boot                                                        # noqa: E402
TORCH = rl_boot.bootstrap(verbose=False)
import numpy as np                                                    # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "tools"))
import value_head_eval as V                                           # noqa: E402
sys.path.insert(0, ROOT)
import rl_action_space as A                                           # noqa: E402
import rl_policy as RP                                                # noqa: E402

d = np.load(os.path.join(ROOT, "_tmp", "value_data100k.npz"))
X, y = d["X"], d["y_score"]
n = len(X); cut = int(n * 0.85)
Xtr, ytr, Xva, yva = X[:cut], y[:cut], X[cut:], y[cut:]

net = RP.make_net({"state_dim": 370, "cand_turn_dim": A.CAND_TURN_DIM,
                   "cand_win_dim": A.CAND_WIN_DIM, "hidden": 128})
for p in net.parameters():
    p.requires_grad_(False)                # ★ 冻结全部
for p in net.value.parameters():
    p.requires_grad_(True)                 # ★ 只训 value 头

T = TORCH.tensor(Xtr, dtype=TORCH.float32)
Y = TORCH.tensor(ytr, dtype=TORCH.float32)
Ym = float(ytr.mean())                     # 零和标签 ⇒ 均值≈0 ✓
opt = TORCH.optim.Adam(net.value.parameters(), lr=1e-2)
bs = 4096
rng = np.random.default_rng(0)
print("=== 只预热 CandNet.value（冻结 trunk 与打分头）===")
for ep in range(1, 6):
    perm = rng.permutation(len(T))
    tot = 0.0
    for s in range(0, len(T), bs):
        b = perm[s:s + bs]
        # ★ 直接复用 net.trunk 提特征（与 RL 前向同一条 ✓ 不需要 mu/sd ✓）
        with TORCH.no_grad():
            h = net.trunk(T[b])
        pv = net.value(h).squeeze(-1)
        loss = TORCH.nn.functional.mse_loss(pv, Y[b] - Ym)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss) * len(b)
    with TORCH.no_grad():
        hv = net.trunk(TORCH.tensor(Xva, dtype=TORCH.float32))
        pv = net.value(hv).squeeze(-1).numpy() + Ym
    rho = V.spearman(list(pv), list(yva))
    print("  ep%d  训练MSE %.3f ｜ 验证 Spearman %+.4f" % (ep, tot / len(T), rho))
lo, hi = V.boot_ci(list(pv), list(yva))
print("  ★ 末轮 样本外 Spearman = %+.4f ｜ 95%%CI [%+.4f, %+.4f]" % (rho, lo, hi))
print("  ★ 门槛 0.387 ⇒ %s" % ("**超过 ✓ 方案成立**" if lo > 0.387 else "**未超过 ✗ 需换方案**"))
for i, (cnt, pm, ym) in enumerate(V.buckets(list(pv), list(yva), 5)):
    print("     桶%d n=%-5d 预测均 %+8.3f ｜ 实际均分 %+8.3f" % (i + 1, cnt, pm, ym))
TORCH.save({"state": net.state_dict(), "note": "value-head preheat probe"},
           os.path.join(ROOT, "_tmp", "candnet_value_probe.pt"))