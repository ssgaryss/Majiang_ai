#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""价值网络 —— 从自对弈状态预测「这一席最终拿多少分 / 会不会胡」。

为什么第一个网络选「价值」而不是直接学策略：
  1. **动作空间极小**（出牌只有 ~14 种合法），所以拿到一个可信的 V 之后，
     一步前瞻 argmax_d V(打完 d 之后的状态) 就是一个完整、稳定的策略，
     不需要策略梯度的那些坑（高方差、需要 reward shaping、对超参敏感）。
  2. **可验证**。价值网络能直接量「预测准不准」（AUC / 校准 / Brier），
     而策略网络只能靠对局胜负间接看，噪声大得多。
  3. 它替换的正是把 `discard=value` 搞崩的那个手写 `win_prob`
     —— 那个骨架在对拍里是 -5.81 分/局、p=0.0002，说明手写概率模型很差。
     先用学习出来的替换掉它，是收益最确定的一步。

目标同时也对齐官方排名：指南 §判定链 的**主键就是 total_score**，
所以回归头直接预测每局最终得分，而不是用「胜率」这种代理量。

用法：
  python train_value.py --data selfplay_train.npz --out value_net.npz
"""
import argparse
import sys
import time

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------ 网络

class Net:
    """共享躯干 + 两个头（胡牌概率 / 期望得分）。纯 numpy，Adam 优化。"""

    def __init__(self, din, hidden=128, seed=0):
        rng = np.random.default_rng(seed)
        h = hidden
        # He 初始化
        self.W1 = rng.normal(0, np.sqrt(2.0 / din), (din, h)).astype(np.float32)
        self.b1 = np.zeros(h, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / h), (h, h)).astype(np.float32)
        self.b2 = np.zeros(h, dtype=np.float32)
        self.Ww = rng.normal(0, np.sqrt(1.0 / h), (h, 1)).astype(np.float32)
        self.bw = np.zeros(1, dtype=np.float32)
        self.Ws = rng.normal(0, np.sqrt(1.0 / h), (h, 1)).astype(np.float32)
        self.bs = np.zeros(1, dtype=np.float32)
        self.params = ["W1", "b1", "W2", "b2", "Ww", "bw", "Ws", "bs"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.t = 0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0.0)
        logit = (a2 @ self.Ww + self.bw).ravel()
        score = (a2 @ self.Ws + self.bs).ravel()
        return logit, score, (X, z1, a1, z2, a2)

    def backward(self, cache, logit, yw, score, ys, lam):
        X, z1, a1, z2, a2 = cache
        n = X.shape[0]
        p = 1.0 / (1.0 + np.exp(-logit))
        dl = (p - yw) / n                                  # BCE 梯度
        ds = lam * (score - ys) * 2.0 / n                  # MSE 梯度
        g = {}
        g["Ww"] = a2.T @ dl[:, None]
        g["bw"] = np.array([dl.sum()], dtype=np.float32)
        g["Ws"] = a2.T @ ds[:, None]
        g["bs"] = np.array([ds.sum()], dtype=np.float32)
        da2 = dl[:, None] @ self.Ww.T + ds[:, None] @ self.Ws.T
        dz2 = da2 * (z2 > 0)
        g["W2"] = a1.T @ dz2
        g["b2"] = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        g["W1"] = X.T @ dz1
        g["b1"] = dz1.sum(0)
        return g

    def adam(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.t += 1
        for k in self.params:
            gk = g[k]
            if wd and k.startswith("W"):
                gk = gk + wd * getattr(self, k)      # L2 只作用在权重，不碰偏置
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * (gk ** 2)
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + eps))

    def snapshot(self):
        return {k: getattr(self, k).copy() for k in self.params}

    def restore(self, snap):
        for k, val in snap.items():
            setattr(self, k, val)

    def save(self, path, mu, sd):
        np.savez(path, mu=mu, sd=sd,
                 **{k: getattr(self, k) for k in self.params})

    @staticmethod
    def load(path):
        d = np.load(path)
        net = Net(d["W1"].shape[0], d["W1"].shape[1])
        for k in net.params:
            setattr(net, k, d[k])
        return net, d["mu"], d["sd"]


# ------------------------------------------------------------------ 指标

def auc(y, s):
    """秩和法 AUC。y 为 0/1，s 为分数。"""
    y = np.asarray(y)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 并列取平均秩
    s_sorted = np.asarray(s)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


# ------------------------------------------------------------------ 训练

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="selfplay_train.npz")
    ap.add_argument("--out", default="value_net.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.02,
                    help="得分回归头的权重（相对 BCE）")
    ap.add_argument("--wd", type=float, default=1e-4, help="L2 权重衰减")
    ap.add_argument("--patience", type=int, default=4,
                    help="验证集连续多少轮不改善就停")
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()

    d = np.load(a.data)
    X, yw, ys = d["X"], d["y_win"].astype(np.float32), d["y_score"]
    n = len(X)
    print("=== 数据 ===")
    print("  样本 = %d   特征 = %d   胡牌率 = %.1f%%   得分均值 %+.2f / SD %.2f"
          % (n, X.shape[1], 100 * yw.mean(), ys.mean(), ys.std()))

    # 按连续尾部切验证集。样本在同一局内高度相关，所以不能随机切；
    # 尾部切分等于换了一批牌局，是最简单且诚实的做法。
    cut = int(n * (1 - a.val_frac))
    mu = X[:cut].mean(0)
    sd = X[:cut].std(0) + 1e-6
    Xn = (X - mu) / sd

    Xtr, ywtr, ystr = Xn[:cut], yw[:cut], ys[:cut] / 10.0     # 得分缩放到 ~O(1)
    Xva, ywva, ysva = Xn[cut:], yw[cut:], ys[cut:] / 10.0
    print("  训练 %d / 验证 %d（尾部切分，避免同局样本泄漏）" % (cut, n - cut))

    net = Net(X.shape[1], a.hidden, seed=7)
    rng = np.random.default_rng(11)
    t0 = time.time()
    best = None
    best_w = None
    bad = 0
    for ep in range(1, a.epochs + 1):
        idx = rng.permutation(cut)
        tot = 0.0
        for s in range(0, cut, a.bs):
            b = idx[s:s + a.bs]
            logit, score, cache = net.forward(Xtr[b])
            p = 1.0 / (1.0 + np.exp(-logit))
            eps = 1e-7
            loss = (-(ywtr[b] * np.log(p + eps) +
                      (1 - ywtr[b]) * np.log(1 - p + eps)).mean()
                    + a.lam * ((score - ystr[b]) ** 2).mean())
            tot += loss * len(b)
            net.adam(net.backward(cache, logit, ywtr[b], score, ystr[b], a.lam),
                     lr=a.lr, wd=a.wd)
        logit, score, _ = net.forward(Xva)
        pv = 1.0 / (1.0 + np.exp(-logit))
        va_bce = float(-(ywva * np.log(pv + 1e-7) +
                         (1 - ywva) * np.log(1 - pv + 1e-7)).mean())
        va_mse = float(((score - ysva) ** 2).mean())
        va_auc = auc(ywva, pv)
        va_br = brier(ywva, pv)
        improved = best is None or va_bce < best[0] - 1e-5
        if improved:
            # 必须把**最优轮的权重**存下来。早先只记了指标、最后却保存了当前
            # （已经过拟合到第 30 轮）的权重 —— 验证 BCE 从 0.48 涨到 0.83 的那份。
            best = (va_bce, ep, pv.copy(), score.copy())
            best_w = net.snapshot()
            bad = 0
        else:
            bad += 1
        if ep % 2 == 0 or ep == 1:
            print("  ep%-3d 训练损失 %.4f  验证 BCE %.4f  AUC %.4f  Brier %.4f  "
                  "得分 MSE %.4f  %s(%.0fs)"
                  % (ep, tot / cut, va_bce, va_auc, va_br, va_mse,
                     "★" if improved else "  ", time.time() - t0))
        if bad >= a.patience:
            print("  早停：验证集连续 %d 轮未改善（最优 ep=%d）" % (bad, best[1]))
            break

    net.restore(best_w)
    print()
    print("=== 验证集评估（已回滚到最优 epoch = %d）===" % best[1])
    pv, sv = best[2], best[3] * 10.0
    print("  基线：恒定预测胡牌率 %.4f -> Brier %.4f" %
          (ywva.mean(), brier(ywva, np.full_like(pv, ywva.mean()))))
    print("  网络：AUC %.4f   Brier %.4f" % (auc(ywva, pv), brier(ywva, pv)))
    # 手写基线：只用向听（特征 68+4+9 位）线性预测
    sh_col = 68 + 4 + 9
    print("  手写基线（仅向听）：AUC %.4f" % auc(ywva, -Xva[:, sh_col]))
    print("  得分回归：MSE %.3f  预测均值 %+.2f  真值均值 %+.2f  相关 %.3f"
          % (float(((sv - ysva * 10) ** 2).mean()), sv.mean(),
             (ysva * 10).mean(), float(np.corrcoef(sv, ysva * 10)[0, 1])))
    print()
    print("  校准表（预测胡牌概率 -> 实测频率）：")
    bins = np.quantile(pv, np.linspace(0, 1, 6))
    bins[-1] += 1e-6
    for i in range(5):
        m = (pv >= bins[i]) & (pv < bins[i + 1])
        if m.sum():
            print("    [%.2f, %.2f)  n=%-6d 预测 %.3f  实测 %.3f"
                  % (bins[i], bins[i + 1], m.sum(), pv[m].mean(),
                     ywva[m].mean()))
    net.save(a.out, mu, sd)
    print()
    print("  已保存 -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
