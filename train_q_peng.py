#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学"碰不碰" —— 用**随机探索**得到的数据做因果估计 + 训 Q（第 123 轮）。

为什么这批数据能回答因果问题（而普通自对弈数据不能）
----------------------------------------------------
`explore_peng_shard.py` 在**每次碰窗口**以概率 ε **从合法动作里随机换一个**。
于是 `explored==1` 的那个子集里，动作是**独立于状态**随机抽的
⇒ 在它上面比"碰 vs 过"的最终得分，就是**无偏的平均处理效应（ATE）**，
不需要任何模型、也不怕混淆。这是本脚本的第一件事，也是**最硬的一条证据**。

第二件事才是训 `Q(X, a)`：把动作 one-hot 接进输入，预测该席本局最终得分。
它的用处是**异质性** —— 平均效应可能是 0，但"该碰的时候碰"可能明显更好。
验证异质性的办法：在**留出的随机化子集**上，看"按 Q 选动作"是否比
"一律碰/一律过"更好。**只在随机化子集上评**，因为只有那里动作不受状态影响。

⚠ **按局分组切分**（`gid`）：同一局的几十个样本共享 `y_score`，
随机切会让标签泄漏 ⇒ 指标虚高。用 `--val-frac` 的**尾部局**做验证。

用法
----
    python train_q_peng.py --data ex*.npz --out q_peng.pt
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "_pkgs")            # torch 在 _pkgs（pip 被沙箱拦）
for _st in ("stdout", "stderr"):
    _s = getattr(sys, _st)
    if getattr(_s, "encoding", None) and _s.encoding.lower() != "utf-8":
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
import torch.nn as nn                                             # noqa: E402

ACTS = [34, 36, 38]                       # 碰 / 杠 / 过
A2I = {a: i for i, a in enumerate(ACTS)}


def load(paths):
    Xs, ys, yw, ac, ex, gid = [], [], [], [], [], []
    for p in paths:
        d = np.load(p)
        Xs.append(d["X"])
        ys.append(d["y_score"].astype(np.float32))
        yw.append(d["y_win"].astype(np.float32))
        ac.append(d["a"].astype(np.int64))
        ex.append(d["explored"].astype(np.int8))
        gid.append(d["gid"].astype(np.int64))
        print("  %-16s %7d 窗口样本" % (os.path.basename(p), len(d["X"])))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(yw),
            np.concatenate(ac), np.concatenate(ex), np.concatenate(gid))


def ate_report(y, a, ex):
    """随机化子集上的 ATE —— **这是最硬的一条证据**。"""
    print()
    print("=== ★ 随机化子集上的因果比较（动作独立于状态）===")
    m = ex == 1
    print("  被探索换过动作的样本 %d / %d = %.1f%%"
          % (m.sum(), len(m), 100 * m.mean()))
    if m.sum() < 50:
        print("  ⚠ 随机化样本太少，不做 ATE")
        return
    name = {34: "碰", 36: "杠", 38: "过"}
    for act in ACTS:
        mm = m & (a == act)
        if mm.sum() < 20:
            print("  %-4s n=%-6d （太少）" % (name[act], mm.sum()))
            continue
        mu = y[mm].mean()
        se = y[mm].std(ddof=1) / np.sqrt(mm.sum())
        print("  %-4s n=%-6d 平均得分 %+7.3f  (SE %.3f)"
              % (name[act], mm.sum(), mu, se))
    for act in (34, 36):
        mm = m & (a == act)
        pp = m & (a == 38)
        if mm.sum() < 20 or pp.sum() < 20:
            continue
        d = y[mm].mean() - y[pp].mean()
        se = np.sqrt(y[mm].var(ddof=1) / mm.sum() + y[pp].var(ddof=1) / pp.sum())
        star = " ★" if abs(d) > 1.96 * se else "（含 0 ⇒ 未分辨）"
        print("  ⇒ %s − 过 = **%+.3f 分/次**  SE %.3f  95%%CI [%+.3f,%+.3f] %s"
              % (name[act], d, se, d - 1.96 * se, d + 1.96 * se, star))


class QNet(nn.Module):
    """Q(X, a)：状态 84 维 + 动作 one-hot 3 维 → 期望最终得分 / 胡牌 logit。"""

    def __init__(self, din, na, hidden=128, depth=2, p_drop=0.1):
        super().__init__()
        layers = []
        d = din + na
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p_drop)]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head_score = nn.Linear(hidden, 1)
        self.head_win = nn.Linear(hidden, 1)

    def forward(self, x, aoh):
        h = self.trunk(torch.cat([x, aoh], dim=1))
        return self.head_score(h).squeeze(-1), self.head_win(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ex*.npz")
    ap.add_argument("--out", default="q_peng.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    paths = ([p for p in a.data.split(",") if p]
             if "," in a.data else sorted(glob.glob(a.data)))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("没有数据：%s" % a.data)
        return 1
    print("=== 数据 ===")
    X, ys, yw, ac, ex, gid = load(paths)
    n = len(X)
    print("  合计 %d 个窗口决策点  来自 %d 局" % (n, len(np.unique(gid))))
    u, c = np.unique(ac, return_counts=True)
    print("  动作分布 %s" % " ".join("%d(%s):%.1f%%"
                                     % (t, {34: "碰", 36: "杠", 38: "过"}.get(t, "?"),
                                        100 * k / n) for t, k in zip(u, c)))
    ate_report(ys, ac, ex)

    # ★ 按局分组切（尾部局），避免同局标签泄漏
    games = np.unique(gid)
    cutg = games[int(len(games) * (1 - a.val_frac))]
    tr = gid < cutg
    va = gid >= cutg
    print()
    print("=== 切分 ===  训练 %d 局/%d 样本   验证 %d 局/%d 样本"
          % (len(np.unique(gid[tr])), tr.sum(), len(np.unique(gid[va])), va.sum()))

    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    def ohe(idx):
        m = np.zeros((len(idx), len(ACTS)), dtype=np.float32)
        for i, t in enumerate(idx):
            m[i, A2I.get(int(t), 2)] = 1.0
        return m

    O = ohe(ac)
    Xtr = torch.from_numpy(Xn[tr]); Otr = torch.from_numpy(O[tr])
    ystr = torch.from_numpy(ys[tr] / 10.0)
    ywtr = torch.from_numpy(yw[tr])
    Xva = torch.from_numpy(Xn[va]); Ova = torch.from_numpy(O[va])
    ysva, ywva = ys[va], yw[va]
    exva = ex[va]
    acva = ac[va]

    torch.manual_seed(a.seed)
    net = QNet(X.shape[1], len(ACTS), a.hidden, a.depth)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    best, best_state, bad = None, None, 0
    t0 = time.time()
    g = torch.Generator().manual_seed(a.seed)
    ntr = int(tr.sum())
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(ntr, generator=g)
        tot = 0.0
        for s in range(0, ntr, a.bs):
            b = perm[s:s + a.bs]
            opt.zero_grad()
            sc, lw = net(Xtr[b], Otr[b])
            loss = (nn.functional.mse_loss(sc, ystr[b])
                    + a.lam * bce(lw, ywtr[b]))
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        net.eval()
        with torch.no_grad():
            vsc, vlw = net(Xva, Ova)
            vsc = vsc.numpy() * 10.0
            vlw = vlw.numpy()
        vmse = float(((vsc - ysva) ** 2).mean())
        improved = best is None or vmse < best - 1e-4
        if improved:
            best, bad = vmse, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print("  ep%-3d 训练 %.4f  验证得分 MSE %7.3f（SD %.2f）  %s(%.0fs)"
                  % (ep, tot / ntr, vmse, ysva.std(), "★" if improved else "  ",
                     time.time() - t0), flush=True)
        if bad >= a.patience:
            print("  早停")
            break

    net.load_state_dict(best_state)
    net.eval()
    # ---- 在**随机化子集**上评：按 Q 选动作 vs 一律碰 / 一律过
    print()
    print("=== ★ 留出集上的策略比较（只在**随机化子集**上，动作不受状态影响）===")
    m = exva == 1
    if m.sum() < 50:
        print("  随机化样本太少（%d）" % m.sum())
    else:
        Xm = torch.from_numpy(Xn[va][m])
        with torch.no_grad():
            qcols = []
            for i in range(len(ACTS)):
                oi = torch.zeros(len(Xm), len(ACTS))
                oi[:, i] = 1.0
                s_i, _ = net(Xm, oi)
                qcols.append(s_i.numpy() * 10.0)
            q = np.stack(qcols, axis=1)
        pick = q.argmax(1)
        # ⚠ 第 124 轮诊断抓到的坑：**必须把 argmax 限制在合法动作集内**。
        #   实测 argmax 有 82.9% 去选"杠"，而杠在实际数据里只占 2.2%
        #   —— 因为 one-hot 让模型对**该状态下根本不合法的杠**也输出 Q，
        #   而"杠"在数据里稀少且被选择性采样（手里有 3 张才出现）⇒ 均值虚高。
        #   运行时的策略**已经**限制了合法集（`mj_policy` 的 `peng==9` 分支），
        #   所以部署没问题；**出错的是这里的离线评估**。
        #   ⇒ 离线评估统一退到 **{碰, 过}** 这个子策略（杠单独看）。
        pick2 = np.where(q[:, 0] >= q[:, 2], 0, 2)
        realised = ysva[m]
        act_real = acva[m]
        beh = realised.mean()
        # ★★ 无偏的策略价值估计：在**随机化子集**里，动作是独立于状态抽的
        #    ⇒ "动作恰好等于该策略的选择" 那个子集的均值，就是**该策略价值的无偏估计**
        #    （选择只依赖随机化的动作，不依赖状态 ⇒ 无混淆）。
        #    唯一能这样评的策略形式：**只依赖 (状态, 合法动作集) 的确定性策略**。
        agree_mask = pick2 == np.array([A2I.get(int(t), 2) for t in act_real])
        print("  %-30s %10s %10s" % ("策略", "n", "无偏价值"))
        print("  %-30s %10d %10.3f" % ("随机均匀（探索基线）", len(realised), beh))
        if agree_mask.sum() >= 30:
            se = realised[agree_mask].std(ddof=1) / np.sqrt(agree_mask.sum())
            print("  %-30s %10d %10.3f  (SE %.3f)"
                  % ("★ 按 Q 取 argmax", agree_mask.sum(),
                     realised[agree_mask].mean(), se))
        # 向听并列子集之外的整体策略价值（所有样本的行为值，供参考）
        print("  （参考）全部样本的行为值 = %+.3f（n=%d，**非随机化**，仅供参考）"
              % (ys[va].mean(), len(ys[va])))
        print("  %-22s %10s %10.3f" % ("一律碰（Q 预测）", "—", q[:, 0].mean()))
        print("  %-22s %10s %10.3f" % ("一律过（Q 预测）", "—", q[:, 2].mean()))
        print("  %-22s %10s %10.3f" % ("★ 按 Q 取 argmax", "—", q.max(1).mean()))
        for t, tag in ((34, "碰"), (38, "过")):
            mm = act_real == t
            if mm.sum() >= 20:
                print("    （随机化子集里实际选%s：n=%d 均值 %+.3f）"
                      % (tag, mm.sum(), realised[mm].mean()))
        agree = agree_mask.mean()
        # ★ 诊断：把 pick 的分布打出来。若"一致率"异常低（例如 8%），
        #   而 Q 的均值又明显偏向某一动作，那就说明**列与动作对不上**
        #   —— 部署一个错位的 Q 会直接把策略改坏，必须先排掉这一种可能。
        import collections
        cnt = collections.Counter(pick.tolist())
        nm = {0: "碰", 1: "杠", 2: "过"}
        print("  argmax 选择分布：%s"
              % "  ".join("%s:%.1f%%" % (nm.get(k, k), 100 * v / len(pick))
                          for k, v in sorted(cnt.items())))
        print("  各动作的 Q 均值：%s"
              % "  ".join("%s:%+.3f" % (nm[i], q[:, i].mean())
                          for i in range(len(ACTS))))
        print("  实际动作分布：%s"
              % "  ".join("%s:%.1f%%" % (nm[A2I.get(int(t), 2)],
                                         100 * (act_real == t).mean())
                          for t in ACTS))
        print("  与行为动作一致率 %.1f%%（偏离越多说明模型越想改）" % (100 * agree))
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                "acts": ACTS, "din": int(X.shape[1]), "hidden": a.hidden,
                "depth": a.depth}, a.out)
    print()
    print("  已保存 -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
