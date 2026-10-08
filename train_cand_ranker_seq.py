#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**顺序版**弃牌候选排序器 —— 用上那条一直没人用过的逐家弃牌序列（`Xseq`）。

为什么要有它（这是与前面三次失败**不同的**一件事）
------------------------------------------------
`gen_cand_shard.py:136` 把**逐家弃牌序列** (`discard_seq(v, 20)`，4×20，有序) 记进了
`gc*.npz` 的 `Xseq`，而 `train_cand_ranker.py` **一处都没引用它**。
⇒ 到目前为止，所有学出来的排序器都只见过**快照**（360 维 `X`），
而快照的预测力**已被独立测过为零**：
  · §Z.120：360 维完备信息、11 万样本 ⇒ 预测力为零，且 gap 翻倍；
  · §Z.123：四个目标、一个受控设计 ⇒ 一致为零。
⇒ **前三次不是"模型不行"，是"喂了已知无效的表示"。** 本轮换表示。

★ 而且 §Z.123 自己写下的下一步就是"**换序列结构**"。

两条与旧版的关键差别（都是有依据的，不是调参）
--------------------------------------------
1. **标签用 `y_score`（本局 ΔScore），不用 `y_win`。**
   旧版默认 `y_win`，理由是"番数在绝大多数胡牌局里都是 ×1"。
   但 **§Z.141 已经实测推翻了这个近似**：上线的 `decline=5` 让
   **自摸率 25.0%→24.8%（降）**、**均番 1.297→1.361（升）**、净 **+0.298 分/局**
   ⇒ 胜率与得分**确实不同向**。goal §7 要的是得分。
2. **把"候选牌 × 对手弃牌"的交互**直接喂进去，而不是指望网络自己学出来：
   除序列嵌入外，另给 8 个显式匹配特征
   —— 每个座位在**最近 5 张 / 全部 20 张**里打出过这张候选牌几次。

判据（**照抄 `train_cand_ranker.py` 里已经写死的那两条，不新增**）
--------------------------------------------------------------
* 分歧率（模型 argmax vs ukeire argmax）**< 5%** ⇒ 它只是 ukeire 的昂贵复制品，
  **没有可测的东西**，不必进因果评测；
* 否则 ⇒ 进**同 seed 配对因果评测**（接成 `discard=rank`），由分说话。
* ⚠ **绝不拿 AUC 当判据**（`mj_policy.py:717` 的教训：AUC 0.75 但策略 −8.08 分/局）。
  AUC 只作**过拟合温度计**。

用法：
    python train_cand_ranker_seq.py --data gc0.npz,gc1.npz,gc2.npz --out cand_rank_seq.pt
"""
import argparse
import glob
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "_pkgs")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402

from features_full import F as F_FULL                                 # noqa: E402

K_DEFAULT = 10
CAND_DIM = 6
NTILE = 34
SEQK = 20


def load(paths):
    Xs, Qs, Cs, Ps, Ns, Es, Ys, Yw, Gs = ([] for _ in range(9))
    for p in paths:
        for fn in sorted(glob.glob(p)):
            z = np.load(fn)
            Xs.append(z["X"])
            Qs.append(z["Xseq"])
            Cs.append(z["cands"])
            Ps.append(z["played"].astype(np.int64))
            Ns.append(z["ncand"].astype(np.int64))
            Es.append(z["explored"].astype(np.int64))
            Ys.append(z["y_score"].astype(np.float32))
            Yw.append(z["y_win"].astype(np.float32))
            Gs.append(z["gid"].astype(np.int64))
            print("  载入 %s：%d 行" % (fn, len(z["y_score"])))
    return (np.concatenate(Xs), np.concatenate(Qs), np.concatenate(Cs),
            np.concatenate(Ps), np.concatenate(Ns), np.concatenate(Es),
            np.concatenate(Ys), np.concatenate(Yw), np.concatenate(Gs))


class SeqRanker(nn.Module):
    """`score(state, history, candidate) -> 实数`。

    ★ 交互是**显式**给的（`match`），不指望网络从序列里自己学出来：
      `match[s, w]` = 座位 s 在最近 w 张里打出过候选牌几次（w ∈ {5, 20}）。
      这正是"这张牌安不安全 / 对手要不要"最直接的信号（goal §2.7）。
    """

    def __init__(self, d_state=F_FULL, hidden=256, seq_h=64, emb=8, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(NTILE + 1, emb)          # +1：把 -1 补位映射到 0
        self.seq_conv = nn.Sequential(
            nn.Conv1d(emb, seq_h, 5, padding=2), nn.ReLU(),
            nn.Conv1d(seq_h, seq_h, 5, padding=2), nn.ReLU(),
        )
        self.state = nn.Sequential(
            nn.Linear(d_state, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.cand_emb = nn.Embedding(NTILE, emb)
        self.head = nn.Sequential(
            nn.Linear(hidden + 2 * seq_h + emb + (CAND_DIM - 1) + 8, hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, q, cand):
        """x:(B,360) q:(B,4,20) cand:(B,6)"""
        B = x.shape[0]
        tok = (q.reshape(B, -1).long() + 1).clamp(0, NTILE)          # -1 -> 0
        e = self.emb(tok).transpose(1, 2)                            # (B,emb,80)
        s = self.seq_conv(e)                                         # (B,seq_h,80)
        s = torch.cat([s.mean(dim=2), s.amax(dim=2)], dim=1)         # (B,2*seq_h)
        tile = cand[:, CAND_DIM - 1].long().clamp(0, NTILE - 1)
        ce = self.cand_emb(tile)                                     # (B,emb)
        cont = cand[:, :CAND_DIM - 1]                                # (B,5)
        # ★ 显式交互：候选牌在座位 s 的最近 5 / 20 张里出现几次
        m5 = (q[:, :, -5:] == tile[:, None, None]).sum(dim=2).float()
        m20 = (q == tile[:, None, None]).sum(dim=2).float()
        match = torch.cat([m5, m20], dim=1)                          # (B,8)
        h = self.state(x)
        return self.head(torch.cat([h, s, ce, cont, match], dim=1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="gc*.npz")
    ap.add_argument("--out", default="cand_rank_seq.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seq-h", type=int, default=64)
    ap.add_argument("--emb", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--target", choices=["score", "win"], default="score")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    print("=== 载入（★ 这次会用到 Xseq）===")
    X, Q, C, P, N, E, YS, YW, G = load([s for s in a.data.split(",") if s])
    print("总行 %d（唯一局 %d）  explored=%.3f" % (len(X), len(set(G.tolist())), E.mean()))
    print("  序列形状 %s（-1=补位）" % (Q.shape,))

    # 训练只用无偏子集（`explored==1` 里动作随机）—— 与旧版同一口径
    e = np.where(E == 1)[0]
    games = np.unique(G[e])
    n_val = max(1, int(len(games) * a.val_frac))
    val_g = set(games[-n_val:].tolist())
    tr = np.array([i for i in e if G[i] not in val_g], dtype=np.int64)
    va = np.array([i for i in e if G[i] in val_g], dtype=np.int64)
    print("  训练 %d 行 / %d 局   验证 %d 行 / %d 局"
          % (len(tr), len(games) - n_val, len(va), n_val))

    def pick(idx):
        k = N[idx]
        j = np.clip(P[idx], 0, k - 1)
        cand = C[idx].reshape(-1, K_DEFAULT, CAND_DIM)[np.arange(len(idx)), j]
        return (torch.from_numpy(X[idx]), torch.from_numpy(Q[idx]),
                torch.from_numpy(cand))

    y = (YS if a.target == "score" else YW)
    Xt, Qt, Ct = pick(tr)
    yt = torch.from_numpy(y[tr])
    Xv, Qv, Cv = pick(va)
    yv = torch.from_numpy(y[va])
    # 标准化标签（score 的 σ≈11.5，标准化让 lr 与 loss 可比）
    ym, ys_ = float(yt.mean()), float(yt.std())
    yt_n = (yt - ym) / ys_
    yv_n = (yv - ym) / ys_

    net = SeqRanker(F_FULL, a.hidden, a.seq_h, a.emb, a.dropout)
    print("  参数量 %d" % sum(p.numel() for p in net.parameters()))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    lossf = nn.MSELoss()
    print("  %5s | %10s %10s %9s" % ("epoch", "tr_loss", "va_loss", "gap"))
    best = None
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            opt.zero_grad()
            l = lossf(net(Xt[b], Qt[b], Ct[b]), yt_n[b])
            l.backward()
            opt.step()
            tot += float(l.detach()) * len(b)
        net.eval()
        with torch.no_grad():
            lva = float(lossf(net(Xv, Qv, Cv), yv_n))
        ltr = tot / len(Xt)
        if best is None or lva < best["lva"]:
            best = {"lva": lva, "ep": ep, "ltr": ltr,
                    "sd": {k: v.clone() for k, v in net.state_dict().items()}}
        if ep <= 3 or ep % 5 == 0:
            print("  %5d | %10.5f %10.5f %+9.5f" % (ep, ltr, lva, lva - ltr))
    print("  最优 epoch %d：train %.5f  val %.5f  **gap %+.5f**"
          % (best["ep"], best["ltr"], best["lva"], best["lva"] - best["ltr"]))

    # ---------------- ★ 判据一：分歧率（不是 AUC）----------------
    net.load_state_dict(best["sd"])
    net.eval()
    with torch.no_grad():
        nv = len(va)
        pred = np.full((nv, K_DEFAULT), -1e9, dtype=np.float64)
        for j in range(K_DEFAULT):
            cand = C[va].reshape(-1, K_DEFAULT, CAND_DIM)[:, j]
            ok = j < N[va]
            if not ok.any():
                continue
            sc = net(torch.from_numpy(X[va]), torch.from_numpy(Q[va]),
                     torch.from_numpy(cand)).numpy()
            pred[ok, j] = sc[ok]
        am = pred.argmax(axis=1)
    uk = C[va].reshape(-1, K_DEFAULT, CAND_DIM)[:, :, 1].copy()
    # ⚠ 掩码必须是 (nv, K_DEFAULT)：第一版写 `np.arange(nv)[:, None] >= N[va][:, None]`
    #   ⇒ 形状是 (nv,1)，对 (nv,10) 的数组做布尔索引 ⇒ IndexError。
    #   （这类广播错在**只有跑到那一步**才现形 —— 而它正是在训练完之后。）
    uk[np.arange(K_DEFAULT)[None, :] >= N[va][:, None]] = -1e9
    uk_am = uk.argmax(axis=1)
    print()
    print("=== ★ 判据一：分歧率（`train_cand_ranker.py` 里已写死）===")
    print("  模型 argmax vs **ukeire** argmax 分歧 = **%.2f%%**"
          % (100.0 * float((am != uk_am).mean())))
    # 与规则真实选择比（`explored==0` 的行里 `played` 就是规则那张）
    allidx = np.where(E == 0)[0]
    if len(allidx):
        sub = allidx[:20000]
        with torch.no_grad():
            pr2 = np.full((len(sub), K_DEFAULT), -1e9)
            for j in range(K_DEFAULT):
                cand = C[sub].reshape(-1, K_DEFAULT, CAND_DIM)[:, j]
                ok = j < N[sub]
                if not ok.any():
                    continue
                sc = net(torch.from_numpy(X[sub]), torch.from_numpy(Q[sub]),
                         torch.from_numpy(cand)).numpy()
                pr2[ok, j] = sc[ok]
            am2 = pr2.argmax(axis=1)
        print("  模型 argmax vs **规则实际**选择分歧 = **%.2f%%**（%d 行，explored==0）"
              % (100.0 * float((am2 != P[sub]).mean()), len(sub)))
    print()
    print("  ⇒ 若分歧 <5%% ⇒ 它只是 ukeire 的复制品，**不必**进因果评测；")
    print("     否则 ⇒ 下一步是同 seed 配对因果评测（接 `discard=rank`）。")
    print("  ⚠ **绝不拿 AUC 当判据**（`mj_policy.py:717`：AUC 0.75 但策略 −8.08 分/局）。")

    torch.save({"sd": net.state_dict(), "d_state": F_FULL, "hidden": a.hidden,
                "seq_h": a.seq_h, "emb": a.emb, "dropout": a.dropout,
                "target": a.target, "features": "candseq",
                "y_mean": ym, "y_std": ys_,
                "val_loss": best["lva"], "gap": best["lva"] - best["ltr"]}, a.out)
    print("  已写 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
