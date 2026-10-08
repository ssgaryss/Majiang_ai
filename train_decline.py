#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**爆头：立即胡 vs 弃胡追倍率** —— 先给因果效应，再训模型（goal §2.3/§2.4/§2.6）。

两步，顺序不能反
----------------
**第 1 步（本脚本的主判据）：无偏 ATE。**
`gen_decline_shard.py` 在 `eps` 概率下**与内层选择无关地**在两支里各 1/2 抽
⇒ `explored==1` 子集里**动作与状态独立** ⇒

    ATE(弃 − 胡) = E[ΔScore | 弃] − E[ΔScore | 胡]        （在 explored 子集上）

这是**无偏**的（不需要建模、不需要 IPS）。它直接回答 goal §2.4 的核心问题：
**"在规则真的在权衡的那些局面上，弃胡追倍率到底值不值？"**
⚠ 自举必须**按局（gid）重抽**：同一局可能有多个决策点、共享同一个 y。

**第 2 步：训一个条件模型** `E[ΔScore | s, a]`（状态 + 动作 → 期望得分），
再看它的 argmax 与规则的一致率。⚠ 判据**不是 AUC/r²**（本仓反复踩过），
而是"接进策略后用**同 seed 配对**测出多少分"。

用法：
    python train_decline.py --data gd0.npz,gd1.npz,gd2.npz --out decline_net.pt
"""
import argparse
import glob
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "_pkgs")

import numpy as np                                                    # noqa: E402

import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402

from features_full import CTX, F as F_FULL                             # noqa: E402


def load(paths):
    """载入分片。★ 第 160 轮补上 `branchB/wall/chain/melds/is_dealer/wild` ——
    紧凑表征（`--features compact`）正是从这六个**记录字段**重建的，
    它们的口径必须与 `gen_decline_shard.py:132-136` 逐字一致。"""
    (Xs, Cs, As, Es, IHs, YS, YW, GS,
     BS, WS, HS, MS, DS, LS) = ([] for _ in range(14))
    for p in paths:
        for fn in sorted(glob.glob(p)):
            z = np.load(fn)
            Xs.append(z["X"])
            Cs.append(z["Xctx"])
            As.append(z["action"].astype(np.int64))
            Es.append(z["explored"].astype(np.int64))
            IHs.append(z["inner_hu"].astype(np.int64))
            YS.append(z["y_score"].astype(np.float32))
            YW.append(z["y_win"].astype(np.float32))
            GS.append(z["gid"].astype(np.int64))
            BS.append(z["branchB"].astype(np.float32))
            WS.append(z["wall"].astype(np.float32))
            HS.append(z["chain"].astype(np.float32))
            MS.append(z["melds"].astype(np.float32))
            DS.append(z["is_dealer"].astype(np.float32))
            LS.append(z["wild"].astype(np.float32))
            print("  载入 %s：%d 行" % (fn, len(z["action"])))
    return (np.concatenate(Xs), np.concatenate(Cs), np.concatenate(As),
            np.concatenate(Es), np.concatenate(IHs), np.concatenate(YS),
            np.concatenate(YW), np.concatenate(GS),
            np.concatenate(BS), np.concatenate(WS), np.concatenate(HS),
            np.concatenate(MS), np.concatenate(DS), np.concatenate(LS))


def build_compact(BS, WS, HS, MS, DS, LS, C):
    """按 `features_full.compact_ctx` 这**唯一一份布局定义**重建训练输入。

    ⚠ 绝不在这里手写一遍索引 —— 那就是第二个副本，而错位的模型**不会报错**，
      只会安静地输出无意义的数（对拍见 `check_decline_compact.py`）。
    """
    from features_full import compact_ctx
    n = len(BS)
    return np.array([compact_ctx(WS[i], HS[i], MS[i], DS[i], LS[i], BS[i], C[i])
                     for i in range(n)], dtype=np.float32)


def boot_by_game(d, gid, n_boot=4000, seed=7):
    """按**局**重抽的自举（同一局的行必须一起进出）。"""
    rng = np.random.RandomState(seed)
    games = np.unique(gid)
    idx_by_g = {g: np.where(gid == g)[0] for g in games}
    out = []
    for _ in range(n_boot):
        pick = games[rng.randint(0, len(games), len(games))]
        idx = np.concatenate([idx_by_g[g] for g in pick])
        out.append(d[idx].mean())
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


class Q(nn.Module):
    """E[y | s, a]：状态 ⊕ 动作 one-hot → 标量（**MSE 口径**）。"""

    def __init__(self, d_in, hidden=256, depth=2, dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TNet(nn.Module):
    """★ `ŷ = b(s) + a · τ(s)` —— **共享基线 + 动作交互头**（第 152 轮新增）。

    为什么需要它（这是 §Z.134(e) 提前标出的那个风险）：
      · MSE 直接回归 `y ~ [s, a]` 时，模型的**首要任务是把状态本身值多少分拟合出来**
        （`σ_状态` 远大于动作效应）⇒ 动作那一维很自然被**收缩到接近 0**。
      · 在纯噪声的合成数据上实测：真值 ATE=+4，MSE 口径训出的模型
        **建议"弃"的比例 = 0.000** —— 等于没学。
      · 实测真实数据 `σ_y ≈ 12.6`，而每机会效应只有 ≈1.5 ⇒ **同一个收缩必然发生**。

    做法：把"状态基线" `b(s)` 与"动作额外效应" `τ(s)` **分成两个头**，
    共享同一个 trunk。于是 `b` 吸走全部状态方差，`τ` 只需解释**动作带来的差**
    ⇒ 不会被压扁。决策就是 `τ(s) > 0 ? 弃 : 胡`（`b` 在比较中自动消掉）。

    ⚠ 这不是"换个损失函数调参"，而是**换目标**：从"预测 y"换成"预测动作的效应"
    —— 与 `mj_policy.py:717` 记的那条教训（**AUC 量跨状态排序、策略要同状态内比较**）
    是同一个道理。
    """

    def __init__(self, d_state, hidden=256, depth=2, dropout=0.3):
        super().__init__()
        layers, d = [], d_state
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head_b = nn.Linear(d, 1)
        self.head_tau = nn.Linear(d, 1)
        # τ 头**小初始化**：起点"动作没有效应"，让它自己长出来
        nn.init.zeros_(self.head_tau.weight)
        nn.init.zeros_(self.head_tau.bias)

    def forward(self, xs, a):
        h = self.trunk(xs)
        return (self.head_b(h).squeeze(-1)
                + a * self.head_tau(h).squeeze(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="gd*.npz")
    ap.add_argument("--out", default="decline_net.pt")
    ap.add_argument("--target", choices=["score", "win"], default="score")
    ap.add_argument("--objective", choices=["mse", "tlearner"], default="tlearner",
                    help="mse = 回归 y~[s,a]（会把动作效应收缩掉）；"
                         "tlearner = b(s)+a·τ(s) 双头（默认，见 TNet 的说明）")
    ap.add_argument("--features", choices=["full", "compact"], default="compact",
                    help="full=360+10（**第一版过拟合到 gap +150** 的那个）；"
                         "compact=18 维充分统计量（默认，见 features_full.compact_ctx）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--cluster", choices=["auto", "game", "room"], default="auto",
                    help="切 train/val 与自举的聚类单元。⚠ room 数据**必须**用 room："
                         "同一房间的多局共享累计分/连庄/对手，按局切会让它们**漏进验证集**"
                         "⇒ val 指标偏乐观（泄漏）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    print("=== 载入 ===")
    (X, C, ACT, EXP, IH, YS, YW, G,
     BRB, WALL, CHN, MLD, ISD, WLD) = load([s for s in a.data.split(",") if s])
    n_all = len(ACT)
    y = (YS if a.target == "score" else YW).astype(np.float64)
    # ★★ 第 164 轮：聚类/切分单元。room 数据里**同一房间的多局共享**
    #   累计分 / 连庄链 / 对手 ⇒ 按局切 train/val 会让它们**漏进验证集**
    #   （val 指标偏乐观），按局自举也会把 CI 算小。判定结果**打印出来**。
    import mj_room
    if a.cluster == "room" or (a.cluster == "auto" and mj_room.is_room_level(G)):
        G = np.array([mj_room.room_of_gid(g) for g in G.tolist()], dtype=np.int64)
        unit = "房间"
        print("  ★ 聚类单元 = **房间**（检测到 room 模式；已用 wall_seed 的逆还原房间号）")
    else:
        unit = "局"
        print("  ★ 聚类单元 = 局（game 模式）")
    print("总决策点 %d（%d %s）  目标=%s" % (n_all, len(np.unique(G)), unit, a.target))
    print("  其中 explored==1（**无偏**）%d  内层选胡比例 %.3f"
          % (int(EXP.sum()), IH.mean()))

    # ---------------- 第 1 步：无偏 ATE ----------------
    e = EXP == 1
    print()
    print("=== ① 无偏 ATE（只在 explored 子集上；动作与状态独立）===")
    if e.sum() < 50:
        print("  ⚠ explored 样本太少（%d），ATE 不可信" % int(e.sum()))
    else:
        a_hu = e & (ACT == 1)
        a_dc = e & (ACT == 0)
        print("  选胡 n=%d  平均 ΔScore %+.3f" % (a_hu.sum(), y[a_hu].mean()))
        print("  弃胡 n=%d  平均 ΔScore %+.3f" % (a_dc.sum(), y[a_dc].mean()))
        d = y[a_dc].mean() - y[a_hu].mean()
        lo, hi = boot_by_game(y[a_dc], G[a_dc], a.boot, a.seed)
        lo2, hi2 = boot_by_game(y[a_hu], G[a_hu], a.boot, a.seed + 1)
        # 差的 CI：用"两支合并后按局重抽"的差分布（更严谨）
        rng = np.random.RandomState(a.seed + 2)
        games = np.unique(G[e])
        idx_by_g = {g: np.where(e & (G == g))[0] for g in games}
        diffs = []
        for _ in range(a.boot):
            pick = games[rng.randint(0, len(games), len(games))]
            idx = np.concatenate([idx_by_g[g] for g in pick])
            ii = ACT[idx]
            yy = y[idx]
            if (ii == 1).sum() < 3 or (ii == 0).sum() < 3:
                continue
            diffs.append(yy[ii == 0].mean() - yy[ii == 1].mean())
        diffs.sort()
        print("  **ATE(弃 − 胡) = %+.3f 分/机会**" % d)
        if diffs:
            lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
            print("  按局自举 95%%CI = [%+.3f, %+.3f]  %s"
                  % (lo, hi, "★ 不含 0" if (lo > 0 or hi < 0) else "含 0 ⇒ 未分辨"))
        print("  ⇒ %s"
              % ("**弃胡在这些局面上是正期望**" if d > 0
                 else "**弃胡在这些局面上是负期望**（规则可能拧错方向）"))
        print("  （对照：`mj_policy.py:1057` 记的 `decline=0 vs 2` 整体效应 +0.358 分/局、"
              "机会率 0.235/局 ⇒ 折算每机会 ≈ +1.52 分）")

    # ---------------- 第 2 步：条件模型 ----------------
    # ★ 第 160 轮：`full` 口径 = 370 维 → 第一版 TNet 约 16 万参数 / 717 训练样本
    #   ⇒ gap +150、退化成"弃 100%"。默认改成 `compact`（18 维）。
    if a.features == "compact":
        Xc = build_compact(BRB, WALL, CHN, MLD, ISD, WLD, C)
        print("  紧凑表征 %d 维（布局定义唯一在 features_full.compact_ctx）" % Xc.shape[1])
    else:
        Xc = X
    Xa = np.concatenate([Xc, C, ACT[:, None].astype(np.float32)], axis=1)
    games = np.unique(G)
    n_val = max(1, int(len(games) * a.val_frac))
    val_g = set(games[-n_val:].tolist())
    is_val = np.array([g in val_g for g in G])
    tr = np.where(e & ~is_val)[0]
    va = np.where(e & is_val)[0]
    print()
    print("=== ② 条件模型（只用无偏子集；**按局切分**；口径 = %s）===" % a.objective)
    print("  训练 %d 行 / %d 局   验证 %d 行 / %d 局"
          % (len(tr), len(games) - n_val, len(va), n_val))
    if len(tr) < 200 or len(va) < 50:
        print("  ⚠ 样本不足以训模型（需要更多局）")
        return 0
    # `tlearner` 的输入**不含**动作列（动作用 a 单独喂）；`mse` 用已含动作列的 `Xa`
    # ★★★ 第 190 轮修掉**第五个**口径错配 —— 这一个让模型**从来没被用过**：
    #   `Xc = build_compact(...)` 已经是 **18 维（8 标量 + ctx 10）**（`COMPACT_DIM` 的注释
    #   与 `features_full.compact_ctx` 都这么写），而这里又 `concat([Xc, C])` 把
    #   **ctx 拼了第二遍** ⇒ 训练是 **28** 维；而推理侧 `mj_policy.py:1083`
    #   只有 `compact_from_view(v, branch)` = **18** 维
    #   ⇒ `X.shape[1] != len(mu)` ⇒ `decline_model_action` **每次都返回 None**。
    #   ⚠ 更糟的是 `mj_policy.py:1362` 那段规则分支**只服务 `decline ∈ (2,3,4,5)`**
    #     ⇒ `decline=9` 在模型不可用时**连规则的弃胡也不会做** ⇒ 它等价于"**永不弃胡**"。
    #   ★ 这就是历史 `decline=9 = −0.2875 ★` 的真正成因（**不是模型能力，也不是符号**）：
    #     实测：在真实决策点上调用 `decline_model_action`，
    #     **99/118 返回 None**，剩下 19 个是"本来就没有可弃的牌"的早退分支。
    #   ⇒ 修法：`compact` 口径**只喂 18 维**（`compact_ctx` 是唯一布局定义，不重复拼 ctx）；
    #     `full` 口径仍是 `360 + ctx 10 = 370`（那个历史 checkpoint 本来就是对的）。
    if a.objective == "tlearner":
        S = Xc if a.features == "compact" else np.concatenate([Xc, C], axis=1)
    else:
        S = Xa
    print("  输入 %d = %s%s   （features=%s）"
          % (S.shape[1],
             "紧凑状态 18（8 标量 + ctx 10，**ctx 不重复拼**）" if a.features == "compact"
             else "完备状态 %d + ctx %d" % (Xc.shape[1], CTX),
             "" if a.objective == "tlearner" else " + 动作 1", a.features))
    mu = S[tr].mean(axis=0)
    sd = S[tr].std(axis=0)
    sd[sd < 1e-6] = 1.0
    Str = torch.from_numpy(((S[tr] - mu) / sd).astype(np.float32))
    Sva = torch.from_numpy(((S[va] - mu) / sd).astype(np.float32))
    atr = torch.from_numpy(ACT[tr].astype(np.float32))
    ava = torch.from_numpy(ACT[va].astype(np.float32))
    ytr = torch.from_numpy(y[tr].astype(np.float32))
    yva = torch.from_numpy(y[va].astype(np.float32))

    if a.objective == "tlearner":
        net = TNet(S.shape[1], a.hidden, a.depth, a.dropout)

        def fwd(x, aa):
            return net(x, aa)
    else:
        net = Q(S.shape[1], a.hidden, a.depth, a.dropout)

        def fwd(x, aa):
            # ⚠ `S` 在 mse 口径下**已经含动作列**（= `Xa`）⇒ 这里不能再拼一次
            #   （拼了就是 372 vs 371 —— 第 152/153 轮同一个坑踩了两次）。
            return net(x)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    lossf = nn.MSELoss() if a.target == "score" else nn.BCEWithLogitsLoss()
    print("  %5s | %10s %10s %9s" % ("epoch", "tr_loss", "va_loss", "gap"))
    best = None
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(Str))
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            opt.zero_grad()
            l = lossf(fwd(Str[b], atr[b]), ytr[b])
            l.backward()
            opt.step()
            tot += float(l.detach()) * len(b)
        net.eval()
        with torch.no_grad():
            lva = float(lossf(fwd(Sva, ava), yva))
        ltr = tot / len(Str)
        if best is None or lva < best["lva"]:
            best = {"lva": lva, "ep": ep, "ltr": ltr,
                    "sd": {k: v.clone() for k, v in net.state_dict().items()}}
        if ep <= 5 or ep % 5 == 0:
            print("  %5d | %10.5f %10.5f %+9.5f" % (ep, ltr, lva, lva - ltr))
    print("  最优 epoch %d：train %.5f  val %.5f  **gap %+.5f**"
          % (best["ep"], best["ltr"], best["lva"], best["lva"] - best["ltr"]))

    # 模型 argmax vs 规则
    net.load_state_dict(best["sd"])
    net.eval()
    zero = torch.zeros(len(va), dtype=torch.float32)
    one = torch.ones(len(va), dtype=torch.float32)
    with torch.no_grad():
        # ★★★ 第 190 轮修正：数据编码是 `A_HU, A_DECLINE = 1, 0`
        #   （`gen_decline_shard.py`）⇒ `a=1` 是**胡**、`a=0` 是**弃**。
        #   TNet 的口径 `ŷ = b(s) + a·τ(s)` ⇒ `net(x,0)` 是**弃**、`net(x,1)` 是**胡**。
        #   原来这里把两者**写反了**，于是打印出的"模型建议弃的比例"是**镜像**的：
        #   在修好数据后它打印 **0.000**，而同一份数据的无偏 ATE 是
        #   **弃 − 胡 = +8.38 ★** —— 自相矛盾。⇒ 详见 `check_decline_sign.py`。
        q_dc = fwd(Sva, zero).numpy()          # a=0 ⇒ 弃
        q_hu = fwd(Sva, one).numpy()           # a=1 ⇒ 胡
    pick_decline = q_dc > q_hu
    rule_decline = IH[va] == 0
    print("  验证集 %d 点：模型建议**弃**的比例 %.3f，规则实际弃的比例 %.3f，"
          "两者一致率 %.3f"
          % (len(va), pick_decline.mean(), rule_decline.mean(),
             (pick_decline == rule_decline).mean()))
    # ★ 线性口径（depth=0）下 τ 的系数**可直接读** —— 这是把"模型说了什么"
    #   摊开并与 CATE 对照的唯一机会（本仓纪律：**凡写代价必须把公式和数摆出来**）。
    if a.objective == "tlearner" and a.depth == 0:
        w = net.head_tau.weight.detach().numpy().reshape(-1)
        names = (["is_dealer", "wild", "melds", "wall/70", "wall>=60",
                  "chain", "branch_b", "wild*is_dealer"]
                 + ["ctx%d" % i for i in range(CTX)]) if a.features == "compact" \
            else ["f%d" % i for i in range(len(w))]
        print()
        print("  --- τ(s) 线性系数（标准化空间；正 = 更该弃）---")
        for nm, wi in sorted(zip(names, w), key=lambda kv: -abs(kv[1]))[:12]:
            print("     %-16s %+8.4f" % (nm, wi))
        print("  ⚠ 只报系数**不**等于可用：判据仍是同 seed 配对比分。")

    torch.save({"sd": net.state_dict(), "mu": mu, "sd_norm": sd,
                "d_in": S.shape[1], "hidden": a.hidden, "depth": a.depth,
                "dropout": a.dropout, "target": a.target,
                "objective": a.objective, "features": a.features,
                "val_loss": best["lva"], "gap": best["lva"] - best["ltr"],
                "decline_rate_val": float(pick_decline.mean())}, a.out)
    print()
    print("  已写 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
