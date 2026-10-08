#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练「弃牌候选排序器」—— 输入**完备逐家编码**，输出每个候选的胜率。

这一格为什么从没被填过（第 145 轮）
------------------------------------
`mj_policy.py:717` 记着那次失败：84 维网络在**全部**合法弃牌里挑 ⇒ 配对对拍
**−8.08 分/局**；而同一网络跨状态 AUC 0.75。根因写得很清楚：

    **"AUC 衡量跨状态排序，策略需要同一状态内对候选动作排序"**。

所有后续尝试都只用 84 维（四家弃牌被汇总成一个计数，**看不到对手出了什么**）。
本脚本用的是 `gen_cand_shard.py` 新产的数据：**360 维完备状态 × 每个并列候选**，
且动作是**干净随机化**的（`explored==1` 里动作与状态独立 ⇒ 标签无偏）。

训练口径（跑之前写死）
----------------------
* **只用 `explored==1` 的行** —— 这是数据里唯一无偏的部分（其余是内层策略选的，
  拿它训练等于**模仿**内层，永远不可能超过内层）。
* **按局切分**验证集（尾 15% 的 `gid`），**绝不按行随机切** ——
  同一局的多个决策点高度相关，按行切会把训练集信息漏进验证集。
* 标签默认用 **`y_win`**（P(胡)）：`y_score` 的 σ 是 12.5，信噪比差一个量级；
  而本仓已有结论是"番数在绝大多数胡牌局里都是 ×1"⇒ 期望得分 ≈ P(胡)×常数。
* **过拟合是被测量的量**：报 `train/val` 的损失与 AUC 之差（gap）。
  这是用户那条硬要求（"避免过拟合"）在训练侧的读法。

输出的判读（也写死在代码里）
----------------------------
⚠⚠ **绝不拿 AUC 当这条路的判据。** 第一版我正是那么写的（"val AUC < 0.53 ⇒ 到此为止"），
那是**把本仓已经踩过的那次错误又犯一遍**：`mj_policy.py:717` 的教训就是
**"AUC 衡量跨状态排序，策略需要同一状态内对候选动作排序"** ——
一个跨状态 AUC 0.75 的网络在策略上可以是 −8.08 分/局。
本脚本里 AUC 只是**过拟合的温度计**（train vs val），不是决策质量的判据。

所以判据只有两条，**都不是 AUC**：
* **分歧率**（模型 argmax  vs  ukeire argmax，在验证集上）：< 5% ⇒ 它只是 ukeire
  的昂贵复制品，**没有可测的东西**，不必进因果评测；
* 否则 ⇒ 进**同 seed 配对因果评测**（把它接成 `discard=rank`），由分说话。

「避免过拟合」这条硬要求在这里读作 **train/val 的损失 gap**：报出来、可复现、可调
（`--dropout` / `--wd` / `--hidden`）。

用法：
    python train_cand_ranker.py --data gc0.npz,gc1.npz,gc2.npz --out cand_rank.pt
"""
import argparse
import glob
import sys

sys.path.insert(0, ".")
# ★★★ 第 268 轮：CUDA 版 torch 解在 `_pkgs_gpu/`（依赖仍在 `_pkgs/`）。
#   只有显式 `MJ_USE_GPU=1` 才把它插到前面 ⇒ **不设时行为与改前逐字一致**。
#   （为什么不用 pip：本沙箱里 `tempfile.mkdtemp` 建的目录不可写 ⇒ pip 解包必失败，
#     实测根因见 `_dl_torch_cuda.py` 的注释；所以走"直接下 wheel + 手工解包"。）
import os as _os_gpu
sys.path.insert(0, "_pkgs")            # torch 在这里（pip 被沙箱拦，见 fetch_wheels.py）
# ⚠⚠ 顺序很关键：这一句**必须**在上面的 `_pkgs` 之后 —— 两者都是 `insert(0, …)`，
#    谁后插谁在前面。第 268 轮第一版写反了 ⇒ 实测打印出「设备 = cpu」，
#    即 GPU 那次其实还在用 CPU 版 torch（1.08× 的"加速"纯属噪声）。
if _os_gpu.environ.get("MJ_USE_GPU") == "1" and _os_gpu.path.isdir("_pkgs_gpu"):
    sys.path.insert(0, "_pkgs_gpu")

import numpy as np                                                    # noqa: E402

import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402

from features_full import F as F_FULL                                 # noqa: E402

K_DEFAULT = 10
CAND_DIM = 6
# `cand_scalars` 的 6 个标量顺序（`features_full.py:94-97`）：
# [sh/9, uk/400, bt, is_wild, is_drawn, **tile**] ⇒ 牌码是第 6 个（下标 5）。
C_TILE = 5
EXTRA_DIM_RULEVARS = 2          # 自然向听 + unseen 修正 ukeire（第 196 轮）


def load(paths):
    Xs, Cs, Ps, Ns, Es, Yw, Ys, Gs = [], [], [], [], [], [], [], []
    for p in paths:
        for fn in sorted(glob.glob(p)):
            z = np.load(fn)
            Xs.append(z["X"])
            Cs.append(z["cands"])
            Ps.append(z["played"].astype(np.int64))
            Ns.append(z["ncand"].astype(np.int64))
            Es.append(z["explored"].astype(np.int64))
            Yw.append(z["y_win"].astype(np.float32))
            Ys.append(z["y_score"].astype(np.float32))
            Gs.append(z["gid"].astype(np.int64))
            print("  载入 %s：%d 行" % (fn, len(z["played"])))
    return (np.concatenate(Xs), np.concatenate(Cs), np.concatenate(Ps),
            np.concatenate(Ns), np.concatenate(Es), np.concatenate(Yw),
            np.concatenate(Ys), np.concatenate(Gs))


def pick_cand_features(cands, played, n, idx=None):
    """取出第 idx 行「候选 played[idx]」的 6 个标量。"""
    K = cands.shape[1] // CAND_DIM
    C = cands.reshape(-1, K, CAND_DIM)
    if idx is None:
        return C[np.arange(n), played, :]
    return C[idx, played[idx], :]


def pick_tile(C, played, idx=None):
    """取出第 idx 行「候选 played[idx]」的**牌码**（`cand_scalars` 的第 6 个标量）。

    ★ 第 196 轮新增：`--extra-feats rulevars` 要拿候选牌码才能算派生特征。
    """
    K = C.shape[1] // CAND_DIM
    Cv = C.reshape(-1, K, CAND_DIM)
    if idx is None:
        return Cv[np.arange(len(played)), played, C_TILE]
    return Cv[idx, played[idx], C_TILE]


def extra_block(X, C, P, idx, mode):
    """把规则的裁决变量算成候选特征（第 196 轮）。`mode != "rulevars"` 时返回 None。

    ⚠ **与推理侧必须共用 `cand_rulevars` 的同一个函数**；
      两端的"取得输入"的途径不同（这边从 `X` 反解），所以 `check_rulevars_identity.py`
      的对拍**不是可选项** —— 本仓这类"两份实现分叉"已经栽过（见 `features_full.py:44-52`）。
    ⚠ 逐行 python 循环，122k 行约几分钟（`ukeire` 内部要枚举 34 张牌）。
    """
    if mode != "rulevars":
        return None
    from cand_rulevars import from_x
    t = pick_tile(C, P, idx)
    out = np.empty((len(idx), 2), dtype=np.float32)
    for j in range(len(idx)):
        e = from_x(X[idx[j]], int(t[j]))
        out[j, 0] = e[0]
        out[j, 1] = e[1]
    return out


def extra_block_cand(X, C, idx, mode, K):
    """**逐候选**派生块，行序为 `[i0]*K, [i1]*K, …`（与 `np.repeat(X[idx],K)` 对齐）。

    ★ 第 263 轮新增。为什么不能用 `extra_block`：那一个取的是
    `pick_tile(C,P,idx)` = **实际打出那张牌**的派生量，对"给全部候选打分"的场景，
    它在**候选之间是常数** —— 那是个近似，而 `PREREG_RANK_RULEVARS.md` §4.1 的**主判据**
    （与规则的分歧率）正是靠这一段量的，且基线 0.726 来自 `--extra-feats none`
    （那一档没有派生块 ⇒ 基线**精确**）⇒ 若这边用近似，**两边不可比**。
    代价：`nv×K` 次 `from_x`（每次要枚举 34 张牌），比 `extra_block` 贵 K 倍。
    """
    if mode != "rulevars":
        return None
    from cand_rulevars import from_x
    Cv = C.reshape(-1, K, CAND_DIM)
    n = len(idx)
    out = np.empty((n * K, 2), dtype=np.float32)
    for j in range(n):
        xi = X[idx[j]]
        for k in range(K):
            e = from_x(xi, int(Cv[idx[j], k, C_TILE]))
            out[j * K + k, 0] = e[0]
            out[j * K + k, 1] = e[1]
    return out


class Ranker(nn.Module):
    def __init__(self, d_in, hidden=256, depth=2, dropout=0.3):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def auc(y, s):
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列：用平均秩
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def corr(y, s):
    """Pearson 相关 —— **回归目标**下用它代替 AUC。

    ⚠ 第 147 轮踩到：`--target score` 时我仍然打印 AUC，而 `auc()` 只做二分类
    （`n1 = (y==1).sum()`），连续标签下 `n1==0` ⇒ 直接返回 **nan**，
    于是输出里出现"train nan → val nan"，**判读段却还在照着 nan 说话**。
    ⇒ 回归目标必须换指标；这与 `train_value_torch.py` 报 score corr 是同一口径。
    """
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if len(y) < 3 or y.std() < 1e-12 or s.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(y, s)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="gc*.npz")
    ap.add_argument("--out", default="cand_rank.pt")
    ap.add_argument("--target", choices=["win", "score"], default="win")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = 全部")
    # ★ 第 196 轮：**把规则自己的裁决变量喂进来**（`PREREG_RANK_RULEVARS.md`）
    #   默认 `none` ⇒ **默认路径与改动前逐字节等价**（本仓纪律：不许顺手改口径）。
    ap.add_argument("--extra-feats", choices=["none", "rulevars"], default="none",
                    help="rulevars = 追加 2 个**派生**候选特征：自然向听 + unseen 修正 ukeire"
                         "（规则并列链的第③⑤步，`cand_scalars` 里没有它们）。"
                         "⚠ 派生而不入库 ⇒ 不需要重采数据；⚠ 跑之前必须先过 "
                         "`check_rulevars_identity.py` 的两条路径对拍。")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    print("=== 载入数据 ===")
    paths = [s for s in a.data.split(",") if s]
    X, C, P, N, E, YW, YS, G = load(paths)
    n_all = len(X)
    print("总决策点 %d（唯一局数 %d）" % (n_all, len(np.unique(G))))

    # ★ 只用无偏的那一半
    keep = np.where(E == 1)[0]
    print("其中 explored==1（无偏）%d（%.1f%%）" % (len(keep), 100 * len(keep) / n_all))
    if a.max_rows and len(keep) > a.max_rows:
        keep = keep[:a.max_rows]
        print("  截到 %d 行（--max-rows）" % len(keep))

    # ★ 按局切分（不是按行）
    games = np.unique(G[keep])
    n_val = max(1, int(len(games) * a.val_frac))
    val_games = set(games[-n_val:].tolist())
    is_val = np.array([g in val_games for g in G[keep]])
    tr = keep[~is_val]
    va = keep[is_val]
    print("按局切分：训练 %d 行 / %d 局   验证 %d 行 / %d 局"
          % (len(tr), len(games) - n_val, len(va), n_val))

    def build(idx):
        cf = pick_cand_features(C, P, n_all, idx)
        parts = [X[idx], cf]
        ex = extra_block(X, C, P, idx, a.extra_feats)     # ★ 第 196 轮，默认 None
        if ex is not None:
            parts.append(ex)
        parts.append(N[idx][:, None].astype(np.float32))
        xx = np.concatenate(parts, axis=1)
        yy = (YW[idx] if a.target == "win" else YS[idx]).astype(np.float32)
        return xx.astype(np.float32), yy

    Xtr, ytr = build(tr)
    Xva, yva = build(va)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    d_in = Xtr.shape[1]
    _ex = EXTRA_DIM_RULEVARS if a.extra_feats == "rulevars" else 0
    print("输入维度 %d = 状态 %d + 候选 %d + 派生 %d + ncand 1（extra-feats=%s）"
          % (d_in, F_FULL, CAND_DIM, _ex, a.extra_feats))
    if a.extra_feats == "rulevars":
        print("  ⚠ 派生特征 = 规则并列链的第③⑤步（自然向听 / unseen 修正 ukeire）")
        print("  ⚠ 推理侧**必须**同样追加这两个（`mj_policy.best_discard_rank`），")
        print("     否则 `X.shape[1] != len(mu)` ⇒ 会**报错**（不是静默退化）—— 这是故意的。")

    # ★ 第 268 轮：设备选择（默认仍是 cpu；`MJ_USE_GPU=1` 且 cuda 可用才走 GPU）
    dev = "cuda" if (_os_gpu.environ.get("MJ_USE_GPU") == "1"
                     and torch.cuda.is_available()) else "cpu"
    print("  设备 = %s%s" % (dev, ("（%s）" % torch.cuda.get_device_name(0))
                             if dev == "cuda" else ""))
    net = Ranker(d_in, a.hidden, a.depth, a.dropout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    lossf = nn.BCEWithLogitsLoss() if a.target == "win" else nn.MSELoss()

    # ★ 第 268 轮：`ttr/ytr_t/tva` 必须与网络**同设备**（第 268 轮首版漏了这步，
    #   GPU 路径当场崩：`mat1 is on cpu, different from other tensors on cuda:0`）。
    #   `yva_t` **故意留在 CPU** —— 它只用于下面的 numpy-loss（`ova` 是 CPU 张量）。
    ttr = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    tva = torch.from_numpy(Xva).to(dev)
    yva_t = torch.from_numpy(yva)                 # ← 保持 CPU

    print()
    print("=== 训练（target=%s，dropout=%.2f，wd=%.1e，hidden=%d×%d）==="
          % (a.target, a.dropout, a.wd, a.hidden, a.depth))
    print("  %5s | %9s %9s %8s | %9s %9s %8s | %8s"
          % ("epoch", "tr_loss", "tr_auc", "tr_gap", "va_loss", "va_auc", "va_gap",
             "Δauc"))
    best = None
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(ttr))
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            opt.zero_grad()
            out = net(ttr[b])
            l = lossf(out, ytr_t[b])
            l.backward()
            opt.step()
            tot += float(l.detach()) * len(b)
        net.eval()
        with torch.no_grad():
            otr = net(ttr).detach().cpu().numpy()
            ova = net(tva).detach().cpu().numpy()
        ltr = tot / len(ttr)
        lva = float(lossf(torch.from_numpy(ova), yva_t))
        # ★ 分类目标报 AUC，回归目标报 Pearson r（见 `corr()` 的说明）
        if a.target == "win":
            atr, ava, mname = auc(ytr, otr), auc(yva, ova), "AUC"
        else:
            atr, ava, mname = corr(ytr, otr), corr(yva, ova), "r"
        gap = lva - ltr
        print("  %5d | %9.5f %9.4f %8s | %9.5f %9.4f %8.4f | %+8.4f"
              % (ep, ltr, atr, "-", lva, ava, gap,
                 (ava - atr) if (ava == ava and atr == atr) else float("nan")))
        if best is None or lva < best["lva"]:
            best = {"lva": lva, "ep": ep,
                    "sd": {k: v.clone() for k, v in net.state_dict().items()},
                    "ava": ava, "atr": atr, "gap": gap}

    print()
    print("=== 结果（指标 = %s）===" % mname)
    print("  最优 epoch %d：train %s %.4f  val %s %.4f  **Δ%s = %+.4f**"
          % (best["ep"], mname, best["atr"], mname, best["ava"], mname,
             best["ava"] - best["atr"]))
    print("  **过拟合 gap（val_loss − train_loss）= %+.5f**" % best["gap"])

    # ---- 决策侧的量：与 ukeire 的分歧率 ----
    net.load_state_dict(best["sd"])
    net.eval()
    with torch.no_grad():
        # 对验证集的每个决策点，给**全部**候选打分，取 argmax
        K = C.shape[1] // CAND_DIM
        Cv = C.reshape(-1, K, CAND_DIM)
        idx = va
        nv = len(idx)
        # ★★★ 第 260 轮修：这里**必须**和 `build()` 逐列对齐 ——
        #   派生特征块（`--extra-feats rulevars`）此前只加在训练侧，
        #   决策侧漏了 ⇒ `cand_all` 宽度 = 全宽 − 2 ⇒
        #   `((cand_all - mu) / sd)` 直接 ValueError（367 vs 369）。
        #   默认 `--extra-feats none` 时派生块为空、两端刚好相等 ⇒ 这个 bug **一直是潜伏的**。
        # ★★★ 第 263 轮再修一层：派生块要**逐候选**算（`extra_block_cand`），
        #   不能用 `extra_block`（那是"打出那张牌"的派生量，候选间不变）——
        #   这一段量的就是**主判据**（与规则的分歧率），而基线 0.726 是 `none` 档、**精确**的，
        #   必须同口径才可比。
        _parts = [np.repeat(X[idx], K, axis=0),
                  Cv[idx].reshape(-1, CAND_DIM)]
        _ex = extra_block_cand(X, C, idx, a.extra_feats, K)
        if _ex is not None:
            _parts.append(_ex)
        _parts.append(np.repeat(N[idx][:, None].astype(np.float32), K, axis=0))
        cand_all = np.concatenate(_parts, axis=1)
        assert cand_all.shape[1] == len(mu), (
            "决策侧列数 %d != mu 宽度 %d ⇒ 又分叉了" % (cand_all.shape[1], len(mu)))
        cand_all = ((cand_all - mu) / sd).astype(np.float32)
        sc = net(torch.from_numpy(cand_all).to(dev)).detach().cpu().numpy().reshape(nv, K)
        mask = np.arange(K)[None, :] < N[idx][:, None]
        sc = np.where(mask, sc, -1e9)
        am = sc.argmax(axis=1)
        # ukeire = 第 2 个标量（见 explore_discard_shard.cand_scalars）
        uk = Cv[idx][:, :, 1]
        uk = np.where(mask, uk, -1e9)
        uk_am = uk.argmax(axis=1)
        dis = float((am != uk_am).mean())
        print("  验证集决策点 %d：模型 argmax 与 **ukeire argmax** 分歧率 %.3f"
              % (nv, dis))
        # 随机化那一半里，模型选的与"实际打出的"对比（仅作参考）
        print("  模型 vs 内层实际选择 分歧率 %.3f"
              % float((am != P[idx]).mean()))
        print("  候选数分布（验证集）: %s"
              % {int(k): int(v) for k, v in
                 zip(*np.unique(N[idx], return_counts=True))})

    print()
    print("=== 判读（写死在代码里；**判据不是 AUC**）===")
    print("  ⚠ %s 只当**过拟合温度计**：train %.4f → val %.4f（Δ%+.4f）。"
          % (mname, best["atr"], ava, ava - best["atr"]))
    print("    在策略上它说明不了任何事 —— `mj_policy.py:717` 那次就是"
          "跨状态 AUC 0.75 而配对 −8.08 分/局。")
    if best["gap"] > 0.05:
        print("  ⚠ **过拟合偏重**：val_loss − train_loss = %+.5f ⇒ "
              "加正则/扩样本再看。" % best["gap"])
    else:
        print("  ✅ 过拟合 gap = %+.5f（可接受区间）" % best["gap"])
    if dis < 0.05:
        print("  ⛔ 与 ukeire 分歧率只有 %.3f < 0.05 ⇒ 它只是 ukeire 的昂贵复制品"
              % dis)
        print("     ⇒ 没有可测的东西；**不必进因果评测**。")
    else:
        print("  ⇒ ⚠ 分歧率 %.3f ≥ 0.05 ⇒ **值得进下一步**：" % dis)
        print("     把它接成 `discard=rank`，用**同 seed 配对**做因果评测，由分说话。")

    torch.save({"sd": net.state_dict(), "mu": mu, "sd_norm": sd, "d_in": d_in,
                "hidden": a.hidden, "depth": a.depth, "dropout": a.dropout,
                "target": a.target, "K": K, "val_auc": ava,
                "train_auc": best["atr"], "gap": best["gap"],
                # ★ 第 196 轮：把派生特征的口径写进 checkpoint。
                #   推理侧**必须**读它并做同样的追加；读不到就应当**报错**，
                #   绝不允许静默退回规则（那会让配对评测量到"规则 vs 规则"=0）。
                "features": ("rulevars" if a.extra_feats == "rulevars" else "flat"),
                "extra_feats": a.extra_feats,
                "disagree_ukeire": dis}, a.out)
    print()
    print("  已写 %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
