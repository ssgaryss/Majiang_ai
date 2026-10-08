#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL 策略网络（候选打分式）+ 可注入 `mj_sim` 的 Policy 对象。

网络形状（**候选打分**，不是"固定 34 维 softmax"）
--------------------------------------------------
本仓的教训（`mj_policy.py:717`）：跨状态排序的指标（AUC 0.75）在策略上可以是
−8.08 分/局，而策略要的正是"**同一状态内对候选动作排序**"。所以这里的结构
天然就是"同状态内排序"：

    state(370) --trunk(MLP)--> h(128)
    score_i = head([h ⊕ cand_i])          ← 逐候选打分，再在本决策点的候选上 softmax
    V(s)    = value(h)                    ← 基线（PPO 的 critic）

回合决策（`on_turn`）与窗口决策（`on_window`）用**两个不同的 head**，
共享同一个 trunk。候选维度不同（13 / 8，见 `rl_action_space`），所以分开更干净。

为什么输出是"候选上的 softmax"而不是"34 张牌上的 softmax"
--------------------------------------------------------
非法动作**不进入候选集**（`rl_action_space` 里逐条对齐了引擎判定），
所以不存在"给非法动作分概率再 mask"的中间态 —— 概率质量天然只落在合法动作上。
例外（只有一处，已写进文档）：`Game._poll_window` 按座位序询问、先答应者得牌，
所以"我们答了碰但被靠前的座位抢走"是可能的；那是**环境**的性质，不是掩码问题。

★ 设备纪律（`train_cand_ranker.py` 的两个坑）
  · 所有进网络的张量都要 `.to(dev)`（训练侧；worker 里推理固定用 CPU 以便并行）；
  · 输出必须 `.detach().cpu()` 之后才能 `.numpy()`。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rl_boot                                            # noqa: E402  必须在 torch 之前

TORCH = rl_boot.bootstrap()
import torch                                              # noqa: E402
import torch.nn as nn                                     # noqa: E402
import numpy as np                                        # noqa: E402

import rl_action_space as A                               # noqa: E402

DEFAULT_CFG = {
    "state_dim": A.STATE_DIM,
    "cand_turn_dim": A.CAND_TURN_DIM,
    "cand_win_dim": A.CAND_WIN_DIM,
    "hidden": 128,
    "temp": 1.0,
}


# ---------------------------------------------------------------- 网络
class CandNet(nn.Module):
    def __init__(self, state_dim=A.STATE_DIM, cand_turn_dim=A.CAND_TURN_DIM,
                 cand_win_dim=A.CAND_WIN_DIM, hidden=128):
        super().__init__()
        self.state_dim = state_dim
        self.cand_turn_dim = cand_turn_dim
        self.cand_win_dim = cand_win_dim
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_turn = nn.Sequential(
            nn.Linear(hidden + cand_turn_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.head_win = nn.Sequential(
            nn.Linear(hidden + cand_win_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value = nn.Linear(hidden, 1)

    def _logits(self, head, s, cands):
        h = self.trunk(s)                                   # (B,H)
        B, K, _C = cands.shape
        z = torch.cat([h.unsqueeze(1).expand(B, K, h.shape[1]), cands], dim=2)
        return head(z).squeeze(-1)                          # (B,K)

    def forward(self, kind, s, cands, mask=None):
        """kind ∈ {"turn","win"}；返回 (logits(B,K), value(B,))。"""
        head = self.head_turn if kind == "turn" else self.head_win
        logits = self._logits(head, s, cands)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        v = self.value(self.trunk(s)).squeeze(-1)
        return logits, v


def make_net(cfg, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return CandNet(cfg.get("state_dim", A.STATE_DIM),
                   cfg.get("cand_turn_dim", A.CAND_TURN_DIM),
                   cfg.get("cand_win_dim", A.CAND_WIN_DIM),
                   cfg.get("hidden", 128))


# ---------------------------------------------------------------- 存取
def sd_to_cpu(sd):
    return {k: v.detach().to("cpu").clone() for k, v in sd.items()}


def load_net(path, device="cpu"):
    """读 checkpoint → (net, cfg, meta)。`weights_only=False`（我们存了 dict/list）。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(DEFAULT_CFG)
    cfg.update(ck.get("cfg") or {})
    net = make_net(cfg)
    net.load_state_dict(ck["sd"])
    net.to(device)
    net.eval()
    meta = {k: v for k, v in ck.items() if k not in ("sd", "cfg")}
    return net, cfg, meta


# ---------------------------------------------------------------- Policy
class NetPolicy(object):
    """把 `CandNet` 包成 `mj_sim` 能用的策略对象，并**顺手记下轨迹**。

    `on_turn(v)` / `on_window(v, phase)` 的两个返回值与规则策略完全同形
    （见 `mj_sim._ask_window` / `play`），所以它可以直接塞进 `Game.play(strategies)`。

    记录内容（供 PPO 用）：
      `points` : list of dict(kind, state, cands, act, logp, aux, tok)
      `trace`  : 动作指纹序列（评估里判"两臂是否逐局不同"）
    """

    def __init__(self, net, cfg=None, device="cpu", greedy=False, record=True,
                 aux=None, name="rl"):
        self.net = net
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update(cfg or {})
        self.device = device
        self.greedy = greedy
        self.record = record
        self.aux = aux                     # None 或 AuxCfg（见 train_selfplay_rl）
        self.name = name
        self.points = []
        self.trace = []
        self.errors = []
        self.fallbacks = 0                 # 由 AuditedGame 回填（见训练器）
        self._temp = float(self.cfg.get("temp", 1.0) or 1.0)

    # -------------------------------------------------- 内部
    def _decide(self, kind, v, cands, feats):
        s = np.asarray(A.encode_state(v), dtype=np.float32)
        c = np.asarray(feats, dtype=np.float32)
        with torch.no_grad():
            st = torch.from_numpy(s).unsqueeze(0).to(self.device)
            ct = torch.from_numpy(c).unsqueeze(0).to(self.device)
            logits, val = self.net(kind, st, ct)
            z = logits[0] / self._temp
            logp_all = torch.log_softmax(z, dim=-1)
            if self.greedy:
                idx = int(torch.argmax(z).item())
            else:
                p = torch.softmax(z, dim=-1)
                idx = int(torch.multinomial(p, 1).item())
            logp = float(logp_all[idx].item())
            vv = float(val[0].item())
        return idx, logp, vv, s, c

    def _aux_for(self, v, cand, before_counts, after_counts):
        if self.aux is None:
            return 0.0
        return self.aux.step(v, cand, before_counts, after_counts)

    # -------------------------------------------------- 引擎接口
    def on_turn(self, v):
        try:
            cands, feats = A.turn_candidates(v)
            idx, logp, vv, s, c = self._decide("turn", v, cands, feats)
            cand = cands[idx]
            before = list(v.counts)
            after = list(v.counts)
            if cand["kind"] == A.K_DISCARD:
                after[cand["tile"]] -= 1
            elif cand["kind"] == A.K_GANG:
                after[cand["tile"]] -= 4 if before[cand["tile"]] >= 4 else 1
            aux = self._aux_for(v, cand, before, after)
            if self.record:
                self.points.append({"kind": "turn", "state": s, "cands": c,
                                    "act": idx, "logp": logp, "val": vv, "aux": aux})
                self.trace.append(A.action_token(cand))
            return A.to_engine_action(cand)
        except Exception as e:                                   # noqa: BLE001
            # 绝不静默：引擎会吞掉异常并 fallback 弃牌，那样轨迹就废了
            self.errors.append("on_turn: %r" % (e,))
            return None

    def on_window(self, v, phase):
        try:
            cands, feats = A.window_candidates(v, phase)
            idx, logp, vv, s, c = self._decide("win", v, cands, feats)
            cand = cands[idx]
            before = list(v.counts)
            after = list(v.counts)
            if cand["kind"] == A.K_PENG:
                after[cand["tile"]] -= 2
            elif cand["kind"] == A.K_CHI:
                from mj_core import chi_tiles, tile_index
                for code in chi_tiles(before, v.last_discard, int(v.meld_sets or 0)):
                    after[tile_index(code)] -= 1
            aux = self._aux_for(v, cand, before, after)
            if self.record:
                self.points.append({"kind": "win", "state": s, "cands": c,
                                    "act": idx, "logp": logp, "val": vv, "aux": aux})
                self.trace.append("w:%s" % A.action_token(cand))
            return A.to_engine_action(cand)
        except Exception as e:                                   # noqa: BLE001
            self.errors.append("on_window: %r" % (e,))
            return None

    # -------------------------------------------------- 采样（不进引擎，供诊断用）
    def scores_only(self, kind, v, cands, feats):
        s = np.asarray(A.encode_state(v), dtype=np.float32)
        c = np.asarray(feats, dtype=np.float32)
        with torch.no_grad():
            st = torch.from_numpy(s).unsqueeze(0).to(self.device)
            ct = torch.from_numpy(c).unsqueeze(0).to(self.device)
            logits, val = self.net(kind, st, ct)
        return logits[0].detach().cpu().numpy(), float(val[0].item())


def net_params(net):
    return sum(p.numel() for p in net.parameters())


if __name__ == "__main__":
    net = make_net(DEFAULT_CFG, seed=0)
    print("CandNet 参数 = %d（state %d, turn_cand %d, win_cand %d, hidden %d）"
          % (net_params(net), net.state_dim, net.cand_turn_dim,
             net.cand_win_dim, net.hidden))
    x = torch.zeros(3, net.state_dim)
    ct = torch.zeros(3, 7, net.cand_turn_dim)
    lg, vv = net("turn", x, ct)
    print("forward ok:", tuple(lg.shape), tuple(vv.shape))
