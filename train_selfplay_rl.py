#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**自我对弈强化学习训练器**（纯 RL：策略梯度直接优化 `max_π E[Σ γ^t R_t]`）。

与仓库里此前的 `train_*.py` 的区别（这是本文件存在的理由）
----------------------------------------------------------
此前所有 `train_*.py` 都是**监督学习 / 离策略价值拟合**：拿既有策略打出来的
(状态, 该席最终得分) 去拟合回归器，再用它排序候选。那不是 RL，也**没有策略改进保证**
（`rl_stage0.py` 头注：跨状态 AUC 0.75 的网络拿去排动作是 **−8.08 分/局**）。
本文件是**在策略**的（on-policy）策略梯度：轨迹由**当前策略自己**产生，return 用
**本局真实 ΔScore**（`Game.play(...)["scores"][seat]`）当 terminal reward。

目标函数（写死在这里，便于核对）
--------------------------------
    J(θ) = E_{τ~π_θ}[ Σ_t γ^t · r_t ] ,   r_T = R_terminal = 该席本局 ΔScore
    r_t  = λ_scale · (hand + eff + mult − risk)   ← goal §4 辅助项，**默认关闭**（`--aux`）
    优化器：PPO（clip + value 基线 + 熵正则），一次 rollout 内做 `--epochs` 遍更新。

★★ 三条硬纪律（本仓的，不是我的偏好）
--------------------------------------
1. **不许用 loss/AUC 说事**。判据只有一条：**用同一批 seed 跑配对的 ΔScore**。
   ⇒ 每个 iteration 都对**固定验证 seed 集**做配对评估；`--eval-ckpt` 可单独复现。
2. **分叉率必须同时报**。若 Δμ ≡ 0 **且** 分叉率 = 0，那不是"打平"，而是
   **两臂恒等**（本仓 3,600 seed 全平栽过一次）。`--eval-null-test`（两臂都是线上
   基因组）**必须**报 Δμ=0 / CI=[0,0] / 分叉率=0 —— 那是这条纪律的自检；
   `--arm-a genome:first:...` 是**阳性对照**（必须报出显著的分叉与 Δ）。
3. **不许针对固定牌山训练**。每个 iteration 用一段**全新的、互不重叠的 seed 区间**；
   对手池**轮换**（最近 K 个 checkpoint + **恒定包含**线上基因组
   `shanten:1:1:2:5:1:1:0`）；训练 seed 与验证 seed **按 seed 段切分**（不是按决策点切分）。

自我对弈的座位安排
------------------
    被学习席位 = seed % 4        （与 `rl_env.seat_and_dealer` 同约定）
    庄家      = (seed // 4) % 4  （同上）
    另 3 席   = 对手池；**每局保证恰好 1 席是线上基因组**，且它在 3 个对手位之间轮换，
                另外 2 席从"最近 K 个 checkpoint 快照"里轮换取。
⇒ 训练分布与评估口径同源：评估时被评估席也是 `seed % 4`，另外 3 席全是线上基因组。

动作空间
--------
见 `rl_action_space.py` 的头注（逐条对齐了 `mj_sim.Game` 的判定；13 维候选编码；
回合候选 = 1(胡) + ≤3(杠) + ≤14(弃)，窗口恒 2 = {过, 碰/吃}）。

启动（★ 与 `train_cand_ranker.py` 同一套 sys.path 纪律，见 `rl_boot.py`）
------------------------------------------------------------------------
    $env:MJ_USE_GPU='1'
    .\\_venv\\Scripts\\python.exe train_selfplay_rl.py --selfcheck      # 先跑这个
    .\\_venv\\Scripts\\python.exe train_selfplay_rl.py --smoke          # 冒烟（~40s）
    .\\_venv\\Scripts\\python.exe train_selfplay_rl.py --iters 20 --games 600 --nproc 4
    # 单独复现配对评估（本仓认可的唯一判据）：
    .\\_venv\\Scripts\\python.exe train_selfplay_rl.py --eval-ckpt rl_latest.pt \\
        --eval-n 2000 --arm-a ckpt:rl_latest.pt --arm-b live
    .\\_venv\\Scripts\\python.exe train_selfplay_rl.py --eval-null-test --eval-n 300

并发纪律：本机今晚 20:00 有正式赛，线上客户端有 **1 秒**的碰/吃决策窗口 ⇒
`--nproc` **默认 4、硬上限 6**（要更高必须显式 `--allow-more`）。若线上日志出现
`窗口错过`/`409`，立刻把 `--nproc` 降到 2。
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TMP = os.path.join(HERE, "rl_tmp")          # 子进程任务/结果的中转目录（跑完即删）

import rl_boot                                                       # noqa: E402
TORCH = rl_boot.bootstrap()
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402

import rl_action_space as A                                         # noqa: E402
import rl_policy as P                                               # noqa: E402
import aux_reward as AR                                             # noqa: E402
from mj_core import tile_index                                      # noqa: E402
from mj_sim import Game                                             # noqa: E402
from rl_env import GenomePolicy                                     # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 线上基因组：`live_ab_arms.txt` 的当前内容（线上 A/B 热切换文件，每房读一次）。
# ⚠ 口径提醒：`run_live.py:GENOME` 与 `rl_env.BASE_GENOME` 写的是 `...:2:1:1:0`
#   （decline=2），而 arms 文件一旦存在就**优先**，所以当前线上是 decline=5。
#   本脚本默认用 arms 文件里的那一个，并把两者都打进日志（口径不靠猜）。
BASE_GENOME = "shanten:1:1:2:5:1:1:0"
ALT_GENOME = "shanten:1:1:2:2:1:1:0"

MAX_NPROC = 6                    # 硬上限：线上客户端有 1s 窗口，别把 CPU 占满
INMEM_KEY = "__current__"        # 训练中当对手/被评估者的"当前策略"在 sds 里的键


# ================================================================ 审计用对局
class AuditedGame(Game):
    """`_fallback_discard` 计数器 —— **合法集与引擎是否分叉**的哨兵。

    引擎在动作非法/策略抛异常时会**静默**改成"打刚摸的那张"（`mj_sim.play`）。
    那会让记录下来的 (状态, 动作) 与真正执行的动作不一致 ⇒ 策略梯度在优化另一个策略。
    所以训练/评估一律用这个子类，`fallbacks > 0` 就当失败来报（`--selfcheck` 断言）。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fallbacks = 0

    def _fallback_discard(self, seat):
        self.fallbacks += 1
        return Game._fallback_discard(self, seat)


# ================================================================ 辅助 reward
def _opp_melds_max(v):
    snap = v.snap or {}
    melds = snap.get("melds") or []
    best = 0
    for s, row in enumerate(melds[:4]):
        if s == v.seat:
            continue
        best = max(best, len(row or []))
    return best


class AuxCfg(object):
    """goal §4 的四个辅助信号的**权重/开关容器**（信号本身在 `aux_reward.py`）。

    ★ 默认 `None`（= `--aux` 不打开）⇒ return **逐字等于** terminal ΔScore。
      打开时 `r_t = --aux-scale · composite(...)`，并**实测**打印
      `|Σ aux| / |R_terminal|` —— 这是"辅助必须显著低于 terminal"那句话的可核对读法，
      而不是一句口头承诺。
    """

    def __init__(self, scale=0.3, lam=None):
        self.scale = float(scale)
        self.lam = dict(AR.LAMBDA)
        if lam:
            self.lam.update({k: v for k, v in lam.items() if v is not None})
        self.acc = {"hand": 0.0, "eff": 0.0, "mult": 0.0, "risk": 0.0,
                    "abs_sum": 0.0, "n": 0}

    def step(self, v, cand, before, after):
        k = cand["kind"]
        melds = int(v.meld_sets or 0)
        unseen = A.unseen_counts(v)
        melds_after = melds + (1 if k in (A.K_PENG, A.K_CHI, A.K_GANG) else 0)
        hand = 0.0 if k == A.K_HU else AR.hand_progress(before, melds, after, unseen)
        eff = 0.0 if k == A.K_HU else AR.efficiency(after, melds_after, unseen)
        piao = int((v.god or {}).get("piao_count") or 0)
        mult = AR.multiplier_potential(before, melds, wall=v.wall, chain=v.chain,
                                       piao=piao, baotou=v.baotou)
        sh_after = 0 if k == A.K_HU else A._sh(after, melds_after)
        declined = bool(k != A.K_HU and (v.drawn or "") and A.can_win(before, melds))
        rv = AR.risk(declined_win=declined, wall=v.wall, tenpai=(sh_after == 0),
                     opp_melds_max=_opp_melds_max(v), shanten=sh_after)
        val = AR.composite(hand, eff, mult, rv, self.lam) * self.scale
        self.acc["hand"] += hand
        self.acc["eff"] += eff
        self.acc["mult"] += mult
        self.acc["risk"] += rv
        self.acc["abs_sum"] += abs(val)
        self.acc["n"] += 1
        return val


# ================================================================ 轨迹记录
class Tracer(object):
    """记录一个座位上**真实发生**的动作序列（评估里判"两臂是否逐局不同"）。"""

    def __init__(self, inner, seat, store):
        self.inner = inner
        self.seat = seat
        store[seat] = []
        self.store = store
        self.canon = 0

    def _tok(self, a, pre=""):
        t = "pass" if not a else "%s:%s" % (a.get("action"), a.get("tile", "") or "")
        self.store[self.seat].append(pre + t)

    def on_turn(self, v):
        a = self.inner.on_turn(v)
        # 抓打圈下引擎会**静默**把非法弃牌换成"打刚摸的那张"（`_fallback_discard`）
        # ⇒ 指纹要记**真正会被执行**的动作，否则两臂会因"返回不同、执行相同"而虚报分叉。
        if a and a.get("action") == "discard" and getattr(v, "restricted", False):
            dr = getattr(v, "drawn", "") or ""
            if a.get("tile") and dr and a.get("tile") != dr:
                self.canon += 1
                a = {"action": "discard", "tile": dr}
        self._tok(a)
        return a

    def on_window(self, v, phase):
        a = self.inner.on_window(v, phase)
        self._tok(a, "w:")
        return a


# ================================================================ 对手池
def parse_arm(spec, base_genome):
    """`'live'` | `'genome:<str>'` | `'ckpt:<key>'` | 裸基因组串 → `(kind, 值)`。"""
    if isinstance(spec, (tuple, list)):
        return tuple(spec)
    spec = (spec or "live").strip()
    if spec in ("live", "base"):
        return ("genome", base_genome)
    if spec.startswith("genome:"):
        return ("genome", spec[len("genome:"):])
    if spec.startswith("ckpt:"):
        return ("ckpt", spec[len("ckpt:"):])
    if ":" in spec:                       # 裸基因组串，如 'shanten:1:1:2:2:1:1:0'
        return ("genome", spec)
    raise ValueError("看不懂的 arm 规格 %r（支持 live / genome:<str> / ckpt:<path>）" % spec)


def read_g1_threshold(default=0.80):
    """从 `PREREG_RL_DEPLOY.md` 的 G1 行里读**当前**门槛（值只在预注册文件里维护）。

    为什么这么做：门槛被收紧过一次（+0.69 → +0.80，§1.0），而脚本里还写死着旧值
    ⇒ "判词与预注册不一致"是**两处维护**的必然结果，这里改成单点。
    """
    import re
    path = os.path.join(HERE, "PREREG_RL_DEPLOY.md")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lstrip().startswith("| **G1**") or line.lstrip().startswith("|**G1**"):
                    m = re.search(r"\*\*≥ \+?([0-9.]+)\*\*", line)
                    if m:
                        return float(m.group(1)), line.strip()[:150]
    except OSError:
        pass
    return default, "（读不到预注册文件，回落默认 %s）" % default


def parse_pool(s, base_genome):
    """逗号分隔的臂规格 → 元组列表（`--opp-pool` / `--eval-league` 共用）。"""
    return [parse_arm(x, base_genome) for x in str(s).split(",") if x.strip()]


# ★★ 2026-09-23（父 agent §Z.326）：**默认训练池改成只有 `live`**，
#   弱档全部去掉，且每局 `live` 席数提到 `--live-seats`（默认 2）。
#   为什么改：16 轮 PPO 后训练回报摆动 1.27，而**对 live 的 val Δμ 纹丝不动**（≈ −1.0），
#   而池子里 8 档多数远弱于 live（`first:…` ≈ −3.7 的地板）
#   ⇒ 预注册假设：梯度主要在奖励"刷弱对手"。判别准则（父 agent 钉死，不许临场改）：
#     · 去掉弱档后 val Δμ **朝 0 移动**（哪怕只到 −0.4~−0.2）⇒ 假设成立；
#     · 仍平坦在 ≈ −1.0 ⇒ 不是对手池问题，瓶颈在容量/信用分配/奖励尺度。
#   （下面这个"强弱都有"的池仍可用 `--opp-pool <串>` 显式启用，供对照。）
# `PREREG_RL_DEPLOY.md` §3.3：「对手池必须含'比我强'的对手」，否则自对弈退化成打自己。
# ★ `pool[0]` 恒为 live（线上基因组），`opp_specs_for` 保证**每房至少 `--live-seats` 席** live。
# 其余覆盖：decline（弃胡博财飘）2/0/9、peng 1/2、gang 0、ukeire 0、route 0，
# 外加一条**公认的弱基线** `discard=first`（防止"只会欺负克隆人"）。
DEFAULT_OPP_POOL_FULL = (
    "live,"
    "genome:shanten:1:1:2:2:1:1:0,"
    "genome:shanten:1:1:2:0:1:1:0,"
    "genome:shanten:1:1:2:9:1:1:0,"
    "genome:shanten:1:1:1:5:1:1:0,"
    "genome:shanten:2:1:2:5:1:1:0,"
    "genome:shanten:1:1:2:5:0:1:0,"
    "genome:shanten:1:1:2:5:1:0:0,"
    "genome:first:0:0:0:0:0:0:0")
DEFAULT_OPP_POOL = "live"

# **标准夹具**（`probe_posture_frontier._play`，G1 用）：被评估席 vs **3 席 BASE**
FIXTURE_OPP = "genome:shanten:1:1:2:2:1:1:0"
# **困难联赛**（G3）：3 席异质，其中一席是线上最强（live）。固定座位序。
DEFAULT_LEAGUE = "live,genome:shanten:1:1:2:2:1:1:0,genome:shanten:2:1:2:9:1:1:0"


def build_spec_policy(spec, base_genome, sds, cfg, greedy=True):
    kind, val = parse_arm(spec, base_genome)
    if kind == "genome":
        return GenomePolicy(val, fresh=True)
    net = P.make_net(cfg)
    net.load_state_dict(sds[val])
    net.eval()
    return P.NetPolicy(net, cfg, device="cpu", greedy=greedy, record=False)


def opp_specs_for(i, pool, live_seats=1):
    """第 i 个房间的 3 个对手规格。

    `pool[0]` 恒为 `"live"`（调用方保证）⇒ **至少 `live_seats` 席是线上基因组**，
    位置在 3 个对手位上轮换；其余席位在 `pool[1:]`（= 最近 K 个 checkpoint 快照）上轮换。
    `live_seats=1` 时与旧行为**逐字一致**（`slot = i % 3`）。
    """
    oth = list(pool[1:])
    ls = max(0, min(3, int(live_seats)))
    start = i % 3
    live_slots = set((start + k) % 3 for k in range(ls))
    out = []
    for k in range(3):
        if k in live_slots or not oth:
            out.append(pool[0])
        else:
            out.append(oth[(i + k) % len(oth)])
    return out



# ================================================================ 房间层（R1）
def _play_room(seed, policies, rounds, room_mode=True, first_dealer="rot",
               game_cls=None, marks=None, salt=0):
    """房间循环 —— 与 `mj_room.play_room` 同构，但多两样东西：

    ① **手边界回调** `marks(t, wall_seed, dealer)`：每一手**开始前**调用，
       用来把该手的 trajectory / 动作序列切开、逐手算 return、逐手判分叉；
    ② **流局判定的修正**（见下面 ★★）。

    `wall_seed` 直接调 `mj_room.wall_seed`（唯一实现，不许抄第二份）；
    `--selfcheck` 里对拍 `mj_room.play_room` 的逐手 rows，**故意**只在流局那一项上
    允许不同（并把差异打印出来）。

    room_mode=False ⇒ 旧口径：一手一局、`Game(seed=seed, dealer=(seed//4)%4)`、
    不传 carry/streak/round_no（= 父 agent 的"配置 A"），用来做死输入 A/B 对照。
    """
    from mj_room import wall_seed
    gc = game_cls or AuditedGame
    if not room_mode:
        rounds = 1
        dealer = (seed // 4) % 4
    else:
        dealer = seed % 4 if first_dealer == "seat" else (seed // 4) % 4
    out, total, streak, fb = [], [0.0, 0.0, 0.0, 0.0], 0, 0
    for t in range(rounds):
        d = dealer if room_mode else dealer
        ws = wall_seed(seed, t, salt) if room_mode else seed
        if marks is not None:
            marks(t, ws, d)
        g = gc(seed=ws, dealer=d, carry_scores=list(total), dealer_streak=streak,
               round_no=t + 1, rounds=rounds)
        r = g.play(policies)
        fb += getattr(g, "fallbacks", 0)
        total = [total[i] + float(r["scores"][i]) for i in range(4)]
        out.append({"round": t, "wall_seed": ws, "dealer": d,
                    "winner": r.get("winner"), "scores": list(r["scores"]),
                    "total": list(total), "dealer_streak": streak,
                    "fan": r.get("fan"), "draw": bool(r.get("draw"))})
        if room_mode:
            w = r.get("winner")
            # ★★ 这里与 `mj_room.play_room:116` **故意不同**：
            #   `mj_sim.Game.result()["winner"]` 在流局时是 **-1**（`mj_sim.py:124` 初始化，
            #   `_settle_draw` 不写它），**不是 None**。而 `mj_room` 写的是 `if w is None`
            #   ⇒ "流局连庄"那一支**永远不进**，并且 `dealer = w` 会把庄家设成 **-1**，
            #   下一手 `Game(dealer=-1)` 里 `hands[-1]` = 3 号位摸第 14 张、
            #   且 `self.dealer=-1` 让**赔付结构全部退化成闲家 ×1**（庄家 8f 那一路消失）。
            #   实测证据见 `_tmp/probe_room_draw.py` 与日志里打印的对照。
            #   ⇒ 本训练器按**规则原文**（"流局庄家连庄"）判：`w is None or w < 0`。
            #   ⚠ `mj_room.py` 本身**没有被改**（它是父 agent 正在跑的探针在用的共享件），
            #     这条差异在自检里打印、在报告里点名。
            if w is None or int(w) < 0:
                streak += 1                      # 流局：庄家不动（连庄）
            elif int(w) == int(d):
                streak += 1                      # 庄家胡 ⇒ 连庄
            else:
                streak = 0                       # 换庄
            dealer = int(w) if (w is not None and int(w) >= 0) else dealer
    return out, fb


def _hand_slices(marks_log, n):
    """把 marks 记录下来的长度列表变成 [(b0, b1), …]（每手一个）。"""
    out = []
    for k in range(n):
        out.append((marks_log[k], marks_log[k + 1]))
    return out


# ================================================================ worker 侧
def _zero(shape):
    return np.zeros(shape, dtype=np.float32)


def rollout_task(task):
    """跑 `n` 个**房间**的自我对弈，返回**被学习席**的决策点样本 + 统计。

    `task["rounds"]` 手为一房（默认 8）；`room_mode=False` 时退化成"一手一局"的旧口径。
    逐手 terminal reward = 该手真实 ΔScore；return 在房间内**跨手累计**
    （γ=1 时就是"该手及其后各手的累计"，见文件头）。
    """
    torch.set_num_threads(1)
    cfg = task["cfg"]
    gamma = float(cfg.get("gamma", 1.0))
    base, sds, pool = task["base_genome"], task["sds"], task["pool"]
    rounds, room_mode = int(task.get("rounds", 1)), bool(task.get("room_mode", False))
    rr = rounds if room_mode else 1
    aux_on = bool(task["aux"])
    net = P.make_net(cfg)
    net.load_state_dict(task["sd"])
    net.eval()

    out = {k: {"s": [], "c": [], "a": [], "lp": [], "r": [], "n": []}
           for k in ("turn", "win")}
    scores, tgts, refs, n_pts, n_empty = [], [], [], 0, 0
    fallbacks = errors = 0
    pure_gap = 0.0
    # ★★ 第 395 轮修 bug：`aux_acc` 以前恒为 `None`，而下面 `if aux is not None:
    #   _fold_aux(aux_acc, aux)` 会直接 `acc.get(...)` ⇒ **AttributeError**。
    #   ⇒ 即 `goal §4` 的辅助 reward **在这条路径上从来没跑起来过**；
    #     因为仓库默认 `--aux` 关，日志永远只打印 `η(辅助/终局) = 0.0000`，bug 被藏住了。
    #   现在按 `aux_on` 初始化成与 `AuxCfg.acc` **同键**的零字典。
    aux_acc = ({"hand": 0.0, "eff": 0.0, "mult": 0.0, "risk": 0.0,
                "abs_sum": 0.0, "n": 0} if aux_on else None)
    t0 = time.time()

    for i in range(task["n"]):
        seed = task["seed0"] + i
        seat = seed % 4
        aux = AuxCfg(task["aux_scale"], task["aux_lam"]) if aux_on else None
        learner = P.NetPolicy(net, cfg, device="cpu", greedy=False, record=True,
                             aux=aux, name="learner")
        specs = opp_specs_for(task["gi0"] + i, pool, task.get("live_seats", 1))
        strategies = [None, None, None, None]
        strategies[seat] = learner
        k = 0
        for s in range(4):
            if s == seat:
                continue
            strategies[s] = build_spec_policy(specs[k], base, sds, cfg, greedy=True)
            k += 1
        marks_log = []

        def _mark(t, ws, d, _m=marks_log, _p=learner):
            _m.append(len(_p.points))

        rows, fb = _play_room(seed, strategies, rounds, room_mode=room_mode,
                              first_dealer=task.get("first_dealer", "rot"),
                              marks=_mark)
        fallbacks += fb
        # ★★★ 第 700 轮：**同 seed 配对优势**（`task["ref_sd"]` 存在时启用）。
        #   由来（这是被逼出来的，不是口味）：
        #     `--per-hand-return`（按手截断）与 `--nstep`（手内自举）都只解决**部分**方差；
        #     实测：`rl11`（截断）/ `rl12b`（+nstep）/ `rl12c`（+clip 0.05）三次，
        #     每轮 `val Δμ` 始终在 0 附近震荡（`rl12c` 七点 −0.300/+0.203/+0.221/
        #     +0.168/+0.226/+0.030/−0.009）⇒ **信号仍被发牌运气淹没**。
        #   本仓已有使能条件（`train_selfplay_rl.py:330`）：
        #     `ws = wall_seed(seed, t, salt)` 是 `(seed,t,salt)` 的**纯函数**
        #     ⇒ **同一 seed 下不同策略拿到完全相同的牌** ⇒ 配对是**可计算**的。
        #   做法：**同一批 seed 再跑一遍**，但把学习席换成**冻结参考策略**，
        #     逐手取 `R_ref` ⇒ 训练目标是 `R − R_ref`（**同期望、发牌运气相消**）。
        #   ⚠ 统计读数（`scores`/`train μ`）**仍记真实的 `R`** ⇒ 口径不变、新旧可比。
        #   ⚠ 参考策略 `record=False`（只要逐手 `rows`，不需要决策点）。
        #   ⚠ 默认关闭（`ref_sd` 缺省）⇒ 与既有行为**逐字一致**（纪律：不悄悄改读数）。
        ref_sd = task.get("ref_sd")
        ref_hand = None
        if ref_sd is not None:
            ref_net = P.make_net(cfg)
            ref_net.load_state_dict(ref_sd)
            ref_net.eval()
            ref_pol = P.NetPolicy(ref_net, cfg, device="cpu", greedy=True,
                                  record=False, name="ref")
            ref_strats = list(strategies)
            ref_strats[seat] = ref_pol
            ref_rows, _rfb = _play_room(seed, ref_strats, rounds,
                                        room_mode=room_mode,
                                        first_dealer=task.get("first_dealer", "rot"))
            ref_hand = [float(ref_rows[kk]["scores"][seat]) for kk in range(rr)]
        errors += len(learner.errors)
        pts_all = learner.points
        bounds = marks_log + [len(pts_all)]
        hands = []
        for kk in range(rr):
            r_real = float(rows[kk]["scores"][seat])
            # ★★ 第 700 轮：训练目标 = 配对差（`ref_hand` 缺省时就是 `r_real`，逐字保持旧行为）
            r_tgt = r_real if ref_hand is None else (r_real - ref_hand[kk])
            hands.append((pts_all[bounds[kk]:bounds[kk + 1]], r_tgt))
            # ⚠ 统计读数**仍记真实 R** ⇒ `train μ` 口径不变 ⇒ 新旧训练可比
            scores.append(r_real)
            tgts.append(r_tgt)          # ★ 配对目标（供自检 ✓）
            refs.append(float(ref_hand[kk]) if ref_hand is not None else float("nan"))
        if aux is not None:
            _fold_aux(aux_acc, aux)
        # ★ 第 518 轮：按手截断的开关（见下面 `per_hand` 的用法与理由）
        per_hand = bool(task.get("per_hand", False))
        if not pts_all:
            n_empty += 1
            continue
        # ★★★ 第 526 轮：**手内的 n-step 自举**（`--nstep N`，默认 0 = 保持原样）。
        #   由来（接 §Z.524 已确认的那条）：
        #     §Z.524 实测"按手截断"把 PPO 相对 `rl_bc6` 的伤害从 −0.48 收到 −0.07（约 85%），
        #     ⇒ **跨手累加确实是噪声来源**。但**手内**仍然是纯蒙特卡洛：
        #       由于奖励**只在手末**给，手内每个决策的 MC 回报**都是同一个 `R_hand`**，
        #       于是**开局第一步和离胡牌只差一步的决策，拿到的目标方差完全一样**。
        #   标准解法就是自举（n-step / GAE）：用 `γ^n · V(s_{t+n})` 替代远端那部分。
        #   而本仓**已经具备条件**：`NetPolicy` 在每个决策点都记了 `val`（`rl_policy.py:184-185`），
        #   只是 `_collect` 之前**没有透传**（本行下方就是新增的透传）。
        #   ★ 公式在本仓特别干净：**手内没有任何中间奖励**，所以
        #       i + n 已经到达手末 ⇒ ret = R_hand（退化成 MC）
        #       否则               ⇒ ret = γ^n · V(s_{i+n})
        #   ⚠ 默认 0 = 与既有行为**逐字一致**（纪律：不能悄悄改掉别人的读数）。
        nstep = int(task.get("nstep", 0) or 0)
        if nstep > 0:
            for pts, R in hands:
                L = len(pts)
                for i, p in enumerate(pts):
                    j = i + nstep
                    if j >= L:
                        p["ret"] = R
                    else:
                        # ⚠ **不静默兜底**：缺 `val` 说明用的不是 `NetPolicy` 的记录
                        #   （例如 BC 的 `GenomeLabelPolicy`），那时 n-step **无法成立**。
                        #   若这里用 `0.0` 顶替，会悄悄把目标换成"γ^n·0"，读数却照常输出 ——
                        #   正是本会话反复抓的"安静的错"。
                        if "val" not in p:
                            raise RuntimeError(
                                "--nstep 需要每个决策点带 `val`，但这条轨迹里没有"
                                "（多半不是 NetPolicy 记录的）。不要静默兜底。")
                        p["ret"] = (gamma ** nstep) * float(p["val"])
        # ★ 逐手 terminal reward，房间内跨手累计（γ=1 ⇒ "该手及其后各手的累计"）
        # ★★★ 第 518 轮：**加一个 `per_hand` 选项** —— 只累计"本手之内"的回报。
        #   为什么（这是被四条实测逼出来的，不是口味问题）：
        #     · §Z.510/514：PPO 的改动**集中在开局**（深度 1-3 一致率 90.3%，
        #       n=10,699），而 §Z.509：KL 提了 5 倍、配对读数**零变化**
        #       ⇒ 它在**方差最大的地方动得最多，却没有方向**。
        #     · 而方差的来源就在这里：`gamma=1` + **跨手累加** ⇒ 开局某个决策的 return
        #       含着**其后 7 手**的结果。可**每一手有自己的 `wall_seed`**（见 §Z.300 附近
        #       的 `wall_seed(seed, t, salt)`）⇒ 未来各手的结果对这个决策**只贡献方差、
        #       几乎不贡献信号**（唯一耦合是 `carry_scores`/连庄，二阶小量）。
        #   ⇒ 所以"按手截断"**同期望、大幅降方差**，是冲着已测到的失效点去的。
        #   ⚠ 默认仍是 1（保持既有行为逐字不变），要用显式打开 —— 免得悄悄改掉所有人的读数。
        nxt = 0.0
        for pts, R in reversed(hands):
            if per_hand:
                nxt = R                      # 只从本手的结果起算
            else:
                nxt = R + gamma * nxt
            for p in reversed(pts):
                nxt = p["aux"] + gamma * nxt
                # ⚠ n-step 分支在上面已经写过 `p["ret"]`；这里**不要覆盖**它
                #   （否则 `--nstep` 会被静默忽略 —— 那正是本会话反复抓的那类"安静的错"）。
                if nstep <= 0:
                    p["ret"] = nxt
        if not aux_on:
            # 纯度自检：第 k 手的 return 必须 = Σ_{j>=k} R_j（无辅助、无折扣时逐字相等）
            expect = 0.0
            for kk in range(len(hands) - 1, -1, -1):
                expect += hands[kk][1]
                for p in hands[kk][0]:
                    pure_gap = max(pure_gap, abs(p["ret"] - expect))
        for p in pts_all:
            d = out[p["kind"]]
            d["s"].append(p["state"])
            c = _zero((A.MAX_TURN_CAND if p["kind"] == "turn" else A.N_WIN_CAND,
                       A.CAND_TURN_DIM if p["kind"] == "turn" else A.CAND_WIN_DIM))
            c[:len(p["cands"])] = p["cands"]
            d["c"].append(c)
            d["n"].append(len(p["cands"]))
            d["a"].append(p["act"])
            d["lp"].append(p["logp"])
            d["r"].append(p["ret"])
        n_pts += len(pts_all)

    res = {}
    for k in ("turn", "win"):
        d = out[k]
        res[k] = {
            "s": np.asarray(d["s"], dtype=np.float32) if d["s"] else _zero((0, A.STATE_DIM)),
            "c": np.asarray(d["c"], dtype=np.float32) if d["c"] else _zero(
                (0, A.MAX_TURN_CAND if k == "turn" else A.N_WIN_CAND,
                 A.CAND_TURN_DIM if k == "turn" else A.CAND_WIN_DIM)),
            "n": np.asarray(d["n"], dtype=np.int64) if d["n"] else np.zeros(0, np.int64),
            "a": np.asarray(d["a"], dtype=np.int64) if d["a"] else np.zeros(0, np.int64),
            "lp": np.asarray(d["lp"], dtype=np.float32) if d["lp"] else _zero((0,)),
            "r": np.asarray(d["r"], dtype=np.float32) if d["r"] else _zero((0,)),
        }
    res.update({"scores": np.asarray(scores, dtype=np.float32),
                "targets": np.asarray(tgts, dtype=np.float32),
                "ref_scores": np.asarray(refs, dtype=np.float32),
                "n_games": task["n"] * rr, "n_rooms": task["n"],
                "n_pts": n_pts, "n_empty": n_empty,
                "fallbacks": fallbacks, "errors": errors, "pure_gap": pure_gap,
                "secs": time.time() - t0,
                "aux": (dict(aux_acc) if aux_acc else None)})
    return res


def _fold_aux(acc, aux):
    for k in ("hand", "eff", "mult", "risk", "abs_sum", "n"):
        acc[k] = acc.get(k, 0.0) + aux.acc[k]


def _fold_aux_dict(acc, d):
    for k in ("hand", "eff", "mult", "risk", "abs_sum", "n"):
        acc[k] = acc.get(k, 0.0) + d[k]


def _trace_key(store):
    """整局的"全席位动作序列"指纹（`store` 是 {seat: [token,…]}）。"""
    return tuple(tuple(store[s]) for s in sorted(store))


def _play_eval_room(seed, spec_own, opp_specs, base, sds, cfg, rounds, room_mode,
                    first_dealer="rot", greedy=True):
    """跑一个**房间**做评估：`seat = seed % 4` 席放 `spec_own`，另 3 席按 `opp_specs`（座位序）。

    返回 (逐手列表, 房间总分, fallbacks)。逐手列表里含有**该手的动作序列**，
    用来逐手判"两臂是否不同"（分叉率）。
    """
    seat = seed % 4
    store = {}
    strategies = [None, None, None, None]
    others = [s for s in range(4) if s != seat]
    for k, s in enumerate(others):
        spec = opp_specs[k] if k < len(opp_specs) else opp_specs[-1]
        strategies[s] = Tracer(build_spec_policy(spec, base, sds, cfg, greedy),
                               s, store)
    own = opp_specs[0] if spec_own is None else spec_own
    strategies[seat] = Tracer(build_spec_policy(own, base, sds, cfg, greedy),
                              seat, store)
    marks_log = []
    rr = int(rounds) if room_mode else 1

    def _mark(t, ws, d, _m=marks_log, _st=store):
        _m.append([len(_st[s]) for s in range(4)])

    rows, fb = _play_room(seed, strategies, rr, room_mode=room_mode,
                          first_dealer=first_dealer, marks=_mark)
    marks_log.append([len(store[s]) for s in range(4)])
    hands = []
    for k in range(rr):
        b0, b1 = marks_log[k], marks_log[k + 1]
        hands.append({
            "score": float(rows[k]["scores"][seat]),
            "dealer": int(rows[k]["dealer"]),
            "winner": rows[k]["winner"],
            "draw": bool(rows[k]["draw"]),
            "is_dealer": int(rows[k]["dealer"]) == seat,
            "total": float(rows[k]["total"][seat]),
            "dealer_streak": int(rows[k]["dealer_streak"]),
            "trace_seat": tuple(store[seat][b0[seat]:b1[seat]]),
            "trace_all": tuple(tuple(store[s][b0[s]:b1[s]]) for s in range(4)),
        })
    return hands, sum(h["score"] for h in hands), fb, sum(t.canon for t in strategies)


def eval_task(task):
    """配对评估（**按房间 seed**）：同一批 seed 跑 A / B 两臂，逐手给 Δ 与分叉。"""
    torch.set_num_threads(1)
    cfg, base, sds = task["cfg"], task["base_genome"], task["sds"]
    opp = task["opp_specs"]
    rounds, room_mode = int(task.get("rounds", 1)), bool(task.get("room_mode", False))
    fd = task.get("first_dealer", "rot")
    d, fl, fa, sa, sb, fb, la, lb = [], [], [], [], [], [], [], []
    droom, isd, roomA, roomB = [], [], [], []
    # ★★★★★ 逐小局胜负（2026-09-29 新增）：`_play_eval_room` 的每手已带 `winner`/`draw` ✓
    #   为什么要它：线上诊断（`GOAL_STATUS.md §Z.802`）证明**漏点是「胡牌率」而不是「放铳」**，
    #   而 Δ分 是重尾量（σ≈26~37）⇒ 收敛极慢；**胡牌率是 0/1 量，收敛快得多** ✓
    #   ⇒ 让 L1 离线判据同时报「胡牌率差」，就能**小时级**筛掉"更保守"的候选 ✓
    wa, na, wb, nb, opp_w, opp_n = 0, 0, 0, 0, 0, 0
    canon = 0
    for seed in task["seeds"]:
        sk = seed % 4
        ha, ra, fa_i, ca_i = _play_eval_room(seed, task["arm_a"], opp, base, sds,
                                            cfg, rounds, room_mode, fd)
        hb, rb, fb_i, cb_i = _play_eval_room(seed, task["arm_b"], opp, base, sds,
                                            cfg, rounds, room_mode, fd)
        canon += ca_i + cb_i
        for k in range(len(ha)):
            d.append(ha[k]["score"] - hb[k]["score"])
            fl.append(0 if ha[k]["trace_seat"] == hb[k]["trace_seat"] else 1)
            fa.append(0 if ha[k]["trace_all"] == hb[k]["trace_all"] else 1)
            sa.append(ha[k]["score"])
            sb.append(hb[k]["score"])
            isd.append(1 if ha[k]["is_dealer"] else 0)
            la.append(len(ha[k]["trace_seat"]))
            lb.append(len(hb[k]["trace_seat"]))
            # 逐小局：被评估席是否胡牌（draw=流局 ⇒ 不计入分母 ✓）
            if not ha[k]["draw"]:
                na += 1
                if ha[k]["winner"] == sk:
                    wa += 1
                opp_n += 1
                if ha[k]["winner"] not in (sk, None):
                    opp_w += 1
            if not hb[k]["draw"]:
                nb += 1
                if hb[k]["winner"] == sk:
                    wb += 1
        droom.append(ra - rb)
        roomA.append(ra)
        roomB.append(rb)
        fb.append(fa_i + fb_i)
    rr = rounds if room_mode else 1
    return {"delta": np.asarray(d, dtype=np.float32),
            "delta_room": np.asarray(droom, dtype=np.float32),
            # ★ 规范排序用的种子（见 `paired_eval` 的注释：装配顺序曾经不确定）
            "hand_seed": np.repeat(np.asarray(task["seeds"], dtype=np.int64), rr),
            "room_seed": np.asarray(task["seeds"], dtype=np.int64),
            "fork_learner": np.asarray(fl, dtype=np.int8),
            "fork_all": np.asarray(fa, dtype=np.int8),
            "sa": np.asarray(sa, dtype=np.float32),
            "sb": np.asarray(sb, dtype=np.float32),
            "is_dealer": np.asarray(isd, dtype=np.int8),
            "room_a": np.asarray(roomA, dtype=np.float32),
            "room_b": np.asarray(roomB, dtype=np.float32),
            # ★ 胡牌率口径（新增 ✓）：A/B 各自的"非流局小局数"与"被评估席胡牌数"
            "win_a": np.int64(wa), "win_b": np.int64(wb),
            "nrd_a": np.int64(na), "nrd_b": np.int64(nb),
            "opp_win": np.int64(opp_w), "opp_n": np.int64(opp_n),
            "fallbacks": int(sum(fb)), "canon": int(canon),
            "len_a": float(np.mean(la)) if la else 0.0,
            "len_b": float(np.mean(lb)) if lb else 0.0,
            "n": len(d), "n_rooms": len(droom), "secs": 0.0}


# ================================================================ BC 热启动（行为克隆）
#
# 为什么要 BC（父 agent 的判据）：随机初始化的网络起点是**垃圾** ——
# 实测 iter1 训练回报 **−6.076 分/局**，而"永远打第一张"（`discard=first`）约 −3.7
# ⇒ 从零开始的 PPO 要先把"别乱打"学会，那一大段位移与麻将目标无关。
# ⇒ 先用**线上基因组自己的决策**做监督预训练（候选打分 ⇒ 同状态内排序，交叉熵），
#   把起点抬到 live 水平，再进 PPO。
def _match_label(cands, act):
    """规则策略的动作 → 候选集下标；**找不到返回 None**（= 掩码与规则分叉，必须计数）。

    ⚠ 这个计数必须 ~0：它 >0 就说明"规则策略会做而我们的动作空间里没有"的动作存在，
      那类样本不能进 BC（标签无意义），也不能进 PPO（掩码与引擎分叉）。
    """
    if not act:
        return None
    k = act.get("action")
    for i, c in enumerate(cands):
        if c["kind"] != k:
            continue
        # ⚠ `tile` 可能是**空串**（`mj_policy.Evolved` 的 hu 返回 {'action':'hu','tile':''}）
        #   ⇒ 必须用 `not at` 判"没有牌"，用 `is None` 会把空串当牌码去解析、然后 ValueError
        #   被吞掉，结果是**所有 hu 标签都被误判成"落不进候选集"**（实测 24 例里 4/6 是它）。
        at = act.get("tile")
        if not at or c["tile"] is None:
            return i                      # 胡 / 碰 / 吃：同名候选唯一
        try:
            if tile_index(at) == c["tile"]:
                return i
        except (ValueError, TypeError):
            pass
    return None


class GenomeLabelPolicy(object):
    """BC 数据采集：在**被学习席**上跑规则基因组，把它的动作当标签。

    候选集由 `rl_action_space` 唯一提供 ⇒ 标签落不进候选集的**比例**就是
    "动作空间 vs 规则策略"的分叉率，逐轮打印。
    """

    def __init__(self, genome, record=True, labeler=None, inner_policy=None):
        # ★★ 第 442 轮（DAgger）：把**执行者**与**打标签者**分开。
        #   动机（§Z.441 的定量结论）：BC 的缺口是**复合误差** ——
        #     每步 9.1% 的偏差，在"一副大牌 ≈ 连续 13.9 步"上被放大成 0.909^13.9 = **0.264 倍**，
        #     与实测的番>=4 比值（1.46/5.52 = 0.264）**完全一致**。
        #   ⇒ 病根是"BC 只在**老师的状态分布**上训练"，模型一旦偏离就没见过那些状态。
        #   ⇒ DAgger 的解法：让**模型自己去走**（得到它自己的状态分布），
        #     但每一步**问老师**该怎么做（老师只给标签，不接管牌局）。
        #   `inner_policy`：驱动牌局的策略（DAgger 时 = 当前模型）
        #   `labeler`     ：只用来产出标签的策略（DAgger 时 = 老师）；None ⇒ 用 inner（= 原 BC 行为）
        self.inner = inner_policy if inner_policy is not None else GenomePolicy(
            genome, fresh=True)
        self.labeler = labeler
        self.record = record
        self.points = []
        self.dropped = 0            # 答了但候选集里没有（真·掩码分叉）
        self.unsupported = 0        # 环境**不支持的**动作（窗口明杠）—— 不是掩码错
        self.pass_only = 0
        self.drops = []             # 前几例的原始形状（诊断用，必须能解释每一条）
        self.errors = []

    def _note_drop(self, where, v, cands, act):
        if len(self.drops) < 4:
            self.drops.append(
                "%s act=%r cand_kinds=%s cand_tiles=%s restricted=%s drawn=%r wall=%s "
                "melds=%s counts_nonzero=%s"
                % (where, act, [c["kind"] for c in cands],
                   [(c["kind"], c["tile"]) for c in cands
                    if c["kind"] in (A.K_HU, A.K_GANG)],
                   bool(getattr(v, "restricted", False)), getattr(v, "drawn", ""),
                   getattr(v, "wall", None), getattr(v, "meld_sets", None),
                   [t for t in range(34) if v.counts[t] > 0]))

    def _push(self, kind, v, cands, feats, lab):
        if self.record:
            # ★ 第 419 轮：多记一个 **局号** —— 尾部加权 BC 要用它把
            #   "该决策点"与"该局最终倍率"对上（§Z.418：BC 在稀有高倍率动作上退化，
            #   番>=4 只有老师的 1.82/5.61 = 32%）。
            self.points.append({"kind": kind, "state": A.encode_state(v),
                                "cands": feats, "act": lab,
                                "rno": int(getattr(v, "round_no", 0) or 0)})

    def on_turn(self, v):
        try:
            cands, feats = A.turn_candidates(v)
            act = self.inner.on_turn(v)                 # ← 驱动牌局（DAgger 时 = 模型）
            # ★ 第 442 轮：标签来自 `labeler`（DAgger 时 = 老师），没有则回落到 inner（原 BC 行为）
            lab_act = self.labeler.on_turn(v) if self.labeler is not None else act
            lab = _match_label(cands, lab_act)
            if lab is None:
                self.dropped += 1
                self._note_drop("turn", v, cands, act)
                if not act:
                    lab = None
                # 抓打圈下规则策略可能返回非法弃牌（引擎会换成打刚摸那张）
            else:
                self._push("turn", v, cands, feats, lab)
            return act
        except Exception as e:                                   # noqa: BLE001
            self.errors.append("bc on_turn: %r" % (e,))
            return None

    def on_window(self, v, phase):
        try:
            cands, feats = A.window_candidates(v, phase)
            act = self.inner.on_window(v, phase)        # ← 驱动牌局（DAgger 时 = 模型）
            # ★ 第 442 轮：标签来自 `labeler`（DAgger 时 = 老师）
            lab_act = (self.labeler.on_window(v, phase)
                       if self.labeler is not None else act)
            lab = _match_label(cands, lab_act)
            if lab is None:
                if lab_act and lab_act.get("action") == "gang":
                    # ★ 规则策略在**碰窗口**里会返回「明杠」（从别人打出的牌杠）。
                    #   而 `mj_sim` 的窗口只认 peng/chi（`_ask_window` 要求 action==kind，
                    #   `window_act` 也只接受当前窗口类型）⇒ 引擎把它**静默当 pass**。
                    #   线上平台是有明杠的（`audit_minggang.py`），所以这是**模拟器的动作缺口**，
                    #   不是我们的掩码错。⇒ 单独计数，不混进 dropped。
                    self.unsupported += 1
                elif lab_act:                   # 答了碰/吃但候选里没有 ⇒ 才算分叉
                    self.dropped += 1
                    self._note_drop("win/" + str(phase), v, cands, lab_act)
                else:
                    lab = 0                   # 过 = 候选 0
                    self.pass_only += 1
            if lab is not None:
                self._push("win", v, cands, feats, lab)
            return act
        except Exception as e:                                   # noqa: BLE001
            self.errors.append("bc on_window: %r" % (e,))
            return None


def _round_fan(vec, seat):
    """从一局的**四家结算向量**反解"我们那一手的番"；不是我们胡则返回 None。

    ★ 第 419 轮新增（尾部加权 BC 用）。口径与 `analyze_eval_tail.py`、
      `analyze_final4_rounds.py` **完全一致**，依据同一条结算恒等式：
    ```
      庄家赢：(+24f, −8f, −8f, −8f)    ← 输的三家**全相同**
      闲家赢：(+10f, −8f,  −f,  −f)    ← 输的三家里**有一家不同**（就是庄）
      流局  ：全 0
    ```
    ⇒ **庄闲不用额外信息，看"输的三家的形状"就能定**：
      三家全等 ⇒ 庄赢 ⇒ `f = me/24`；有一家不同 ⇒ 闲赢 ⇒ `f = me/10`。
    ⚠ 只认"最大元素 == 我们"；其余（别人胡 / 流局）返回 None ——
      **不猜**，猜错会把别人家的倍率当成自己的，直接污染权重。
    """
    try:
        if len(vec) != 4:
            return None
        v = [float(x) for x in vec]
        # ★ 先用**仓库已有的不变量**挡掉非法向量（`mj_bot` 自测里就有"每局四家得失之和恒为 0"）：
        #   少了这两道，`[10,0,0,0]` 这种非结算向量会被算成"庄赢 f=0.417"，
        #   直接把垃圾权重喂进 BC。
        if abs(sum(v)) > 1e-6:
            return None
        me = v[seat]
        others = [v[i] for i in range(4) if i != seat]
        if me <= 0 or any(x >= 0 for x in others):
            return None                      # 输家必须为负 ⇒ 挡掉流局与非法向量
        if any(x >= me for x in others):
            return None                      # 不是我们赢
        uniq = len(set(round(x, 6) for x in others))
        if uniq == 1:
            return me / 24.0                 # 庄家自摸：三家各 −8f
        if uniq == 2:
            return me / 10.0                 # 闲家自摸：庄 −8f，另两家 −f
        return None
    except Exception:                                                # noqa: BLE001
        return None


def bc_task(task):
    """采集 BC 数据（被学习席 = 规则基因组 `teacher`，对手 = 对手池）。"""
    torch.set_num_threads(1)
    base, sds, pool = task["base_genome"], task["sds"], task["pool"]
    # ★ 第 419 轮：**尾部加权 BC**（`goal §2.2/§2.3/§2.6`；§Z.418 的实测动机）
    #   实测（24,000 局同 seed 配对）：模型的 `番>=4` 率 = **1.82%**，
    #   而它模仿的老师 = **5.61%** ⇒ **BC 是退化模仿，且恰恰退化在稀有高倍率动作上**。
    #   机制：`cross_entropy` 最大化**平均**对数似然 ⇒ 常见决策（打哪张/碰不碰）
    #   样本极多、梯度占绝对主导；稀有但关键的动作（弃胡转爆头、财飘）
    #   样本极少 ⇒ 被平均掉。
    #   ⇒ 干预：按**该局最终倍率**给每个决策点加权 `w = 1 + α·max(0, fan−1)`。
    #     `alpha=0` 时与旧行为**逐字一致**（w 恒为 1）⇒ 可 A/B、可回退。
    alpha = float(task.get("tail_alpha", 0.0) or 0.0)
    out = {k: {"s": [], "c": [], "n": [], "a": [], "w": [], "rno": []}
           for k in ("turn", "win")}
    rows, scores, drops = [], [], []
    dropped = errors = fallbacks = unsupported = 0
    t0 = time.time()
    rounds, room_mode = int(task.get("rounds", 1)), bool(task.get("room_mode", False))
    rr = rounds if room_mode else 1
    for i in range(task["n"]):
        # ★ 第 438 轮：采集阶段的进度输出（落到 `_tmp/<tag>_log_<i>.txt`，
        #   由 `run_children` 的心跳带出来）。原实现**全程不打一个字**，
        #   导致日志停在"拉起 6 个子进程"，我因此两次误判"还在采集"（§Z.437）。
        if i and i % 40 == 0:
            print("      [bc] %d/%d 局 ｜ 已用 %.0fs ｜ 决策点 %d"
                  % (i, task["n"], time.time() - t0,
                     sum(len(v["a"]) for v in out.values())), flush=True)
        seed = task["seed0"] + i
        seat = seed % 4
        # ★★ 第 442 轮 DAgger：若给了 `dagger_ckpt`，则**执行者是那个模型**、
        #   **标签仍由老师给**。这样采到的是**模型自己会遇到的状态**（§Z.441 的病根所在）。
        # ★ 第 456 轮补：**混合比例** `dagger_mix`（默认 1.0 = 全部由模型驱动）。
        #   为什么需要：标准 DAgger 用的是**聚合数据集** D = D_老师 ∪ D_模型，
        #   若**全部**由模型驱动，训练集就只覆盖模型自己的状态分布 ⇒
        #   可能**忘掉老师在它自己分布上的好行为**（而模型的分布本身就是从老师出发漂移来的）。
        #   ⇒ `mix=0.5` 表示一半的局仍由老师驱动（标签同样是老师，所以标签口径不变），
        #     这样一份数据里**两种分布都有**，是更稳的迭代方式。
        _dag = task.get("dagger_ckpt")
        _mix = float(task.get("dagger_mix", 1.0) or 0.0)
        if _dag:
            _use_model = (i % 100) < int(round(_mix * 100))
            if _use_model:
                _inner = build_spec_policy("ckpt:" + _dag, base, sds, task["cfg"],
                                           greedy=False)
            else:
                _inner = GenomePolicy(task["teacher"], fresh=True)
            _labeler = GenomePolicy(task["teacher"], fresh=True)
            learner = GenomeLabelPolicy(task["teacher"], inner_policy=_inner,
                                        labeler=_labeler)
        else:
            learner = GenomeLabelPolicy(task["teacher"])
        specs = opp_specs_for(task["gi0"] + i, pool, task.get("live_seats", 1))
        strategies = [None, None, None, None]
        strategies[seat] = learner
        k = 0
        for s in range(4):
            if s == seat:
                continue
            strategies[s] = build_spec_policy(specs[k], base, sds, task["cfg"],
                                              greedy=True)
            k += 1
        rws, fb = _play_room(seed, strategies, rounds, room_mode=room_mode,
                             first_dealer=task.get("first_dealer", "rot"))
        fallbacks += fb
        dropped += learner.dropped
        unsupported += learner.unsupported
        errors += len(learner.errors)
        if learner.drops and len(drops) < 6:
            drops.extend(learner.drops[:2])
        scores.extend(float(x["scores"][seat]) for x in rws)
        rows.append(len(learner.points))
        # ★★ 第 435 轮修**两个静默 bug**（第 427 轮那个修复只修了一半）：
        #   实测 `_play_room` 的行（`rounds=4` 的样例）：
        #     round | winner | fan | scores
        #         0 |      2 |   2 | [-16,-2,20,-2]
        #         1 |      0 |   1 | [10,-1,-8,-1]
        #   ⇒ **bug A：`fan` 是"赢家"的番，不一定是我们的** ——
        #        原代码直接取 `x["fan"]` ⇒ 我们没胡的那一局，会被**对手的番**加权。
        #   ⇒ **bug B：`round` 是 0 起，而 `View.round_no` 是 1 起** ——
        #        原代码用 `p["rno"]`（1..8）去查 `x["round"]`（0..7）⇒ **整体错位一轮**。
        #        ⚠ 这比"没加权"更糟：**大牌局的权重被加到了下一轮的决策上**。
        #   ⇒ 两处一起修：只认"我们赢"的行，并把 0 起的 `round` 转成 1 起。
        fan_by_rno = {}
        for x in rws:
            try:
                rnd0 = x.get("round")
                if rnd0 is None:
                    continue
                if x.get("winner") != seat:      # ★ bug A：不是我们胡 ⇒ 这一局不该按番加权
                    continue
                fan = x.get("fan")
                if fan is None:
                    fan = _round_fan(x.get("scores") or [], seat)   # 兜底：反解
                if fan is not None:
                    fan_by_rno[int(rnd0) + 1] = float(fan)          # ★ bug B：0 起 -> 1 起
            except Exception:                                        # noqa: BLE001
                continue
        for p in learner.points:
            d = out[p["kind"]]
            d["s"].append(p["state"])
            c = _zero((A.MAX_TURN_CAND if p["kind"] == "turn" else A.N_WIN_CAND,
                       A.CAND_TURN_DIM if p["kind"] == "turn" else A.CAND_WIN_DIM))
            c[:len(p["cands"])] = p["cands"]
            d["c"].append(c)
            d["n"].append(len(p["cands"]))
            d["a"].append(p["act"])
            # ★ 第 448 轮：把**局号**也导出（诊断探针要按"本局内第几步"分层，
            #   用来独立检验 §Z.441 的复合误差说法 —— 那条链是否真的越走越偏）
            d["rno"].append(int(p.get("rno") or 0))
            f = fan_by_rno.get(int(p.get("rno") or 0))
            d["w"].append(1.0 + alpha * max(0.0, (f or 1.0) - 1.0))
    res = {}
    for k in ("turn", "win"):
        d = out[k]
        res[k] = {
            "s": np.asarray(d["s"], dtype=np.float32) if d["s"] else _zero((0, A.STATE_DIM)),
            "c": np.asarray(d["c"], dtype=np.float32) if d["c"] else _zero(
                (0, A.MAX_TURN_CAND if k == "turn" else A.N_WIN_CAND,
                 A.CAND_TURN_DIM if k == "turn" else A.CAND_WIN_DIM)),
            "n": np.asarray(d["n"], dtype=np.int64) if d["n"] else np.zeros(0, np.int64),
            "a": np.asarray(d["a"], dtype=np.int64) if d["a"] else np.zeros(0, np.int64),
            # ★ 第 419 轮：尾部加权 BC 的样本权重（`alpha=0` 时恒为 1.0 ⇒ 与旧行为逐字一致）
            "w": np.asarray(d["w"], dtype=np.float32) if d["w"] else np.zeros(0, np.float32),
            "rno": np.asarray(d["rno"], dtype=np.int64) if d["rno"] else np.zeros(0, np.int64),
        }
    res.update({"rows": rows, "scores": np.asarray(scores, dtype=np.float32),
                "drops": drops,
                "dropped": dropped, "unsupported": unsupported,
                "errors": errors, "fallbacks": fallbacks,
                "n_games": task["n"] * rr, "n_rooms": task["n"],
                "secs": time.time() - t0,
                "n_pts": sum(len(v["a"]) for v in res.values() if isinstance(v, dict))})
    return res


# ================================================================ 主进程工具
def split_blocks(items, n):
    items = list(items)
    n = max(1, min(n, len(items)))
    per = len(items) // n
    out = [items[i * per:(i + 1) * per] for i in range(n)]
    for j, x in enumerate(items[per * n:]):
        out[j].append(x)
    return [b for b in out if b]


class Log(object):
    def __init__(self, path, also_stdout=True):
        self.path = path
        self.also = also_stdout
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        self.f.write(msg + "\n")
        self.f.flush()
        if self.also:
            print(msg)
            sys.stdout.flush()

    def close(self):
        try:
            self.f.close()
        except OSError:
            pass

    def report_eval(self, tag, o):
        self("  [配对评估%s] 房间 %d 个 / 共 %d 手  Δμ(分/局) = %+.4f   "
              "95%%CI [%+.4f, %+.4f]   σ_paired = %.3f"
              % (tag, o.get("n_rooms", 0), o["n"], o["mu"], o["lo"], o["hi"], o["sd"]))
        if o.get("mu_dealer") == o.get("mu_dealer"):
            self("             按庄位分层（R3）：被评估席**坐庄** %d 手 Δμ = %+.3f "
                  "(SE %.3f) ｜ **做闲** %d 手 Δμ = %+.3f (SE %.3f)  ｜ 房间总分 Δμ = %+.3f"
                  % (o.get("n_dealer", 0), o.get("mu_dealer", float("nan")),
                     o.get("se_dealer", float("nan")), o.get("n_nodealer", 0),
                     o.get("mu_nodealer", float("nan")), o.get("se_nodealer", float("nan")),
                     o.get("mu_room", float("nan"))))
        if "se_naive" in o:
            self("             朴素 SE = %.4f   块间 SE(%d 块) = %.4f   "
                  "块间 95%%CI [%+.4f, %+.4f]%s"
                  % (o["se_naive"], o.get("n_blocks", 0), o.get("se_block", float("nan")),
                     o.get("lo_block", float("nan")), o.get("hi_block", float("nan")),
                     "   ⚠ 两者差 >2× ⇒ **只认块间 SE**"
                     if (o.get("se_block") == o.get("se_block")
                         and o.get("se_naive", 0) > 0
                         and o["se_block"] / o["se_naive"] > 2.0) else ""))
        self("             分叉率（被评估席）= %.4f   分叉率（四席全序列）= %.4f   "
              "A 均值 %+.3f / B 均值 %+.3f   fallback=%d   用时 %.1fs"
              % (o["fork_learner"], o["fork_all"], o["mean_a"], o["mean_b"],
                 o["fallbacks"], o["secs"]))
        if o["fork_learner"] == 0.0 and abs(o["mu"]) < 1e-9:
            self("             ⛔ **Δμ ≡ 0 且分叉率 = 0 ⇒ 两臂恒等，不是「打平」**"
                 "（本仓在 3,600 seed 上栽过一次）⇒ 这条评估没有任何信息量。")
        elif o["fork_learner"] == 0.0:
            self("             ⚠ 分叉率 0 而 Δμ≠0 ⇒ 两臂动作序列相同、结果却不同，"
                 "说明环境里有**未被记录的随机源**，先查它再信这个数。")
        else:
            self("             ⇒ 两臂**确实不同**（分叉率 %.1f%%），Δμ 的符号与大小可以读。"
                 % (100 * o["fork_learner"]))
        # ★ 胡牌率口径（新增 ✓）：诊断证明漏点是「胡牌率」而非「放铳」（§Z.802）
        #   ⇒ 它是 0/1 量、收敛远快于重尾的 Δ分 ⇒ 可当 **L1.5** 快速筛选用 ✓
        if o.get("nrd_a") and o.get("nrd_b"):
            wra = 100.0 * o["win_a"] / o["nrd_a"]
            wrb = 100.0 * o["win_b"] / o["nrd_b"]
            ra = wra / 100.0
            rb = wrb / 100.0
            se_d = (ra * (1 - ra) / o["nrd_a"] + rb * (1 - rb) / o["nrd_b"]) ** 0.5
            dd = (wra - wrb) / 100.0
            self("             ★ 胡牌率：A **%.2f%%**（%d/%d）｜ B **%.2f%%**（%d/%d）"
                 " ⇒ 差 **%+.2fpp** ｜ SE %.2fpp ｜ 95%%CI [%+.2f, %+.2f]pp"
                 % (wra, o["win_a"], o["nrd_a"], wrb, o["win_b"], o["nrd_b"],
                    100 * dd, 100 * se_d, 100 * (dd - 1.96 * se_d), 100 * (dd + 1.96 * se_d)))
            if o.get("opp_n"):
                self("               对手（3 席合计）胡牌率 %.2f%%（%d/%d）⇒ A 相对对手 %+.2fpp"
                     % (100.0 * o["opp_win"] / o["opp_n"], o["opp_win"], o["opp_n"],
                        100 * (ra - o["opp_win"] / o["opp_n"])))


def paired_eval(seeds, arm_a, arm_b, opp_specs, base_genome, cfg, sds, nproc,
                log=None, tag="", tmp_tag="pev", blocks_want=8, rounds=1,
                room_mode=False, first_dealer="rot", dump=""):
    """配对评估：同一批 seed 跑 A/B 两臂，逐 seed 求 Δ 并统计分叉率。

    ★ 唯一认可的判据（`--eval-null-test` 是它的空对照，`--arm-a genome:first:…` 是阳性对照）。
    """
    seeds = list(seeds)
    blocks = split_blocks(seeds, nproc)
    jobs = [{"seeds": b, "arm_a": arm_a, "arm_b": arm_b,
             "opp_specs": opp_specs, "base_genome": base_genome,
             "cfg": cfg, "sds": sds, "rounds": rounds, "room_mode": room_mode,
             "first_dealer": first_dealer} for b in blocks]
    t0 = time.time()
    res = run_children("eval", jobs, tmp_tag, log=log)
    secs = time.time() - t0
    # ★★ G4 抓到过的缺陷：逐 seed 数组的**装配顺序**不确定（两次跑的聚合逐位相同、
    #   而逐元素是同一批数据的置换）⇒ 任何"按位置对齐"的分析都会错。
    #   这里按 seed **规范排序**，让产出与分块/装配顺序完全无关。
    def _cat(key, shape0, dtype):
        return (np.concatenate([r[key] for r in res]) if res
                else np.zeros(shape0, dtype=dtype))

    hs = _cat("hand_seed", (0,), np.int64)
    rs = _cat("room_seed", (0,), np.int64)
    ho = np.argsort(hs, kind="stable") if len(hs) else np.zeros(0, np.int64)
    ro = np.argsort(rs, kind="stable") if len(rs) else np.zeros(0, np.int64)
    d = _cat("delta", (0,), np.float32)[ho]
    dr = _cat("delta_room", (0,), np.float32)[ro]
    idd = _cat("is_dealer", (0,), np.int8)[ho]
    if len(hs):
        assert np.array_equal(hs[ho], np.sort(hs)), "排序后仍不单调 —— 逻辑错"
    n, n_rooms = len(d), len(dr)
    fl_sorted = _cat("fork_learner", (0,), np.int8)[ho]
    fa_sorted = _cat("fork_all", (0,), np.int8)[ho]
    mu = float(d.mean()) if n else float("nan")
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    # R3：按 is_dealer 分层（被评估席这一手是不是庄）
    def _strat(mask):
        x = d[mask]
        if len(x) == 0:
            return float("nan"), float("nan"), 0
        se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
        return float(x.mean()), se, len(x)
    mu_de, se_de, n_de = _strat(idd == 1)
    mu_nd, se_nd, n_nd = _strat(idd == 0)
    # ★ `PREREG_RL_DEPLOY.md` §2：必须**同时**给朴素 SE 与**块间 SE**（≥8 块）。
    #   同一段 seed 上的局与局之间有相关性（同一批牌风/连续 seed 的牌山相关），
    #   块间 SE 是这条相关性的**保守**读数；两者差 >2× 时只认块间 SE。
    # ★ 块间 SE 按**房间**聚类（同一房间的 8 手共享累计分/连庄链 ⇒ 不是独立的；
    #   `mj_room.room_of_gid` 的注释就是为这件事写的）。每块 = 若干房间的**逐手均值**。
    nb = max(1, min(blocks_want, max(1, n_rooms)))
    # ★★ 块 = 若干**房间**；块均值必须换算成**分/局**（÷ rounds）才能与逐手 Δ 的
    #   朴素 SE 同量纲 —— 第一版忘了除，块间 SE 就大了 rounds 倍，误报"11× 过散"。
    bmeans = [float(np.mean(x)) / max(1, rounds) for x in np.array_split(dr, nb)] \
        if n_rooms else []
    se_block = float(np.std(bmeans, ddof=1) / np.sqrt(len(bmeans))) if len(bmeans) > 1 \
        else float("nan")
    ci_block = 1.96 * se_block

    def m(k):
        """逐任务的**数组**键（fork_*）要拼接后求均值 —— 各块手数不同，
        直接 `np.mean([array,…])` 会 `inhomogeneous shape` 报错（实测白跑一次 815s）。"""
        if not res:
            return float("nan")
        vals = [r[k] for r in res]
        if all(np.ndim(v) == 0 for v in vals):
            return float(np.mean(vals))
        return float(np.concatenate([np.asarray(v).ravel() for v in vals]).mean())
    out = {"n": n, "mu": mu, "sd": sd, "ci95": float(ci), "lo": mu - ci, "hi": mu + ci,
           "se_naive": float(sd / np.sqrt(n)) if n else float("nan"),
           "se_block": se_block, "n_blocks": len(bmeans),
           "ci_block95": ci_block, "lo_block": mu - ci_block,
           "hi_block": mu + ci_block,
           "n_hands": n, "n_rooms": n_rooms,
           "mu_room": float(dr.mean()) if n_rooms else float("nan"),
           "mu_dealer": mu_de, "se_dealer": se_de, "n_dealer": n_de,
           "mu_nodealer": mu_nd, "se_nodealer": se_nd, "n_nodealer": n_nd,
           "fork_learner": float(fl_sorted.mean()) if n else float("nan"),
           "fork_all": float(fa_sorted.mean()) if n else float("nan"),
           "canon": int(sum(r.get("canon", 0) for r in res)),
           "mean_a": float(np.concatenate([r["sa"] for r in res]).mean()) if n else
           float("nan"),
           "mean_b": float(np.concatenate([r["sb"] for r in res]).mean()) if n else
           float("nan"),
           "fallbacks": int(sum(r["fallbacks"] for r in res)),
           "len_a": m("len_a"), "len_b": m("len_b"), "secs": secs}
    # ★ 胡牌率口径（新增 ✓）：A/B 各自「非流局小局数」与「被评估席胡牌数」
    #   ⇒ L1 离线判据从此能报 **胡牌率差**（0/1 量、收敛远快于重尾的 Δ分 ✓）
    out["win_a"] = int(sum(int(r.get("win_a", 0)) for r in res))
    out["win_b"] = int(sum(int(r.get("win_b", 0)) for r in res))
    out["nrd_a"] = int(sum(int(r.get("nrd_a", 0)) for r in res))
    out["nrd_b"] = int(sum(int(r.get("nrd_b", 0)) for r in res))
    out["opp_win"] = int(sum(int(r.get("opp_win", 0)) for r in res))
    out["opp_n"] = int(sum(int(r.get("opp_n", 0)) for r in res))
    out["_raw"] = {"delta": d, "delta_room": dr, "is_dealer": idd,
                   "sa": _cat("sa", (0,), np.float32)[ho],
                   "sb": _cat("sb", (0,), np.float32)[ho],
                   "room_a": _cat("room_a", (0,), np.float32)[ro],
                   "room_b": _cat("room_b", (0,), np.float32)[ro],
                   "hand_seed": hs[ho], "room_seed": rs[ro],
                   "fork_learner": _cat("fork_learner", (0,), np.int8)[ho],
                   "fork_all": _cat("fork_all", (0,), np.int8)[ho],
                   # ★ 胡牌率口径（标量，供 `analyze_*` 直接读 ✓）
                   "win_a": np.int64(out["win_a"]), "win_b": np.int64(out["win_b"]),
                   "nrd_a": np.int64(out["nrd_a"]), "nrd_b": np.int64(out["nrd_b"]),
                   "opp_win": np.int64(out["opp_win"]), "opp_n": np.int64(out["opp_n"]),
                   "canon": out["canon"]}
    if dump:
        # ★★ 2026-10-09：**先建目录再落盘** ★★
        #   缺陷：默认 dump 路径是 `_tmp/eval_<ts>.npz`（见 run_eval 的调用点），
        #   而 `_tmp/` 是**运行期临时目录**、干净检出里并不存在
        #   ⇒ `np.savez_compressed` 抛 FileNotFoundError。
        #   代价特别大：它在**全部配对评估跑完之后**才崩
        #   （实测 120 seed / 4 进程 / 488 秒算完），而报告是在落盘**之后**才打印
        #   ⇒ 结果全部丢掉、只剩一条 traceback。
        #   本仓其它工具（build_teacher_data / export_* / merge_live …）都写了
        #   `os.makedirs(..., exist_ok=True)`，只有这一处漏了。
        _d = os.path.dirname(os.path.abspath(dump))
        if _d:
            os.makedirs(_d, exist_ok=True)
        np.savez_compressed(dump, **{k: v for k, v in out["_raw"].items()})
        if log is not None:
            log("      （逐手原始数组已落盘 -> %s）" % dump)
    if log is not None:
        log.report_eval(tag, out)
    return out


# ================================================================ PPO 更新
# ★ ③ 独立价值基线（`--value-init`）的**懒加载缓存**（2026-10-03，第 945 轮 / §Z.1321）。
#   · 默认 `--value-init` 为空 ⇒ `_VB_CACHE` 恒为 None ⇒ **行为逐字不变** ✓
#   · 维度守卫见 `ppo_update`：ckpt 的 `din` 必须等于 `A.STATE_DIM`，否则忽略 + 告警 ✓
_VB_CACHE = None
_VB_TRIED = False


def ppo_update(net, opt, batch, hp, dev, log=None):
    """PPO（clip）。return/advantage 按**批内 μ/σ**归一化（纯尺度变换，见文件头）。

    ★★★★★ 2026-09-30（`GOAL_STATUS.md §Z.919`）：新增 `log` 参数。
      为什么必须（吃过亏 ✓）：`_tick` 原来**只用 `print`** ⇒ 计时只进 stdout ✗、
      **不进 `rl_*.log`** ✗。于是 2026-09-30 训练"更新阶段 25~60 分钟"时，
      我从日志里**看不到任何一步计时** ✗，误判成"卡死在子进程收尾"✗ 并**白白重启一次** ✓。
      真相是：100 个梯度步里个别步 138~275s ✗（其余 0.0~0.4s ✓）——
      而这个真相**一直打在 stdout 里**，只是没进日志 ✓。
      ⇒ 现在同时走 `log()`（若调用点传入）与 `print()` ✓，两条路都在 ✓。
    """
    _tt = [time.time()]

    def _tick(lab):
        _tt.append(time.time())
        _line = "      [计时·update] %-22s %.1fs" % (lab, _tt[-1] - _tt[-2])
        if log is not None:
            try:
                log(_line)
            except Exception:
                pass
        print(_line, flush=True)

    # ★★★★★ 2026-10-03（第 945 轮 / `GOAL_STATUS.md §Z.1321`）：③ **独立价值基线**的懒加载。
    #   目的：把 PPO advantage 的 **baseline** 从"网络自带 value 头"换成**独立价值头**
    #         （`tools/value_baseline.py`；离线 Spearman 0.4436 ／ 优势方差降幅 17.46%）。
    #   为什么懒加载：`ppo_update` 在父进程被调（`:1835`/`:1839` ✓），但把**路径字符串**放进
    #         `hp` 更稳（跨进程/未来调用点都安全 ✓），且**默认空 ⇒ 走原路径** ✓。
    #   ★ 维度守卫（吃过亏 ✗ 见 §Z.1320 的 371/370）：ckpt 的 `din` 必须等于当前 `A.STATE_DIM`；
    #         不符 ⇒ **忽略基线 + 告警一次** ✓（绝不在维度不符时硬接 ⇒ 那会静默换掉实验 ✗）。
    global _VB_CACHE, _VB_TRIED
    if not _VB_TRIED:
        _VB_TRIED = True
        _vi = str(hp.get("value_init") or "").strip()
        if _vi:
            try:
                import importlib.util as _ilu
                _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "tools", "value_baseline.py")
                _spec = _ilu.spec_from_file_location("value_baseline", _p)
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                # ★ 兼容两种布局：`load` 是**模块级函数**（本仓现状 ✓），也可能是类方法 ✓
                _loader = getattr(_mod, "load", None) or getattr(_mod.ValueBaseline, "load", None)
                _vb0 = _loader(_vi, torch=TORCH, np=np)
                _din = int((getattr(_vb0, "meta", None) or {}).get("din") or 0)
                if _din != int(A.STATE_DIM):
                    print("      [value-init] ✗ 维度不符：ckpt din=%d ≠ 当前 %d ⇒ **忽略基线**"
                          "（行为不变 ✓）" % (_din, int(A.STATE_DIM)), flush=True)
                    _VB_CACHE = None
                else:
                    _VB_CACHE = _vb0
                    _sc = float(getattr(_vb0, "scale", 1.0))
                    print("      [value-init] ✓ 启用独立价值基线 %s（din=%d，scale=%.4f）"
                          "⇒ advantage 的 baseline 改用它" % (_vi, _din, _sc), flush=True)
                    if abs(_sc - 1.0) < 1e-9:
                        # ★ 单位守卫（吃过两次亏 ✗）：训练器把**目标除以 10** 后再学 ⇒
                        #   价值头原始输出与 return 差一个 ≈10 的因子。若 `scale` 未拟合
                        #   （恒为 1.0）⇒ baseline 会**差一个量级** ✗ ⇒ 必须吼出来 ✓
                        print("      [value-init] ⚠⚠ `scale` 未拟合（=1.0）⇒ baseline 可能差 ≈10 倍！"
                              "先用 `ValueBaseline.fit_scale(X, y)` 拟合并重存 ckpt，再重跑 ✓"
                              "（本次仍按现 scale 运行，但请把这条当成未完成项 ✗）", flush=True)
            except Exception as _e:            # noqa: BLE001
                print("      [value-init] ✗ 载入失败：%r ⇒ **忽略基线**（行为不变 ✓）" % (_e,),
                      flush=True)
                _VB_CACHE = None
    _vb_use = _VB_CACHE

    st = {"loss": 0.0, "vloss": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0,
          "ratio_abs": 0.0, "n": 0}
    eps = float(hp["clip"])
    temp = float(hp["temp"] or 1.0)
    for kind in ("turn", "win"):
        b = batch[kind]
        if len(b["a"]) == 0:
            continue
        r = b["r"]
        # ★★★★★ 第 767+ 轮：**return 标准化的口径可切换**（默认 `batch` = 逐字保持旧行为 ✓）
        #   动机（读代码 + 观测得到，`GOAL_STATUS.md §Z.833`）：
        #     · 旧实现 `mu, sd = r.mean(), r.std()` 是**按本批**算的 ⇒
        #       同一个状态的回归目标在**批与批之间会漂移**（这批 +1、下批 −1）✗
        #     · 而 `value_loss` 长期停在 **0.66~0.91**（标准化目标下"完全预测不了"= 1.0
        #       ⇒ R² 仅 0.1~0.34）且**从不下降** ✗ ⇒ 与"批评家在追一个移动靶"一致 ✓
        #   三个口径：
        #     `batch`  （默认）逐批标准化 —— 旧行为 ✓ 可比 ✓
        #     `running`      跨批 **EMA** 标准化 ⇒ 目标近似平稳 ⇒ 批评家能真的收敛 ✓
        #     `none`         不标准化 ⇒ 保留绝对尺度 ✓
        _rn_mode = str(hp.get("ret_norm", "batch") or "batch")
        if _rn_mode == "none":
            rn = TORCH.from_numpy(r.astype(np.float32)).to(dev)
        elif _rn_mode == "running":
            # ⚠ 必须**按 kind 分开**统计：`turn` 与 `win` 两个头的 return 尺度不同 ✗
            #   （共用一份 EMA ⇒ 两个头互相污染 ⇒ 目标仍是错的）
            _rs = hp.setdefault("_rs_" + kind, {"mean": 0.0, "var": 1.0, "init": False})
            _bm, _bv = float(r.mean()), float(r.var())
            if not _rs["init"]:
                _rs["mean"], _rs["var"], _rs["init"] = _bm, max(_bv, 1e-6), True
            else:
                _a = 0.05                      # EMA 的慢漂移 ⇒ 目标近似平稳 ✓
                _rs["mean"] = (1 - _a) * _rs["mean"] + _a * _bm
                _rs["var"] = (1 - _a) * _rs["var"] + _a * max(_bv, 1e-6)
            rn = TORCH.from_numpy(((r - _rs["mean"]) / (_rs["var"] ** 0.5 + 1e-6))
                                  .astype(np.float32)).to(dev)
        else:
            mu, sd = float(r.mean()), float(r.std() + 1e-6)
            rn = TORCH.from_numpy(((r - mu) / sd).astype(np.float32)).to(dev)
        S = TORCH.from_numpy(b["s"]).to(dev)
        _tick("%s: 传 S(%d×%d) 上卡" % (kind, b["s"].shape[0], b["s"].shape[1]))
        # ★ ③ 独立价值基线：**每个 update batch 只算一次**（不随 epoch 重复 ✓）。
        #   `_vbv` 形状 (N,) ⇒ 与 `rn[sel]` 同形 ⇒ 直接相减 ✓。`None` ⇒ 走原路径 ✓。
        if _vb_use is not None:
            with TORCH.no_grad():
                _vbv = TORCH.from_numpy(
                    np.asarray(_vb_use.predict(b["s"]), dtype=np.float32)).to(dev).reshape(-1)
        else:
            _vbv = None
        C = TORCH.from_numpy(b["c"]).to(dev)
        M = TORCH.from_numpy(b["m"]).to(dev)
        ACT = TORCH.from_numpy(b["a"]).to(dev)
        OLD = TORCH.from_numpy(b["lp"]).to(dev)
        _tick("%s: 传 C/M/ACT/OLD 上卡" % kind)
        n = len(b["a"])
        for _ep in range(int(hp["epochs"])):
            perm = TORCH.randperm(n)
            _tick("%s: epoch %d randperm(%d)" % (kind, _ep + 1, n))
            for i in range(0, n, int(hp["batch"])):
                sel = perm[i:i + int(hp["batch"])]
                logits, val = net(kind, S[sel], C[sel], M[sel])
                lsm = TORCH.log_softmax(logits / temp, dim=-1)
                logp = lsm.gather(1, ACT[sel].unsqueeze(1)).squeeze(1)
                ratio = TORCH.exp(logp - OLD[sel])
                with TORCH.no_grad():
                    # ★ ③：baseline = 独立价值头预测（若已启用且维度相符）；否则 = 网络自带 value ✓
                    _base = _vbv[sel] if _vbv is not None else val.detach()
                    adv = rn[sel] - _base
                    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                obj = TORCH.min(ratio * adv,
                                TORCH.clamp(ratio, 1 - eps, 1 + eps) * adv)
                p = TORCH.softmax(logits / temp, dim=-1)
                ent = -(p * lsm).sum(-1).mean()
                loss = (-obj.mean() + float(hp["vf_coef"]) * 0.5
                        * ((val - rn[sel]) ** 2).mean()
                        - float(hp["ent_coef"]) * ent)
                # ★★ 2026-10-06（§Z.1554）：**自模仿项**（Self-Imitation Learning ✓）
                #   对「结果好于基线」（`adv > 0` ✓ 注意 `adv` 已归一化 ⇒ `>0` = 优于均值 ✓）
                #   的决策点，**额外做一次监督**：`+ β · mean(-logp | adv>0)` ✓
                #   · 为什么这样有效 ✓：PPO 的 `adv` 在**手内被同一个 `R_hand` 抹平** ✗（手内无区分度 ✗）
                #     ⇒ 而自模仿**只挑出"确实比基线好"的那些决策** ✓ 直接给它们一个**低方差**的
                #       监督信号 ✓ ⇒ 部分绕开了信用分配问题 ✓
                #   · 语义 = 本仓既有结论「**用【结果】当老师**」✓（`:18472` ✓）
                #   · ★ 默认 `β = 0.0` ⇒ 本段**完全不执行** ⇒ 与既有行为**逐字一致** ✓
                _si = float(hp.get("self_imit", 0.0) or 0.0)
                if _si > 0.0:
                    _m = (adv > 0.0).to(logp.dtype)
                    _nn = float(_m.sum())
                    if _nn > 0:
                        sil = -((logp * _m).sum() / _m.sum().clamp_min(1.0))
                        loss = loss + _si * sil
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                _tick("%s: ep%d step@%d 前向+反传+更新" % (kind, _ep + 1, i))
                with TORCH.no_grad():
                    k = len(sel)
                    st["loss"] += float(loss.detach()) * k
                    st["vloss"] += float(((val - rn[sel]) ** 2).mean()) * k
                    st["ent"] += float(ent) * k
                    st["kl"] += float((OLD[sel] - logp).mean()) * k
                    st["clip"] += float(((ratio - 1).abs() > eps).float().mean()) * k
                    st["ratio_abs"] += float((ratio - 1).abs().mean()) * k
                    st["n"] += k
    for k in ("loss", "vloss", "ent", "kl", "clip", "ratio_abs"):
        st[k] = st[k] / max(1, st["n"])
    return st


def _tile_batch(b, k):
    """把 batch 按行复制 k 倍（只用于**设备计时**，不改变内容分布）。"""
    out = {}
    for kind in b:
        out[kind] = {}
        for kk, vv in b[kind].items():
            out[kind][kk] = np.tile(vv, (k,) + (1,) * (vv.ndim - 1)) if len(vv) else vv
    return out


def make_batch(res, dev=None):
    """把多个 worker 的样本拼成一个 batch（含合法性掩码）。"""
    batch = {}
    for kind in ("turn", "win"):
        dim = A.MAX_TURN_CAND if kind == "turn" else A.N_WIN_CAND
        s = np.concatenate([r[kind]["s"] for r in res]) if res else _zero((0, A.STATE_DIM))
        c = np.concatenate([r[kind]["c"] for r in res]) if res else _zero((0, dim, 1))
        n = np.concatenate([r[kind]["n"] for r in res]) if res else np.zeros(0, np.int64)
        mask = (np.arange(dim)[None, :] < n[:, None]) if len(n) else \
            np.zeros((0, dim), dtype=bool)
        batch[kind] = {"s": s, "c": c, "n": n,
                       "a": np.concatenate([r[kind]["a"] for r in res]) if res
                       else np.zeros(0, np.int64),
                       "lp": np.concatenate([r[kind]["lp"] for r in res]) if res
                       else _zero((0,)),
                       "r": np.concatenate([r[kind]["r"] for r in res]) if res
                       else _zero((0,)),
                       "m": mask}
    return batch


# ================================================================ 并行（子进程 + 文件回传）
#
# ★★ 为什么**不用** `multiprocessing.Pool`（本机沙箱实测的硬约束，别改成 Pool）：
#   Windows 上 `Pool()` 的 IPC 走**命名管道**（`multiprocessing.connection.Pipe`
#   → `_winapi.CreateFile(r"\\.\pipe\pyc-…")`），而本沙箱（workspace-write）
#   禁止打开命名管道 ⇒ 实测 `PermissionError: [WinError 5] 拒绝访问`。
#   而 `subprocess` 用**文件句柄**接 stdio 是允许的（`_tmp/probe_subprocess.py`
#   实测：带 `stdout=<file>` 的 Popen rc=0；`capture_output=True` 则同样 WinError 5）。
#   ⇒ 并行只能走"一个子进程一个任务 + 结果写文件"，代价是每轮多一次进程启动
#     （≈5~8s 的 torch import，子进程并发所以只付一次）。
def _tail(path, n=25):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(读不到日志)"


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def run_children(mode, jobs, tag, log=None, keep=False, timeout=None):
    """并行跑 `jobs`（每个 job 一个子进程），返回结果列表（顺序与 jobs 一致）。"""
    os.makedirs(TMP, exist_ok=True)
    procs = []
    for i, job in enumerate(jobs):
        tp = os.path.join(TMP, "%s_task_%d.pt" % (tag, i))
        op = os.path.join(TMP, "%s_out_%d.pt" % (tag, i))
        lp = os.path.join(TMP, "%s_log_%d.txt" % (tag, i))
        TORCH.save({"mode": mode, "job": job, "out": op}, tp)
        lf = open(lp, "w", encoding="utf-8")
        p = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "train_selfplay_rl.py"),
             "--run-task", tp],
            stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
        procs.append((p, lf, op, lp, tp))
    if log is not None:
        log("      [并行] %s：拉起 %d 个子进程（mode=%s）" % (tag, len(procs), mode))
    res, t0 = [], time.time()
    # ★★ 第 438 轮新增：**采集阶段的进度心跳**。
    #   为什么需要（工具缺陷，害我误判两次）：
    #     原实现是 `for p in procs: p.wait()` —— 在**全部子进程结束之前一个字都不打**，
    #     于是日志永远停在"拉起 6 个子进程"。
    #     我因此两次以为"还在采集"，而其中一次**其实已经进到 epoch 27 了**（§Z.437）。
    #   ⇒ 现在每 30 秒报一次"已完成 k/n"，并把每个子进程日志的**最后一行**带出来
    #     （子进程里 `bc_task` 也会定期打进度，见那里的 `_prog`）。
    import threading as _th
    done_flag = {"n": 0}
    stop_hb = _th.Event()

    def _heartbeat():
        while not stop_hb.wait(30.0):
            k = sum(1 for (pp, _lf, _op, _lp, _tp) in procs if pp.poll() is not None)
            done_flag["n"] = k
            tails = []
            for (_pp, _lf, _op, lp2, _tp) in procs:
                t = _tail(lp2, n=1).strip()
                if t:
                    tails.append(t)
            if log is not None:
                log("      [进度 %s] 子进程完成 %d/%d ｜ 用时 %.0fs ｜ %s"
                    % (tag, k, len(procs), time.time() - t0,
                       " ／ ".join(tails[:6]) if tails else "(暂无子进程输出)"))

    hb = _th.Thread(target=_heartbeat, daemon=True)
    hb.start()
    for p, lf, op, lp, tp in procs:
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -999
        lf.close()
        if rc != 0 or not os.path.exists(op):
            stop_hb.set()
            raise RuntimeError("子进程失败 rc=%s（任务 %s）\n--- 子进程日志尾部 ---\n%s"
                               % (rc, tp, _tail(lp)))
        res.append(TORCH.load(op, map_location="cpu", weights_only=False))
        if not keep:
            _rm(op)
    stop_hb.set()
    if log is not None:
        log("      [并行] %s：全部 %d 个子进程完成，用时 %.0fs"
            % (tag, len(procs), time.time() - t0))
    for (_p, _lf, _op, _lp, tp) in procs:
        _rm(tp)
    return res


def cleanup_tmp(keep=False):
    """清掉 `rl_tmp/` 里的中转文件（保留目录本身）。"""
    if keep or not os.path.isdir(TMP):
        return
    n = 0
    for fn in os.listdir(TMP):
        if any(s in fn for s in ("_task_", "_out_", "_log_")) and \
                fn.endswith((".pt", ".txt")):
            _rm(os.path.join(TMP, fn))
            n += 1
    return n


def run_task_main(task_path):
    """子进程入口：跑一个任务并把结果写回文件。"""
    d = TORCH.load(task_path, map_location="cpu", weights_only=False)
    mode, job, out = d["mode"], d["job"], d["out"]
    if mode == "rollout":
        res = rollout_task(job)
    elif mode == "eval":
        res = eval_task(job)
    elif mode == "bc":
        res = bc_task(job)
    else:
        raise ValueError("未知 mode %r" % (mode,))
    TORCH.save(res, out)
    return 0


def cfg_from_ckpt(cfg, ck, where=""):
    """把**检查点里的结构参数**覆盖进 `cfg`，返回被改动的项（供打印）。

    ★★★ 第 473 轮：把一段**被写了三遍**的逻辑抽成唯一实现。

    为什么必须有它（本会话的真实事故链）：
    ```
      同一个陷阱 = "**结构来自命令行默认、权重来自检查点**" ⇒ 必然 size mismatch。
        ① `--eval-ckpt`      §Z.439 查过，本来就是"以检查点为准" ✅
        ② `--init`           §Z.469 **实测崩了**（11 处 shape 不符）才补上
        ③ 两臂都用 `ckpt:`    §Z.471 预检时才发现，原来只读了 `sd`、没带 `cfg`
      ⇒ 三次都犯同一个错，根因不是"没查"，而是**没有单一实现**：
        每加一个入口就重写一遍，于是"上一处修好了"对下一处毫无约束。
    ```
    ⇒ 本函数是那个**单一实现**。以后任何"从检查点取权重"的入口都必须经过它。
    ⚠ 只覆盖**结构**类键（hidden/state_dim/cand_*_dim/temp），不动超参（lr/epochs/…）——
      结构必须与权重一致，超参是本次训练的选择，两者不能混。
    """
    c2 = (ck or {}).get("cfg") or {}
    up = {k: c2[k] for k in ("hidden", "state_dim", "cand_turn_dim",
                             "cand_win_dim", "temp") if k in c2}
    bad = {k: (cfg.get(k), v) for k, v in up.items() if cfg.get(k) != v}
    if up:
        cfg.update(up)
    if bad and where:
        print("[cfg] %s：结构参数以检查点为准 %s" % (where, bad))
    return bad


def pick_dev(args):
    """设备选择：`--dev auto|cuda|cpu`。`auto` = `MJ_USE_GPU=1` 且 cuda 可用时用 GPU。

    ⚠ 为什么要有 `cpu` 这一档：`--bench-device` 实测在本网络规模下
      **CPU 的 1 遍 PPO 更新比 GPU 快约 2.3 倍**（1e5 参数的小 MLP，GPU 的 kernel
      启动 + 传输把它吃回去了）。GPU 路径本身是通的（自检有断言），
      但要它赚回来需要更大的网络/批量。默认仍按任务要求走 `auto`（即 GPU）。
    """
    want = getattr(args, "dev", "auto")
    if want == "cpu":
        return "cpu"
    if want == "cuda":
        if not TORCH.cuda.is_available():
            raise SystemExit("--dev cuda 但 torch.cuda.is_available() = False")
        return "cuda"
    return rl_boot.pick_device(TORCH)



# ================================================================ 死输入审计（R1 验收）
def _dead_audit(rooms, rounds, room_mode, first_dealer, nproc=3, log=None):
    """跑若干房间，收集**被学习席**的状态向量，逐维 std，报死维数（按段拆开）。

    这是父 agent `p297_dead.txt` 那个审计的等价物（分段口径照 `features_full.F` 的布局）：
      手牌 34 / 未见 34 / 逐家弃牌 136 / 逐家副露 136 / 副露数 4 / 座位 4 /
      标量 12 / CTX 10（= 累计分 4 + 分差 3 + 名次 1 + 连庄 1 + 局号 1）
    """
    cfg = dict(P.DEFAULT_CFG, hidden=32)
    sd = P.sd_to_cpu(P.make_net(cfg, seed=0).state_dict())
    jobs = []
    for b in split_blocks(range(rooms), nproc):
        jobs.append({"sd": sd, "cfg": dict(cfg, gamma=1.0), "base_genome": BASE_GENOME,
                     "pool": ["live"], "sds": {}, "seed0": 7000000 + b[0],
                     "n": len(b), "gi0": b[0], "aux": False, "aux_scale": 0.3,
                     "aux_lam": None, "rounds": rounds, "room_mode": room_mode,
                     "first_dealer": first_dealer})
    res = run_children("rollout", jobs, "dead", log=log)
    S = np.concatenate([r["turn"]["s"] for r in res] +
                       [r["win"]["s"] for r in res])
    n = len(S)
    std = S.std(axis=0)
    segs = [("手牌", 0, 34), ("未见牌", 34, 68), ("逐家弃牌", 68, 204),
            ("逐家副露", 204, 340), ("副露数", 340, 344), ("座位 one-hot", 344, 348),
            ("标量", 348, 360), ("CTX（分/差/名次/连庄/局号）", 360, 370)]
    out = {"n": n, "dead": int((std < 1e-9).sum()), "segs": []}
    for name, a, b in segs:
        d = int((std[a:b] < 1e-9).sum())
        out["segs"].append((name, b - a, d, float(std[a:b].min()),
                            float(std[a:b].max())))
    out["ctx_std"] = [float(x) for x in std[360:370]]
    out["std"] = std
    return out


def dead_inputs(args):
    """`--dead-inputs N`：对**开/关房间层**两种配置各跑 N 个房间，打印对照表。"""
    log = Log(args.log)
    nproc = args.nproc if args.allow_more else min(args.nproc, MAX_NPROC)
    log("=" * 78)
    log("=== 死输入审计（R1 验收：CTX 必须活）===")
    log("  配置 A = `--no-room`（旧口径：一手一局、不传 carry/streak/round_no）")
    log("  配置 B = 房间层（Rounds=%d，庄位继承，carry_scores/dealer_streak/round_no/rounds 全传）"
        % args.rounds)
    res = {}
    for name, room_mode in (("A 单局", False), ("B 房间层", True)):
        rr = args.dead_inputs if room_mode else args.dead_inputs * args.rounds
        o = _dead_audit(args.dead_inputs, args.rounds, room_mode,
                        args.first_dealer, nproc, log=log)
        res[name] = o
        log("")
        log("  --- 配置 %s：%d 个决策点，370 维中死维 %d（%.1f%%）---"
            % (name, o["n"], o["dead"], 100.0 * o["dead"] / 370.0))
        log("    %-34s %5s %6s %10s %10s" % ("维度段", "维数", "死维", "min std", "max std"))
        for nm, w, d, lo, hi in o["segs"]:
            log("    %-34s %5d %6d %10.6f %10.6f%s"
                % (nm, w, d, lo, hi, "   ← **整段全死**" if d == w else ""))
        log("    CTX 10 维逐维 std: %s" % " ".join("%.4f" % x for x in o["ctx_std"]))
        log("    ⇒ CTX 死维 = %d / 10" % int(sum(1 for x in o["ctx_std"] if x < 1e-9)))
    log("")
    log("=== 结论 ===")
    ca = int(sum(1 for x in res["A 单局"]["ctx_std"] if x < 1e-9))
    cb = int(sum(1 for x in res["B 房间层"]["ctx_std"] if x < 1e-9))
    log("  CTX 死维：配置 A = %d/10，配置 B = %d/10 ⇒ %s"
        % (ca, cb, "✅ R1 的验收成立（房间层把 CTX 变活）" if cb == 0
           else "**✗ 还有 %d 维是死的，先查**" % cb))
    log.close()
    return 0 if cb == 0 else 1


def room_crosscheck(args, log, rooms=16):
    """对拍：本训练器的 `_play_room` vs `mj_room.play_room` 的逐手 rows。

    ★ 必须传**同一个首手庄家**：`mj_room.play_room` 默认 `first_dealer = seed % 4`，
      本训练器默认 `--first-dealer rot` = `(seed//4)%4`。
      第一版漏了这个，于是 96 处"不一致"**全是这个约定差异**（不是循环漂移）——
      记在这里，免得下次又把它读成 bug。

    判据：逐字段**完全相同**。差异分两类：
      · **硬差异**（wall_seed / scores / total / dealer / dealer_streak / draw）
        ⇒ 必须为 0（= 循环真的是照抄的）；
      · **schema 差异**（winner / fan）：`mj_room` 把 `mj_sim` 的流局 `-1` **原样透传**，
        若哪一侧改成 `None` 就会出现 —— 允许，但**必须打印出来**（`p302` 那次修的就是这个语义）。
    另外断言两边的 `dealer ∉ 0..3` 手数都为 0（流局连庄 ⇒ 庄家不变）。
    """
    from mj_arena import make_strategy
    from mj_room import play_room
    rounds = args.rounds if not args.no_room else 1
    FIELDS = ("wall_seed", "winner", "scores", "total", "dealer_streak", "fan",
              "draw", "dealer")
    SCHEMA = ("winner", "fan")
    n_rooms = n_hands = draw_hands = 0
    hard = schema = my_neg = mj_neg = 0
    hard_f, schema_f, examples = {}, {}, []
    for s in range(rooms):
        fd = (s // 4) % 4 if args.first_dealer == "rot" else s % 4
        mine, _fb = _play_room(s, [make_strategy(BASE_GENOME) for _ in range(4)],
                               rounds, room_mode=not args.no_room,
                               first_dealer=args.first_dealer)
        theirs = play_room(s, [make_strategy(BASE_GENOME) for _ in range(4)],
                           rounds, mode="succ", first_dealer=fd)
        n_rooms += 1
        for x, y in zip(mine, theirs):
            n_hands += 1
            draw_hands += 1 if x["draw"] else 0
            my_neg += 1 if x["dealer"] < 0 else 0
            mj_neg += 1 if y["dealer"] < 0 else 0
            for f in FIELDS:
                if x[f] != y[f]:
                    if f in SCHEMA:
                        schema += 1
                        schema_f[f] = schema_f.get(f, 0) + 1
                    else:
                        hard += 1
                        hard_f[f] = hard_f.get(f, 0) + 1
                    if len(examples) < 3:
                        examples.append((s, x["round"], f, x[f], y[f]))
    ok = (hard == 0 and my_neg == 0 and mj_neg == 0)
    msg = ("%d 房 / %d 手（流局 %d 手）：硬差异 = %d %s；schema 差异 = %d %s；"
           "dealer<0：本实现 %d / mj_room %d"
           % (n_rooms, n_hands, draw_hands, hard, hard_f or "无", schema,
              schema_f or "无", my_neg, mj_neg))
    if log is not None:
        log("      %s" % msg)
        if examples:
            log("      差异样例（房, 手, 字段, 本实现, mj_room）= %s" % examples)
        if schema:
            log("      （winner/fan 的差异 = **schema 约定**：`mj_room` 透传 `mj_sim` 流局的 "
                "-1，本实现若归一成 None 就会在这里露出来；不计入硬差异）")
        log("      ✅ 循环与 `mj_room.play_room` 逐字段一致（首手庄家已对齐）" if ok
            else "      ✗ 有硬差异 ⇒ 循环漂移了，先修")
    return {"ok": ok, "msg": msg, "neg_dealer": my_neg, "mj_neg_dealer": mj_neg,
            "hard": hard, "schema": schema, "hard_fields": hard_f,
            "mj_msg": ("mj_room dealer<0 = %d" % mj_neg) if mj_neg else "无"}


def reaggregate(args):
    """从 npz 重新打印配对评估读数（改判读口径不必重算 —— 一次 815s 的教训）。"""
    log = Log(args.log)
    z = np.load(args.reaggregate)
    d = z["delta"]
    dr = z["delta_room"]
    idd = z["is_dealer"]
    fl = z["fork_learner"]
    fa = z["fork_all"]
    n, n_rooms = len(d), len(dr)
    mu = float(d.mean()) if n else float("nan")
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    nb = max(1, min(args.eval_blocks, max(1, n_rooms)))
    bmeans = [float(np.mean(x)) / max(1, args.rounds) for x in np.array_split(dr, nb)] \
        if n_rooms else []
    se_block = float(np.std(bmeans, ddof=1) / np.sqrt(len(bmeans))) if len(bmeans) > 1 \
        else float("nan")

    def _strat(mask):
        x = d[mask]
        if len(x) == 0:
            return float("nan"), float("nan"), 0
        se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
        return float(x.mean()), se, len(x)

    mu_de, se_de, n_de = _strat(idd == 1)
    mu_nd, se_nd, n_nd = _strat(idd == 0)
    out = {"n": n, "n_rooms": n_rooms, "mu": mu, "sd": sd, "ci95": float(ci),
           "lo": mu - ci, "hi": mu + ci,
           "se_naive": float(sd / np.sqrt(n)) if n else float("nan"),
           "se_block": se_block, "n_blocks": len(bmeans),
           "lo_block": mu - 1.96 * se_block, "hi_block": mu + 1.96 * se_block,
           "mu_room": float(dr.mean()) if n_rooms else float("nan"),
           "mu_dealer": mu_de, "se_dealer": se_de, "n_dealer": n_de,
           "mu_nodealer": mu_nd, "se_nodealer": se_nd, "n_nodealer": n_nd,
           "fork_learner": float(fl.mean()) if n else float("nan"),
           "fork_all": float(fa.mean()) if n else float("nan"),
           "canon": int(z["canon"]) if "canon" in z else 0,
           "fallbacks": -1, "len_a": float("nan"), "len_b": float("nan"), "secs": 0.0,
           "mean_a": float("nan"), "mean_b": float("nan")}
    log("=== 从 %s 重新聚合 ===" % args.reaggregate)
    log.report_eval("", out)
    log.close()
    return 0


# ================================================================ 自检
def selfcheck(args):
    """正确性自检：**这一节比训练本身重要**（训练不学只是没收益，自检不过则是假数据）。"""
    log = Log(args.log)
    ok = True

    def chk(name, good, extra=""):
        nonlocal ok
        ok = ok and bool(good)
        log("  %s %-56s %s" % ("ok  " if good else "FAIL", name, extra))

    log("=" * 78)
    log("=== 自我对弈 RL 训练器 · 正确性自检 ===")
    dev = rl_boot.pick_device(TORCH)
    log("  " + rl_boot.device_report(TORCH, dev))
    # ★ 这一条**必须能失败**：上一版写成 `(MJ_USE_GPU != "1") or (...)` ⇒ 不设变量时
    #   恒为真、照印 "ok"，而实际跑在 CPU 上（假通过）。修法：三条分支分开说。
    gpu_want = os.environ.get("MJ_USE_GPU") == "1"
    gpu_ok = TORCH.cuda.is_available() and "+cpu" not in TORCH.__version__
    if gpu_want and not gpu_ok:
        chk("MJ_USE_GPU=1 且拿到 CUDA 版 torch 且 cuda 可用", False,
            "**FAIL**：torch %s / cuda=%s ⇒ sys.path 顺序或被沙箱挡了，"
            "别用这个配置训练" % (TORCH.__version__, TORCH.cuda.is_available()))
        log("  ⇒ GPU 自检失败，**停止**（不自欺：CPU 上跑 RL 会慢一个量级）。")
        log.close()
        return 2
    if gpu_want:
        chk("MJ_USE_GPU=1 且拿到 CUDA 版 torch 且 cuda 可用", True,
            "torch %s / %s" % (TORCH.__version__,
                               TORCH.cuda.get_device_name(0)))
    else:
        log("  ⚠⚠ **未设 MJ_USE_GPU=1 ⇒ 正在用 CPU 训练**（本机有 RTX 4060 Ti，"
            "_pkgs_gpu 已就绪）")
        log("     本自检的 GPU 项标为 **N/A（未启用）**，不是 ok —— "
            "上一版在这里假通过过。")
        log("     torch %s / cuda.is_available=%s / 设备=%s"
            % (TORCH.__version__, TORCH.cuda.is_available(), dev))

    log("  --- ① 动作空间编码层（rl_action_space._selftest）---")
    chk("编码层自测", A._selftest(10))

    cfg = dict(P.DEFAULT_CFG, hidden=64)
    hp = {"clip": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "temp": 1.0,
          "epochs": 1, "batch": 512, "self_imit": 0.0}
    net = P.make_net(cfg, seed=1)
    sd = P.sd_to_cpu(net.state_dict())
    nw = min(3, MAX_NPROC)
    clean = not args.keep_tasks
    try:
        log("  --- ② 合法集 vs 引擎（AuditedGame._fallback_discard 计数）---")
        t = run_children("rollout", [{
            "sd": sd, "cfg": dict(cfg, gamma=1.0), "base_genome": BASE_GENOME,
            "pool": ["live"], "sds": {}, "seed0": 7000000, "n": 8, "gi0": 0,
            "aux": False, "aux_scale": 0.3, "aux_lam": None,
            "rounds": args.rounds, "room_mode": not args.no_room,
            "first_dealer": args.first_dealer}], "sc_roll", log=log,
            keep=not clean)[0]
        chk("回合动作被引擎拒绝 / 策略抛异常的次数 = 0",
            t["fallbacks"] == 0 and t["errors"] == 0,
            "fallbacks=%d errors=%d（>0 ⇒ 掩码与引擎分叉）" % (t["fallbacks"], t["errors"]))
        chk("决策点样本非空", t["n_pts"] > 0,
            "%d 房间 / %d 手 / %d 决策点 / %.1fs（%.2f 手/秒·单进程）"
            % (t["n_rooms"], t["n_games"], t["n_pts"], t["secs"],
               t["n_games"] / max(1e-9, t["secs"])))

        log("  --- ③ 只用 terminal 的纯度（--aux 关闭时）---")
        chk("max|return − R_terminal| < 1e-6", t["pure_gap"] < 1e-6,
            "max|G−R| = %.3g" % t["pure_gap"])

        log("  --- ④ 环境/对手的确定性（配对评估的前提）---")
        fixture = parse_pool(FIXTURE_OPP, BASE_GENOME)
        base_task = {"seeds": list(range(12)), "arm_a": None, "arm_b": None,
                     "opp_specs": fixture, "base_genome": BASE_GENOME,
                     "cfg": cfg, "sds": {}, "rounds": args.rounds,
                     "room_mode": not args.no_room, "first_dealer": args.first_dealer}
        e1, e2 = run_children("eval", [base_task, base_task], "sc_eval", log=log,
                              keep=not clean)
        # ★ 先断言 n>0：空数组下 `.max()` 会抛异常、而 `np.array_equal` 会**空真通过**
        chk("空对照样本数 n > 0（防「空真通过」）", len(e1["delta"]) > 0,
            "n=%d（对手 = %s）" % (len(e1["delta"]), FIXTURE_OPP))
        chk("两臂都是线上基因组时 Δ 恒为 0",
            len(e1["delta"]) > 0 and float(np.abs(e1["delta"]).max()) == 0.0,
            "max|Δ| = %s" % (float(np.abs(e1["delta"]).max())
                             if len(e1["delta"]) else "n/a"))
        chk("两臂都是线上基因组时分叉率恒为 0",
            float(e1["fork_all"].mean()) == 0.0 and float(e1["fork_learner"].mean()) == 0.0,
            "fork_all=%.4f fork_learner=%.4f"
            % (float(e1["fork_all"].mean()), float(e1["fork_learner"].mean())))
        chk("同 seed 重跑逐字复现（无隐藏随机源）",
            len(e1["sa"]) > 0 and len(e2["sa"]) > 0
            and bool(np.array_equal(e1["sa"], e2["sa"])
                     and np.array_equal(e1["sb"], e2["sb"])),
            "两次跑 60 seed 的得分序列完全相同" if
            np.array_equal(e1["sa"], e2["sa"]) else "**有随机源**")

        log("  --- ④b 房间层对拍 `mj_room.play_room`（口径不许分叉）---")
        droom = room_crosscheck(args, log)
        chk("房间层逐手 rows 与 `mj_room.play_room` 一致（只允许流局那一项不同）",
            droom["ok"], droom["msg"])
        chk("流局时庄家**不变成 -1**（`mj_room` 的 `w is None` 分支进不去）",
            droom["neg_dealer"] == 0,
            "本实现 dealer<0 的手数 = %d；`mj_room` 的 = %d（%s）"
            % (droom["neg_dealer"], droom["mj_neg_dealer"], droom["mj_msg"]))

        log("  --- ⑤ 阳性对照：评估口径是活的吗（必须能测出已知差异）---")
        c = paired_eval(list(range(600000, 600030)), "live",
                        "genome:first:0:0:0:0:0:0:0", fixture, BASE_GENOME, cfg, {},
                        nw, log=None, tmp_tag="sc_pos", rounds=args.rounds,
                        room_mode=not args.no_room, first_dealer=args.first_dealer)
        log("      live vs discard=first：Δμ = %+.3f  95%%CI [%+.3f, %+.3f]  "
            "分叉率 %.4f  n=%d" % (c["mu"], c["lo"], c["hi"], c["fork_learner"], c["n"]))
        chk("阳性对照：分叉率 > 0.5 且 CI 不含 0",
            c["fork_learner"] > 0.5 and (c["lo"] > 0 or c["hi"] < 0),
            "fork=%.3f CI=[%+.2f,%+.2f]" % (c["fork_learner"], c["lo"], c["hi"]))

        log("  --- ⑥ 训练侧张量设备（把漏 .to(dev) 的坑抓出来）---")
        try:
            net2 = P.make_net(cfg).to(dev)
            opt = TORCH.optim.Adam(net2.parameters(), lr=1e-3)
            b0 = make_batch([t])
            if args.bench_device:
                for nd, nm in ((dev, dev), ("cpu", "cpu")):
                    nn_ = P.make_net(cfg).to(nd)
                    o2 = TORCH.optim.Adam(nn_.parameters(), lr=1e-3)
                    bb = _tile_batch(b0, max(1, 16384 // max(1, len(b0["turn"]["a"]))))
                    tt = time.time()
                    ppo_update(nn_, o2, bb, hp, nd)
                    log("      设备实测 %-4s：%d 个决策点的 1 遍更新 = %.2fs（%.0f 样本/秒）"
                        % (nm, len(bb["turn"]["a"]), time.time() - tt,
                           len(bb["turn"]["a"]) / max(1e-6, time.time() - tt)))
            st = ppo_update(net2, opt, b0, hp, dev)
            chk("PPO 更新在 %s 上跑通（无 cpu/cuda 混用）" % dev,
                bool(np.isfinite(st["loss"])),
                "loss=%+.4f value_loss=%.4f kl=%+.5f clip=%.3f ent=%.4f"
                % (st["loss"], st["vloss"], st["kl"], st["clip"], st["ent"]))
        except Exception as e:                                   # noqa: BLE001
            chk("PPO 更新在 %s 上跑通" % dev, False, repr(e))
    finally:
        cleanup_tmp(keep=not clean)

    log("")
    log("=== 自检结论：%s ===" % ("全部通过 ✅" if ok
                                 else "**有失败项 ❌ —— 先修它，别拿去训练**"))
    log("=" * 78)
    log.close()
    return 0 if ok else 1


# ================================================================ 训练
def train(args):
    log = Log(args.log)
    dev = rl_boot.pick_device(TORCH)
    nproc = args.nproc
    if nproc > MAX_NPROC and not args.allow_more:
        log("!! --nproc %d 超过硬上限 %d（线上客户端有 1s 决策窗口）⇒ 收到 %d"
            % (nproc, MAX_NPROC, MAX_NPROC))
        nproc = MAX_NPROC

    cfg = dict(P.DEFAULT_CFG)
    cfg.update({"hidden": args.hidden, "temp": args.temp})
    hp = {"clip": args.clip, "ent_coef": args.ent_coef, "vf_coef": args.vf_coef,
          "temp": args.temp, "epochs": args.epochs, "batch": args.batch,
            # ★ ③ 独立价值基线：只传**路径字符串**（跨进程安全 ✓）；默认空 ⇒ 行为不变 ✓
            "value_init": getattr(args, "value_init", ""),
          # ★ 第 767+ 轮：return 标准化口径（`batch`=旧行为 ✓ / `running` / `none`）✓
          "ret_norm": getattr(args, "ret_norm", "batch"),
          # ★★ 2026-10-06（§Z.1554）：**自模仿（Self-Imitation Learning）** ✓
          #   动机（本仓实测 ✓ `GOAL_STATUS.md:18454-18455`）：
          #     「奖励只在手末 ⇒ 手内每个决策的 MC 回报**都是同一个 `R_hand`**
          #       ⇒ 开局第一步与将胡牌的决策**方差完全一样** ⇒ 手内无区分度 ⇒ 梯度几乎全是噪声」
          #   ⇒ 自模仿 = ★ 对「**结果好于基线**」（`adv > 0`）的决策**额外做一次监督** ✓
          #     ⇒ 标签来自**结果**（不是老师 ✓）⇒ 天花板不是老师 ✓（`:18451` ✓）
          #     ⇒ 是**监督学习**（本仓唯一被证明能 work 的机制 ✓）⇒ 实现小 ✓（`:18452` ✓）
          #   ★ 默认 **0.0 = 关闭** ⇒ **与既有行为逐字一致** ✓（正在跑的任务不受影响 ✓）
          "self_imit": float(getattr(args, "self_imit", 0.0))}
    # 对手池：`pool[0]` **必须**是 live（`opp_specs_for` 靠这条保证每局恰好一席 live）
    opp_base = parse_pool(args.opp_pool, args.base_genome)
    live_t = ("genome", args.base_genome)
    if not opp_base or opp_base[0] != live_t:
        opp_base = [live_t] + [x for x in opp_base if x != live_t]
    ev_opp = parse_pool(args.fixture_opp, args.base_genome) if args.fixture_opp \
        else [live_t] * 3

    TORCH.manual_seed(args.seed)
    np.random.seed(args.seed)
    # ★★ 第 469 轮：`--init` 必须**从检查点里取结构**，否则一定 size mismatch。
    #   实测（`--init rl_bc6.pt`，该检查点是 hidden=512）：
    #     RuntimeError: size mismatch for trunk.0.weight:
    #       copying a param with shape torch.Size([512, 370]) from checkpoint,
    #       the shape in current model is torch.Size([128, 370]).
    #   ⇒ 根因：这里用 `args.hidden`（默认 128）建网，而检查点由 BC 训练时用 `--hidden 512` 存的。
    #   ⚠ 评测路径（`--eval-ckpt`）**早就**是"从检查点读 cfg"的（§Z.439 专门查过并确认），
    #     而 `--init` 这条路径**没有** —— **同一个隐患我只在一条路径上排除了，另一条漏了**。
    #     ⇒ 修法：与评测路径同一口径（`hidden/state_dim/cand_turn_dim/cand_win_dim/temp`）。
    if args.init:
        _ck0 = TORCH.load(args.init, map_location="cpu", weights_only=False)
        # ★ 结构以检查点为准（唯一实现见 `cfg_from_ckpt`；事故链见那里的注释）
        cfg_from_ckpt(cfg, _ck0, "--init " + os.path.basename(args.init))
        # ★★★ 第 767 轮修正：**state_dim 必须用当前环境的 A.STATE_DIM**
        #   否则 cfg_from_ckpt 会把 state_dim 拉回检查点旧值（如 370），
        #   而 encode_state 按 A.STATE_DIM（如 371）编码 ⇒ 前向时形状不符 ✗（已实测）
        #   对齐后模型按新维度建网 ⇒ load_state_dict 必然不匹配 ⇒ **部分热启动**接管 ✓
        _sd_now = int(getattr(A, "STATE_DIM", 0) or 0)
        if _sd_now and int(cfg.get("state_dim", 0)) != _sd_now:
            log("[init] state_dim 对齐：检查点 %s → 当前 %s（将走部分热启动 ✓）"
                % (cfg.get("state_dim"), _sd_now))
            cfg["state_dim"] = _sd_now
    net = P.make_net(cfg, seed=args.seed).to(dev)
    # ★★★ 第 700 轮：**同 seed 配对的冻结参考策略**（`--ref-pair 1` 时启用）。
    #   参考 = `--init` 的检查点（= 出发时的策略）⇒ 训练目标是 `R_当前 − R_参考`
    #   ⇒ **同 seed、同发牌** ⇒ 发牌运气相消（设计与理由见 `rollout_task` 里的注释）。
    #   ⚠ 未开 `--ref-pair` 或没有 `--init` ⇒ 保持 `None` ⇒ 行为逐字不变 ✓
    ref_sd = None
    if args.init:
        ck0 = TORCH.load(args.init, map_location="cpu", weights_only=False)
        # ★★ 第 690 轮（GOAL_STATUS §Z.1091）：**部分热启动**（按列对齐）。
        #   动机：表示层升维（如 `MJ_CTX_DEALER=1` ⇒ 370→371）后，
        #     旧检查点的 `trunk.0.weight` 是 `[h, 370]`，新网是 `[h, 371]`
        #     ⇒ 原本 `load_state_dict` **响亮崩溃**（`rl_action_space.py:87/91` 警告的那件事 ✓）。
        #     但不必从零训 ✓：**重叠列整列拷贝、新列小随机**即可保留全部既有能力 ✓。
        #   ⚠ 默认惰性：形状一致时走原来的 `load_state_dict` ⇒ 行为**逐字不变** ✓
        #     （只有原本会抛异常的情形才走新分支 ✓ 可回退 ✓）
        try:
            net.load_state_dict(ck0["sd"])
        except RuntimeError as _e:
            _tgt = net.state_dict()
            _copied = _rand = _skip = 0
            for _k, _v in ck0["sd"].items():
                if _k not in _tgt:
                    _skip += 1
                    continue
                _t = _tgt[_k]
                if tuple(_t.shape) == tuple(_v.shape):
                    _t.copy_(_v)
                    _copied += 1
                elif (_t.dim() == 2 and _v.dim() == 2 and _t.shape[0] == _v.shape[0]
                      and _t.shape[1] > _v.shape[1]):
                    _t[:, :_v.shape[1]].copy_(_v)              # 重叠列 ✓
                    _t[:, _v.shape[1]:].normal_(0.0, 0.01)     # 新列小随机 ✓
                    _rand += 1
                else:
                    _skip += 1
            print("[init] 维度不匹配（%s）⇒ **部分热启动**：拷贝 %d ／ 按列对齐 %d ／ 跳过 %d"
                  % (str(_e)[:80], _copied, _rand, _skip))
            log("[init] 维度不匹配 ⇒ **部分热启动**：拷贝 %d ／ 按列对齐 %d ／ 跳过 %d（新列小随机 ✓）"
                % (_copied, _rand, _skip))
        print("[init] 从 %s 载入权重（iter=%s）—— BC 热启动" % (args.init, ck0.get("iter")))
        log("[init] 从 %s 载入权重（iter=%s）—— BC 热启动" % (args.init, ck0.get("iter")))
        if int(getattr(args, "ref_pair", 0) or 0):
            ref_sd = P.sd_to_cpu(ck0["sd"])
            print("[ref-pair] **同 seed 配对优势已启用**：参考 = %s（冻结、同发牌）"
                  % os.path.basename(args.init))
            log("[ref-pair] **同 seed 配对优势已启用**：参考 = %s（冻结、同发牌）"
                  % os.path.basename(args.init))
    opt = TORCH.optim.Adam(net.parameters(), lr=args.lr)

    tr_lo, tr_hi = args.seed_base, args.seed_base + args.games * max(1, args.iters)
    va_lo, va_hi = args.val_base, args.val_base + max(args.val_n, 1)
    overlap = (tr_lo < va_hi and va_lo < tr_hi)

    log("=" * 78)
    log("=== 自我对弈 RL（PPO）· 训练启动 ===")
    log("  目标：max_π E[Σ γ^t R_t]，R_terminal = 该席本局真实 ΔScore")
    log("       （`Game.play(...)[\"scores\"][seat]`；不是胡牌率、不是向听数）")
    log("  " + rl_boot.device_report(TORCH, dev))
    log("  网络：CandNet（**候选打分**，同状态内排序）参数 = %d，hidden=%d，temp=%.2f"
        % (P.net_params(net), cfg["hidden"], cfg["temp"]))
    log("  超参：lr=%g epochs=%d clip=%.2f gamma=%.3f ent=%.4f vf=%.2f batch=%d ret_norm=%s"
        " self_imit=%.3f"
        % (args.lr, args.epochs, args.clip, args.gamma, args.ent_coef, args.vf_coef,
           args.batch, getattr(args, "ret_norm", "batch"),
           float(getattr(args, "self_imit", 0.0))))
    log("  辅助 reward：%s（scale=%g，lam=%s）—— goal §4，**默认关闭**"
        % ("开" if args.aux else "关", args.aux_scale,
           (json.dumps(args.lam, ensure_ascii=False) if args.lam else "aux_reward.LAMBDA")))
    log("  对手池（训练）：线上基因组 %s（**每房恒定 %d 席**，位置轮换）+ 最近 %d 个 "
        "checkpoint 轮换（弱档池见 `--opp-pool`）"
        % (args.base_genome, args.live_seats, args.pool_snapshots))
    log("    基础池 %d 项 = %s" % (len(opp_base), [x[1] for x in opp_base]))
    _fc = {}
    for i in range(max(1, args.games)):
        for sp in opp_specs_for(i, [x for x in opp_base] + ["ckpt"], args.live_seats):
            _fc[sp if isinstance(sp, str) else sp[1]] = \
                _fc.get(sp if isinstance(sp, str) else sp[1], 0) + 1
    log("    实际出现频次（前 %d 局，确定性轮换）= %s"
        % (max(1, args.games), {k: v for k, v in sorted(_fc.items())}))
    log("  评估夹具（标准夹具 G1）：3 席 = %s；被评估席 = seed%%4"
        % ([x[1] for x in ev_opp]))
    np_est = int(args.games * (args.rounds if not args.no_room else 1) * 6)
    log("  PPO 每轮梯度步数 ≈ %d（batch=%d，epochs=%d，样本/轮 ≈ %d = %d 房 × %d 手 × ~6）"
        % (max(1, int(np_est / max(1, args.batch))) * args.epochs, args.batch,
           args.epochs, np_est, args.games,
           args.rounds if not args.no_room else 1))
    log("  ⚠ 口径提醒：`live_ab_arms.txt` = %s（线上 A/B 臂，优先）；"
        "`run_live.py:GENOME`/`rl_env.BASE_GENOME` = %s。本脚本用前者。"
        % (BASE_GENOME, ALT_GENOME))
    log("  房间层：%s（Rounds=%d，庄位继承=赢家下一手坐庄 / 流局连庄，carry_scores/"
        "dealer_streak/round_no/rounds 全传进 Game ⇒ CTX 不再是死输入）"
        % ("**关**（--no-room，只用于 A/B 对照）" if args.no_room
           else "开", args.rounds if not args.no_room else 1))
    log("  进程数 = %d（硬上限 %d）  训练 seed 段 [%d, %d)  验证 seed 段 [%d, %d)  重叠=%s"
        % (nproc, MAX_NPROC, tr_lo, tr_hi, va_lo, va_hi,
           "**是 ⇒ 硬错误**" if overlap else "否 ✅"))
    log("  checkpoint = %s<iter>.pt / %slatest.pt   日志 = %s   历史 = %s"
        % (args.out_prefix, args.out_prefix, args.log, args.history))
    log("=" * 78)

    snap_sds, snap_order = {}, []
    # ★★★★★ 2026-09-30：**训练路径**也要预载 `--opp-pool` 里的 `ckpt:<path>`（`§Z.856`）
    #   现场崩溃：`KeyError: 'rl_kld_b20.pt'`（`build_spec_policy` 直接 `sds[val]` ✗）
    #   ⚠ 第一次我把它打在了 `--eval-*` 分支的 `sds` 上 ⇒ **没生效** ✗（训练用 `snap_sds` ✓）
    #   ⇒ 凡 `ckpt:` 指向**存在的文件**，就 `snap_sds[path] = sd` ✓
    #   ⚠ 这里**不**改 `cfg`：对手用主配置结构 ✓；结构不一致时 `load_state_dict` 会响亮报错 ✓
    for _spec in parse_pool(args.opp_pool, args.base_genome):
        if (isinstance(_spec, tuple) and len(_spec) == 2 and _spec[0] == "ckpt"
                and isinstance(_spec[1], str) and os.path.exists(_spec[1])):
            _n3, _c3, _m3 = P.load_net(_spec[1], device="cpu")
            snap_sds[_spec[1]] = P.sd_to_cpu(_n3.state_dict())
            # ⚠ **不要** append 进 `snap_order`：那个列表是**快照轮换的驱逐队列** ✗
            #   （`while len(snap_order) > pool_snapshots: snap_sds.pop(snap_order.pop(0))`）
            #   ⇒ 一旦入队，第 2 轮就会被驱逐 ⇒ 再崩 `KeyError` ✗（实测过 ✓）
            #   ⇒ 对手池的预载项必须**常驻** ✓
            log("  ★ snap_sds 预载对手池 %s（iter=%s）" % (_spec[1], _m3.get("iter")))
    curve = []
    # ★★ 证据文件必须能区分"它记录的是哪一次运行"（本会话第三次栽在这类事上）：
    #   每次运行**归档旧文件**并加 `run` 字段，绝不追加到同一个文件里。
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-%04d" % (os.getpid() % 10000)
    if os.path.exists(args.history) and os.path.getsize(args.history) > 0:
        arch = "%s_%s.jsonl" % (os.path.splitext(args.history)[0], run_id)
        os.replace(args.history, arch)
        log("  （旧 history 已归档 -> %s；本次 run=%s，不再追加到同一个文件）"
            % (arch, run_id))
    hist_f = open(args.history, "w", encoding="utf-8")
    hist_f.write(json.dumps({"run": run_id, "kind": "header",
                             "iters": args.iters, "games": args.games,
                             "rounds": args.rounds, "room_mode": not args.no_room,
                             "val_base": args.val_base, "val_n": args.val_n,
                             "val_every": args.val_every, "lr": args.lr,
                             "epochs": args.epochs, "batch": args.batch,
                             "init": args.init, "base_genome": args.base_genome,
                             "opp_pool": args.opp_pool,
                             "fixture_opp": args.fixture_opp,
                             "seed_base": args.seed_base}, ensure_ascii=False) + "\n")
    hist_f.flush()
    try:
        for it in range(1, max(1, args.iters) + 1):
            t_it = time.time()
            sd = P.sd_to_cpu(net.state_dict())
            pool_specs = list(opp_base) + ["ckpt:" + p for p in snap_order]
            tasks = []
            gi0 = (it - 1) * args.games
            for b in split_blocks(range(args.games), nproc):
                tasks.append({"sd": sd, "cfg": dict(cfg, gamma=args.gamma),
                              "base_genome": args.base_genome, "pool": pool_specs,
                              "sds": snap_sds, "seed0": args.seed_base + b[0],
                              "n": len(b), "gi0": gi0 + b[0], "aux": args.aux,
                              "aux_scale": args.aux_scale, "aux_lam": args.lam,
                              "rounds": args.rounds, "room_mode": not args.no_room,
                              "first_dealer": args.first_dealer,
                              "live_seats": args.live_seats,
                              # ★ 第 518 轮：按手截断的 return（见 `_collect` 里的长注释）
                              "per_hand": bool(getattr(args, "per_hand_return", 0)),
                              # ★ 第 526 轮：手内 n-step 自举（见 `_collect` 里的长注释）
                              "nstep": int(getattr(args, "nstep", 0) or 0),
                             # ★ 第 700 轮：同 seed 配对优势（None = 关闭，行为逐字不变）
                             "ref_sd": ref_sd})
            t0 = time.time()
            res = run_children("rollout", tasks, "rt%d" % it, log=log,
                               keep=args.keep_tasks)
            roll_secs = time.time() - t0
            # ★★ 第 385 轮：更新阶段出现 **2711s 实测 vs 27s 基准** 的 100 倍异常，
            #   而按代码读（纯 numpy 拼接 + 30MB 存盘 + 简单归约）解释不了。
            #   ⇒ 在这里打三个分段计时，**先量再改**。
            _t1 = time.time()
            batch = make_batch(res)
            _t2 = time.time()
            stats = ppo_update(net, opt, batch, hp, dev, log=log)
            _t3 = time.time()
            log("           [计时] 拼接 %.1fs ｜ 更新 %.1fs ｜ (rollout %.1fs)"
                % (_t2 - _t1, _t3 - _t2, roll_secs))

            scores = np.concatenate([r["scores"] for r in res])
            n_pts = sum(r["n_pts"] for r in res)
            fb = sum(r["fallbacks"] for r in res)
            er = sum(r["errors"] for r in res)
            gap = max([r["pure_gap"] for r in res] or [0.0])
            aux_acc = None
            for r in res:
                if r["aux"]:
                    aux_acc = aux_acc or {"hand": 0.0, "eff": 0.0, "mult": 0.0,
                                          "risk": 0.0, "abs_sum": 0.0, "n": 0}
                    _fold_aux_dict(aux_acc, r["aux"])
            mean_R = float(scores.mean()) if len(scores) else float("nan")
            aux_ratio = (aux_acc["abs_sum"] / max(1.0, aux_acc["n"]) /
                         max(1e-9, abs(mean_R))) if aux_acc else 0.0

            ev = None
            if args.val_n > 0 and (it % args.val_every == 0):
                ev = paired_eval(range(args.val_base, args.val_base + args.val_n),
                                 ("ckpt", INMEM_KEY), "live", ev_opp,
                                 args.base_genome, cfg, {INMEM_KEY: sd}, nproc,
                                 log=log, tag=" 迭代%d" % it, tmp_tag="ev%d" % it,
                                 blocks_want=args.eval_blocks, rounds=args.rounds,
                                 room_mode=not args.no_room,
                                 first_dealer=args.first_dealer)
                curve.append((it, mean_R, ev["mu"], ev["lo"], ev["hi"],
                              ev["fork_learner"], stats["ent"]))
                ev = {k: v for k, v in ev.items() if not k.startswith("_")}

            ckpt = "%s%04d.pt" % (args.out_prefix, it)
            TORCH.save({"sd": sd, "cfg": cfg, "iter": it, "args": vars(args),
                        "eval": ev, "train_scores": scores.tolist()}, ckpt)
            TORCH.save({"sd": sd, "cfg": cfg, "iter": it, "args": vars(args),
                        "eval": ev}, args.out_prefix + "latest.pt")

            it_secs = time.time() - t_it
            log("")
            log("[iter %3d] 训练回报（被学习席 ΔScore 均值）= %+.3f   "
                "决策点 %d（%d 局，空轨迹 %d）"
                % (it, mean_R, n_pts, len(scores), sum(r["n_empty"] for r in res)))
            log("           熵 = %.4f   KL = %+.5f   裁剪比例 = %.4f   |ratio−1| = %.4f"
                % (stats["ent"], stats["kl"], stats["clip"], stats["ratio_abs"]))
            # ★★★★★ 2026-09-30：**`--ref-pair` 的自检读数**（硬规则 ⑤ 同族 ✓）
            #   机制：同 seed 跑一遍冻结参考席，训练目标 = `R − R_ref` ⇒ 发牌运气相消 ✓
            #   ⚠ 没有这一行就**完全看不出它有没有起作用** ✗（只能相信注释 ✗）
            # ★★★★★ 2026-09-30：**`--ref-pair` 的自检读数**（硬规则 ⑤ 同族 ✓）
            #   为什么要它：**没有这一行就完全看不出配对有没有起作用** ✗（只能相信注释 ✗）
            #   ★ 判据（相对**单臂**基线 ✓，这是训练信号关心的口径 ✓）：
            #       Var(R−R_ref) / Var(R) = 2(1−ρ)·(σ_ref/σ_R 的修正项) ⇒ **只有 ρ > 0.5 才降** ✓
            if res and all("targets" in _r for _r in res):
                _rr = np.asarray(scores, dtype=np.float32)
                _tt = np.concatenate([np.asarray(_r["targets"], dtype=np.float32) for _r in res])
                _r0 = next((_r for _r in res if "ref_scores" in _r), {})
                _rr2 = np.asarray(_r0.get("ref_scores", []), dtype=np.float32)
                if len(_rr) > 1 and len(_tt) > 1:
                    _v1, _v2 = float(_rr.std()), float(_tt.std())
                    _has_ref = (len(_rr2) == len(_rr) and bool(np.isfinite(_rr2).all())
                                and float(_rr2.std()) > 0)
                    _rho = float(np.corrcoef(_rr, _rr2)[0, 1]) if _has_ref else float("nan")
                    log("           ★ 配对优势自检：sd(R)=%.2f ｜ sd(R_ref)=%s ｜ sd(R−R_ref)=%.2f"
                        " ｜ corr=%s ｜ **目标方差比=%.2f**（`--ref-pair` %s）"
                        % (_v1, ("%.2f" % _rr2.std()) if _has_ref else "n/a", _v2,
                           ("%.3f" % _rho) if _has_ref else "n/a",
                           (_v2 ** 2) / max(_v1 ** 2, 1e-9),
                           "已启用 ✓" if ref_sd is not None else "**未启用** ✗"))
                    if _has_ref:
                        log("              ⇒ ★ 配对**降方差的条件是 corr > 0.5** ✓（相对单臂基线 ✓）；"
                            "本配置 corr=%.3f ⇒ %s"
                            % (_rho, "**降方差** ✓" if _rho > 0.5 else "**放大方差** ✗"))
            log("           loss = %+.4f   value_loss = %.4f   η(辅助/终局) = %.4f"
                % (stats["loss"], stats["vloss"], aux_ratio))
            # ★★ 第 395 轮：`--aux` 打开时纯度自检**根本不执行**（`if not aux_on`），
            #   但这一行以前照样打印 `max|G−R| = 0` ⇒ **看起来像"检查通过"，其实是"没检查"**。
            #   与 §Z.370 那个"预算 16"标签同一类问题：**读数必须反映"量了什么"，
            #   而不是"想让它显示什么"。** 现在 aux 模式下明确写"未检查"。
            _gap_s = ("未检查（aux 开，return 故意不等于 terminal）"
                      if args.aux else ("%.2g" % gap))
            log("           耗时：rollout %.1fs + 更新/评估 %.1fs = %.1fs/轮   "
                "fallback = %d   策略异常 = %d   max|G−R| = %s"
                % (roll_secs, it_secs - roll_secs, it_secs, fb, er, _gap_s))
            if ev:
                log("           曲线 iter=%3d | train μ=%+.3f | val Δμ=%+.3f "
                    "[%+.3f,%+.3f] | fork=%.3f | H=%.3f"
                    % (it, mean_R, ev["mu"], ev["lo"], ev["hi"],
                       ev["fork_learner"], stats["ent"]))
            else:
                log("           曲线 iter=%3d | train μ=%+.3f | val Δμ=  （本轮未评估）"
                    " | H=%.3f" % (it, mean_R, stats["ent"]))
            log("           checkpoint -> %s" % ckpt)
            if fb or er:
                log("           !! fallback/异常 > 0 ⇒ 掩码与引擎分叉或策略抛异常，"
                    "这批 (状态,动作) 不可信 —— **先查再训**")
            hist_f.write(json.dumps({
                "run": run_id, "iter": it, "mean_train_score": mean_R, "n_pts": n_pts,
                "n_games": len(scores), "entropy": stats["ent"], "kl": stats["kl"],
                "clip_frac": stats["clip"], "ratio_abs": stats["ratio_abs"],
                "loss": stats["loss"], "value_loss": stats["vloss"],
                "aux_ratio": aux_ratio, "rollout_secs": roll_secs,
                "iter_secs": it_secs, "fallbacks": fb, "errors": er,
                "pure_gap": gap, "eval": ev, "ckpt": ckpt}, ensure_ascii=False) + "\n")
            hist_f.flush()

            snap_sds[ckpt] = sd
            snap_order.append(ckpt)
            while len(snap_order) > max(0, args.pool_snapshots):
                snap_sds.pop(snap_order.pop(0), None)
    finally:
        hist_f.close()
        cleanup_tmp(keep=args.keep_tasks)

    log("")
    if curve:
        log("=== 验证集 μ 曲线（固定 seed 段 [%d, +%d)，与训练段不重叠）==="
            % (args.val_base, args.val_n))
        for it, tr, mu, lo, hi, fk, ent in curve:
            log("  iter %3d | train μ=%+.3f | val Δμ=%+.3f [%+.3f,%+.3f] | fork=%.3f | H=%.3f"
                % (it, tr, mu, lo, hi, fk, ent))
        first, last = curve[0][2], curve[-1][2]
        mid = (first + last) / 2.0
        cw = [abs(c[4] - c[2]) for c in curve if c[4] == c[4]]
        se_hand = (sum(cw) / len(cw) / 1.96) if cw else float("nan")
        n_hand = max(1, args.val_n * (args.rounds if not args.no_room else 1))
        sd_hand = se_hand * (n_hand ** 0.5) if se_hand == se_hand else float("nan")
        need_hand = (1.96 * sd_hand / 0.3) ** 2 if sd_hand == sd_hand else float("nan")
        need_rooms = need_hand / max(1, args.rounds if not args.no_room else 1)
        log("  G5 读数：首 %+.3f → 末 %+.3f（差 %+.3f）；'末值 ≥ 首末均值 + 0.3' ⇔ 差 ≥ +0.6"
            % (first, last, last - first))
        log("     本曲线的实测 per-hand SD ≈ %.3f（⇒ 本轮 %d 房的 95%%CI 半宽 ≈ %.3f）；"
            "要判 0.3 需 ≈ %.0f 手 ≈ %.0f 房" % (sd_hand, args.val_n, 1.96 * se_hand,
                                                need_hand, need_rooms))
        if args.val_n >= need_rooms:
            log("     ⇒ G5 = %s（样本量足够）"
                % ("**过**" if last >= mid + 0.3 else "**不过**"))
        else:
            log("     ⇒ **G5 = 未分辨**：验证集 %d 房 < 所需 %.0f 房（噪声是阈值的 %.1f 倍），"
                "**不给它盖章** —— 曲线只当描述，不作判据。"
                % (args.val_n, need_rooms,
                   (1.96 * se_hand) / 0.3 if se_hand == se_hand else float("nan")))
    log("=== 训练结束。下一步（**本仓唯一认可的判据**）===")
    log("    .\\_venv\\Scripts\\python.exe train_selfplay_rl.py --eval-ckpt %slatest.pt "
        "--eval-n 2000 --arm-a ckpt:%slatest.pt --arm-b live"
        % (args.out_prefix, args.out_prefix))
    log("    （先跑 --eval-null-test 确认评估口径本身没坏：Δμ 必须 ≡ 0 且分叉率 0）")
    log.close()
    return 0


# ================================================================ BC 训练（热启动）
def _bc_dump(path, res_tr, res_va, log=None):
    """把采集到的 BC 数据落盘（`--bc-dump`）。

    ★ 为什么需要它：**采集是贵的，训练是便宜的。**
      实测（2026-10-09）：600 房采集 ≈ 41 分钟（`--nproc 3`），
      而 4 个 epoch 的 BC 训练只要几分钟。
      而第 10 轮的实测结论要求**扫 `--bc-kl-beta`**
      （β=0.1 时候选相对原模型的分叉率已达 **0.9792** ⇒ 没有起到"减少遗忘"的作用）。
      ⇒ 不解耦的话，扫 3 个 β 就要重复 3 次 41 分钟的采集，而且三次的
        **局面分布还未必一样**（seed 段固定，但采集有并行调度）——
        那样扫出来的差异里混着"数据不同"的成分 ✗。
      ⇒ 落盘后，同一次采集可以喂给任意组超参，**对照组之间数据完全一致** ✓

    格式：每个子进程一份、每个头（turn/win）四个数组（s / c / n / a），
    与 `--bc-from-npz` 的 `_task()` 结构同构 ⇒ 加载后直接可用。
    """
    import numpy as _np
    d = {"n_tr": _np.int64(len(res_tr)), "n_va": _np.int64(len(res_va))}
    for tag, rs in (("tr", res_tr), ("va", res_va)):
        for i, r in enumerate(rs):
            d["%s%d_npts" % (tag, i)] = _np.int64(r.get("n_pts", 0))
            for kind in ("turn", "win"):
                for f in ("s", "c", "n", "a"):
                    d["%s%d_%s_%s" % (tag, i, kind, f)] = _np.asarray(r[kind][f])
    _dd = os.path.dirname(os.path.abspath(path))
    if _dd:
        os.makedirs(_dd, exist_ok=True)
    _np.savez(path, **d)
    if log is not None:
        log("  ★ 采集数据已落盘 -> %s（%.1f MB）—— 之后扫超参用 `--bc-load` 免采集"
            % (path, os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0))


def _bc_load(path, log=None):
    """读回 `_bc_dump` 落盘的数据 ⇒ `(res_tr, res_va)`。"""
    import numpy as _np
    z = _np.load(path, allow_pickle=False)
    out = []
    for tag in ("tr", "va"):
        rs = []
        for i in range(int(z["n_" + tag])):
            r = {"n_pts": int(z["%s%d_npts" % (tag, i)]),
                 "dropped": 0, "fallbacks": 0, "errors": 0,
                 "n_games": 0, "secs": 0.0}
            for kind in ("turn", "win"):
                r[kind] = {f: z["%s%d_%s_%s" % (tag, i, kind, f)]
                           for f in ("s", "c", "n", "a")}
            rs.append(r)
        out.append(rs)
    if log is not None:
        log("  ★ 复用已落盘的采集数据：%s（训练 %d 段 / 验证 %d 段）—— **跳过采集**"
            % (os.path.basename(path), len(out[0]), len(out[1])))
    return out[0], out[1]


def run_bc(args):
    """行为克隆：用规则基因组自己的决策，监督"同状态内候选排序"（交叉熵）。

    验收（父 agent 的判据，**不是**最终判据）：BC 之后**不做任何 PPO 更新**，
    `--eval-n 400 --arm-a ckpt:<bc.pt> --arm-b live` 的 **|Δμ| 应很小**
    （CI 含 0，或 |Δμ| < 0.15）—— 这同时证明"BC 真学到了规则策略"与"评估口径无偏"。
    """
    log = Log(args.bc_log)
    dev = pick_dev(args)
    cfg = dict(P.DEFAULT_CFG, hidden=args.hidden, temp=args.temp)
    # ★★ 第 442 轮 DAgger：`--bc-dagger-ckpt <path>` ⇒
    #   ① 用**那个检查点**当执行者（采集"模型自己的状态分布"）；
    #   ② 它的 sd 要放进 `sds`，`build_spec_policy("ckpt:...")` 才取得到；
    #   ③ **cfg 必须改成检查点里的**（否则 hidden 不匹配会 size mismatch）。
    sds_bc = {}
    dagger = getattr(args, "bc_dagger_ckpt", "") or ""
    if dagger:
        _d = TORCH.load(dagger, map_location="cpu", weights_only=False)
        sds_bc[dagger] = _d["sd"]
        _c = _d.get("cfg") or {}
        cfg.update({k: _c[k] for k in ("hidden", "state_dim", "cand_turn_dim",
                                       "cand_win_dim", "temp") if k in _c})
        log("  ★ DAgger 模式：执行者 = **%s**（iter=%s, hidden=%s），标签仍由老师给"
            % (os.path.basename(dagger), _d.get("iter"), cfg.get("hidden")))
    # ★★ 第 563 轮修正：**学生**的输入维度必须用**当前**的 `A.STATE_DIM`，
    #   而不能沿用检查点的 `state_dim`。为什么（这是一次 7.2 小时的真实事故）：
    #     上面那行把检查点的 `state_dim` 也复制进了 `cfg` —— 那是为了让**执行者**
    #     （`rl_bc6.pt`，370 维）能加载它的 SD。但**学生**的输入是由
    #     `A.encode_state(v)` 现场编码的，维度由当前代码决定（开 `MJ_CTX2` 时是 376）。
    #     ⇒ 若沿用 370，则 `make_net` 建出 370 的 trunk，而喂进去的是 376 的数据 ⇒
    #       `RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x376 and 370x512)`
    #       在**训练第一步**才炸 —— 而**采集已经跑完**（那次白烧 25,831s）。
    #   ⇒ 因此这里**分开两份 cfg**：`cfg`（执行者用，含它的 state_dim）与
    #     `cfg_student`（学生用，state_dim = 当前值）。
    #   ⚠ 两者的 `hidden / cand_*` 仍然一致（BC 从头训，不加载执行者权重）。
    cfg_student = dict(cfg)
    cfg_student["state_dim"] = int(A.STATE_DIM)
    if int(cfg_student["state_dim"]) != int(cfg.get("state_dim", 0)):
        log("  ★ 学生输入维度 = **%d**（执行者 = %s）—— 两者不同是有意的："
            "学生的输入由当前 `encode_state` 决定"
            % (cfg_student["state_dim"], cfg.get("state_dim")))
    nproc = args.nproc if args.allow_more else min(args.nproc, MAX_NPROC)
    opp_base = parse_pool(args.opp_pool, args.base_genome)
    live_t = ("genome", args.base_genome)
    if not opp_base or opp_base[0] != live_t:
        opp_base = [live_t] + [x for x in opp_base if x != live_t]
    n_tr = max(1, int(args.bc_games * (1.0 - args.bc_val_frac)))
    n_va = max(1, args.bc_games - n_tr)
    log("=" * 78)
    log("=== BC 热启动（行为克隆：候选打分头的交叉熵）===")
    log("  老师 = %s" % args.teacher)
    log("  数据：训练 %d 局（seed 从 %d）/ 验证 %d 局（seed 从 %d）"
        " —— **按 seed 段切分，不按决策点**"
        % (n_tr, args.seed_base, n_va, args.val_base))
    log("  对手池 %d 项 = %s（每局恰好 1 席 live，位置轮换）"
        % (len(opp_base), [x[1] for x in opp_base]))
    log("  " + rl_boot.device_report(TORCH, dev))
    log("=" * 78)

    def _jobs(tag, s0, n):
        out = []
        for b in split_blocks(range(n), nproc):
            out.append({"teacher": args.teacher, "cfg": cfg,
                # ★ 第 419 轮：把尾部加权系数带进每个采集任务
                "tail_alpha": float(getattr(args, "bc_tail_alpha", 0.0) or 0.0),
                # ★ 第 442 轮 DAgger：把执行者的检查点路径带进每个任务
                "dagger_ckpt": dagger,
                # ★ 第 456 轮：混合比例（1.0 = 全模型驱动；0.5 = 一半仍由老师驱动）
                "dagger_mix": float(getattr(args, "bc_dagger_mix", 1.0) or 0.0),
                        "base_genome": args.base_genome, "pool": opp_base,
                        "sds": sds_bc, "seed0": s0 + b[0], "n": len(b), "gi0": b[0],
                        "rounds": args.rounds, "room_mode": not args.no_room,
                        "first_dealer": args.first_dealer,
                        "live_seats": args.live_seats})
        return out

    # ★ 两段**分开跑**：一次只让 nproc 个子进程在跑（并发纪律 ≤6）
    # ★★★★★ 第 710 轮：**`--bc-from-npz`** —— 用**外部生成的专家标签**做 BC。
    #   由来（用户 2026-09-29：「我指的落后，是**落后其他对手的模型策略**」）：
    #     `GOAL_LOOP.md §8` 给出四条结构性原因，其中
    #       ① 能力来源太弱（模型只模仿**手写规则老师** ⇒ 上限 = 规则的上限）
    #       ③ RL 从未成功（PPO 族试 8 次全部"未分辨" ⇒ 信号被方差淹没）
    #     本开关服务 goal §8.2 的 **分歧点取标签（Expert Iteration）**：
    #       数据由 `gen_expert_data.py` 生成（同一 seed 跑两次 ⇒ 谁那一手得分高就学谁），
    #       **标签来自"结果"而非老师** ✅ —— 而 BC 是本仓**唯一成功过的机制** ✅
    #   ★ I4 合规：数据生产用采样（I4 原文明确允许「搜索可以产生训练目标」），
    #     线上仍是**一次前向** ✅
    #   ⚠ npz 必须含 `s` / `c` / `act` / `kind`（见 `gen_expert_data.py` 的写出格式）；
    #     `n`（每条的候选数）由 `c` 的每条长度推出 —— `stack()` 需要它 ✓
    #   ⚠ **必须切分 train/val**：BC 训练对空验证集是**静默跳过**（`continue`）✗，
    #     所以这里按 85/15 切分并**打印**切分结果 ✓
    res_tr = res_va = None
    if getattr(args, "bc_from_npz", ""):
        import numpy as _np
        z = _np.load(args.bc_from_npz, allow_pickle=True)
        S_all, C_all = z["s"], z["c"]
        A_all, K_all = z["act"], z["kind"]
        n_all = len(A_all)
        if n_all == 0:
            log("  !! `--bc-from-npz` 的数据为空 ⇒ 中止")
            return 2
        rng = _np.random.RandomState(0)
        perm = rng.permutation(n_all)
        n_va = max(1, int(round(0.15 * n_all)))
        va_ids, tr_ids = perm[:n_va], perm[n_va:]

        def _pack(ids):
            # ★★★★★ 关键：`c`（候选特征）**必须按 kind pad 到固定长度**。
            #   为什么（第 4 次踩坑才看清）：
            #     `stack()` 用 `np.concatenate([res[i][kind]["c"] ...])` ⇒ 要求各条**等长**；
            #     而 `tens()` 用 `mask = arange(dim) < n` 把 padding 掩掉 ⇒
            #     **它假设 `c` 已经是 (N, dim, feat) 的已 padding 数组** ✓
            #   原采集路径天然等长（pad 到 `MAX_TURN_CAND` / `N_WIN_CAND`）⇒ 我这里照做 ✓
            #   ⚠ 变长直接 concatenate ⇒ `inhomogeneous shape`（前两轮踩到）；
            #     object 数组 ⇒ `TORCH.from_numpy` 拒绝（第三轮踩到）✗
            dim_of = {"turn": A.MAX_TURN_CAND, "win": A.N_WIN_CAND}
            out = {k: {"s": [], "c": [], "n": [], "a": []} for k in ("turn", "win")}
            for i in ids:
                k = "turn" if int(K_all[i]) == 0 else "win"
                cc = _np.asarray(C_all[i], dtype=_np.float32)
                if cc.ndim == 1:
                    cc = cc[None, :]
                dim = int(dim_of[k])
                n_c = int(cc.shape[0])
                if n_c > dim:                     # 不该发生，但要**响亮**而不是静默截断
                    raise SystemExit("候选数 %d 超过 %s 的 dim=%d ⇒ 数据/配置不一致"
                                     % (n_c, k, dim))
                pad = _np.zeros((dim, cc.shape[1]), dtype=_np.float32)
                pad[:n_c] = cc
                out[k]["s"].append(_np.asarray(S_all[i], dtype=_np.float32))
                out[k]["c"].append(pad)           # ← 已 padding，等长 ✓
                out[k]["n"].append(n_c)
                out[k]["a"].append(int(A_all[i]))
            for k in ("turn", "win"):
                if not out[k]["a"]:               # 空头 ⇒ 保持空（训练段会打印并跳过）
                    out[k] = {"s": _np.zeros((0, 370), _np.float32),
                              "c": _np.zeros((0, int(dim_of[k]), 14), _np.float32),
                              "n": _np.zeros((0,), _np.int64),
                              "a": _np.zeros((0,), _np.int64)}
                    continue
                out[k]["s"] = _np.stack(out[k]["s"]).astype(_np.float32)
                out[k]["c"] = _np.stack(out[k]["c"]).astype(_np.float32)
                out[k]["n"] = _np.asarray(out[k]["n"], dtype=_np.int64)
                out[k]["a"] = _np.asarray(out[k]["a"], dtype=_np.int64)
            return out

        def _task(ids):
            d = _pack(ids)
            return {"turn": d["turn"], "win": d["win"], "n_pts": len(ids),
                    "dropped": 0, "fallbacks": 0, "errors": 0,
                    "n_games": 0, "secs": 0.0}

        res_tr, res_va = [_task(tr_ids)], [_task(va_ids)]
        log("  ★★ **`--bc-from-npz`**：%s ｜ 总标签 %d ⇒ 训练 %d / 验证 %d"
            % (os.path.basename(args.bc_from_npz), n_all, len(tr_ids), len(va_ids)))
        log("     其中 turn 头 %d 条 / win 头 %d 条"
            % (int((K_all == 0).sum()), int((K_all == 1).sum())))
    if res_tr is None and getattr(args, "bc_load", ""):
        res_tr, res_va = _bc_load(args.bc_load, log=log)
    if res_tr is None:
        res_tr = run_children("bc", _jobs("tr", args.seed_base, n_tr), "bc_tr", log=log,
                              keep=args.keep_tasks)
        res_va = run_children("bc", _jobs("va", args.val_base, n_va), "bc_va", log=log,
                              keep=args.keep_tasks)
        if getattr(args, "bc_dump", ""):
            _bc_dump(args.bc_dump, res_tr, res_va, log=log)
    res = res_tr + res_va
    idx_tr = list(range(len(res_tr)))
    idx_va = list(range(len(res_tr), len(res)))
    n_pts = sum(r["n_pts"] for r in res)
    dropped = sum(r["dropped"] for r in res)
    fb = sum(r["fallbacks"] for r in res)
    er = sum(r["errors"] for r in res)
    log("  采集完成：训练 %d 局 / 验证 %d 局，共 %d 个决策点，用时 %.1fs"
        % (sum(res[i]["n_games"] for i in idx_tr),
           sum(res[i]["n_games"] for i in idx_va), n_pts,
           sum(r["secs"] for r in res)))
    uns = sum(r.get("unsupported", 0) for r in res)
    log("  ★ 标签落不进候选集的比例 = %.4f%%（%d / %d）—— >0.5%% 就先查动作空间"
        % (100.0 * dropped / max(1, n_pts), dropped, n_pts))
    log("    其中**环境不支持**的动作（碰窗口里的明杠，`mj_sim` 只认 peng/chi ⇒ "
        "引擎静默当 pass）= %d 例（%.4f%%）；扣掉它之后的真·分叉 = %d 例（%.4f%%）"
        % (uns, 100.0 * uns / max(1, n_pts), dropped,
           100.0 * dropped / max(1, n_pts)))
    log("    引擎 fallback = %d（规则策略在抓打圈下会打出非法弃牌，引擎会换成打刚摸那张）"
        "  策略异常 = %d" % (fb, er))
    seen = []
    for r in res:
        for d in (r.get("drops") or []):
            if len(seen) < 6:
                seen.append(d)
    if seen:
        log("    丢弃样本的原始形状（前 %d 例，必须逐条解释得通）：" % len(seen))
        for d in seen:
            log("      %s" % d)

    def stack(idxs, kind):
        return {kk: np.concatenate([res[i][kind][kk] for i in idxs])
                for kk in ("s", "c", "n", "a")}

    def tens(b, kind):
        dim = A.MAX_TURN_CAND if kind == "turn" else A.N_WIN_CAND
        mask = np.arange(dim)[None, :] < b["n"][:, None]
        w = b.get("w")
        if w is None or len(w) != len(b["a"]):
            w = np.ones(len(b["a"]), dtype=np.float32)   # 旧数据/旧任务兼容：权重恒 1
        return (TORCH.from_numpy(b["s"]).to(dev), TORCH.from_numpy(b["c"]).to(dev),
                TORCH.from_numpy(mask).to(dev),
                TORCH.from_numpy(b["a"]).to(dev),
                TORCH.from_numpy(w).to(dev))

    # ★★★★★ 第 710 轮（**必修**）：BC 热启动**必须**先走 `cfg_from_ckpt`。
    #   事故：我第一版直接 `load_state_dict` ⇒
    #     `size mismatch: trunk.0.weight [512,370] vs [128,370]`
    #     —— 因为 `cfg_student` 来自**命令行默认**（hidden=128），而权重来自检查点（hidden=512）✗
    #   这正是 `cfg_from_ckpt` 注释里写过的那个陷阱：
    #     「结构来自命令行默认、权重来自检查点 ⇒ **必然** size mismatch；
    #       唯一实现是 `cfg_from_ckpt`，**任何从检查点取权重的入口都必须经过它**」
    #   ⇒ 严格照做 ✓（先对齐结构，再建网，再载权重）
    if args.init:
        _ck_cfg = TORCH.load(args.init, map_location="cpu", weights_only=False)
        cfg_from_ckpt(cfg_student, _ck_cfg, "--init(BC) " + os.path.basename(args.init))
    net = P.make_net(cfg_student, seed=args.seed).to(dev)
    # ★★★★★ 第 710 轮：**BC 也支持从检查点热启动**（`--init`）。
    #   为什么必须有（否则 Expert Iteration 会被自己毁掉）：
    #     · `rl_bc6` 是用 **1700 局（~十几万决策点）** 的 BC 训出来的；
    #     · 而 `gen_expert_data.py` 一次只产出**数千条**标签（40 房 ⇒ 2278 条）⇒ 少 ~50 倍；
    #     ⇒ 若 BC **从随机初始化**开跑，它会**丢掉 `rl_bc6` 全部已有能力** ✗
    #       ⇒ 得到的新模型必然远弱于起点 ✗（那样测出"EI 无效"是**假结论**）
    #   ⇒ 正确做法：EI 标签当"**追加的专家样本**"，在强模型权重上做**增量训练** ✅
    #   ⚠ 只覆盖权重，不动结构（结构仍由 `cfg_from_ckpt` 统一处理 ✓）
    if args.init:
        _cki = TORCH.load(args.init, map_location="cpu", weights_only=False)
        try:
            net.load_state_dict(_cki["sd"])
            log("  ★ BC 热启动：从 %s 载入权重（增量训练，而非从零）"
                % os.path.basename(args.init))
        except Exception as _e:                                   # noqa: BLE001
            log("  !! BC 热启动失败（%r）⇒ 退回随机初始化（**读数会不可比**）" % (_e,))
            raise
    opt = TORCH.optim.Adam(net.parameters(), lr=args.bc_lr, weight_decay=args.bc_wd)
    # ★★★★★ 第 767 轮（**队列 ⑤「信任域蒸馏」的核心机制**）：
    #   `--bc-kl-beta β > 0` ⇒ 在 BC 损失上加 **β · KL(π_ref ‖ π_cur)**（参考 = `--init` 的权重，冻结）
    #   为什么必须有它（`GOAL_STATUS.md §Z.791` 的失败诊断）：
    #     用 52,361 条真场强手标签做 plain BC ⇒ 判据 **−4.19**（CI 不含 0 ⇒ **有害**）✗
    #     诊断：**部分模仿**（acc 仅 62%）会**覆盖掉模型已有能力** ✗
    #       （对照：`expert_40` 学**确定性**规则老师、acc 86% ⇒ 覆盖得干净 ✓）
    #   ⇒ 正确做法：把强手标签当"**追加的专家样本**"，但**限制它偏离起点** ✓
    #     即 **信任域 / 锚定蒸馏**：既能吸收新信号，又不允许把旧能力冲掉 ✓
    #   ⚠ β = 0（默认）⇒ **逐字等于旧行为** ✓（不改动任何既有实验的可比性 ✓）
    beta_kl = float(getattr(args, "bc_kl_beta", 0.0) or 0.0)
    ref_net = None
    if beta_kl > 0.0:
        if not args.init:
            log("  !! `--bc-kl-beta > 0` 但没有 `--init` ⇒ 无参考策略 ⇒ **忽略该参数** ✓")
        else:
            import copy as _copy
            ref_net = _copy.deepcopy(net)
            ref_net.eval()
            for _p in ref_net.parameters():
                _p.requires_grad_(False)
            log("  ★★ 信任域已启用：β·KL(π_ref‖π_cur)，β = %g（参考 = %s，冻结 ✓）"
                % (beta_kl, os.path.basename(args.init)))
    data = {}
    for kind in ("turn", "win"):
        tr, va = stack(idx_tr, kind), stack(idx_va, kind)
        if len(tr["a"]) == 0 or len(va["a"]) == 0:
            log("  !! %s 头数据为空（train %d / val %d）⇒ 跳过"
                % (kind, len(tr["a"]), len(va["a"])))
            continue
        data[kind] = (tens(tr, kind), tens(va, kind), len(tr["a"]), len(va["a"]))
        log("  %s 头：训练 %d 行 / 验证 %d 行" % (kind, len(tr["a"]), len(va["a"])))

    temp = float(cfg.get("temp", 1.0) or 1.0)
    log("")
    log("  %5s | %9s %8s | %9s %8s | %8s" % ("epoch", "tr_CE", "tr_acc",
                                             "va_CE", "va_acc", "gap"))
    best = None
    for ep in range(1, int(args.bc_epochs) + 1):
        tot, acc, nrow = 0.0, 0, 0
        kl_run = []                      # ★ 信任域 KL 的可观测性（硬规则 ⑤：不怕崩，怕静默错 ✓）
        for kind, (tr, va, ntr, nva) in data.items():
            S, C, M, LAB, WGT = tr
            perm = TORCH.randperm(ntr)
            for i in range(0, ntr, int(args.bc_batch)):
                sel = perm[i:i + int(args.bc_batch)]
                logits, _v = net(kind, S[sel], C[sel], M[sel])
                # ★ 第 419 轮：**尾部加权**的交叉熵
                #   `reduction="none"` 后按 WGT 加权求均值（WGT 恒 1 时与旧代码逐字等价）。
                ce = nn.functional.cross_entropy(logits / temp, LAB[sel],
                                                 reduction="none")
                w = WGT[sel]
                loss = (ce * w).sum() / TORCH.clamp(w.sum(), min=1e-6)
                # ★ 信任域惩罚（β=0 时不执行 ⇒ 旧行为逐字不变 ✓，见上方注释）
                if ref_net is not None:
                    with TORCH.no_grad():
                        _rl, _rv = ref_net(kind, S[sel], C[sel], M[sel])
                    _lg = TORCH.log_softmax(logits / temp, dim=-1)
                    _rg = TORCH.log_softmax(_rl / temp, dim=-1)
                    kl = (_rg.exp() * (_rg - _lg)).sum(-1)        # KL(π_ref ‖ π_cur) ✓
                    loss = loss + beta_kl * (kl * w).sum() / TORCH.clamp(w.sum(), min=1e-6)
                    # ★ 必须 detach：否则把这个带图的 tensor 挂在 list 上
                    #   ⇒ 整条训练会**累积计算图 ⇒ 显存泄漏** ✗
                    #   （冒烟时被抓到：UserWarning "requires_grad=True to a scalar" ✓）
                    kl_run.append(float(kl.detach().mean()))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                with TORCH.no_grad():
                    tot += float(ce.mean()) * len(sel)
                    acc += int((logits.argmax(-1) == LAB[sel]).sum())
                    nrow += len(sel)
        tr_ce, tr_acc = tot / max(1, nrow), acc / max(1, nrow)
        vtot, vacc, vrow = 0.0, 0, 0
        with TORCH.no_grad():
            for kind, (tr, va, ntr, nva) in data.items():
                S, C, M, LAB, _W = va
                logits, _v = net(kind, S, C, M)
                vtot += float(nn.functional.cross_entropy(logits / temp, LAB)) * nva
                vacc += int((logits.argmax(-1) == LAB).sum())
                vrow += nva
        va_ce, va_acc = vtot / max(1, vrow), vacc / max(1, vrow)
        if kl_run:
            log("  %5d | %9.4f %8.4f | %9.4f %8.4f | %+8.4f | KL(信任域) = %.5f"
                % (ep, tr_ce, tr_acc, va_ce, va_acc, va_ce - tr_ce,
                   sum(kl_run) / len(kl_run)))
        else:
            log("  %5d | %9.4f %8.4f | %9.4f %8.4f | %+8.4f"
                % (ep, tr_ce, tr_acc, va_ce, va_acc, va_ce - tr_ce))
        if best is None or va_ce < best[0]:
            best = (va_ce, ep, {k: v.clone() for k, v in net.state_dict().items()})
    if best:
        net.load_state_dict(best[2])
    log("")
    if best:
        log("  最优 epoch %d（val CE %.4f）；过拟合 gap = %+.4f"
            % (best[1], best[0], best[0] - tr_ce))
    else:
        log("  没有可训练的数据")

    TORCH.save({"sd": P.sd_to_cpu(net.state_dict()), "cfg": cfg_student, "iter": 0,
                "args": vars(args),
                "bc": {"teacher": args.teacher, "n_train_games": n_tr,
                       "n_val_games": n_va, "n_pts": n_pts,
                       "dropped_frac": dropped / max(1, n_pts),
                       "val_ce": best[0] if best else None,
                       "val_acc": va_acc if best else None}},
               args.bc_out)
    log("  已写 %s" % args.bc_out)
    log("")
    log("=== BC 验收（**不是**最终判据，只证明它学到了规则策略）===")
    log("    .\\_venv\\Scripts\\python.exe train_selfplay_rl.py --eval-ckpt %s "
        "--eval-n 400 --arm-a ckpt:%s --arm-b live" % (args.bc_out, args.bc_out))
    log("    判据：CI 含 0 或 |Δμ| < 0.15")
    log("  然后拿它当 PPO 起点：")
    log("    ... train_selfplay_rl.py --init %s --iters ... --games ..." % args.bc_out)
    log.close()
    return 0


# ================================================================ eval 入口
def run_eval(args):
    log = Log(args.log)
    dev = rl_boot.pick_device(TORCH)
    cfg = dict(P.DEFAULT_CFG)
    sds = {}
    arm_a, arm_b = args.arm_a, args.arm_b
    if args.eval_null_test:
        arm_a, arm_b = "live", "live"
        log("=== 空对照（null test）：A 臂 = B 臂 = 线上基因组 %s ===" % args.base_genome)
        log("    预期：Δμ ≡ 0、95%CI = [0,0]、分叉率 = 0。若不然 ⇒ 评估代码里有随机源。")
    if args.eval_ckpt:
        net, cfg2, meta = P.load_net(args.eval_ckpt, device="cpu")
        # ★ 结构以检查点为准（唯一实现 `cfg_from_ckpt`）
        cfg_from_ckpt(cfg, {"cfg": cfg2}, "--eval-ckpt " + os.path.basename(args.eval_ckpt))
        sds[args.eval_ckpt] = P.sd_to_cpu(net.state_dict())
        if not args.arm_a_given:
            arm_a = "ckpt:" + args.eval_ckpt
        log("=== 载入 %s（iter=%s, hidden=%s, temp=%s）==="
            % (args.eval_ckpt, meta.get("iter"), cfg["hidden"], cfg.get("temp")))
    # 两臂里凡是 ckpt:<path> 的，若没预先载入就从盘上读
    # ★★ 第 471 轮：**同时把结构参数从检查点带进 `cfg`**。
    #   这是"cfg 用命令行默认、权重来自检查点"这个陷阱的**第三条路径** ——
    #     · `--eval-ckpt`：早就从检查点读 cfg ✅（§Z.439 查过）
    #     · `--init`     ：§Z.469 才补上（实测 size mismatch 崩溃）
    #     · **本处（两臂都是 ckpt 时）**：原来只读了 `sd`，`cfg` 仍是命令行默认
    #       ⇒ 用一个 512 的检查点当 arm 时会以 64/128 建网 ⇒ **必然 size mismatch**。
    #   ⚠ 这一条是 §Z.470 里"要比 rl7 vs rl_bc6"时才必须走到的路径 ——
    #     如果没提前查，就会在**要出结论的那一刻**才崩。
    #   ⇒ 一次修干净：凡是从检查点取的权重，结构参数一律以检查点为准。
    for spec in (arm_a, arm_b):
        if isinstance(spec, str) and spec.startswith("ckpt:"):
            path = spec[len("ckpt:"):]
            if path not in sds:
                n2, c2, m2 = P.load_net(path, device="cpu")
                sds[path] = P.sd_to_cpu(n2.state_dict())
                # ★ 结构以检查点为准（唯一实现 `cfg_from_ckpt`）
                cfg_from_ckpt(cfg, {"cfg": c2}, "arm " + os.path.basename(path))
                log("  载入 arm 快照 %s（iter=%s）" % (path, m2.get("iter")))
    # ★★★★★ 2026-09-30：**`--opp-pool` 里的 `ckpt:<path>` 也必须预载进 `sds`**
    #   现场崩溃（`GOAL_STATUS.md §Z.856`）：`KeyError: 'rl_kld_b20.pt'`
    #     `build_spec_policy` 直接 `sds[val]` 取权重 ✗，而**外部文件从未被载入** ✗
    #     ⇒ 对手子进程 `rc=1` 崩掉、训练中断（`--opp-pool live,ckpt:rl_kld_b*.pt` 时必现 ✓）
    #   修法：与上面"两臂"那条路**同构** —— 凡 `ckpt:` 指向存在的文件，就 `sds[path] = sd` ✓
    #   ⚠ 这里**不**调用 `cfg_from_ckpt`：对手必须用**主配置**的结构 ✓
    #     （结构不一致时 `load_state_dict` 会**响亮报错** ✓，不会静默走偏 ✗）
    for _spec in parse_pool(args.opp_pool, args.base_genome):
        if (isinstance(_spec, tuple) and len(_spec) == 2 and _spec[0] == "ckpt"
                and isinstance(_spec[1], str) and os.path.exists(_spec[1])
                and _spec[1] not in sds):
            _n3, _c3, _m3 = P.load_net(_spec[1], device="cpu")
            sds[_spec[1]] = P.sd_to_cpu(_n3.state_dict())
            log("  载入对手池快照 %s（iter=%s）" % (_spec[1], _m3.get("iter")))
    if args.eval_league:
        opp_specs = parse_pool(args.eval_league, args.base_genome)
        if len(opp_specs) != 3:
            raise SystemExit("--eval-league 必须正好 3 项（逗号分隔），收到 %d 项"
                             % len(opp_specs))
        kind_eval = "G3 困难联赛"
    else:
        opp_specs = parse_pool(args.fixture_opp, args.base_genome)
        if len(opp_specs) == 1:
            opp_specs = opp_specs * 3
        kind_eval = "G1 标准夹具"
    log("=== 配对评估（%s）===" % kind_eval)
    log("  A 臂 = %s" % arm_a)
    log("  B 臂 = %s" % arm_b)
    log("  对手（3 席，**两臂相同**，按座位序固定）= %s"
        % [x[1] for x in opp_specs])
    log("  seed 段 = [%d, +%d)，被评估席 = seed%%4，庄家 = (seed//4)%%4"
        % (args.val_base, args.eval_n))
    log("  " + rl_boot.device_report(TORCH, dev))
    nproc = args.nproc if args.allow_more else min(args.nproc, MAX_NPROC)
    try:
        out = paired_eval(range(args.val_base, args.val_base + args.eval_n),
                          arm_a, arm_b, opp_specs, args.base_genome, cfg, sds, nproc,
                          log=log, tag="", tmp_tag="evcli",
                          blocks_want=args.eval_blocks, rounds=args.rounds,
                          room_mode=not args.no_room,
                          first_dealer=args.first_dealer,
                          dump=args.eval_dump or ("_tmp/eval_%d.npz" % int(time.time())))
    finally:
        cleanup_tmp(keep=args.keep_tasks)
    log("")
    log("=== 判读（本仓纪律：不许用 loss/AUC 说事；`PREREG_RL_DEPLOY.md`）===")
    log("  Δμ = %+.4f   95%%CI [%+.4f, %+.4f]   seed 数 = %d   分叉率 = %.4f"
        % (out["mu"], out["lo"], out["hi"], out["n"], out["fork_learner"]))
    log("  朴素 SE = %.4f（±%.4f）   块间 SE = %.4f（%d 块，±%.4f）   %s"
        % (out["se_naive"], 1.96 * out["se_naive"], out["se_block"],
           out["n_blocks"], 1.96 * out["se_block"],
           "⚠ 块间/朴素 > 2× ⇒ 只认块间 SE"
           if out["se_naive"] > 0 and out["se_block"] / out["se_naive"] > 2.0 else ""))
    g2 = out["fork_learner"] > 0.90
    log("  G2（分叉率 > 0.90）：%.4f ⇒ %s" % (out["fork_learner"], "过" if g2 else "**不过**"))
    if kind_eval.startswith("G1"):
        thr, src = read_g1_threshold()
        g1 = (out["mu"] >= thr) and (out["lo"] > 0)
        log("  G1（标准夹具下 Δμ ≥ +%.2f 且 CI 不含 0）：%+.4f [%+.4f,%+.4f] ⇒ %s"
            % (thr, out["mu"], out["lo"], out["hi"], "过" if g1 else "**不过**"))
        log("     门槛来自预注册文件：%s" % src)
    else:
        g3 = (out["mu"] >= 0) and (out["hi_block"] > 0 or out["hi"] > 0)
        log("  G3（困难联赛下 Δμ ≥ 0 且 CI 不含 0 的负侧）：%+.4f [%+.4f,%+.4f] ⇒ %s"
            % (out["mu"], out["lo"], out["hi"], "过" if g3 else "**不过**"))
    if out["fork_learner"] == 0.0 and abs(out["mu"]) < 1e-9:
        log("  ⛔ 两臂恒等（分叉率 0 且 Δμ≡0）⇒ **这不是打平，是没有信息**。")
    if out["lo"] <= 0 <= out["hi"]:
        need = (1.96 * out["sd"] / 0.5) ** 2 if out["sd"] else float("nan")
        log("  ⇒ 这个 seed 数下**未能分辨**（CI 含 0）。按实测 σ_paired=%.2f，"
            "要分辨 ±0.5 分/局约需 %.0f 个 seed。" % (out["sd"], need))
    else:
        log("  ⇒ CI 不含 0，符号 = %s。" % ("A 更好" if out["mu"] > 0 else "A 更差"))
    log.close()
    return 0


# ================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(
        description="自我对弈强化学习训练器（PPO；terminal reward = 本局真实 ΔScore）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "复现配对评估（本仓唯一认可的判据）：\n"
            "  set MJ_USE_GPU=1\n"
            "  .\\_venv\\Scripts\\python.exe train_selfplay_rl.py --eval-ckpt rl_latest.pt "
            "--eval-n 2000 --arm-a ckpt:rl_latest.pt --arm-b live\n"
            "空对照（证明评估口径本身没坏 ⇒ Δμ 必须 ≡ 0 且分叉率 0）：\n"
            "  ... --eval-null-test --eval-n 300\n"
            "阳性对照（证明评估口径是活的）：\n"
            "  ... --arm-a genome:first:0:0:0:0:0:0:0 --arm-b live --eval-n 400\n"
            "BC 热启动（起点从 −6 抬到 live 水平）：\n"
            "  ... --bc --bc-games 1500 --bc-epochs 12      # -> rl_bc.pt\n"
            "  ... --eval-ckpt rl_bc.pt --eval-n 400 --arm-a ckpt:rl_bc.pt --arm-b live\n"
            "  ... --init rl_bc.pt --iters 20 --games 1200 --nproc 4   # BC -> PPO\n"
            "G1 标准夹具（3 席 shanten:1:1:2:2:1:1:0，默认）\n"
            "G3 困难联赛：--eval-league live,genome:shanten:1:1:2:2:1:1:0,"
            "genome:shanten:2:1:2:9:1:1:0\n"))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games", type=int, default=600,
                    help="每个 iteration 的**房间数**（一间 = --rounds 手）")
    ap.add_argument("--rounds", type=int, default=8,
                    help="一个房间的手数（线上 auto 房 Rounds=8；`--no-room` 时强制 1）")
    ap.add_argument("--no-room", action="store_true",
                    help="关掉房间层（一手一局、不传 carry/streak/round_no）—— "
                         "只用于死输入 A/B 对照，**不要**用它训练")
    ap.add_argument("--first-dealer", default="rot", choices=["rot", "seat"],
                    help="房间第一手的庄家：rot=(seed//4)%%4（与历史夹具同约定，"
                         "被评估席不恒为庄）；seat=seed%%4（mj_room 的默认）")
    ap.add_argument("--nstep", type=int, default=0,
                    help="手内 n-step 自举（0 = 关闭，保持原 MC 行为）。"
                         "奖励只在手末给 ⇒ 手内每个决策的 MC 回报都是同一个 R_hand，"
                         "**开局第一步与离胡牌一步的决策方差完全一样**；"
                         "n-step 用 gamma^n * V(s_{i+n}) 替代远端部分以降低方差。"
                         "需要 `NetPolicy` 记录的 `val`（`rl_policy.py:184-185` 已有）")
    ap.add_argument("--eval-dump", default="",
                    help="把配对评估的**逐手原始数组**落盘（npz）⇒ 以后可用 "
                         "`--reaggregate` 只重印读数，不必重算")
    # ★★ 2026-10-06（§Z.1554）：**自模仿（Self-Imitation Learning）** ✓
    #   出处 ✓：`GOAL_STATUS.md:18453-18472` 的先前研判把它列为高优先（★ 与"修 PPO 信用分配"同级 ✓）
    #     「标签来自**结果**而非老师的动作 ⇒ **天花板不是老师** ✓
    #       ＋ **监督学习** ⇒ 走本仓**唯一被证明能 work** 的机制 ✓」
    #   机制 ✓：对「结果好于基线」（`adv > 0` ✓）的决策点**额外做一次监督** ✓
    #     `L = L_PPO + β · mean(-log π(a|s) | adv > 0)` ✓
    #   ★ **默认 0.0 = 关闭** ⇒ 与既有行为**逐字一致** ✓（不影响任何既有 run ✓）
    ap.add_argument("--self-imit", type=float, default=0.0,
                    help="自模仿（SIL）系数 β：对 `adv>0`（结果好于基线）的决策额外做一次"
                         "监督（`+β·mean(-logp | adv>0)`）。"
                         "理由（§Z.1554）：奖励只在手末给 ⇒ **手内每个决策的 MC 回报都是同一个 "
                         "`R_hand`** ⇒ 手内无区分度（`:18454-18455`）；自模仿只挑'确实比基线好'的"
                         "决策，给它们**低方差**的监督信号 ⇒ 部分绕开信用分配问题。"
                         "★ 标签来自**结果**（不是老师）⇒ 天花板不是老师。默认 0.0 = 行为逐字不变")
    ap.add_argument("--per-hand-return", type=int, default=0,
                    help="1 = return **只累计本手之内**的回报（不跨手求和）。"
                         "理由（§Z.518）：gamma=1 + 跨手累加会让**开局**的 return 含着其后 7 手的结果，"
                         "而每一手有自己的 wall_seed ⇒ 未来各手几乎只贡献**方差**、不贡献信号。"
                         "§Z.510 实测 PPO 在开局动得最多（深度1-3 一致率 0.9031，n=10,699），"
                         "§Z.509 实测这些改动**零收益** ⇒ 冲着这个方差来源去。"
                         "默认 0 = 保持既有行为逐字不变")
    ap.add_argument("--reaggregate", default="",
                    help="从 `--eval-dump` 的 npz 重新打印读数（秒级）")
    ap.add_argument("--dead-inputs", type=int, default=0,
                    help="跑 N 个房间的**死输入审计**（逐维 std，报告 CTX 死维数）后退出")
    ap.add_argument("--nproc", type=int, default=4,
                    help="并行进程数（默认 4；硬上限 %d，线上有 1s 决策窗口）" % MAX_NPROC)
    ap.add_argument("--allow-more", action="store_true",
                    help="允许突破 %d 进程（默认不允许）" % MAX_NPROC)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=3, help="每轮 rollout 的 PPO 更新遍数")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0,                    help="折扣（默认 1.0 = 不折扣回合制；terminal 只在局末给）")
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ret-norm", default="batch", choices=["batch", "running", "none"],
                    help="★ return 标准化口径：batch（默认=旧行为 ✓）/ running（跨批 EMA ⇒ "
                         "目标平稳，批评家能真收敛 ✓）/ none（不标准化）")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--temp", type=float, default=1.0,
                    help="策略 softmax 温度（采样与更新同用，保证重要性比自洽）")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux", action="store_true",
                    help="打开 goal §4 辅助 reward（默认关 = 纯 terminal；打开会明显变慢）")
    ap.add_argument("--aux-scale", type=float, default=0.3)
    ap.add_argument("--lam", type=json.loads, default=None,
                    help='覆盖 aux_reward.LAMBDA，如 \'{"hand":0.05,"risk":0.0}\'')
    # ★★★ 第 700 轮：**同 seed 配对优势**（默认 0 = 关闭，行为逐字不变）。
    #   1 ⇒ 每个 seed 额外跑一遍【冻结的 --init 策略】，训练目标用 `R_当前 − R_参考`
    #   （同发牌 ⇒ 运气相消）。代价：rollout ×2。理由见 `rollout_task` 里的长注释。
    ap.add_argument("--ref-pair", type=int, default=0,
                    help="1 = 同 seed 配对优势（需要 --init；代价 rollout ×2）")
    ap.add_argument("--base-genome", default=BASE_GENOME, help="线上基因组（对手/基线）")
    ap.add_argument("--pool-snapshots", type=int, default=3,                    help="对手池里保留的历史 checkpoint 数（另有恒定的线上基因组）")
    ap.add_argument("--seed-base", type=int, default=7000000, help="训练 seed 段起点")
    ap.add_argument("--val-base", type=int, default=2000000, help="验证 seed 段起点")
    ap.add_argument("--val-n", type=int, default=240,
                    help="每轮配对评估的 seed 数（0 = 跳过）")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--out-prefix", default="rl_ckpt_")
    ap.add_argument("--log", default="rl_train.log")
    ap.add_argument("--history", default="rl_history.jsonl")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：小规模、短时间跑通并写出 checkpoint")
    ap.add_argument("--selfcheck", action="store_true", help="正确性自检（**先跑这个**）")
    ap.add_argument("--bc", action="store_true", help="行为克隆热启动（先 BC，再 PPO）")
    ap.add_argument("--bc-from-npz", default="", help="用 gen_expert_data.py 产出的 npz 直接做 BC（Expert Iteration，跳过仿真采集；goal §8.2）")
    ap.add_argument("--bc-games", type=int, default=1500)
    ap.add_argument("--bc-epochs", type=int, default=12)
    ap.add_argument("--bc-lr", type=float, default=3e-3)
    ap.add_argument("--bc-kl-beta", type=float, default=0.0,
                    help="★ 信任域蒸馏（队列 ⑤）：BC 损失加 β·KL(π_ref‖π_cur)，"
                         "参考 = `--init` 的权重（冻结）。0 = 关闭（逐字等于旧行为 ✓）")
    ap.add_argument("--bc-wd", type=float, default=1e-5)
    ap.add_argument("--bc-batch", type=int, default=512)
    ap.add_argument("--bc-val-frac", type=float, default=0.15)
    ap.add_argument("--bc-out", default="rl_bc.pt")
    # ★★ 第 442 轮 DAgger（§Z.441 的对症修法）：
    #   病根 = **复合误差** —— 每步 9.1% 的分叉，在"一副大牌 ≈ 连续 13.9 步"上
    #   被放大成 0.909^13.9 = 0.264 倍（与实测的番>=4 比值 1.46/5.52 **完全一致**）。
    #   病根的原因是"BC 只在**老师的状态分布**上训练"。
    #   ⇒ 传一个检查点进来，就变成 DAgger：**执行者是它**（采模型自己的状态），
    #     **标签仍由老师给**（`--teacher`）。迭代若干轮即可。
    ap.add_argument("--bc-dagger-ckpt", default="",
                    help="DAgger：用这个检查点当执行者、老师当标签者；空 = 普通 BC")
    # ★ 第 456 轮：DAgger 的**混合比例**。
    #   1.0（默认）= 全部由模型驱动（只覆盖模型的状态分布）；
    #   0.5        = 一半仍由老师驱动 ⇒ 一份数据里**两种分布都有**
    #                （标准 DAgger 用的是聚合数据集 D_老师 ∪ D_模型，这一档就是在近似它）。
    ap.add_argument("--bc-dagger-mix", type=float, default=1.0,
                    help="DAgger 中由**模型**驱动的局数占比（1.0 = 全部；0.5 = 一半）")
    # ★ 第 419 轮：**尾部加权 BC** 的系数 α。
    #   `w = 1 + α·max(0, fan−1)`，fan 由该局结算向量反解（`_round_fan`）。
    #   `α=0`（默认）⇒ 权重恒 1 ⇒ **与旧行为逐字一致**（可 A/B、可回退）。
    #   动机（§Z.418 实测）：模型的 `番>=4` 率 = 1.82%，老师 = 5.61%
    #   ⇒ BC 在**稀有高倍率动作**上退化；`cross_entropy` 的平均化把它抹掉了。
    #   评它时必须用**尾部指标**（`analyze_eval_tail.py`），不能再用 μ。
    ap.add_argument("--bc-tail-alpha", type=float, default=0.0,
                    help="尾部加权系数：w = 1 + α·max(0, 番−1)；0 = 关闭（默认）")
    ap.add_argument("--bc-log", default="rl_bc.log")
    ap.add_argument("--bc-dump", default="",
                    help="★ 把采集到的 BC 数据落盘（npz）。**采集贵、训练便宜**："
                         "落盘后可用 `--bc-load` 复用同一份数据扫超参（如 --bc-kl-beta），"
                         "既不重复花 41 分钟采集，又能保证各对照组**数据完全一致**")
    ap.add_argument("--bc-load", default="",
                    help="直接复用 `--bc-dump` 落盘的数据，**跳过采集**")
    ap.add_argument("--teacher", default=BASE_GENOME,
                    help="BC 的老师（默认线上基因组）")
    ap.add_argument("--eval-ckpt", default="")
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--eval-null-test", action="store_true")
    ap.add_argument("--arm-a", default="live", help="live | genome:<str> | ckpt:<path>")
    ap.add_argument("--arm-b", default="live", help="live | genome:<str> | ckpt:<path>")
    ap.add_argument("--live-seats", type=int, default=2,
                    help="每个房间 3 个对手位里，**恒定是线上基因组**的席数（默认 2；"
                         "1 = 旧行为）。其余席位从 `--opp-pool` 的 pool[1:] 轮换（= 历史快照）")
    ap.add_argument("--opp-pool", default=DEFAULT_OPP_POOL,
                    help="**训练**对手池（逗号分隔：live / genome:<str> / ckpt:<path>）；"
                         "pool[0] 恒为 live ⇒ 每局恰好 1 席线上基因组，其余轮换")
    ap.add_argument("--fixture-opp", default=FIXTURE_OPP,
                    help="配对评估的对手（座标固定 3 席）；默认 = 标准夹具 3 席 "
                         "`shanten:1:1:2:2:1:1:0`")
    ap.add_argument("--eval-league", default="",
                    help="G3 困难联赛：3 个逗号分隔的对手规格（覆盖 --fixture-opp）。"
                         "默认关闭。例：live,genome:shanten:1:1:2:2:1:1:0,"
                         "genome:shanten:2:1:2:9:1:1:0")
    ap.add_argument("--eval-blocks", type=int, default=8,
                    help="块间 SE 的块数（预注册要求 ≥8）")
    ap.add_argument("--init", default="", help="从 checkpoint 热启动（BC 产物）")
    ap.add_argument("--value-init", default="", help="独立价值基线（value_v2.pt）；默认空 ⇒ 行为逐字不变")
    ap.add_argument("--dev", default="auto", choices=["auto", "cuda", "cpu"],
                    help="训练侧设备；auto = MJ_USE_GPU=1 且 cuda 可用时用 GPU。"
                         "**小网络实测 CPU 反而更快**（见 --bench-device）")
    ap.add_argument("--keep-tasks", action="store_true",
                    help="保留 rl_tmp/ 里的子进程任务与日志（排查用）")
    ap.add_argument("--bench-device", action="store_true",
                    help="自检里额外量一次 cuda vs cpu 的更新耗时（诚实报告设备收益）")
    ap.add_argument("--run-task", default="", help="（内部）子进程入口：跑一个任务文件")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.run_task:
        return run_task_main(args.run_task)
    args.arm_a_given = ("--arm-a" in sys.argv)
    args.arm_b_given = ("--arm-b" in sys.argv)
    # ★★ 2026-10-09：**防一个代价很大的静默分派** ★★
    #   实测（我自己踩的）：只写 `--arm-a ... --arm-b ...` 而漏了 `--eval-ckpt`，
    #   下面的分派会**掉进 `train(args)`** —— 也就是**静默启动一次完整的 PPO 训练**：
    #   拉起 4 个子进程、写 `rl_ckpt_*.pt` / `rl_history.jsonl` / `rl_train.log`，
    #   同时把正在跑的另一个训练拖慢，而**屏幕上看起来"评估正在跑"**。
    #   ⇒ 与其"猜用户想要评估"，不如**直接报错**：口径歧义时宁可停下。
    if not (args.eval_null_test or args.eval_ckpt) and (args.arm_a_given or args.arm_b_given):
        ap.error("--arm-a / --arm-b 只对**配对评估**有意义，而评估还需要 "
                 "--eval-ckpt（或 --eval-null-test）。\n"
                 "  · 现在这样会静默进入**训练**分支（本检查就是为防这个）。\n"
                 "  · 要比较两个检查点：\n"
                 "      --eval-ckpt <A.pt> --arm-a ckpt:<A.pt> --arm-b ckpt:<B.pt> "
                 "--eval-n <N>\n"
                 "  · 要跑空对照：--eval-null-test --eval-n <N>")
    if args.smoke:
        # 8 房 × 8 手 = 64 手；验证 6 房 × 8 手 = 48 手（两臂共 96 手）
        args.iters, args.games, args.nproc = 2, 8, 4
        args.val_n, args.epochs, args.hidden = 6, 2, 64
        args.log = "rl_smoke.log"
        args.history = "rl_smoke_history.jsonl"
        args.out_prefix = "rl_smoke_ckpt_"
    if args.reaggregate:
        return reaggregate(args)
    if args.dead_inputs:
        return dead_inputs(args)
    if args.selfcheck:
        return selfcheck(args)
    if args.bc:
        return run_bc(args)
    if args.eval_null_test or args.eval_ckpt:
        return run_eval(args)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
