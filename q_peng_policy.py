#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**碰不碰交给学到的 Q** —— 强化学习策略的部署侧（第 124 轮）。

为什么用"基因值"而不是新策略类来接入
------------------------------------
`mj_arena.make_strategy(name)` 对不认识的字符串会回落到
`Evolved(parse_genome(name))` ⇒ 只要让 `peng=9` 这个**新档位**表示
"碰窗口交给学到的 Q"，那么**全项目现有的评测工具全部立即可用**
（`probe_posture_frontier._play`、座位拆分、`run_shards`、配对夹具），
**不需要注册任何新类、不需要改评测代码**。这与 `decline=5` 是同一个手法。

⚠ **必须优雅降级**：模型文件缺失时**不能抛异常**，否则会拖垮
`make_strategy("shanten:9:...")` 的每一次构造 —— 而线上循环也在用同一个
工厂函数。⇒ 拿不到模型就**退回基因组原本的碰逻辑**（保守、确定、不会更差）。

用法
----
    genome = "shanten:9:1:2:2:1:1:0"     # peng=9 → 用 Q 决定碰/杠/过
"""
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "_pkgs")            # torch 在 _pkgs（pip 被沙箱拦）

A_PENG, A_CHI, A_GANG, A_HU, A_PASS = 34, 35, 36, 37, 38

_M = {"net": None, "mu": None, "sd": None, "acts": None, "tried": False}


def available():
    """模型是否可用（**不抛异常**）。"""
    load()
    return _M["net"] is not None


def load(path="q_peng.pt"):
    if _M["tried"]:
        return _M["net"] is not None
    _M["tried"] = True
    try:
        import torch
        import train_q_peng as TQ
        d = torch.load(path, map_location="cpu", weights_only=False)
        net = TQ.QNet(d["din"], len(d["acts"]), d["hidden"], d["depth"], 0.0)
        net.load_state_dict(d["state"])
        net.eval()
        _M.update(net=net, mu=d["mu"], sd=d["sd"], acts=d["acts"])
        return True
    except Exception as e:                                        # noqa: BLE001
        # 静默降级是对的：调用方会退回基因组逻辑。
        # 但**第一次**要留一行痕迹，免得"模型没加载上"变成又一个静默失真。
        sys.stderr.write("[q_peng] 模型不可用（%r）⇒ 退回基因组碰逻辑\n" % (e,))
        return False


def q_values(v):
    """(3,) 的 Q 值，顺序与 `acts` 一致（碰/杠/过）。取不到返回 None。"""
    if not load():
        return None
    import numpy as np
    import torch
    from features import features
    try:
        f = np.asarray([features(v)], dtype=np.float32)
    except Exception:                                             # noqa: BLE001
        return None
    f = (f - _M["mu"]) / _M["sd"]
    with torch.no_grad():
        q = []
        for i in range(len(_M["acts"])):
            oh = torch.zeros(1, len(_M["acts"]))
            oh[0, i] = 1.0
            sc, _w = _M["net"](torch.from_numpy(f), oh)
            q.append(float(sc[0]))
    return np.asarray(q)


def decide(v, legal=(A_PENG, A_PASS)):
    """在给定合法动作里按 Q 取 argmax。取不到模型或 Q 时返回 None（调用方降级）。"""
    q = q_values(v)
    if q is None:
        return None
    idx = [i for i, a in enumerate(_M["acts"]) if a in legal]
    if not idx:
        return None
    best = max(idx, key=lambda i: q[i])
    return _M["acts"][best]
