#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_tmp_torch_probe.py - 验证 run_train.py 启动通道 + torch 可用 + 能读价值数据。"""
import json, os, sys
import torch
print("[probe] torch", torch.__version__, "cuda:", torch.cuda.is_available())
x = torch.randn(4, 3); y = torch.randn(4, 1)
w = torch.zeros(3, 1, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)
for _ in range(5):
    loss = ((x @ w - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print("[probe] 梯度下降 OK, loss =", float(loss))
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_tmp", "value_data.jsonl")
p = os.path.abspath(p)
n = 0
with open(p, encoding="utf-8") as f:
    for ln in f:
        if not ln.strip():
            continue
        d = json.loads(ln); n += 1
        if n == 1:
            print("[probe] 首行维度 =", len(d["s"]), " seat =", d["seat"], " y =", d["y"])
        if n >= 3:
            break
print("[probe] 读价值数据 OK（前 3 行）")
print("[probe] ⇒ 启动通道可用 ✓")