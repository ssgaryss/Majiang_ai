#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_decline_gate_realpath.py - 验证门控的**真实调用路径**（离线 ✓）。

★ 为什么必须有这个测试：
   包装器 `_decline_gate_ok` 在异常时**保守返回 False**（= 强制胡 ✗）。
   若 `mj_policy.decline_gate_ok` 在真实 View 上**总是抛异常** ✗
   ⇒ 门控一开就会**无条件强制胡** ✗✗ —— 这是最危险的失败模式 ✓
   本测试直接对拍「真实调用」与「包装器」的返回值 ⇒ 只要二者一致，就说明**没落 except** ✓
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rl_action_space as A
from mj_policy import decline_gate_ok

fails = []


def mk(**kw):
    d = dict(restricted=False, wall=80, dealer=0, seat=1)
    d.update(kw)
    return types.SimpleNamespace(**d)


CASES = [
    ("正常局面（闲家 / 牌墙够）", mk(), True),
    ("restricted=True ⇒ 闸门不成立", mk(restricted=True), False),
    ("牌墙不足（wall-20<4）", mk(wall=22), False),
    ("庄位 ＋ allow_dealer=True ⇒ 放开", mk(dealer=0, seat=0), True),
]

print("=== 真实调用路径对拍（防「异常⇒永远强制胡」）===")
for label, v, want in CASES:
    try:
        direct = bool(decline_gate_ok(v, allow_dealer=True))
    except Exception as e:                                        # noqa: BLE001
        direct = "EXC:" + type(e).__name__
    try:
        wrapped = bool(A._decline_gate_ok(v))
    except Exception as e:                                        # noqa: BLE001
        wrapped = "EXC:" + type(e).__name__
    ok = (direct == want) and (wrapped == direct)
    print("  %-4s %-40s direct=%-6s wrapped=%-6s want=%s"
          % ("ok" if ok else "FAIL", label, direct, wrapped, want))
    if not ok:
        fails.append(label)

print("⇒ " + ("**全部通过** ✓（真实路径可用，未落 except）" if not fails
             else "**失败 %d 项** ✗ %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)