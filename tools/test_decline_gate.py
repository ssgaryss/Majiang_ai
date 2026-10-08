#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_decline_gate.py - 定向验证 MJ_DECLINE_GATE 包装器的**逻辑**（离线 ✓ 不碰线上 ✓）。

做法：用桩替换 `_turn_candidates_impl`（返回合成候选）⇒ 只测包装器本身 ✓
      `decline_gate_ok` 也用桩控制 ⇒ 三种情形可枚举 ✓
为什么这样测：自检的统计口径对候选过滤**不敏感**（§Z.1154 ✗）⇒ 必须直接断言候选集合 ✓
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rl_action_space as A

HU, GA, DI = A.K_HU, A.K_GANG, A.K_DISCARD
fails = []


def stub(cands):
    A._turn_candidates_impl = lambda v: (list(cands), [[0.0]] * len(cands))


def run(env, gate_ok, cands, label, want_kinds):
    A._turn_candidates_impl = lambda v: (list(cands), [[0.0]] * len(cands)); os.environ["MJ_DECLINE_GATE"] = env
    A._decline_gate_ok = lambda v: gate_ok
    got, feats = A.turn_candidates(object())
    kinds = [c["kind"] for c in got]
    ok = (kinds == want_kinds) and (len(feats) == len(got))
    print("  %-4s %-46s got=%-22s want=%-22s" % ("ok" if ok else "FAIL", label, kinds, want_kinds))
    if not ok:
        fails.append(label)


print("=== 定向测试：MJ_DECLINE_GATE 包装器 ===")
C = [{"kind": HU, "tile": 5}, {"kind": GA, "tile": None}, {"kind": DI, "tile": 7}]
run("0", False, C, "门控关 ⇒ 必须原样透传（线上零影响）", [HU, GA, DI])
run("1", True,  C, "门控开 + 闸门成立 ⇒ 原样透传（交给网络）", [HU, GA, DI])
run("1", False, C, "★ 门控开 + 闸门不成立 ⇒ 只剩 hu（强制胡）", [HU])
C2 = [{"kind": GA, "tile": None}, {"kind": DI, "tile": 7}]
run("1", False, C2, "门控开 + 闸门不成立 但**不能胡** ⇒ 原样透传", [GA, DI])
run("1", False, [{"kind": DI, "tile": 1}], "门控开 + 只有打牌候选 ⇒ 原样透传", [DI])

print("⇒ " + ("**全部通过** ✓" if not fails else "**失败 %d 项** ✗ %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)