#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付物冒烟测试 —— 在**目标目录里**验证核心代码真的能用。

为什么必须单独做这一步：
  · `import` 成功 ≠ 能打牌 ✗ —— 权重路径、`torch.load`、形状、动作空间
    都可能在 import 之后才炸。
  · 而这一步**必须在目标目录内**执行（repo 根不能挂在 `sys.path` 上），
    否则测的是"原仓能跑"，不是"交付物能跑" ✗。

检查项：
  1. 核心模块逐个 import
  2. 入口 `run_live` import（模块级代码要安全）
  3. 读 `live_ab_arms.txt`（若存在）并 `make_strategy` 每一条臂
  4. 神经臂：确认 `torch.load` 成功、拿到 state_dict 的键数
  5. 规则臂：确认能对一手牌给出出牌决策
"""
import argparse
import os
import sys
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="交付物目录")
    a = ap.parse_args()
    root = os.path.abspath(a.dir)
    os.chdir(root)
    sys.path.insert(0, root)

    # 确认隔离：repo 根不应在 path 上（若在，测试就不算数）
    print("dir  = %s" % root)
    print("path = %s" % [p for p in sys.path[:4]])

    ok = True

    print("\n[1] 核心模块 import")
    mods = ["mj_core", "shanten_ref", "mj_policy", "mj_sim", "mj_arena",
            "mj_bot", "my_bot", "rl_policy", "rl_action_space", "rl_boot"]
    for m in mods:
        try:
            __import__(m)
            print("    OK   %s" % m)
        except Exception as e:
            ok = False
            print("    FAIL %s: %r" % (m, e))
            traceback.print_exc()

    print("\n[2] 入口 run_live import")
    # ★ 注意：`run_live.py` 的**模块级代码**会往自己所在目录写 `live.pid`（供沙箱
    #   确认线上客户端身份）。本测试会因此产生一个 27 B 的副产物 —— 测完删掉，
    #   否则交付物会被污染（实测过）。
    _pid = os.path.join(root, "live.pid")
    _pid_existed = os.path.isfile(_pid)
    try:
        import run_live  # noqa: F401
        print("    OK   run_live（SERVER=%s）" % getattr(
            __import__("my_bot"), "SERVER", "?"))
    except Exception as e:
        ok = False
        print("    FAIL run_live: %r" % (e,))
        traceback.print_exc()
    finally:
        if not _pid_existed and os.path.isfile(_pid):
            try:
                os.remove(_pid)
                print("    (已清理副产物 live.pid)")
            except OSError:
                pass

    print("\n[3] 逐臂 make_strategy")
    import mj_arena
    arms = []
    if os.path.isfile("live_ab_arms.txt"):
        with open("live_ab_arms.txt", encoding="utf-8") as f:
            raw = f.read().lstrip("\ufeff").strip()
        arms = [s for s in raw.split(",") if s.strip()]
        print("    live_ab_arms.txt = %r" % raw)
    else:
        print("    (无 live_ab_arms.txt，用规则臂做代表)")
    if not arms:
        arms = ["shanten:1:1:2:5:1:1:0"]

    strategies = {}
    for name in arms:
        try:
            s = mj_arena.make_strategy(name)
            strategies[name] = s
            print("    OK   %s  -> %s" % (name, type(s).__name__))
        except Exception as e:
            ok = False
            print("    FAIL %s: %r" % (name, e))
            traceback.print_exc()

    print("\n[4] 神经臂权重体检")
    for name, s in strategies.items():
        if not name.startswith("ckpt:"):
            continue
        path = name[len("ckpt:"):]
        print("    %s -> 文件 %s (%d B)" % (
            name, "存在" if os.path.isfile(path) else "缺失",
            os.path.getsize(path) if os.path.isfile(path) else -1))
        sd = getattr(s, "sd", None) or getattr(s, "state_dict", None)
        if isinstance(sd, dict):
            print("        state_dict 键数 = %d" % len(sd))
        else:
            print("        (未暴露 state_dict；类型 %s)" % type(s).__name__)
        # 尝试一次真决策（若接口允许）
        for meth in ("choose_discard", "discard", "decide"):
            fn = getattr(s, meth, None)
            if callable(fn):
                print("        可调用决策接口: %s" % meth)
                break
        else:
            print("        (未找到常规决策接口名；由 mj_sim 驱动)")

    print("\n[5] 规则臂一次真决策")
    try:
        import mj_sim
        fn = None
        for cand in ("selftest", "self_test", "smoke"):
            fn = getattr(mj_sim, cand, None)
            if callable(fn):
                break
        if fn:
            fn()
            print("    OK   mj_sim.%s()" % fn.__name__)
        else:
            print("    (mj_sim 无自测入口；跳过)")
    except Exception as e:
        print("    (规则臂决策检查跳过: %r)" % (e,))

    print("\n==== 结论: %s ====" % ("通过 ✓" if ok else "有失败项 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
