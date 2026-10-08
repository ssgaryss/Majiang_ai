#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core_manifest.py - 静态算出「线上核心代码」的依赖闭包与体积（不执行任何东西）。

为什么需要：交付要求「核心代码 <= 20MB」，但仓库里混着数据/依赖/一次性工具。
本工具只做 AST 解析（**绝不 import/执行 run_live.py**，否则会启动第二个客户端 ✗）。

用法:
  python tools/core_manifest.py                 # 打印清单 + 合计
  python tools/core_manifest.py --copy _dist    # 另把闭包拷到 _dist/
"""
import ast, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def local_mods(path):
    """返回该文件 import 到的「本地」模块名（有同名 .py 的才算）。"""
    out = set()
    try:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except Exception as e:
        print("  [warn] parse failed:", path, e); return out
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: out.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.level == 0 and n.module: out.add(n.module.split(".")[0])
    return out

def main():
    args = sys.argv[1:]
    dest = None
    if "--copy" in args:
        dest = args[args.index("--copy") + 1]
    # ★★★ 2026-10-06（§Z.1568）：**改成「多入口」** ✗
    #   ★ 缺陷 ✓：此前 `entry = "run_live"` **只有一个入口** ✗
    #     ⇒ 闭包只覆盖**线上推理**那条链 ✗ ⇒ ★★ **训练脚本完全不在里面** ✗✗
    #     ⇒ ★★★ 这就是 §Z.1555 缺陷① 的**真正根因** ✓：
    #        `.gitignore` 的白名单自称"由本工具静态算出，共 24 个" ✓
    #        而 `train_selfplay_rl.py`（**唯一的主训练脚本** ✓）**从未被算进去** ✗
    #        ⇒ 它**从未进入版本控制** ✗（★ 对它的修改有丢失风险 ✗）
    #   ★ 修法 ✓：**入口是一组** ✓ —— 线上与训练是**两条独立入口** ✓
    #     （★ 两者之间没有 import 关系 ⇒ 单入口必然漏掉另一条 ✓）
    #   ★ 仍然**不 import/执行任何东西** ✓（纯 AST ✓ ⇒ 不会启动第二个客户端 ✓ 见文件头 :6 ✓）
    ENTRIES = [
        "run_live",            # 线上 A/B 客户端 ✓
        "run_room",            # 单房演练 ✓
        "train_selfplay_rl",   # ★★ 唯一的主训练脚本（PPO／自对弈 ✓）
        "train_value",         # 价值头 ✓
        "train_cand_ranker",   # 候选排序头 ✓
        "analyze_live_paired",  # ★ 唯一有权宣布"模型更强"的判据工具 ✓
        "mj_bot",              # 机器人入口 ✓
    ]
    seen, queue = set(), list(ENTRIES)
    while queue:
        m = queue.pop()
        if m in seen: continue
        f = os.path.join(ROOT, m + ".py")
        if not os.path.isfile(f): continue
        seen.add(m)
        for d in local_mods(f):
            if os.path.isfile(os.path.join(ROOT, d + ".py")): queue.append(d)
    # 运行期还会读这些（非 .py）
    extra = ["live_ab_arms.txt"]
    rows, total = [], 0
    for m in sorted(seen):
        f = os.path.join(ROOT, m + ".py"); sz = os.path.getsize(f)
        rows.append((m + ".py", sz)); total += sz
    for e in extra:
        f = os.path.join(ROOT, e)
        if os.path.isfile(f):
            sz = os.path.getsize(f); rows.append((e, sz)); total += sz
    print("=== 核心代码闭包（静态 AST 扫描，未执行任何代码）===")
    for name, sz in rows:
        print("  %-28s %7.1f KB" % (name, sz / 1024.0))
    print("  %-28s %7.2f MB" % ("[小计 代码]", total / 1024.0 / 1024.0))
    print("  %-28s %7.2f MB" % ("[+1 个模型权重]", 3.81))
    print("  %-28s %7.2f MB   (预算 20 MB)" % ("[合计]", total / 1024.0 / 1024.0 + 3.81))
    if dest:
        d = os.path.join(ROOT, dest)
        os.makedirs(d, exist_ok=True)
        for name, _ in rows:
            shutil.copy2(os.path.join(ROOT, name), os.path.join(d, name))
        print("已拷贝到", d)

if __name__ == "__main__":
    main()