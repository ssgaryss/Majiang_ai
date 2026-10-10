# -*- coding: utf-8 -*-
"""自足性检查：本包能不能**只靠自己**跑起来（不依赖任何其它目录）。

★ 为什么需要它
  本包是从开发仓抽取出来的，抽取时**漏过文件**，而且漏得很难发现：
  `run_live.py` 会 spawn `review_room.py`，后者又 import `room_data` / `mj_replay`……
  链条上任意一环缺失，都**只在跑到那一步才炸**（复盘失败还会被 except 吞掉，
  于是"看上去在跑"，其实数据源已经断了）。
  ⇒ 直接从入口做**传递依赖闭包**，一次把缺口找全。

判据（对每个被 import 的顶层名字 X）：
  · `sys.stdlib_module_names` 里有      ⇒ 标准库 ✓ 不管
  · 本目录有 `X.py` / `X/__init__.py`   ⇒ 本包自带 ✓
  · 在本包的第三方白名单里              ⇒ 依赖包 ✓（见 `_pkgs/`、`requirements.txt`）
  · 以上都不是                          ⇒ **缺失 ✗**（这就是会在运行时炸的那种）

用法：  python tools/check_selfcontained.py            # 退出码 0 = 自足
        python tools/check_selfcontained.py --verbose
"""
import argparse
import ast
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 线上路径 + 训练路径的入口
ENTRIES = ["run_live.py", "review_room.py", "mj_bot.py", "train_selfplay_rl.py"]

# `_pkgs/` 与 requirements.txt 提供的第三方包（不是本包源码）
THIRD_PARTY = {"numpy", "torch", "sympy", "networkx", "jinja2", "filelock",
               "fsspec", "mpmath", "markupsafe", "functorch", "torchgen"}


def find_module(name):
    for cand in (os.path.join(ROOT, name + ".py"),
                 os.path.join(ROOT, name, "__init__.py"),
                 os.path.join(ROOT, "tools", name + ".py")):
        if os.path.exists(cand):
            return cand
    return None


def imports_of(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except Exception as e:                                          # noqa: BLE001
        print("  ⚠ 解析失败 %s: %s" % (path, e))
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [al.name.split(".")[0] for al in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                out.append(node.module.split(".")[0])
    return out


def main():
    ap = argparse.ArgumentParser(description="本包自足性检查（传递依赖闭包）")
    ap.add_argument("--verbose", action="store_true", help="把闭包里的每个文件都列出来")
    args = ap.parse_args()

    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    seen, missing, order = set(), {}, []
    stack = list(ENTRIES)

    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            found = find_module(os.path.splitext(rel)[0])
            if not found:
                missing[rel] = True
                continue
            path = found
        order.append(os.path.relpath(path, ROOT))
        for name in imports_of(path):
            if name in stdlib or name in THIRD_PARTY:
                continue
            if name in seen:
                continue
            found = find_module(name)
            if found:
                stack.append(os.path.relpath(found, ROOT))
            else:
                missing[name] = False

    print("=" * 68)
    print("本包自足性检查（根目录 = %s）" % ROOT)
    print("=" * 68)
    print("入口：%s" % ", ".join(ENTRIES))
    print("闭包里的本地模块 %d 个" % len(order))
    if args.verbose:
        for o in sorted(order):
            print("   ✓ %s" % o)
    print("第三方依赖（来自 _pkgs/，非本包源码）：%s" % ", ".join(sorted(THIRD_PARTY)))
    print("")
    if missing:
        print("✗ **缺失 %d 个** —— 这些会在运行时炸：" % len(missing))
        for k in sorted(missing):
            print("   ✗ %s" % k)
        return 1
    print("✓ 自足：闭包内所有本地模块都在本包内，没有任何外部目录依赖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
