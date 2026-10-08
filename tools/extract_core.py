#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从入口脚本出发，**自动算传递导入闭包**，把核心代码抽到独立目录。

为什么必须自动算（而不是手写白名单）：
  · 本仓顶层有 400+ 个 .py，其中绝大多数是探针/诊断/一次性分析脚本，
    手写清单**一定会漏**（漏一个 ⇒ 交付物 import 失败 ⇒ 比赛现场炸）。
  · 导入关系是**客观事实**，可以从源码用 ast 精确解析 —— 比任何人工清单都可靠。

做法：
  1. 从 --roots 里的每个入口开始，用 ast 解析 `import X` / `from X import Y`
     / `from . import Z`，只看**本仓内**的模块（顶层 .py 或包目录）。
  2. 递归收集，直到闭包不动。
  3. 复制到 --out，保持相对路径；可选地把 --assets 里的非代码文件一起带上。
  4. 打印：文件数、总字节、逐个大小、以及**任何解析不到的 import**（用于人工确认
     它是不是第三方库）。

★ 只读本仓；不修改任何源文件。
"""
import argparse
import ast
import os
import shutil
import sys

# ★ 控制台编码兜底（Windows GBK 控制台打印中文会崩；本仓其它脚本同做法）
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ★ 标准库白名单不用手写：用 sys.stdlib_module_names（3.10+）判定即可。
STDLIB = set(getattr(sys, "stdlib_module_names", ()))


class ExtendAction(argparse.Action):
    """让重复出现的 `nargs='*'` 选项**累加**而不是互相覆盖。

    为什么必须这样：argparse 对同一个 `nargs='*'` 选项出现多次时，
    **最后一次会覆盖前面所有次**（默认 store 语义）。实测代价 ——
    `--map A=B --map C=D` 只生效 C=D，A=B **被静默丢弃且不报错** ✗。
    这与本仓记录过的「传了参数而输出没变 ⇒ 立刻怀疑参数没被用」是同一类坑，
    所以在工具侧根治：重复给就累加。"""
    def __call__(self, parser, namespace, values, option_string=None):
        cur = getattr(namespace, self.dest, None) or []
        setattr(namespace, self.dest, list(cur) + list(values))


def local_module_name(repo, mod_top):
    """把顶层模块名解析成仓内文件/包；不是本仓的就返回 None。"""
    py = os.path.join(repo, mod_top + ".py")
    if os.path.isfile(py):
        return py
    pkg = os.path.join(repo, mod_top, "__init__.py")
    if os.path.isfile(pkg):
        return pkg
    return None


def imports_of(path):
    """返回该文件里所有**可能是本仓模块**的顶层名（已去掉相对导入的点）。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        print("  !! 语法错误，跳过解析: %s (%s)" % (path, e))
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level>0 = 相对导入；本仓是**扁平的**（顶层 .py），相对导入基本不出现，
            # 但仍然尝试用 module 名解析，解析不到就报出来由人工确认。
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def closure(repo, roots):
    seen = {}          # 绝对路径 -> None
    missing = {}       # 名字 -> 引用它的文件集合
    queue = []
    for r in roots:
        p = r if os.path.isabs(r) else os.path.join(repo, r)
        if not os.path.isfile(p):
            print("  !! 入口不存在: %s" % p)
            continue
        queue.append(p)
    while queue:
        cur = queue.pop()
        key = os.path.normcase(os.path.abspath(cur))
        if key in seen:
            continue
        seen[key] = os.path.abspath(cur)
        for name in imports_of(cur):
            if name in STDLIB:
                continue
            target = local_module_name(repo, name)
            if target:
                queue.append(target)
            else:
                missing.setdefault(name, set()).add(os.path.basename(cur))
    return sorted(seen.values()), missing


def scan_assets(files, repo):
    """在闭包文件里找出**字符串字面量形式**的资源名（.pt/.npz/.json/…），
    并报告它是否存在。用途：防止「代码抽全了、权重忘了带」这种现场事故。"""
    import re
    pat = re.compile(r"""["']([A-Za-z0-9_][A-Za-z0-9_\-.]*\.(?:pt|npz|pth|pkl|json|txt|csv))["']""")
    found = {}
    for p in files:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        try:
            tree = ast.parse(src, filename=p)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                m = pat.fullmatch(node.value)
                if m:
                    found.setdefault(m.group(1), set()).add(os.path.relpath(p, repo))
    rows = []
    for name in sorted(found):
        exists = os.path.isfile(os.path.join(repo, name))
        rows.append((name, exists, sorted(found[name])))
    return rows


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.2f %s" % (n, unit)
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="入口脚本（相对 repo）")
    ap.add_argument("--assets", nargs="*", default=[], action=ExtendAction,
                    help="额外要带上的非代码文件（相对 repo，缺省忽略不存在的）★ 可重复，累加 ✓")
    ap.add_argument("--dirs", nargs="*", default=[], action=ExtendAction,
                    help="额外要整体带上的目录（相对 repo；自动跳过 __pycache__/*.pyc）★ 可重复，累加 ✓")
    ap.add_argument("--map", nargs="*", default=[], action=ExtendAction,
                    help="改名搬运 SRC=DST（相对 repo 的源 -> 交付物内的目标路径）。"
                         "用途：手写的交付物 README 在仓里叫 README_BOT.md（仓根已有"
                         "历史 README.md，不能覆盖），进交付物时改成 README.md。"
                         "★ 可重复，累加 ✓")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="逐个列出闭包文件")
    a = ap.parse_args()

    repo = os.path.abspath(a.repo)
    files, missing = closure(repo, a.roots)

    print("=== 导入闭包 ===")
    total = 0
    groups = {}
    for p in files:
        rel = os.path.relpath(p, repo)
        sz = os.path.getsize(p)
        total += sz
        top = rel.split(os.sep)[0] if os.sep in rel else "(顶层)"
        g = groups.setdefault(top, [0, 0])
        g[0] += 1
        g[1] += sz
    for top in sorted(groups):
        c, s = groups[top]
        print("  %-14s %3d 个  %10s" % (top, c, human(s)))
    print("  %-14s %3d 个  %10s" % ("合计", len(files), human(total)))

    if a.list:
        print("\n=== 闭包文件逐个列出 ===")
        for p in sorted(files, key=lambda x: os.path.relpath(x, repo)):
            print("  %-46s %9s" % (os.path.relpath(p, repo), human(os.path.getsize(p))))

    print("\n=== 解析不到的 import（应为第三方库；请人工确认）===")
    if not missing:
        print("  (无)")
    for name in sorted(missing):
        who = sorted(missing[name])
        print("  %-24s <- %s" % (name, ", ".join(who[:4]) + (" …" if len(who) > 4 else "")))

    assets = []
    for rel in a.assets:
        p = os.path.join(repo, rel)
        if os.path.isfile(p):
            assets.append(p)
        else:
            print("  (资源不存在，跳过) %s" % rel)
    atotal = sum(os.path.getsize(p) for p in assets)

    print("\n=== 代码引用的资源名（★ 检查有没有忘带）===")
    for name, exists, who in scan_assets(files, repo):
        mark = "在" if exists else "缺"
        print("  [%s] %-30s <- %s" % (mark, name, ", ".join(who[:3])))

    print("\n=== 合计 ===")
    extra = list(assets)
    for d in a.dirs:
        root_d = os.path.join(repo, d)
        if not os.path.isdir(root_d):
            print("  (目录不存在，跳过) %s" % d)
            continue
        for dirpath, dirnames, filenames in os.walk(root_d):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in filenames:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                extra.append(os.path.join(dirpath, fn))
    etotal = sum(os.path.getsize(p) for p in extra)
    print("  代码 %d 个 = %s" % (len(files), human(total)))
    print("  资源 %d 个 = %s（含 --assets 与 --dirs）" % (len(extra), human(etotal)))
    print("  ★ 总计 = %s" % human(total + etotal))

    if a.dry_run:
        print("\n(--dry-run：未复制)")
        return 0

    out = os.path.abspath(a.out)
    if os.path.isdir(out):
        print("\n清空已存在的输出目录: %s" % out)
        shutil.rmtree(out)
    os.makedirs(out)
    for p in files:
        rel = os.path.relpath(p, repo)
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(p, dst)
    for p in extra:
        rel = os.path.relpath(p, repo)
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(p, dst)
    for spec in a.map:
        if "=" not in spec:
            print("  !! --map 需要 SRC=DST 形式，跳过: %r" % spec)
            continue
        src_rel, dst_rel = spec.split("=", 1)
        src = os.path.join(repo, src_rel)
        if not os.path.isfile(src):
            print("  !! --map 源不存在，跳过: %s" % src_rel)
            continue
        dst = os.path.join(out, dst_rel)
        os.makedirs(os.path.dirname(dst) or out, exist_ok=True)
        shutil.copy2(src, dst)
        print("  --map %s -> %s" % (src_rel, dst_rel))
    print("\n已写出: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
