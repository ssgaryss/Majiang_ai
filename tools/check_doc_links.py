#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 Markdown 文档里的**本地链接**是否都存在（死链检查）。

为什么需要：交付物说明书会被别人照着操作 ⇒ 一个指向不存在文件的链接
就是一次把读者带进沟里。而链接是**手写**的，最容易在改名/删文件后失修
（本项目刚发生过：`SUBMIT.md` 并入 README 后仍留引用）。

只检查**本地**链接：
  · `[文字](路径)` —— 跳过 http(s):// 、mailto: 、纯锚点 `#...`
  · 去掉路径里的 `#锚点` 与查询串
  · 相对路径按**文档所在目录**解析（与阅读者一致）

用法:
  python tools/check_doc_links.py README.md [...]
  python tools/check_doc_links.py --dir mahjong_ai_submit      # 检查目录下所有 .md
"""
import argparse
import os
import re
import sys

# ★ 控制台编码兜底：本工具会打印 ✓/✗ 与中文。Windows 默认 GBK 控制台下
#   print("✓") 会抛 UnicodeEncodeError 直接崩（实测过）。本仓其它脚本
#   （my_bot.py / mj_core.py / run_live.py）都有这段，这里也必须补上。
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# [文字](目标)  —— 目标里不出现右括号
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# `反引号` 里像**文件路径**的东西（README 里大量文件名是这样写的，不是 markdown 链接）
CODE = re.compile(r"`([^`\n]+)`")
FILEISH = re.compile(r"^[\w\-./\\]+\.(?:py|md|txt|pt|npz|jsonl|log|json|csv|pth|whl)$")


def check(path, verbose=False):
    base = os.path.dirname(os.path.abspath(path))
    bad, total = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            for target in LINK.findall(line):
                t = target.strip()
                if t.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                # 去掉锚点与查询串（标题锚点、?x=1）
                t = t.split("#", 1)[0].split("?", 1)[0].strip()
                if not t:
                    continue
                total += 1
                resolved = os.path.normpath(os.path.join(base, t))
                if not os.path.exists(resolved):
                    bad.append((lineno, target.strip(), resolved))
                elif verbose:
                    print("    OK  %s:%d -> %s" % (os.path.basename(path), lineno, t))
    return total, bad


def check_paths(path, extra_roots=()):
    """扫**反引号**里像文件路径的引用。

    与 `check()` 的区别：这些不是 markdown 链接，所以**不算死链错误**，
    只列成"待人工确认"——因为文档里合法地会出现通配符、示例路径、
    以及运行期才存在的产物名（如 `live.pid`）。但**指向本包内不存在的
    源文件**必须被抓出来，那正是改名/删文件后的失修特征。"""
    base = os.path.dirname(os.path.abspath(path))
    roots = [base] + [os.path.abspath(r) for r in extra_roots]
    miss, total = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            for tok in CODE.findall(line):
                t = tok.strip()
                if "*" in t or not FILEISH.match(t):
                    continue          # 通配符/示例名，跳过
                total += 1
                t2 = t.replace("\\", "/").lstrip("/")
                if any(os.path.exists(os.path.normpath(os.path.join(r, t2)))
                       for r in roots):
                    continue
                miss.append((lineno, t))
    return total, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dir", default="")
    ap.add_argument("--roots", nargs="*", default=[],
                    help="反引号路径引用可以额外在这些目录里解析（如开发仓根）")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    targets = list(a.files)
    if a.dir:
        for dp, _dn, fns in os.walk(a.dir):
            for fn in fns:
                if fn.endswith(".md"):
                    targets.append(os.path.join(dp, fn))
    if not targets:
        print("没有要检查的文档")
        return 0

    rc = 0
    for p in targets:
        if not os.path.isfile(p):
            print("  !! 文档不存在: %s" % p)
            rc = 1
            continue
        total, bad = check(p, a.verbose)
        name = os.path.relpath(p)
        if bad:
            rc = 1
            print("  ✗ %s：%d 个本地链接，其中 %d 个是死链" % (name, total, len(bad)))
            for lineno, raw, resolved in bad:
                print("      :%d  %s" % (lineno, raw))
                print("          -> 找不到 %s" % resolved)
        else:
            print("  ✓ %s：%d 个本地链接全部有效" % (name, total))

        # 反引号里的路径引用（不算死链，但包内源文件缺失要报出来）
        ptotal, pmiss = check_paths(p, extra_roots=a.roots)
        if pmiss:
            print("      · 反引号路径引用 %d 个，其中 %d 个在本包/仓内找不到"
                  "（可能是示例名或运行期产物，请人工确认）："
                  % (ptotal, len(pmiss)))
            for lineno, tok in pmiss:
                print("          :%d  %s" % (lineno, tok))
        else:
            print("      · 反引号路径引用 %d 个全部可解析" % ptotal)
    return rc


if __name__ == "__main__":
    sys.exit(main())
