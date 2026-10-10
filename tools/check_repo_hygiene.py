# -*- coding: utf-8 -*-
"""tools/check_repo_hygiene.py -- 仓库卫生自检 ✓（防 §Z.1555 的两类缺陷复发 ✓）

★★ 设计要点（**不调用 git** ✗）：
   ★ 试过用 `subprocess` 调 `git ls-files` ⇒ **rc=127** ✗
     （= 沙箱对「程序捕获另一个程序的管道输出」的**已知边界** ✓ ⇒ 见 harness 说明 ✓）
   ⇒ ★★ 改为**纯文件读取** ✓ —— ★ 这已经足够 ✓，因为
     §Z.1555 两类缺陷的**根因都在 `.gitignore` 的白名单** ✓✓：
       · 缺陷① 漏白名单 ✗ ⇒ 白名单里**找不到**核心文件 ✓（可检出 ✓）
       · 缺陷② 误放行敏感文件 ✗✗ ⇒ 白名单里**出现** `global_token.txt` ✓（可检出 ✓）

★ 检查项 ✓：
   A. 放行行（`!…`）里**不得**出现敏感名（token/secret/key/password/credential/…）✗
   B. `tools/core_manifest.py`（若存在）＋ **内置最小清单**里的核心文件**必须**出现在放行行里 ✓
   C. `.gitignore` 必须以 `/*` 起手（白名单式 ✓）且**必须**有 `!.gitignore` ✓

★ 退出码 ✓：0 = 干净 ✓；1 = 发现问题 ✗（可直接挂 pre-commit ✓）
用法 ✓：  .\\_venv\\Scripts\\python.exe tools\\check_repo_hygiene.py
"""
from __future__ import annotations

import io
import os
import re
import sys

# ★ 控制台编码兜底：Windows 默认 GBK 控制台打印 ✓/✗ 会抛 UnicodeEncodeError ✗
#   （本仓其它脚本都有这一句，本文件此前漏了 ⇒ 工具自己崩在最后一行 ✓ 已修 ✓）
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SENSITIVE = re.compile(r"(token|secret|passwd|password|credential|\.pem$|id_rsa|apikey|api_key)",
                       re.I)

# ★ 内置的最小核心清单 ✓（§Z.1555 漏掉的第一个就是它 ✗）
CORE_MIN = [
    "train_selfplay_rl.py",     # ★ 唯一的主训练脚本 ✓（含 --aux／--nstep／--self-imit ✓）
    "run_live.py",
    "mj_core.py",
    "mj_arena.py",
    "rl_action_space.py",
    "rl_policy.py",
    "analyze_live_paired.py",   # ★ 唯一有权宣布"模型更强"的工具 ✓
]


def _whitelist(lines):
    """返回 `.gitignore` 里的放行集合（`!/path` → 去掉 `!` 与 `/` 的**基名** ✓）✓"""
    out = set()
    for ln in lines:
        s = ln.strip()
        if s.startswith("!") and not s.startswith("!#"):
            out.add(os.path.basename(s[1:].lstrip("/")))
    return out


def main():
    bad, notes = [], []
    ign = ".gitignore"
    if not os.path.exists(ign):
        print("  ✗ 找不到 .gitignore ⇒ 无法自检")
        return 1
    lines = io.open(ign, encoding="utf-8", errors="replace").read().splitlines()
    wl = _whitelist(lines)

    # ── A. 放行行里不得有敏感名 ✗ ────────────────────────────────────────
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("!") and SENSITIVE.search(s):
            bad.append("A ✗ .gitignore:%d **放行了敏感文件** ✗：%s" % (i, s))

    # ── B. 核心文件必须出现在放行集合里 ✓ ───────────────────────────────
    names = []
    mani = os.path.join("tools", "core_manifest.py")
    if os.path.exists(mani):
        # ★★ 2026-10-06（§Z.1568）：★ 我用**正则**提了两版都漏 ✗（先要求 `.py` ✗ ⇒ 只得 7 个；
        #    再要求引号后紧跟逗号 ✗ ⇒ 只得 11 个；放宽后仍只 12 个 ✗）
        #    ⇒ ★★ 依我自己的新硬规则（§Z.1564：**别猜，读全** ✗）⇒ ★ **改用 AST** ✓✓
        #    ★ 好处 ✓：① 不会被注释/换行/写法骗到 ✓ ② **纯解析，不执行** ✓
        #      （★ 与 `core_manifest.py` 自己的原则一致 ✓ 见其文件头 :6 ✓ ⇒ 不会启动第二个客户端 ✓）
        try:
            import ast
            tree = ast.parse(io.open(mani, encoding="utf-8", errors="replace").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    v = node.value.strip()
                    if not v or " " in v or os.sep in v:
                        continue                      # 跳过说明文字/路径 ✓
                    if v.endswith(".py"):
                        names.append(v)
                    elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
                        names.append(v + ".py")       # 裸模块名 ✓
        except Exception as e:                        # noqa: BLE001
            bad.append("B ! 解析 core_manifest.py 失败：%r" % (e,))
    names += CORE_MIN
    seen = set()
    missing = []
    for n in names:
        b = os.path.basename(n)
        if b in seen:
            continue
        seen.add(b)
        if not os.path.exists(b):
            continue
        if b not in wl:
            bad.append("B ✗ 核心文件**不在放行白名单** ✗：%s（应加一行 `!/%s` ✓）" % (b, b))

    # ── C. 白名单式结构与自保 ✓ ─────────────────────────────────────────
    if not any(ln.strip() == "/*" for ln in lines):
        bad.append("C ✗ 缺少 `/*`（白名单式 .gitignore 的第一行必须忽略一切 ✓）")
    if ".gitignore" not in wl:
        bad.append("C ✗ 放行集合里没有 `.gitignore` 自身 ✗（它必须自保 ✓）")
    # ★ 反向检查 ✓：令牌文件**不得**在白名单里 ✗（§Z.1555 缺陷② ✓）
    for cand in ("global_token.txt", "token.txt"):
        if cand in wl:
            bad.append("C ✗✗ **令牌文件被放行** ✗：%s（= §Z.1555 的安全缺陷 ✗）" % cand)

    notes.append("放行白名单条目数 = %d" % len(wl))
    notes.append("检查的核心文件数 = %d" % len(seen))
    for n in notes:
        print("  " + n)
    if bad:
        for b in bad:
            print("  " + b)
        print("  => **不合格** ✗（§Z.1555 的两类缺陷任一出现都会到这里 ✓）")
        return 1
    print("  ✓ 干净：白名单里无敏感文件 ✓；核心文件全部在放行集合里 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
