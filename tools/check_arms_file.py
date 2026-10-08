# -*- coding: utf-8 -*-
"""tools/check_arms_file.py -- 复刻客户端的解析方式，验证臂文件格式 ✓

★ 为什么要这个 ✓：`run_live.py:514` 的解析是
      `open(ARMS_FILE, encoding="utf-8").read().strip().split(",")`
   ⇒ ★★ **逗号分隔**（**不是按行** ✗）。
   ★ 我（§Z.1543）曾把它写成**两行** ✗ ⇒ `split(",")` 只得到 1 个元素 ✗
   ⇒ ★ **臂名非法 ⇒ 解析失败** ✗ ⇒ 差点把线上搞坏 ✗✗
   ⇒ ★ 所以：**改臂文件后必须跑本脚本** ✓（与 `py_compile` 同等地位 ✓）
"""
import io
import os
import sys

F = sys.argv[1] if len(sys.argv) > 1 else "live_ab_arms.txt"

if not os.path.exists(F):
    print("  FATAL 文件不存在：%s" % F)
    sys.exit(1)

raw = io.open(F, encoding="utf-8").read()
s = raw.strip()
arms = s.split(",")
print("  原文 = %r" % raw)
print("  strip 后 = %r" % s)
print("  split(',') => %r" % (arms,))
print("  臂数 = %d" % len(arms))

bad = []
if "\n" in s:
    bad.append("★ strip 后仍含换行 ⇒ 臂名会被污染 ✗（必须单行 ✓）")
for i, a in enumerate(arms):
    if a != a.strip():
        bad.append("★ 第 %d 个臂名首尾有空白：%r ✗" % (i, a))
    if "\ufeff" in a:
        bad.append("★ 第 %d 个臂名含 BOM ✗（§Z.1429 那个 bug 的同类 ✓）" % i)
    if not (a.startswith("ckpt:") or a.startswith("shanten:")
            or a.startswith("rank") or a.startswith("genome:")):
        bad.append("★ 第 %d 个臂名不像合法臂：%r ✗" % (i, a))
    if a.endswith("_latest.pt"):
        bad.append("★ 第 %d 个臂指向 `_latest` ✗（**必须冻结** ✗ 见 §Z.1360 ✓）" % i)

if bad:
    for b in bad:
        print("  " + b)
    print("  => **不合格** ✗ 不要上线 ✗")
    sys.exit(2)

print("  ✓ 合格：%s" % " ｜ ".join(arms))
