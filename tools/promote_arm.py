# -*- coding: utf-8 -*-
"""tools/promote_arm.py -- ★★ 把「换臂」做成一条命令 ✓（§Z.1600 的 5 步手工流程 ✓）

★★ 为什么需要它 ✓（★ 手工换臂**已经差点出过事** ✗）：
   · §Z.1543 ✗：★ 我把臂文件写成**两行** ✗ ⇒ 客户端 `split(",")` 只得到 1 个元素 ⇒ **臂名非法** ✗
   · §Z.1429 ✗：★ `Set-Content -Encoding UTF8` **会写 BOM** ✗ ⇒ 历史 BOM bug 的源头 ✗
   · §Z.1546 ✗：★ 回退目标必须**真的存在** ✗ ⇒ 否则线上没有可用臂 ✗
   ⇒ ★★ 这些**都是机械检查** ✓ ⇒ ★ 应该由脚本做 ✗ 不该靠我每次记得 ✓

★ 它做 7 件事 ✓（**只做机械动作，不做判断** ✓）：
   ① ★ **核验权重文件存在** ✓ ＋ 体积**合理** ✓（★ 本仓模型约 3.8 MB ✓ 过大/过小都拒 ✗）
   ② ★ **拒绝 `_latest`** ✗（★ 它会**随训练变化** ✗ ⇒ 臂必须**冻结** ✓ 见 `NEXT_RUNBOOK` ✓）
   ③ ★ **备份**现臂文件 ✓（★ 带时间戳 ✓ 落到 `_tmp/` ✓）
   ④ ★ **写入**臂文件 ✓：★ **单行逗号分隔** ✓ ＋ **无 BOM 的 UTF-8** ✓（§Z.1543 ✓）
   ⑤ ★ **回读校验** ✓：★ 用**与客户端相同**的解析方式（`strip().split(",")` ✓）验元素数 ✓
   ⑥ ★ **跑** `tools/check_arms_file.py` ✓（★ 复用既有检查器 ✓ 不重复实现 ✓）
   ⑦ ★ **打印**回退命令 ✓（★ 出事时一条命令回到原臂 ✓）

★ ★★ **它【不】做** ✗（★ 刻意的 ✓）：★ 不判断"该不该换"✗、不动客户端进程 ✗（硬规则② ✓）
   ⇒ ★ 换臂的**时机与理由**仍由人／台账决定 ✓（★ 见 `NEXT_RUNBOOK.md` §Z.1600 ✓）

★ 用法 ✓：
   # 干跑（默认 ✓ 只检查不写）
   .\\_venv\\Scripts\\python.exe tools\\promote_arm.py --ckpt rl_aux_0004.pt
   # 真换（★ 确认无误后再加 --yes ✓）
   .\\_venv\\Scripts\\python.exe tools\\promote_arm.py --ckpt rl_aux_0004.pt --yes
   # 换回（★ 回退 ✓）
   .\\_venv\\Scripts\\python.exe tools\\promote_arm.py --ckpt rl_rn_0002.pt --yes
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

ARMS = "live_ab_arms.txt"
RULE = "shanten:1:1:2:5:1:1:0"          # ★ 规则臂**永不下场** ✓（goal §4.3 ✓）
MIN_MB, MAX_MB = 1.0, 12.0             # ★ 本仓权重约 3.8 MB ✓ ⇒ 超出即拒 ✗


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="权重文件名（如 rl_aux_0004.pt）")
    ap.add_argument("--arms", default=ARMS)
    ap.add_argument("--yes", action="store_true", help="★ 不加就只干跑 ✓")
    a = ap.parse_args()

    ck = a.ckpt.strip()
    fail = []

    # ── ① 权重存在 ＋ 体积合理 ✓ ────────────────────────────────────────
    if not os.path.exists(ck):
        fail.append("① 权重不存在：%s" % ck)
    else:
        mb = os.path.getsize(ck) / 1048576.0
        print("  ① 权重 ✓ %s  %.2f MB" % (ck, mb))
        if not (MIN_MB <= mb <= MAX_MB):
            fail.append("① 体积异常（%.2f MB 不在 %.1f~%.1f）⇒ 可能是半成品/损坏" % (mb, MIN_MB, MAX_MB))

    # ── ② 拒绝 _latest ✗ ────────────────────────────────────────────────
    if "_latest" in ck:
        fail.append("② **拒绝 `_latest`** ✗（它会随训练变化 ⇒ 臂必须冻结）")
    else:
        print("  ② 非 `_latest` ✓（已冻结 ✓）")

    # ── ③ 备份现臂 ✓ ────────────────────────────────────────────────────
    bak = None
    cur = ""
    if os.path.exists(a.arms):
        cur = open(a.arms, encoding="utf-8-sig").read().strip()
        print("  ③ 现臂 = %r" % cur)
        os.makedirs("_tmp", exist_ok=True)
        bak = os.path.join("_tmp", "live_ab_arms.bak_%s.txt" % time.strftime("%H%M%S"))
        shutil.copy2(a.arms, bak)
        print("     备份 ✓ %s" % bak)
    else:
        print("  ③ 臂文件不存在 ⇒ 将新建 ✓")

    # ── ④/⑤ 写入 ＋ 回读校验（★ 客户端同款解析 ✓）────────────────────────
    content = "ckpt:%s,%s" % (ck, RULE)
    parts = content.strip().split(",")
    print("  ④ 目标内容 = %r" % content)
    print("  ⑤ 客户端同款解析 ⇒ 元素数 = %d" % len(parts))
    if len(parts) != 2:
        fail.append("⑤ 解析出 %d 个元素（应为 2）⇒ 客户端会得到非法臂名" % len(parts))
    for p in parts:
        if p.startswith("ckpt:") and not os.path.exists(p[5:]):
            fail.append("⑤ 臂引用的权重不存在：%s" % p[5:])

    if fail:
        print("  => ✗ **不合格，未做任何改动** ✓：")
        for f in fail:
            print("     " + f)
        return 1

    if not a.yes:
        print("  => ✓ 全部检查通过 ✓（**干跑，未写入** ✓）")
        print("     要真换 ⇒ 加 `--yes` ✓")
        return 0

    # ★ **无 BOM 的 UTF-8** ✓（§Z.1543 ✗：`Set-Content -Encoding UTF8` 会写 BOM ✗）
    with open(a.arms, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(content)
    back = open(a.arms, encoding="utf-8-sig").read()
    print("  ⑥ 已写入 ✓ 回读 = %r" % back.strip())
    if back.strip() != content:
        print("  => ✗ 回读不一致 ⇒ **已回滚** ✓")
        if bak:
            shutil.copy2(bak, a.arms)
        return 1
    with open(a.arms, "rb") as fb:
        raw = fb.read(3)
    print("     BOM 检查 ✓ %s" % ("**有 BOM** ✗" if raw == b"\xef\xbb\xbf" else "无 BOM ✓"))

    # ── ⑥ 复用既有检查器 ✓ ──────────────────────────────────────────────
    print("  ⑦ 跑 tools/check_arms_file.py ✓")
    rc = subprocess.call([sys.executable, os.path.join("tools", "check_arms_file.py"), a.arms])
    print("     rc = %d" % rc)

    # ── ⑦ 打印回退命令 ✓ ────────────────────────────────────────────────
    old = ""
    if "ckpt:" in cur:
        old = cur.split(",")[0][5:]
    print("  ================================")
    print("  ✓ **已换臂** ⇒ `%s`" % content)
    print("  ★ **回退命令**（出事时一条命令 ✓）：")
    if old:
        print("     .\\_venv\\Scripts\\python.exe tools\\promote_arm.py --ckpt %s --yes" % old)
    if bak:
        print("     （或直接还原备份：%s ✓）" % bak)
    print("  ★ 客户端**每房重读** ✓ ⇒ 零中断 ✓（`run_live.py:504` ✓）")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
