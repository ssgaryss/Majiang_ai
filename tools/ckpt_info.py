#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ckpt_info.py —— 列出仓里所有神经网络检查点，并打印它们的**出身证据**。

为什么需要：交付物里到底该放哪个 `.pt`，取决于它是**哪条训练链**的产物、
训了多少轮、状态维度多少。这些信息**都在文件里**（`cfg`），不需要猜。

用法:
  python tools/ckpt_info.py                       # 列全部
  python tools/ckpt_info.py rl_aux_0004.pt        # 只看某几个
  python tools/ckpt_info.py --prefix rl_self      # 只看某条训练链
"""
import argparse
import glob
import os
import sys

# ★ 控制台编码兜底（Windows GBK 控制台打印中文会崩；本仓其它脚本同做法）
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def info(path):
    """读一个 ckpt，返回 (ok, 摘要 dict 或 错误字符串)。"""
    sys.path.insert(0, ROOT)
    import rl_boot
    try:
        torch = rl_boot.bootstrap(verbose=False)
    except Exception as e:
        return False, "torch 不可用: %r" % (e,)
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, "load 失败: %r" % (e,)
    if not isinstance(d, dict):
        return True, {"类型": type(d).__name__}
    cfg = d.get("cfg") or {}
    out = {
        "顶层键": sorted(d.keys()),
        "iter": d.get("iter"),
        "state_dim": cfg.get("state_dim") if isinstance(cfg, dict) else None,
    }
    if isinstance(cfg, dict):
        # 只挑能说明"训练设定"的键，避免刷屏
        for k in ("hidden", "n_hidden", "layers", "actions", "n_actions",
                  "per_hand_return", "nstep", "ent_coef", "ret_norm",
                  "self_imit", "aux", "aux_ratio", "live_seats", "seed"):
            if k in cfg:
                out[k] = cfg[k]
        out["cfg 键数"] = len(cfg)
    sd = d.get("sd") or d.get("state_dict") or d.get("model")
    if isinstance(sd, dict):
        out["参数张量数"] = len(sd)
        out["参数总量"] = int(sum(v.numel() for v in sd.values()
                                if hasattr(v, "numel")))
    return True, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--prefix", default="", help="只列该前缀的 .pt")
    a = ap.parse_args()

    if a.files:
        cands = [f if os.path.isabs(f) else os.path.join(ROOT, f) for f in a.files]
    else:
        cands = sorted(glob.glob(os.path.join(ROOT, a.prefix + "*.pt")))

    print("=== 神经网络检查点清单（%d 个）===" % len(cands))
    for p in cands:
        name = os.path.relpath(p, ROOT)
        sz = os.path.getsize(p) if os.path.isfile(p) else -1
        mt = ""
        if os.path.isfile(p):
            import datetime
            mt = datetime.datetime.fromtimestamp(
                os.path.getmtime(p)).strftime("%m-%d %H:%M")
        print("\n  %s" % name)
        print("    大小 = %s ｜ 改动 = %s" % (
            "%.2f MB" % (sz / 1024.0 / 1024.0) if sz >= 0 else "缺失", mt))
        if sz < 0:
            continue
        ok, res = info(p)
        if not ok:
            print("    !! %s" % res)
            continue
        for k in sorted(res, key=lambda x: (x != "iter", x)):
            print("    %-14s = %s" % (k, res[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
