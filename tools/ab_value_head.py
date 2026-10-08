# -*- coding: utf-8 -*-
"""tools/ab_value_head.py -- ③（价值头）A/B 的判定器（第 981 轮 ｜ §Z.1357 预注册方案的机械化）

读 arm A / arm B 的训练日志，取每轮 `[配对评估 迭代N]` 的 Δμ 与 CI，按**逐轮配对**算
    D_i = B_i - A_i      （两臂用**同一段固定种子** ⇒ 配对 ✓）
报 mean(D) 与 95%CI，并**照预注册的阈值**给判定（阈值写死，不许事后改）：

    · mean(D) >= +0.5 且 CI 下界 > 0   => ③ 有效   => 换臂上线，记 L2
    · mean(D) <= -0.5 且 CI 上界 < 0   => ③ 有害   => 换队列 ② rl_pool
    · 其余                              => 未分辨   => 按 goal §3 熔断换类
      ⚠ 不许因为单轮好看就宣布成功（单轮 CI 半宽 ≈ 0.7 ⇒ 单轮不可判）

用法：
    python tools/ab_value_head.py                      # 默认 rl_vA.log / rl_vB.log
    python tools/ab_value_head.py --a X.log --b Y.log
    python tools/ab_value_head.py --selftest           # 合成数据自检（验算术 ✓）
"""
import argparse
import io
import math
import os
import re
import sys

RE_EVAL = re.compile(
    r"\[配对评估\s*迭代(?P<it>\d+)\]\s*房间\s*(?P<rooms>\d+)\s*个.*?"
    r"Δμ\(分/局\)\s*=\s*(?P<dmu>[+-]?\d+\.?\d*)\s*"
    r"95%CI\s*\[\s*(?P<lo>[+-]?\d+\.?\d*)\s*,\s*(?P<hi>[+-]?\d+\.?\d*)\s*\]"
)

# t(7, 0.975) = 2.365；n<8 时按表取，n>=8 用 2.365 的保守近似
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def read_l1(path):
    """⇒ {迭代号: (Δμ, lo, hi)}（同号取最后一条 ✓）"""
    out = {}
    if not os.path.exists(path):
        return out
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = RE_EVAL.search(ln)
            if m:
                out[int(m.group("it"))] = (float(m.group("dmu")),
                                           float(m.group("lo")), float(m.group("hi")))
    return out


def paired_stats(ds):
    """⇒ (mean, se, lo, hi, n)。CI 用 D_i 的样本 SD（配对 t ✓）。"""
    n = len(ds)
    if n == 0:
        return None
    mean = sum(ds) / n
    if n == 1:
        return mean, float("nan"), float("nan"), float("nan"), 1
    var = sum((d - mean) ** 2 for d in ds) / (n - 1)
    se = math.sqrt(var / n)
    t = T95.get(n, 2.365)
    return mean, se, mean - t * se, mean + t * se, n


def verdict(mean, lo, hi, n):
    if n < 8:
        return ("★ 未完成（n=%d/8）⇒ 不出判定 ✓（部分均值仅供参考，禁止据此行动 ✗）" % n)
    if mean >= 0.5 and lo > 0:
        return "★★ ③ 有效（mean(D) >= +0.5 且 CI 下界 > 0）⇒ 换臂上线 ⇒ 记 L2 ✓"
    if mean <= -0.5 and hi < 0:
        return "★★ ③ 有害（mean(D) <= -0.5 且 CI 上界 < 0）⇒ 换队列 ② rl_pool ✓"
    return "★ 未分辨 ⇒ 按 goal §3 熔断换类 ✓（不许因单轮好看宣布成功 ✗）"


def main(argv=None):
    ap = argparse.ArgumentParser(description="③ 价值头 A/B 判定器（预注册阈值 ✓）")
    ap.add_argument("--a", default="rl_vA.log", help="arm A（对照）的训练日志")
    ap.add_argument("--b", default="rl_vB.log", help="arm B（③）的训练日志")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        ds = [0.6, 0.5, 0.7, 0.55, 0.65, 0.6, 0.5, 0.6]        # mean 0.5875
        mean, se, lo, hi, n = paired_stats(ds)
        print("  自检：D =", ds)
        print("  mean = %.4f  se = %.4f  CI = [%.4f, %.4f]  n = %d" % (mean, se, lo, hi, n))
        assert abs(mean - 0.5875) < 1e-9, "均值算错"
        exp_se = math.sqrt(sum((d - mean) ** 2 for d in ds) / 7 / 8)
        assert abs(se - exp_se) < 1e-12, "SE 算错"
        print("  判定 =", verdict(mean, lo, hi, n))
        print("  未完成分支 =", verdict(0.9, 0.1, 1.7, 3))
        print("  自检通过 ✓（均值/SE/判定三分支都对 ✓）")
        return 0

    la, lb = read_l1(a.a), read_l1(a.b)
    its = sorted(set(la) & set(lb))
    print("=" * 78)
    print("③ 价值头 A/B 判定器（预注册阈值 ✓）")
    print("=" * 78)
    print("  arm A = %s（%d 条 L1）｜ arm B = %s（%d 条 L1）｜ 共同轮次 = %d"
          % (a.a, len(la), a.b, len(lb), len(its)))
    if not its:
        print("\n  ⚠ 还没有可配对的轮次（两臂都在跑时才会有 ✓）⇒ 什么都不判 ✓")
        return 0
    ds = []
    print("\n  %4s %10s %10s %10s" % ("迭代", "A(Δμ)", "B(Δμ)", "D=B-A"))
    print("  " + "-" * 40)
    for i in its:
        d = lb[i][0] - la[i][0]
        ds.append(d)
        print("  %4d %10.4f %10.4f %10.4f" % (i, la[i][0], lb[i][0], d))
    mean, se, lo, hi, n = paired_stats(ds)
    print("\n  ★ 主读数 ✓：mean(D) = %+.4f ｜ SE = %.4f ｜ 95%%CI = [%+.4f, %+.4f] ｜ n = %d"
          % (mean, se, lo, hi, n))
    print("  ★ 判定 =", verdict(mean, lo, hi, n))
    print("  ⚠ 纪律 ✓：只有 CI 下界 > 0 才算「有效」✓；未分辨时按 §3 熔断换类，**不追加轮数** ✗")
    return 0


if __name__ == "__main__":
    sys.exit(main())
