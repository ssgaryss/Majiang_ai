#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对显著性检验 —— 判断「A 真的比 B 强吗」，而不是凭裸分高低做淘汰。

为什么必须做：
  · 四家在同一桌打同一副牌，得分天然配对 —— 配对检验的方差远小于独立样本；
  · 单轮只有 M=10 局，实测同一策略两轮摆动 71 分（+60 -> -11）；
  · 用裸分排名做淘汰 = 在噪声上做选择，会把好策略随机淘汰掉。

方法：对每对策略取「共同出场的每一局」，算逐局分差 -> 单样本 t 检验 + 自助法 95% CI。
CI 跨 0 即差异不显著，不足以据此淘汰。

数据来源：arena_log.json 每轮的 rows[].scores（game_id -> 该席本局得分）。
注意：口径 v2 早先的批次没记 scores，会被自动跳过。

用法：
  python stats.py                 # 全部策略对
  python stats.py --top shanten:1:1:0:0   # 只比某个基准 vs 其他
"""
import argparse
import json
import math
import os
import random
import sys

from mj_policy import GENE_ORDER, GENE_SCHEMA, parse_genome

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_log.json")


def gene_schema_of(rd):
    return rd.get("gene_schema", 1)


def ambiguous_in(rd, genome):
    """该基因组名在「生成它的 schema」与「当前 schema」下语义是否不同。

    只有 v1 -> v2 那次是**原位改含义**（位置 5：baotou -> ukeire），
    故仅 schema=1 且位置 5 非 0 的名字有歧义，必须剔除。
    v2 -> v3 是**追加一位**：名字串长度就不同（6 段 vs 7 段），
    新旧根本不会合并，无需任何处理 —— 早先把判据写成「schema < 当前」是错的，
    会把本来有效的 v2 数据（ukeire=1 的那批）一起误剔。
    """
    if gene_schema_of(rd) >= 2:
        return False
    parts = str(genome).split(":")
    return len(parts) >= 6 and parts[5] != "0"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(name):
    """归一化策略名。

    为什么需要：同一个基因组在不同写入路径下会被记成两个名字 ——
    mj_arena 写 strategy.name（Evolved 的展示名，形如 "G[shanten:1:1:0:0:0]"），
    mj_evolve 直接写基因组串（"shanten:1:1:0:0:0"）。不归一化的话，
    同一策略的证据会被劈成两半：一边 60 局均值 +4.88，一边 120 局均值 -0.88，
    连「最强者是谁」都会选错。存量日志也靠这个函数就地修复。
    """
    s = str(name)
    if s.startswith("G[") and s.endswith("]"):
        return s[2:-1]
    return s


def canon(rd, name):
    """把旧 schema 的基因组名规范到当前 schema —— 仅在**行为完全等价**时。

    v3 只是在末尾追加了 route 位。route=0 时：
      · best_discard_shanten 走 else 分支（与原规则逐字一致）
      · on_window 跳过七对承诺检查
    故 v2 的 "a:b:c:d:e:f" 与 v3 的 "a:b:c:d:e:f:0" 是同一个策略，
    合并它们能把关键基因组的样本量翻倍。这是先前「名字分裂」bug 的镜像修复。
    """
    if gene_schema_of(rd) == 2 and len(name.split(":")) == 6:
        return name + ":0"
    return name


def collect(log):
    """把每轮的逐局得分按「策略对」聚合成配对分差序列。

    返回 ({策略对: [分差...]}, {策略: [本局得分...]}, 用了多少轮, 跳过了多少轮)
    """
    pairs, solo = {}, {}
    used = skipped = 0
    dropped_legacy = []
    for rd in log.get("rounds", []):
        rows = {}
        for r in rd.get("rows", []):
            nm = canon(rd, norm(r["strategy"]))
            if ambiguous_in(rd, nm):
                dropped_legacy.append(nm)
                continue          # 旧基因语义下的同名声，不能与新版混算
            rows[nm] = r
        names = sorted(rows)
        got = False
        for nm in names:
            sc = rows[nm].get("scores") or {}
            if sc:
                solo.setdefault(nm, []).extend(sc.values())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                sa = rows[a].get("scores") or {}
                sb = rows[b].get("scores") or {}
                d = [sa[g] - sb[g] for g in sa if g in sb]
                if d:
                    pairs.setdefault((a, b), []).extend(d)
                    got = True
        if got:
            used += 1
        else:
            skipped += 1
    return pairs, solo, used, skipped, dropped_legacy


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return 0, 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return n, m, math.sqrt(var)


def boot_ci(xs, iters=4000, alpha=0.05, seed=12345):
    rng = random.Random(seed)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    ms = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        ms.append(s / n)
    ms.sort()
    return ms[int(iters * alpha / 2)], ms[min(iters - 1, int(iters * (1 - alpha / 2)))]


def verdict(lo, hi):
    if lo > 0:
        return "显著更优"
    if hi < 0:
        return "显著更差"
    return "不显著（噪声内）"


def gene_effects(solo):
    """按基因位聚合 —— 把共享同一基因取值的所有基因组样本合并。

    为什么需要：单基因组比较的分辨率不够（每个 60~360 局，CI 宽度约 ±3 分），
    而基因空间只有 6 位、每位 2~3 个取值。合并同取值的样本能把 n 抬到数百局，
    是同一批数据里信息量最大的切法。
    代价：这是**边际效应**，可能被其他基因的交互污染（ukeire 与 chi 就明显交互）。
    所以它用来「提出假设」，定论仍要靠单基因隔离实验。
    """
    out = {}
    for nm, xs in solo.items():
        if ":" not in nm:
            continue
        try:
            g = parse_genome(nm)
        except (ValueError, KeyError):
            continue
        for k in GENE_ORDER:
            out.setdefault(k, {}).setdefault(str(g.get(k)), []).extend(xs)
    return out


def collect_shanten(log):
    """按策略聚合「每局平均打出后向听数」—— 低方差中间指标。

    得分单局标准差约 16、自摸率约 0.43（二项），而每局平均向听数把约 17 次
    出牌平均掉，单局标准差小一个量级，同样局数下分辨率高得多，且直接度量出牌质量。
    """
    out = {}
    for rd in log.get("rounds", []):
        for r in rd.get("rows", []):
            nm = canon(rd, norm(r["strategy"]))
            if ambiguous_in(rd, nm):
                continue
            sh = r.get("shanten") or {}
            if sh:
                out.setdefault(nm, []).extend(sh.values())
    return out


def main():
    p = argparse.ArgumentParser(description="配对显著性检验")
    p.add_argument("--log", default=LOG)
    p.add_argument("--top", default=None, help="只比较该策略 vs 其他")
    p.add_argument("--iters", type=int, default=4000)
    a = p.parse_args()

    if not os.path.exists(a.log):
        raise SystemExit("找不到 %s" % a.log)
    log = load(a.log)
    pairs, solo, used, skipped, dropped_legacy = collect(log)

    print("=== 数据 ===")
    print("  可用轮次 %d，跳过（无逐局数据）%d" % (used, skipped))
    if dropped_legacy:
        print("  剔除旧基因语义样本 %d 行（%s）—— 同名但行为不同，不能混算"
              % (len(dropped_legacy), ", ".join(sorted(set(dropped_legacy)))))
    if not pairs:
        print("  没有逐局配对数据 —— 需用记录 scores 的版本跑批次。")
        return

    print()
    print("=== 各策略自身（vs 0）===")
    print("  %-24s %6s %9s %20s  %-8s %8s %18s"
          % ("策略", "局数", "均分/场", "95% CI", "结论", "自摸率", "自摸率 95%CI"))
    for nm in sorted(solo, key=lambda k: -mean_sd(solo[k])[1]):
        xs = solo[nm]
        n, m, sd = mean_sd(xs)
        lo, hi = boot_ci(xs, a.iters)
        # 自摸率：得分 > 0 即本局自摸。得分本身被庄家 ×8 与番型 ×1~×512 主导，
        # 单局标准差约 16，而策略间真实差异只有 1~2 分 —— 分辨率天然不够。
        # 自摸率是二项量（p≈0.25），标准误小得多，是同一批数据的第二视角。
        w = [1.0 if x > 0 else 0.0 for x in xs]
        wr = sum(w) / n if n else 0.0
        wlo, whi = boot_ci(w, a.iters)
        print("  %-24s %6d %9.2f  [%7.2f, %7.2f]  %-8s %7.1f%%  [%5.1f%%, %5.1f%%]"
              % (nm, n, m, lo, hi, verdict(lo, hi), wr * 100,
                 wlo * 100, whi * 100))

    print()
    print("=== 配对比较（同副牌逐局做差）===")
    print("  %-46s %5s %9s %20s  %s"
          % ("A vs B", "局数", "均分差", "95% CI", "结论"))
    keys = sorted(pairs, key=lambda k: -mean_sd(pairs[k])[1])
    for (x, y) in keys:
        if a.top and a.top not in (x, y):
            continue
        xs = pairs[(x, y)]
        n, m, sd = mean_sd(xs)
        lo, hi = boot_ci(xs, a.iters)
        tag = verdict(lo, hi)
        short = "%s vs %s" % (x, y)
        print("  %-46s %5d %9.2f  [%7.2f, %7.2f]  %s"
              % (short, n, m, lo, hi, tag))

    print()
    print("=== 低方差指标：每局平均打出后向听数（越小越好）===")
    sh = collect_shanten(log)
    if not sh:
        print("  无数据（需用带 shanten 埋点的版本跑批次）。")
    else:
        print("  %-30s %6s %10s %20s  %s"
              % ("策略", "局数", "平均向听", "95% CI", "结论"))
        ref = None
        for nm in sorted(sh, key=lambda k: mean_sd(sh[k])[1]):
            xs = sh[nm]
            n, m, sd = mean_sd(xs)
            lo, hi = boot_ci(xs, a.iters)
            if ref is None:
                ref = (nm, m)
            # vs 最优（向听最小）：CI 整体 > 0 表示显著更差
            print("  %-30s %6d %10.3f  [%8.3f, %8.3f]  %s"
                  % (nm, n, m, lo, hi, "" if nm == ref[0] else "参考最优"))

    print()
    print("=== 番数代理：平均每胡得分 ===")
    print("  自摸率已接近四方均势的必然值（四家水平相当则各 25%），")
    print("  提升空间在番数：七对 ×2 / 豪华七对 ×4 / 双豪华 ×8，相对平胡 ×1 是成倍差。")
    print("  %-30s %6s %8s %12s %18s"
          % ("策略", "局数", "胡牌数", "平均每胡", "95% CI"))
    for nm in sorted(solo, key=lambda k: -(sum(x for x in solo[k] if x > 0)
                                           / max(1, sum(1 for x in solo[k] if x > 0)))):
        xs = solo[nm]
        w = [x for x in xs if x > 0]
        if not w:
            print("  %-30s %6d %8d %12s" % (nm, len(xs), 0, "—"))
            continue
        n, m, sd = mean_sd(w)
        lo, hi = boot_ci(w, a.iters)
        print("  %-30s %6d %8d %12.1f  [%7.1f, %8.1f]" % (nm, len(xs), n, m, lo, hi))

    print()
    print("=== 基因级边际效应（合并同取值的所有基因组）===")
    print("  注意：这是边际效应，可能含交互污染 —— 用于提出假设，定论靠隔离实验。")
    ge = gene_effects(solo)
    for k in GENE_ORDER:
        if k not in ge:
            continue
        say_line = "  %-9s" % k
        print(say_line)
        for v in sorted(ge[k]):
            xs = ge[k][v]
            n, m, sd = mean_sd(xs)
            w = sum(1 for x in xs if x > 0) / n if n else 0.0
            lo, hi = boot_ci(xs, a.iters)
            print("      %-10s n=%-5d 均分 %+6.2f [%6.2f,%6.2f]  自摸率 %5.1f%%"
                  % (v, n, m, lo, hi, w * 100))

    print()
    print("=== 淘汰判定（相对当前最强者的配对检验）===")
    # 早先的判据是「对池子里所有对手都显著更差」——只在 3 个策略时可用。
    # 池子涨到 8 个、彼此从不同桌之后，这个条件永远不可能满足，输出恒为空。
    # 改为以「均分最高的那个」为参照系做配对检验：可淘汰 ⇔ 相对它 CI 整体 < 0。
    ranked = sorted(solo, key=lambda k: -mean_sd(solo[k])[1])
    if not ranked:
        print("  无数据。")
        return
    ref = ranked[0]
    print("  参照系 = 均分最高者：%s（%.2f/场，%d 局）"
          % (ref, mean_sd(solo[ref])[1], len(solo[ref])))
    doomed = []
    for nm in ranked[1:]:
        key = (nm, ref) if (nm, ref) in pairs else (ref, nm)
        if key not in pairs:
            print("  %-30s 与参照系无同池配对数据，无法判定" % nm)
            continue
        sign = 1 if key[0] == nm else -1
        adj = [sign * v for v in pairs[key]]
        lo, hi = boot_ci(adj, a.iters)
        m = sum(adj) / len(adj)
        if hi < 0:
            doomed.append(nm)
            print("  %-30s 差 %+6.2f  95%%CI [%6.2f, %6.2f]  -> 可淘汰"
                  % (nm, m, lo, hi))
        else:
            print("  %-30s 差 %+6.2f  95%%CI [%6.2f, %6.2f]  -> 保留"
                  % (nm, m, lo, hi))
    print()
    if doomed:
        print("  结论：可淘汰 %d 个 —— %s" % (len(doomed), ", ".join(doomed)))
    else:
        print("  结论：当前样本量下没有策略可被有依据地淘汰。")
        print("  （这正是配对检验的价值：裸分排名会骗你，CI 不会）")


if __name__ == "__main__":
    main()
