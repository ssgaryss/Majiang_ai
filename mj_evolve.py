#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""进化搜索驱动 —— 锚点制淘汰与引入。

为什么不是「按累计均分排名、淘汰末位」：
  · 精英跨世代累积（可能已跑 200 局），新生个体只有当代 40 局 —— 拿 40 局
    的均值去比 200 局的均值，新个体几乎必输，**等于永远无法引入新策略**；
  · 跨世代对手池不同，累计均分本身不可比。

锚点制（本轮采用）：
  · 固定一个「锚点」基因组（现任冠军），它每代都出场，占 1 席；
  · 其余 3 席是挑战者，**跨世代保留直到置信区间收敛**（不每代换人）；
  · 判据是「挑战者 vs 锚点」的逐局配对分差 —— 同副牌、同一批局，天然配对，
    方差远小于独立样本，所以 40~80 局就能分辨；
  · CI 整体 > 0  -> 挑战者更强，晋升为新锚点；
    CI 整体 < 0  -> 挑战者更差，淘汰，换新变异体；
    CI 跨 0      -> 尚不可辨，继续测（最长 --max-tests 代）。

种群与血统存在 population.json。

用法：
  python mj_evolve.py --generations 4 --rounds 4      # 跑 4 代，每代 4 轮
  python mj_evolve.py --show                          # 打印锚点/挑战者/结论
  python mj_evolve.py --reset                         # 重置
"""
import argparse
import json
import os
import random
import sys
import time

from mj_arena import PARSER_VERSION, accumulate, load_log, make_strategy, play_round, save_log
from mj_policy import crossover, genome_id, mutate, parse_genome
from my_bot import SERVER, say
from run_room import load_tokens, preflight
from stats import GENE_SCHEMA, boot_ci

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

POP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "population.json")

ANCHOR_SEED = "shanten:1:1:0:0:0"    # 上一批 60 局实测冠军（+4.88/场，CI [2.18,7.88]）
MAX_TESTS = 3                        # 一个挑战者最多被测几代，超了就换掉
CI_ITERS = 4000


# ---------------------------------------------------------------- 持久化

def new_pop():
    return {"generation": 0, "parser_version": PARSER_VERSION,
            "anchor": {"genome": ANCHOR_SEED, "parent": "seed", "score": 0,
                       "games": 0, "firsts": 0},
            "challengers": [], "retired": [], "log": []}


def load_pop():
    if not os.path.exists(POP):
        return new_pop()
    try:
        with open(POP, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (ValueError, OSError) as e:
        say("种群文件读取失败(%s)，重置。" % e)
        return new_pop()
    if d.get("parser_version") != PARSER_VERSION:
        say("口径 v%s != 当前 v%d，重置。" % (d.get("parser_version"), PARSER_VERSION))
        return new_pop()
    d.setdefault("challengers", [])
    d.setdefault("retired", [])
    d.setdefault("log", [])
    return d


def save_pop(d):
    tmp = POP + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, POP)


# ---------------------------------------------------------------- 配对统计

def paired(diffs):
    return boot_ci(diffs, CI_ITERS) if diffs else (0.0, 0.0)


def verdict(lo, hi):
    if lo > 0:
        return "更强"
    if hi < 0:
        return "更差"
    return "不可辨"


def spawn(parents, rng, taken, n):
    """从父本变异/交叉出 n 个不与 taken 重复的新基因。"""
    out = []
    tries = 0
    while len(out) < n and tries < 200:
        tries += 1
        p = parents[rng.randrange(len(parents))]
        if len(parents) >= 2 and rng.random() < 0.5:
            q = parents[rng.randrange(len(parents))]
            g = crossover(parse_genome(p), parse_genome(q), rng)
            origin = "交叉 %s×%s" % (p, q)
        else:
            g = mutate(parse_genome(p), rng)
            origin = "变异 %s" % p
        gid = genome_id(g)
        if gid in taken or any(gid == o[0] for o in out):
            continue
        out.append((gid, origin))
    return out


def describe(pop):
    say("=== 种群（已完成 %d 代，口径 v%d）===" % (pop["generation"], PARSER_VERSION))
    a = pop["anchor"]
    say("锚点：%s   （累计 %d 局，均分 %.2f，来源 %s）"
        % (a["genome"], a["games"], (a["score"] / a["games"]) if a["games"] else 0.0,
           a.get("parent", "?")))
    if not pop["challengers"]:
        say("挑战者：无（下代会新生成）")
        return
    say("挑战者：")
    say("  %-26s %6s %8s %20s  %s" % ("基因组", "已测代", "均分差", "95% CI", "结论"))
    for c in pop["challengers"]:
        d = c.get("diffs", [])
        lo, hi = paired(d)
        m = (sum(d) / len(d)) if d else 0.0
        say("  %-26s %6d %8.2f  [%7.2f, %7.2f]  %s"
            % (c["genome"], c.get("tests", 0), m, lo, hi,
               verdict(lo, hi) if d else "待测"))


# ---------------------------------------------------------------- 主流程

def main():
    p = argparse.ArgumentParser(description="锚点制进化搜索")
    p.add_argument("--tokens", default="tokens.txt")
    p.add_argument("--server", default=SERVER)
    p.add_argument("--generations", type=int, default=1)
    p.add_argument("--rounds", type=int, default=4, help="每代连打几轮（10 局/轮）")
    p.add_argument("--max-tests", type=int, default=MAX_TESTS)
    p.add_argument("--challengers", default=None,
                   help="逗号分隔，直接指定 3 个挑战者（定向实验用；缺省随机变异）")
    p.add_argument("--anchor", default=None, help="直接指定锚点基因组")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    if a.reset:
        if os.path.exists(POP):
            os.remove(POP)
        say("已删除 %s" % POP)
        return

    pop = load_pop()
    if a.show:
        describe(pop)
        return

    rng = random.Random(a.seed) if a.seed is not None else random.Random()
    tokens = load_tokens(a.tokens)
    room = preflight(a.server, tokens)

    # 指定锚点（定向实验）
    if a.anchor:
        parse_genome(a.anchor)                    # 先校验可解析
        if a.anchor != pop["anchor"]["genome"]:
            say("锚点改为：%s（原 %s）" % (a.anchor, pop["anchor"]["genome"]))
            pop["anchor"] = {"genome": a.anchor, "parent": "specified",
                             "score": 0, "games": 0, "firsts": 0}
            pop["challengers"] = []

    # 指定挑战者（定向实验）：直接设成给定基因组，清空其历史统计
    if a.challengers:
        gs = [x.strip() for x in a.challengers.split(",") if x.strip()]
        for g in gs:
            parse_genome(g)                       # 先校验可解析
        gs = [g for g in gs if g != pop["anchor"]["genome"]][:3]
        pop["challengers"] = [{"genome": g, "parent": "specified", "tests": 0,
                               "score": 0, "games": 0, "firsts": 0, "diffs": []}
                              for g in gs]
        say("指定挑战者：%s" % " | ".join(gs))
    elif len(pop["challengers"]) < 3:
        taken = {pop["anchor"]["genome"]} | {c["genome"] for c in pop["challengers"]}
        need = 3 - len(pop["challengers"])
        for gid, origin in spawn([pop["anchor"]["genome"]] +
                                 [c["genome"] for c in pop["challengers"]],
                                 rng, taken, need):
            pop["challengers"].append({"genome": gid, "parent": origin, "tests": 0,
                                       "score": 0, "games": 0, "firsts": 0,
                                       "diffs": []})
            say("引入挑战者：%s  <- %s" % (gid, origin))

    say("房间 %s；每代 %d 轮 = %d 局/个体" % (room, a.rounds, a.rounds * 10))

    for gen in range(pop["generation"] + 1, pop["generation"] + 1 + a.generations):
        genomes = [pop["anchor"]["genome"]] + [c["genome"] for c in pop["challengers"]]
        say("")
        say("################ 世代 %d ################" % gen)
        say("出场：[锚点] %s" % genomes[0])
        for i, g in enumerate(genomes[1:], 1):
            say("      [挑战%d] %s" % (i, g))

        strategies = [make_strategy(g) for g in genomes]
        n = len(strategies)
        agg = {i: {"score": 0, "games": 0, "firsts": 0, "missed_hu": 0}
               for i in range(n)}
        seat_map = {i: {} for i in range(n)}
        score_map = {i: {} for i in range(n)}
        sh_map = {i: {} for i in range(n)}
        seen = set()
        elog = load_log()
        t0 = time.time()
        for k in range(a.rounds):
            say("  -- 第 %d/%d 轮 --" % (k + 1, a.rounds))
            players = play_round(a.server, tokens, strategies, room, trace=a.trace)
            # 逐轮单独累计 + 落盘：进化跑出来的对局同样是宝贵证据，必须进统计池，
            # 否则 stats.py 看不到（早先直接累加进 agg，这 120 局全丢了）。
            r_agg = {i: {"score": 0, "games": 0, "firsts": 0, "missed_hu": 0}
                     for i in range(n)}
            r_seat = {i: {} for i in range(n)}
            r_score = {i: {} for i in range(n)}
            r_sh = {i: {} for i in range(n)}
            accumulate(r_agg, r_seat, players, seen, r_score, r_sh)
            for i in range(n):
                for key in ("score", "games", "firsts", "missed_hu"):
                    agg[i][key] += r_agg[i][key]
                seat_map[i].update(r_seat[i])
                for gid, sc in r_score[i].items():
                    score_map[i][gid] = sc
                sh_map[i].update(r_sh[i])
            if any(r_agg[i]["games"] for i in range(n)):
                elog["rounds"].append({
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "room": room,
                    "parser_version": PARSER_VERSION,
                    "gene_schema": GENE_SCHEMA,
                    "evolve_gen": gen,
                    "rows": [dict(r_agg[i], seat=i, strategy=genomes[i],
                                  seats=r_seat[i], scores=r_score[i],
                                  shanten=r_sh[i])
                             for i in range(n)],
                })
                save_log(elog)
        say("  世代用时 %.0f 秒" % (time.time() - t0))

        if not any(agg[i]["games"] for i in range(n)):
            say("本代没接到场次，中止。")
            break

        # 累计到锚点/挑战者
        pop["anchor"]["score"] += agg[0]["score"]
        pop["anchor"]["games"] += agg[0]["games"]
        pop["anchor"]["firsts"] += agg[0]["firsts"]

        say("")
        say("  %-26s %8s %6s %6s" % ("出场", "本代分", "场次", "头名"))
        for i in range(n):
            tag = "锚点" if i == 0 else "挑战%d" % i
            say("  %-26s %8d %6d %6d"
                % ("%s %s" % (tag, genomes[i]), agg[i]["score"],
                   agg[i]["games"], agg[i]["firsts"]))

        # 逐局配对：挑战者 - 锚点（同一 gid 即同一局）
        base = score_map[0]
        say("")
        say("  %-26s %6s %8s %20s  %s"
            % ("挑战者 vs 锚点", "本代局", "本代差", "累计 95% CI", "累计结论"))
        for i in range(1, n):
            cur = [score_map[i][g] - base[g] for g in base if g in score_map[i]]
            c = pop["challengers"][i - 1]
            c["tests"] += 1
            c["score"] += agg[i]["score"]
            c["games"] += agg[i]["games"]
            c["firsts"] += agg[i]["firsts"]
            c["diffs"].extend(cur)
            lo, hi = paired(c["diffs"])
            m = sum(c["diffs"]) / len(c["diffs"])
            say("  %-26s %6d %8.2f  [%7.2f, %7.2f]  %s"
                % (c["genome"], len(cur), (sum(cur) / len(cur)) if cur else 0.0,
                   lo, hi, verdict(lo, hi)))

        pop["generation"] = gen

        # --- 晋升判断 ---
        promoted = None
        for c in pop["challengers"]:
            lo, hi = paired(c["diffs"])
            if c["diffs"] and lo > 0:
                promoted = c
                break
        if promoted:
            say("")
            say(">>> 晋升：%s 显著强于锚点 %s -> 成为新锚点"
                % (promoted["genome"], pop["anchor"]["genome"]))
            pop["log"].append({"gen": gen, "event": "promote",
                               "from": pop["anchor"]["genome"],
                               "to": promoted["genome"]})
            pop["anchor"] = {"genome": promoted["genome"],
                             "parent": promoted.get("parent", "?"),
                             "score": promoted["score"], "games": promoted["games"],
                             "firsts": promoted["firsts"]}
            pop["challengers"] = []
        else:
            # 淘汰判据明确的弱挑战者；测满 max-tests 的也换掉
            keep, drop = [], []
            for c in pop["challengers"]:
                lo, hi = paired(c["diffs"])
                if c["diffs"] and hi < 0:
                    drop.append((c, "显著更差"))
                elif c["tests"] >= a.max_tests:
                    drop.append((c, "测满 %d 代仍不可辨" % c["tests"]))
                else:
                    keep.append(c)
            for c, why in drop:
                say("淘汰：%s（%s）" % (c["genome"], why))
                if c["genome"] not in pop["retired"]:
                    pop["retired"].append(c["genome"])   # 别再重新生成已否决的基因
            since = {"event": "drop", "gen": gen,
                     "items": [{"genome": c["genome"], "why": w} for c, w in drop]}
            if drop:
                pop["log"].append(since)
            taken = ({pop["anchor"]["genome"]} | {c["genome"] for c in keep}
                     | set(pop["retired"]))
            parents = [pop["anchor"]["genome"]] + [c["genome"] for c in keep]
            for gid, origin in spawn(parents, rng, taken, 3 - len(keep)):
                keep.append({"genome": gid, "parent": origin, "tests": 0,
                             "score": 0, "games": 0, "firsts": 0, "diffs": []})
                say("引入：%s  <- %s" % (gid, origin))
            pop["challengers"] = keep

        save_pop(pop)
        say("已写入 %s" % POP)

    say("")
    describe(pop)

    log = load_log()
    log.setdefault("evolve", []).append(
        {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "generation": pop["generation"],
         "anchor": pop["anchor"]["genome"],
         "challengers": [{k: c.get(k) for k in ("genome", "tests", "score", "games",
                                                "parent")}
                         for c in pop["challengers"]]})
    save_log(log)


if __name__ == "__main__":
    main()
