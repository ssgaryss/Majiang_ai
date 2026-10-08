#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""竞技场 —— 让多个策略人格在同一桌对打，收集战绩，维护天梯。

一轮 = 房间 ready 满 4 席 -> M 场并发对局 -> 各场终局快照给出四家得分。
得分按「本人座位」取（座次逐场重洗），累计写入 arena_log.json。

用法：
  python mj_arena.py --ready                          # 开一轮，4 个默认策略对打
  python mj_arena.py --ready --trace                  # 附带每次动作日志
  python mj_arena.py --strategies S1-效率,S1-效率,S2-副露,S0-模板
  python mj_arena.py --standings                      # 只打印天梯，不打
"""
import argparse
import json
import os
import sys
import time

from mj_bot import ALL_STRATEGIES, Player
from mj_policy import GENE_ORDER, Evolved, parse_genome
from my_bot import ApiError, SERVER, say
from run_room import load_tokens, preflight, ready_all
from stats import GENE_SCHEMA

# 默认对阵：一组干净的 A/B/C/D —— 基线 / 现任冠军 / 向听数 / 向听数+吃
DEFAULT_MATCH = ["first:0:0:0:0", "heuristic:1:0:0:0",
                 "shanten:1:0:0:0", "shanten:1:1:0:0"]


def make_strategy(name):
    """具名策略（S0-模板 等）或基因组编码串（'shanten:1:1:0:0'）都能构造。

    ★★★ 第 415 轮新增 `ckpt:<路径>` 分支 —— **让 RL 模型能真的上场**。

    为什么需要它（这是"模型从未上线"的**唯一**技术缺口）
    ----------------------------------------------------
    `rl_policy.NetPolicy` 已经有 `on_turn(self, v)` / `on_window(self, v, phase)`
    —— 与 `mj_policy.Strategy` **接口完全一致**（已实测签名）。
    缺的只是"从名字造出来"这一步：本函数原先只有两个分支
    （具名模板类 ｜ `Evolved(parse_genome(...))`），**没有加载模型文件的路**。

    于是 `run_live.py` 的 `--genome`（它支持**多臂按房间轮换**，见 `:443-475`）
    结构上就只能轮换基因组 —— **模型从来没有机会被真场检验**。

    用法（线上真场 A/B）：
    ```
      python run_live.py --genome "shanten:1:1:2:5:1:1:0,ckpt:rl6_ckpt0011.pt"
    ```
    ⇒ 一半房间用现役规则策略、一半用模型，**同池同时间**，可直接比 μ。

    ⚠ 三条纪律
    -----------
    1. **惰性导入**：`torch`/`rl_policy` 只在真的用到 `ckpt:` 时才导入
       ⇒ 纯基因组路径（现役线上）**一个字节的额外开销都没有**；
    2. **`greedy=True`**：线上要确定性，不做采样；
    3. **`record=False`**：不记训练用的中间量，省内存。
    """
    # ★★ 第 421 轮：**先剥掉 BOM 与首尾空白**。
    #   事故（真实发生过，2026-09-26 01:38~01:42，线上整整 4 轮开不出房）：
    #     `live_ab_arms.txt` 被 PowerShell 5.1 的 `Set-Content -Encoding UTF8`
    #     写成了**带 BOM** 的文件 ⇒ 读出来的基因组串是 `"\ufeffckpt:rl6_ckpt0003.pt"`
    #     ⇒ `startswith("ckpt:")` 判假 ⇒ 落到 `parse_genome` ⇒
    #     `ValueError: discard 只能取 [...]，收到 '\ufeffckpt'` ⇒ **每一轮都抛异常**。
    #   ⚠ 教训有两层：
    #     ① **标签与行为不是一回事** —— 日志里明明打印着
    #        「策略 ﻿ckpt:rl6_ckpt0003.pt」（BOM 不可见），我据此就宣布"模型已上场"，
    #        而实际上客户端**一次都没跑起来**；
    #     ② 这类"一个不可见字节让整条路静默失效"的东西，**要在入口处一次性挡掉**，
    #        不能指望每个调用方都记得别写 BOM。
    name = str(name).lstrip("\ufeff").strip()
    for cls in ALL_STRATEGIES:
        if cls.name == name:
            return cls()
    if name.startswith("ckpt:"):
        return load_ckpt_strategy(name[5:].strip())
    return Evolved(parse_genome(name))


_CKPT_CACHE = {}


def load_ckpt_strategy(path, device="cpu", greedy=True):
    """`ckpt:<路径>` → 一个满足 `Strategy` 接口的策略对象（**只加载一次，带缓存**）。

    ⚠ 检查点的结构（已实测 `rl_bc2.pt` / `rl6_ckpt0011.pt`）：
    ```
      d["sd"]   状态字典（14 项）
      d["cfg"]  网络配置（含 state_dim / cand_turn_dim / cand_win_dim …）
      d["iter"] 训练轮次
    ```
    （注意：**不是** `sds`（那是训练器内部"按轮次索引的多份快照"））
    """
    key = (os.path.abspath(path), device, bool(greedy))
    path = str(path).lstrip("\ufeff").strip()          # ★ 同上：入口处挡掉 BOM/空白
    key = (os.path.abspath(path), device, bool(greedy))
    if key in _CKPT_CACHE:
        return _CKPT_CACHE[key]
    import rl_boot                                     # 惰性：修 sys.path / numpy / torch
    torch = rl_boot.bootstrap(verbose=False)
    import rl_policy as RP
    d = torch.load(path, map_location=device, weights_only=False)
    cfg = d.get("cfg") or {}
    sd = d.get("sd")
    if sd is None:
        raise ValueError("检查点里没有 `sd`（只有 %s）" % sorted(d))
    net = RP.make_net(cfg)
    net.load_state_dict(sd)
    net.eval()
    name = "rl:%s#%s" % (os.path.basename(path), d.get("iter"))
    pol = RP.NetPolicy(net, cfg, device=device, greedy=greedy, record=False,
                       name=name)
    print("  [ckpt] 已加载 %s（iter=%s，cfg.state_dim=%s）-> %s"
          % (os.path.basename(path), d.get("iter"), cfg.get("state_dim"), name))
    _CKPT_CACHE[key] = pol
    return pol

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_log.json")

# 解析口径版本。凡改动 View/手牌拼装/结算口径，必须 +1：
# 天梯只统计与当前版本一致的轮次，避免「口径已变、老数据还混在里面」造成
# 静默污染（v1 的教训：my_hand 多拼一张，胡牌率 0，4 轮数据全是废的却看不出来）。
PARSER_VERSION = 2


# ---------------------------------------------------------------- 日志

def load_log():
    if not os.path.exists(LOG):
        return {"rounds": []}
    try:
        with open(LOG, "r", encoding="utf-8") as f:
            d = json.load(f)
        d.setdefault("rounds", [])
        return d
    except (ValueError, OSError) as e:
        say("警告：%s 读取失败(%s)，本轮另起新档" % (LOG, e))
        return {"rounds": []}


def save_log(d):
    tmp = LOG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, LOG)


# ---------------------------------------------------------------- 一轮

def play_round(server, tokens, strategies, room, trace=False, max_minutes=40):
    """把一轮打完，返回本轮每席战绩。"""
    players = [Player(server, tok, st, room_id=room, trace=trace,
                      max_minutes=max_minutes, label="席%d" % (i + 1))
               for i, (tok, st) in enumerate(zip(tokens, strategies))]
    for p in players:
        p.start()
    time.sleep(3)                      # 先让轮询跑起来再看是否 ready
    ready_all(server, tokens, room)
    for p in players:
        p.join()
    return players


def tally(players):
    """把各 Player 的场次结果汇总成每席战绩。"""
    rows = []
    for i, pl in enumerate(players):
        total, games, firsts = 0, 0, 0
        seats = {}
        for gid, r in sorted(pl.results.items()):
            sc = r.get("scores") or []
            seat = r.get("seat", -1)
            seats[gid] = seat
            if not (0 <= seat < len(sc)):
                continue
            total += sc[seat]
            games += 1
            if sc and sc[seat] == max(sc):
                firsts += 1
        rows.append({"seat": i, "strategy": pl.strategy.name,
                     "score": total, "games": games, "firsts": firsts,
                     "missed_hu": len(pl.missed_hu), "seats": seats})
    return rows


def accumulate(agg, seat_map, players, seen, score_map=None, sh_map=None):
    """把一轮里各 Player 的场次结果累加进跨轮总计。

    为什么需要它：单轮只有 M=10 局，同一策略两轮实测能摆动 71 分（+60 -> -11）。
    在这种噪声下做淘汰等于随机淘汰。测试房支持跨轮复用（4 令牌各 ready 一次
    即开下一轮），所以连打 N 轮把每组对比的样本量抬到 10N 局，是当前唯一杠杆。

    seen 是 (席位下标, game_id) 集合：4 个令牌跨轮不变，新一轮 bot 启动时
    /api/me 可能仍列着上一轮尚未结算完的场次，不去重会把同一场重复计入。

    score_map / sh_map 记录「每局每席得分」与「每局每席平均打出后向听数」。
    四家打的是同一副牌，这些数据天然配对；向听数是低方差中间指标，
    同样的局数下比得分和自摸率分辨率高得多。
    """
    for i, pl in enumerate(players):
        a = agg[i]
        for gid, r in sorted(pl.results.items()):
            if (i, gid) in seen:
                a["dupes"] = a.get("dupes", 0) + 1
                continue
            seen.add((i, gid))
            sc = r.get("scores") or []
            seat = r.get("seat", -1)
            seat_map[i][gid] = seat
            av = r.get("avg_shanten")
            if av is not None:
                a["sh_sum"] = a.get("sh_sum", 0.0) + av
                a["sh_n"] = a.get("sh_n", 0) + 1
                if sh_map is not None:
                    sh_map[i][gid] = av
            if not (0 <= seat < len(sc)):
                continue
            a["score"] += sc[seat]
            a["games"] += 1
            if sc[seat] == max(sc):
                a["firsts"] += 1
            if score_map is not None:
                score_map[i][gid] = sc[seat]
        a["missed_hu"] += len(pl.missed_hu)


def print_rows(rows, title="本轮战绩"):
    say("--- %s ---" % title)
    say("  %-4s %-12s %8s %6s %6s %9s" % ("席位", "策略", "总分", "场次", "头名", "可胡未胡"))
    for r in sorted(rows, key=lambda x: -x["score"]):
        say("  %-4s %-12s %8d %6d %6d %9d"
            % ("席%d" % (r["seat"] + 1), r["strategy"], r["score"],
               r["games"], r["firsts"], r.get("missed_hu", 0)))
    tot = sum(r["score"] for r in rows)
    gs = sorted({r["games"] for r in rows})
    if len(gs) == 1:
        say("  校验：四席总分合计 = %d（零和计分，应为 0）%s"
            % (tot, "" if tot == 0 else "  << 异常！"))
    else:
        say("  校验：各席场次不一致 %s，合计 %d —— 缺场席位的得分没被计入，"
            "合计不为 0 属正常（不是零和被破坏）" % (gs, tot))


def standings(log):
    """跨轮累计天梯（按策略名聚合）。只统计解析口径与当前一致的轮次。"""
    agg = {}
    skipped = 0
    for rd in log.get("rounds", []):
        if rd.get("parser_version", 1) != PARSER_VERSION:
            skipped += 1
            continue
        for r in rd.get("rows", []):
            a = agg.setdefault(r["strategy"], {"score": 0, "games": 0, "firsts": 0,
                                               "rounds": 0})
            a["score"] += r.get("score", 0)
            a["games"] += r.get("games", 0)
            a["firsts"] += r.get("firsts", 0)
            a["rounds"] += 1
    return agg, skipped


def print_standings(log):
    agg, skipped = standings(log)
    if not agg:
        say("天梯为空：没有与当前解析口径 v%d 一致的轮次。" % PARSER_VERSION)
        if skipped:
            say("（已跳过 %d 轮旧口径数据）" % skipped)
        return
    n = max(a["rounds"] for a in agg.values())
    say("=== 天梯（解析口径 v%d，累计 %d 轮%s）==="
        % (PARSER_VERSION, n,
           "；已跳过 %d 轮旧口径" % skipped if skipped else ""))
    say("  %-12s %9s %8s %8s %7s %9s"
        % ("策略", "总分", "场次", "头名", "参赛轮", "均分/场"))
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["score"]):
        per = (a["score"] / a["games"]) if a["games"] else 0.0
        say("  %-12s %9d %8d %8d %7d %9.2f"
            % (name, a["score"], a["games"], a["firsts"], a["rounds"], per))


def print_strategies():
    say("具名策略：")
    for cls in ALL_STRATEGIES:
        say("  %-12s %s" % (cls.name, cls.desc))
    say("")
    say("基因组编码（冒号分隔）：%s" % " : ".join(GENE_ORDER))
    say("  各字段取值：discard=first|heuristic|shanten，其余 0|1")
    say("  例：shanten:1:1:0:0 = 向听数出牌 + 会碰 + 会吃 + 不杠 + 不弃胡")
    say("  默认对阵：%s" % " | ".join(DEFAULT_MATCH))


# ---------------------------------------------------------------- 主流程

def main():
    p = argparse.ArgumentParser(description="麻将策略竞技场")
    p.add_argument("--tokens", default="tokens.txt")
    p.add_argument("--server", default=SERVER)
    p.add_argument("--strategies", default=None,
                   help="逗号分隔的 4 个策略名；缺省用全部内置策略各一席")
    p.add_argument("--ready", action="store_true", help="代提交 register+ready 开赛")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--max-minutes", type=int, default=40)
    p.add_argument("--standings", action="store_true", help="只打印天梯后退出")
    p.add_argument("--list", action="store_true", help="列出可用策略后退出")
    p.add_argument("--rounds", type=int, default=1,
                   help="连打几轮（同一组策略跨轮复用房间），用于堆样本量；默认 1")
    a = p.parse_args()

    if a.list:
        print_strategies()
        return

    log = load_log()
    if a.standings:
        print_standings(log)
        return

    names = ([s.strip() for s in a.strategies.split(",") if s.strip()]
             if a.strategies else list(DEFAULT_MATCH))
    if len(names) != 4:
        raise SystemExit("需要恰好 4 个策略，收到 %d 个：%s" % (len(names), names))
    strategies = [make_strategy(n) for n in names]

    say("=== 竞技场 ===  对局配置：%s" % " | ".join(names))
    tokens = load_tokens(a.tokens)
    room = preflight(a.server, tokens)
    say("房间 %s；计划连打 %d 轮（每轮 M=10 局）" % (room, a.rounds))
    if not a.ready:
        say("未指定 --ready —— 只挂机等待（房间需已被别人 ready 才会开赛）")

    n = len(strategies)
    agg = {i: {"score": 0, "games": 0, "firsts": 0, "missed_hu": 0} for i in range(n)}
    seat_map = {i: {} for i in range(n)}
    seen = set()                      # (席位下标, game_id) 去重，见 accumulate 注释
    score_map = {i: {} for i in range(n)}   # 逐局得分，供事后配对显著性检验
    sh_map = {i: {} for i in range(n)}      # 逐局平均打出后向听数（低方差中间指标）
    t0 = time.time()

    for k in range(max(1, a.rounds)):
        say("")
        say("########## 第 %d/%d 轮 ##########" % (k + 1, max(1, a.rounds)))
        players = play_round(a.server, tokens, strategies, room,
                             trace=a.trace, max_minutes=a.max_minutes)

        # 本轮单独统计：每轮 = 一次完整配对观测（四家同副牌、同 10 局），
        # 逐轮落盘后中途被打断也不丢数据，且事后能做配对显著性检验。
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

        say("--- 本轮 + 累计（跑完 %d 轮）---" % (k + 1))
        say("  %-4s %-22s %7s %7s %6s %6s %9s"
            % ("席位", "策略", "本轮", "累计", "场次", "头名", "可胡未胡"))
        for i in range(n):
            say("  %-4s %-22s %7d %7d %6d %6d %9d"
                % ("席%d" % (i + 1), strategies[i].name, r_agg[i]["score"],
                   agg[i]["score"], agg[i]["games"], agg[i]["firsts"],
                   agg[i]["missed_hu"]))

        if any(r_agg[i]["games"] for i in range(n)):
            r_rows = [dict(r_agg[i], seat=i, strategy=strategies[i].name,
                           seats=r_seat[i], scores=r_score[i], shanten=r_sh[i])
                      for i in range(n)]
            log["rounds"].append({
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "room": room,
                "parser_version": PARSER_VERSION,
                "gene_schema": GENE_SCHEMA,
                "batch_round": k + 1,
                "rows": r_rows,
            })
            save_log(log)             # 逐轮落盘
        else:
            say("本轮一场都没接到，提前退出（房间可能被服务端代打）。")
            break

    rows = [dict(agg[i], seat=i, strategy=strategies[i].name, seats=seat_map[i],
                 scores=score_map[i]) for i in range(n)]
    say("")
    print_rows(rows, "累计战绩")
    say("总用时 %.0f 秒" % (time.time() - t0))

    if not any(r["games"] for r in rows):
        say("一场都没接到 —— 不记入天梯。")
        return

    # 注意：不再追加「累计」条目 —— 每轮都已单独落盘，再记一条累计会让天梯重复计数。
    say("每轮已单独写入 %s（共 %d 条）" % (LOG, max(1, a.rounds)))
    say("")
    print_standings(log)

    try:
        say("")
        st = api_room_status(a.server, tokens[0], room)
        if st:
            say("房间状态：%s" % st)
    except ApiError:
        pass


def api_room_status(server, token, room):
    from my_bot import api
    try:
        t = api("GET", "/api/tournaments/%s" % room, server, token)
        return t.get("status")
    except ApiError:
        return None


if __name__ == "__main__":
    main()
