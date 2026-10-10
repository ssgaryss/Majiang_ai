#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**房间复盘**：每打完一个房间（10 场）就生成统计 + 复盘，落到本地 `reviews/`。

工作约定（用户要求，第 34 轮起）
--------------------------------
> **每次打完十盘需要进行复盘，生成对局统计数据和复盘到本地。**

本脚本就是那个"复盘"。它产出一对文件：

    reviews/<room>_<时间戳>.md     人类可读的复盘
    reviews/<room>_<时间戳>.json   机器可读的统计数据（供后续聚合/画图）

内容
----
1. **两条断言**：结算恒等式（逐局）、庄位继承（逐局相邻）——
   它们是"我们看到的比赛是否符合已知规则"的守门员，**不成立就必须先停下查**；
2. **逐场表**：场号 / 我方座位 / 我方得分 / 四家得分 / 局数；
3. **汇总统计**：总分、场均、每局 μ 与 σ、我方胡牌数、自摸率、均番、
   爆头率、七对率、副露、流局；
4. **复盘要点**：最大赢/输场、番型我方 vs 对手、与基线的对比、
   以及**这一场数据可信度的自校验**（我方逐场求和 == 客户端自报总分）。

用法
----
    python review_room.py --room t_cfcd0480caea --account u_d26d0b6adcb0
    python review_room.py --jsonl room_join2.jsonl        # 房间/账号都从结果文件取
    python review_room.py --jsonl live_r16.jsonl --recent 3   # 最近 3 个房间
    python review_room.py --jsonl room_join2.jsonl --no-net   # 只用本地分数，不取事件

⚠ `--no-net` 时不取 `/events`，因此**没有**恒等式/庄位继承/番型，
   只有逐场得分（来自结果文件）。线上有客户端在打时优先用这个。
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOL = 1e-6
OUTDIR = "reviews"
# ★ 第 70 轮：复盘顺手把**原始** `/events` 落进 `field_cache/`（`field_ranking` 的缓存目录）。
#   复盘本来就要把一个房间的 events 全取下来，原始包以前被丢掉，而
#   `field_ranking.py --fetch` 会为**同一批数据**再取一遍 —— 纯粹取两遍。
#   接上之后 `field_cache/` 随复盘自动长大，μ 的证据**零额外请求**地积累。
RAWCACHE = "field_cache"
# ★ 第 76 轮：把"取一个房间"的请求**摊平**，别再突发。
#   实测（§Z.71）：线上客户端的额度是一个**属于本账号的令牌桶**
#   （44 个空闲段里 28 个回升、**0 个下降**），而 0.35 秒/场的默认节奏
#   会在约 **3.9 秒内连发 ~11 次 GET ≈ 2.8/秒**；客户端自己已经在要 9~10/秒，
#   而额度常常只有 6~8 ⇒ 这 4 秒正好把桶抽干 ⇒ 客户端快照过期 ⇒ **漏窗**。
#   总量不变（还是 11 次），只是**峰值降下来**：3.0 秒/场 ⇒ 峰值 ≈ 0.33/秒。
RAWDELAY = 3.0
# 与夹具/线上对照用的基线（夹具 = 同质桌 live 基因组）
BASE_FIXTURE_MU = -0.380
BASE_FIXTURE_SD = 11.459


def expected_scores(dealer, winner, fan, base=1):
    """`MODEL_SETTLEMENT.md` §1 的恒等式。"""
    s = [0.0] * 4
    if winner is None or not (0 <= winner < 4):
        return s
    pay = float(base) * float(fan)
    if winner == dealer:
        for k in range(4):
            if k != winner:
                s[k] -= pay * 8
                s[winner] += pay * 8
    else:
        for k in range(4):
            if k == winner:
                continue
            mult = 8 if k == dealer else 1
            s[k] -= pay * mult
            s[winner] += pay * mult
    return s


def account_from_log(path):
    """从客户端日志里读我们的 user_id（`令牌 user=u_xxx`）。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()[-400000:]
    except OSError:
        return None
    m = None
    for m in re.finditer(r"令牌 user=(\S+)", txt):
        pass
    return m.group(1) if m else None


def rooms_of(jsonl, recent=None):
    """从结果 jsonl 里取 [(room, total, rows)]，按出现顺序；recent 取最后 N 个。"""
    seen, out = set(), []
    for line in open(jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        r = d.get("room")
        if not r or r in seen:
            continue
        seen.add(r)
        out.append((r, d.get("total"), d.get("rows") or [], d.get("genome")))
    return out[-recent:] if recent else out


def local_stats(rows):
    """只用结果文件（不取网络）算逐场统计。"""
    det = []
    tot = 0.0
    for row in rows:
        mine = row.get("mine")
        if mine is None:
            continue
        tot += float(mine)
        det.append({"game": row.get("game"), "seat": row.get("seat"),
                    "mine": float(mine), "scores": row.get("scores"),
                    "round_nos": len(row.get("round_nos") or [])})
    return tot, det


def md_table(det):
    lines = ["| # | 场 | 我方座位 | 我方得分 | 四家得分 |",
             "|---|---|---|---|---|"]
    for i, d in enumerate(det, 1):
        sc = d.get("scores") or []
        lines.append("| %d | %s | %s | **%+d** | %s |"
                     % (i, str(d.get("game"))[-10:], d.get("seat"),
                        d.get("mine"), ", ".join(str(x) for x in sc)))
    return "\n".join(lines)


def analyse(room, rows, account, R, no_net):
    """返回 (stats_dict, md_text)。"""
    tot_local, det_local = local_stats(rows)
    st = {
        "room": room, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "account": account, "no_net": bool(no_net),
        "games": len(rows), "total": tot_local,
        "per_game": det_local,
    }
    md = []
    md.append("# 复盘：房间 `%s`" % room)
    md.append("")
    md.append("- 生成时间：%s" % st["at"])
    md.append("- 我方账号：`%s`" % account)
    md.append("- 场数：**%d**（结果文件记录）" % len(rows))
    md.append("- 我方总分：**%+d**（场均 %+.2f）"
              % (tot_local, tot_local / max(1, len(rows))))
    md.append("")

    if no_net:
        md.append("## 逐场")
        md.append("")
        md.append(md_table(det_local))
        md.append("")
        md.append("> `--no-net`：未取 `/events`，故无恒等式/庄位继承/番型，"
                  "只有结果文件里的逐场得分。")
        st["self_check"] = None
        return st, "\n".join(md)

    import room_data as RD

    data = RD.fetch_room(R, room, account=account, raw_cache_dir=RAWCACHE,
                         delay=RAWDELAY)
    st["n_fail"] = data["n_fail"]
    st["n_listed"] = data["n_listed"]
    st["config"] = data["config"]
    st["seats"] = data["seats"]

    id_bad, succ, succ_tot = [], 0, 0
    ours, wins, draws, n_round = [], 0, 0, 0
    det_mine, det_opp = Counter(), Counter()
    fan_mine, fan_opp = Counter(), Counter()
    seat_hist = Counter()
    per_game = []
    for g in data["games"]:
        seat = g["seat"]
        if seat is not None:
            seat_hist[seat] += 1
        gs = 0.0
        gwin = 0
        gsum = [0.0] * 4
        rdetail = []
        for i, r in enumerate(g["rounds"]):
            sc = r.get("scores") or []
            if len(sc) != 4:
                continue
            n_round += 1
            for k in range(4):
                gsum[k] += float(sc[k])
            # ★ **把逐局明细也存下来**。只存逐场总分的话，复盘就只是"报告"，
            #   下游想算庄位 EV、自摸率、番型×座位这类东西时**无从下手**
            #   （第 36 轮想做庄位 EV 再估计时才发现 —— 数据已经丢在服务端了，
            #   重新取一遍又是十几次 GET、还要抢预算）。
            #   每条 ~100 字节，一个房间 150 局约 15KB，换来的可分析性远超成本。
            rdetail.append({
                "round_no": r.get("round_no"),
                "dealer": r.get("dealer"),
                "winner": r.get("winner"),
                "fan": r.get("multiplier"),
                "scores": [int(round(x)) for x in sc],
                "mine": (int(round(sc[seat])) if seat is not None else None),
                "we_dealer": (seat is not None and r.get("dealer") == seat),
                "detail": list(r.get("detail") or ()),
            })
            dlr, win, fan = r.get("dealer"), r.get("winner"), r.get("multiplier")
            if win is None:
                draws += 1
            elif seat is not None:
                exp = expected_scores(dlr, win, fan)
                if any(abs(x - y) > TOL for x, y in zip(exp, sc)):
                    id_bad.append((room, r.get("round_no"), dlr, win, fan, sc))
            if seat is not None:
                gs += float(sc[seat])
                if win == seat:
                    wins += 1
                    gwin += 1
                    det_mine[tuple(r.get("detail") or ())] += 1
                    fan_mine[fan] += 1
                elif win is not None:
                    det_opp[tuple(r.get("detail") or ())] += 1
                    fan_opp[fan] += 1
                ours.append(float(sc[seat]))
            # 庄位继承（房间内相邻局）
            if i + 1 < len(g["rounds"]):
                nxt = g["rounds"][i + 1].get("dealer")
                succ_tot += 1
                if (win is None and nxt == dlr) or (win is not None and nxt == win):
                    succ += 1
        per_game.append({"game_id": g["game_id"], "batch": g["batch"],
                         "seat": seat, "score": gs, "n_rounds": len(g["rounds"]),
                         "wins": gwin, "scores": [int(round(x)) for x in gsum],
                         "rounds": rdetail})
    st["per_game"] = per_game
    st["n_rounds"] = n_round
    st["draws"] = draws
    st["identity_bad"] = len(id_bad)
    st["succession"] = [succ, succ_tot]
    st["seats_used"] = dict(sorted(seat_hist.items()))

    n = len(ours)
    mu = sum(ours) / n if n else 0.0
    sd = math.sqrt(sum((x - mu) ** 2 for x in ours) / max(1, n - 1)) if n else 0.0
    st.update({"mu": mu, "sd": sd, "n_seat_rounds": n, "wins": wins,
               "self_draw_rate": (wins / n) if n else 0.0})
    tot_mine = sum(det_mine.values()) or 1
    tot_opp = sum(det_opp.values()) or 1
    st["mine_fan_avg"] = (sum((k or 1) * v for k, v in fan_mine.items())
                          / max(1, sum(fan_mine.values())))
    st["opp_fan_avg"] = (sum((k or 1) * v for k, v in fan_opp.items())
                         / max(1, sum(fan_opp.values())))
    st["mine_baotou"] = sum(v for k, v in det_mine.items() if "爆头" in k) / tot_mine
    st["mine_chiitoi"] = sum(v for k, v in det_mine.items() if "七对" in k) / tot_mine
    st["opp_baotou"] = sum(v for k, v in det_opp.items() if "爆头" in k) / tot_opp
    st["opp_chiitoi"] = sum(v for k, v in det_opp.items() if "七对" in k) / tot_opp

    md.append("## ① 断言（两个守门员）")
    md.append("")
    if id_bad:
        md.append("- ✗ **结算恒等式违例 %d 局** —— 必须立刻查，前 3 例：" % len(id_bad))
        for b in id_bad[:3]:
            md.append("  - `%s`" % (b,))
    else:
        md.append("- ✅ 结算恒等式：**%d/%d 逐项一致**" % (n_round, n_round))
    if succ_tot:
        md.append("- %s 庄位继承「下一局庄家 == 本局赢家」：**%d/%d = %.2f%%**"
                  % ("✅" if succ == succ_tot else "⚠", succ, succ_tot,
                     100.0 * succ / succ_tot))
    md.append("")

    md.append("## ② 逐场")
    md.append("")
    md.append(md_table([{"game": p["game_id"], "seat": p["seat"],
                         "mine": p["score"], "scores": p["scores"]}
                        for p in per_game]))
    md.append("")
    md.append("> 逐场的局数、每局明细与番型见同名 `.json`（`per_game` 字段）。")
    md.append("")

    md.append("## ③ 统计")
    md.append("")
    md.append("| 指标 | 值 |")
    md.append("|---|---|")
    md.append("| 场数 | %d |" % len(data["games"]))
    md.append("| 结算局 | %d（流局 %d） |" % (n_round, draws))
    md.append("| 我方总分 | **%+.0f** |" % tot_local)
    md.append("| 我方场均 | **%+.2f** |" % (tot_local / max(1, len(data["games"]))))
    md.append("| 我方每局 μ | **%+.3f** |" % mu)
    md.append("| 我方每局 σ | **%.3f** |" % sd)
    md.append("| 我方胡牌 | %d 局（占 %.1f%%） |" % (wins, 100.0 * wins / max(1, n)))
    md.append("| 我方均番 | %.3f |" % st["mine_fan_avg"])
    md.append("| 我方爆头/七对 | %.1f%% / %.1f%% |"
              % (100 * st["mine_baotou"], 100 * st["mine_chiitoi"]))
    md.append("| 对手均番 | %.3f |" % st["opp_fan_avg"])
    md.append("| 对手爆头/七对 | %.1f%% / %.1f%% |"
              % (100 * st["opp_baotou"], 100 * st["opp_chiitoi"]))
    md.append("| 我方座位（逐场重排） | %s |" % st["seats_used"])
    md.append("")
    md.append("对照基线：夹具（同质桌）μ = %+.3f、σ = %.3f。"
              % (BASE_FIXTURE_MU, BASE_FIXTURE_SD))
    md.append("")

    md.append("## ④ 复盘要点")
    md.append("")
    if tot_local < 0:
        md.append("- 本房间**净负 %+.0f**，场均 %+.2f —— 低于夹具基线。"
                  % (tot_local, tot_local / max(1, len(data["games"]))))
    else:
        md.append("- 本房间**净胜 %+.0f**，场均 %+.2f。"
                  % (tot_local, tot_local / max(1, len(data["games"]))))
    if per_game:
        best = max(per_game, key=lambda p: p["score"])
        worst = min(per_game, key=lambda p: p["score"])
        md.append("- 最好一场 `%s` **%+.0f**；最差一场 `%s` **%+.0f**。"
                  % (best["game_id"], best["score"], worst["game_id"], worst["score"]))
    if st["mine_chiitoi"] < 0.01 and st["opp_chiitoi"] > 0.02:
        md.append("- ★ 我方七对率 %.1f%% 明显低于对手 %.1f%% —— "
                  "与 `route`/`peng` 互斥那条已知张力一致（见 STATE.md 波 3）。"
                  % (100 * st["mine_chiitoi"], 100 * st["opp_chiitoi"]))
    md.append("- ⚠ **一个房间 = 一个样本**：只用于监控与累积，"
              "**不能**据此改策略参数。")
    md.append("")

    # ---- 自校验 ----
    if account and rows:
        diff = tot_local - sum(p["score"] for p in per_game)
        st["self_check"] = {"jsonl_total": tot_local,
                            "event_sum": sum(p["score"] for p in per_game),
                            "diff": diff}
        md.append("## ⑤ 自校验")
        md.append("")
        md.append("- 结果文件里我方总分 = **%+.0f**；"
                  "按逐场座位从事件流求和 = **%+.0f** ⇒ 差 **%+.0f**"
                  % (tot_local, sum(p["score"] for p in per_game), diff))
        if abs(diff) < TOL:
            md.append("- ✅ **完全一致** ⇒ 逐场座位与回合都读对了。")
        else:
            md.append("- ✗ **不一致** ⇒ 座位或回合有遗漏，**上面所有数都不能用**。")
        md.append("")
    return st, "\n".join(md)


def write_out(room, st, md):
    os.makedirs(OUTDIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    mdp = os.path.join(OUTDIR, "%s_%s.md" % (room, ts))
    jsp = os.path.join(OUTDIR, "%s_%s.json" % (room, ts))
    with open(mdp, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(jsp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return mdp, jsp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default=None)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--account", default=None)
    ap.add_argument("--log", default="room_join2.log")
    ap.add_argument("--recent", type=int, default=None)
    ap.add_argument("--no-net", action="store_true")
    a = ap.parse_args()

    account = a.account or account_from_log(a.log)
    if not account:
        print("⚠ 读不到我们的账号（--account 或 --log）；"
              "没有账号就只能出逐场分数，出不了座位相关的统计。")

    # ★ `--room` **优先**：给了它就只有它一个。
    #   第一版把"`--jsonl` 里的全部房间"和"`--room`"混在一起 ⇒
    #   值守传 `--room X --jsonl Y` 时会把 Y 里**所有**房间都复盘一遍，
    #   既不是调用方要的，也**打穿了 `watch_reviews.py --max-per-cycle`**
    #   的节流（一次叠了 ~150 次 GET，正是第 33 轮漏窗口的那种形状）。
    targets = []
    if a.room:
        rows = []
        if a.jsonl:
            for room, _tot, rws, _g in rooms_of(a.jsonl):
                if room == a.room:
                    rows = rws
                    break
        targets.append((a.room, rows))
    elif a.jsonl:
        for room, _tot, rws, _g in rooms_of(a.jsonl, a.recent):
            targets.append((room, rws))

    if not targets:
        print("要给 --room 或 --jsonl 之一。")
        return 1

    R = None
    if not a.no_net:
        import mj_replay as R  # noqa: F811

    n_ok = 0
    n_bad = 0
    for room, rows in targets:
        print("=== 复盘房间 %s ===" % room)
        st, md = analyse(room, rows, account, R, a.no_net)
        sc = st.get("self_check")
        bad = bool(sc) and abs(sc["diff"]) > TOL
        if bad:
            # ★ 自校验不过 ⇒ **不写盘、返回非 0**。
            #   第 34 轮踩过：值守批量触发时把 API 打到限速，若干场的
            #   `/events` 取失败 ⇒ 复盘少了一部分局 ⇒ 自校验对不上。
            #   那些文件当时**照样写盘并被标记成"已复盘"**，
            #   于是本地留下了一批**看起来正常、数字是错的**复盘。
            #   ⇒ 自校验不过必须当成**失败**，让值守下次重做。
            n_bad += 1
            print("  ✗ 自校验不通过（差 %+.0f）⇒ **不写盘**，留待重做"
                  % sc["diff"])
            if st.get("n_fail"):
                print("     （%d 个对局的事件没取到 —— 多半是请求被限速，"
                      "重做时请放慢）" % st["n_fail"])
            continue
        mdp, jsp = write_out(room, st, md)
        print("  → %s\n  → %s" % (mdp, jsp))
        n_ok += 1
    print()
    print("完成 %d/%d 个房间%s"
          % (n_ok, len(targets), ("，%d 个因自校验失败未写盘" % n_bad) if n_bad else ""))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
