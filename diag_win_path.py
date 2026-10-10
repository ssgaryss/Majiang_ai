#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""胡牌路径分解 —— 把「自摸率低」拆成三个可分别优化的量。

线上 R2（30 场 238 局）里我的自摸率 19.7%，桌面平均约 26.8%；走马 32.5%。
自摸率是**复合量**，只有拆开才知道该改哪里：

    ① 速度   ：多快进入听牌（进入听牌前的平均弃牌数）
    ② 转化率 ：听牌之后真的胡了的比例（P(胡 | 进听)）
    ③ 听口宽度：听牌瞬间的有效牌枚数（宽听更容易转化）

以及贯穿其中的第四条：
    ④ 爆头率 ：听任意 = 转化率接近 1，且番数 ×2

口径（全部用本地已对过 oracle 的 can_win / shanten / is_baotou 复算）：
  · 一局里某席位「进听」= 该席位某次弃牌后（13−3m 张）向听 == 0
  · 听口宽度 = 该时刻 `winning_tiles` 里仍未见过的枚数（用四家弃牌 + 副露扣减）
  · 爆头 = 该时刻 `is_baotou` 为真（这才是「听任意」）
  · 速度 = 从本局该席位第一次摸牌算起，到第一次进听所经过的自己的弃牌数

用法: python diag_win_path.py a_room1 a_room2 ...
      python diag_win_path.py --auto 5      # 取 live_loop.jsonl 里最近 5 个房间
"""
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

sys.path.insert(0, ".")
import shanten_ref as S                                        # noqa: E402
from mj_core import (NTILE, WILD, is_baotou, tile_index,       # noqa: E402
                     to_counts, winning_tiles)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 与 `my_bot.SERVER` 同一口径：可用环境变量 `MJ_SERVER` 覆盖（不写死）。
SERVER = os.environ.get("MJ_SERVER", "https://10.240.169.190:18080")
CTX = ssl._create_unverified_context()
REMOVE = {"peng": 2, "minggang": 3, "gang": 3, "angang": 4, "bugang": 1}
KONG_TYPES = {"gang", "angang", "minggang", "bugang"}


def get(path, retry=5):
    req = urllib.request.Request(SERVER + path, headers={"User-Agent": "diag"})
    for i in range(retry):
        try:
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                raw = r.read().decode("utf-8", "replace")
                return json.loads(raw) if raw.strip() else None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i + 1 < retry:
                time.sleep(2.5 * (i + 1))
                continue
            return {"_err": "HTTP %s" % e.code}
        except Exception as e:                                    # noqa: BLE001
            if i + 1 < retry:
                time.sleep(2.5 * (i + 1))
                continue
            return {"_err": repr(e)[:120]}
    return {"_err": "exhausted"}


class Book:
    def __init__(self):
        self.rounds = 0
        self.tenpai = 0            # 进听的局数
        self.wins = 0
        self.win_after_tenpai = 0
        self.baotou_win = 0
        self.speed = []            # 进听所用弃牌数
        self.widths = []           # 听口宽度（未见枚数）
        self.widths_bt = []        # 其中爆头形的宽度
        self.melds = 0
        self.discards = 0
        self.tenpai_discards = 0   # 进听之后又打了几张（越少越快兑现）


def unseen_from(hands, melds_tiles, discards):
    """已见（死）牌计数：他家弃牌 + 全部副露用掉的牌。

    ★★ 第 52 轮修 bug（第三个，也是藏得最深的那个）：
    传进来的 `melds_tiles` **实际形状是「按座位 → 再按面子 → 牌码」**，
    例如 `[[['1w','2w','3w']], [], [['中','中','中']], []]`。
    旧代码只下钻两层（`for m in melds_tiles: for code in m:`），
    于是 `code` 拿到的是**一个面子（list）**，`tile_index(list)` 抛
    `AttributeError: 'list' object has no attribute 'strip'`。

    而调用方那段是 `try: ... except Exception: pass`，且 `seated` / `first_tp_turn`
    **在 try 块里先赋值**、`widths.append` 在后面 ⇒ 异常被吞掉之后留下
    **"进了听但没有宽度"的静默残缺记录**。

    实测（`probe_width2.py`，25 房 / 4998 个进听席位）：**4729 个（94.6%）宽度为空**。
    ⇒ `diag_win_path` 报的"听口宽度"其实只来自**没人副露的那约 5% 的局**
    （因为只有 `melds` 全为空时旧代码才不会抛异常）—— 那是个被重度选择的子样本，
    所以"听口宽度没有差别"这条结论**不可信**。

    这里改成**两种形状都吃**，避免再依赖调用方的形状假设。
    """
    dead = [0] * NTILE
    for row in discards:
        for code in row:
            try:
                dead[tile_index(code)] += 1
            except (ValueError, TypeError):
                pass
    for m in melds_tiles:
        if not m:
            continue
        if isinstance(m[0], (list, tuple)):        # 这一项是"某座位的面子表"
            rows = m
        else:                                      # 这一项已经是一个面子
            rows = [m]
        for row in rows:
            if isinstance(row, str):               # 防御：单张牌码
                row = [row]
            for code in row:
                try:
                    dead[tile_index(code)] += 1
                except (ValueError, TypeError):
                    pass
    return dead


def merge_rounds(ev):
    """把同一局被**分页切成多个 block** 的事件流按 round_no 合并。

    这是本脚本第二版才发现的坑：回放接口按 128 个事件把一局切成多个 block
    （字段 `truncated` / `seq_start` / `seq_end` 就是为此存在的），一局 8 巡
    的对局大概一半会被切开。review.py 第一版按 block 独立重建手牌 —— 中段的
    block 从「本局起手」重放一段中局事件，牌数永远对不上，于是那些决策被静默
    跳过，只剩每局的第一段被统计。后果是「出牌后向听」这类指标被系统性地
    偏向**开局**（手牌大、向听高），跨玩家比较也就失去意义。

    返回 [(round_no, events, start_hands)]，只保留能确定起手的局。
    """
    by_round = {}
    for b in ev.get("blocks") or []:
        rn = b.get("round_no")
        by_round.setdefault(rn, []).append(b)
    out = []
    for rn, blks in sorted(by_round.items(), key=lambda kv: (kv[0] is None, kv[0])):
        blks.sort(key=lambda b: (b.get("seq_start") or 0))
        evs = []
        for b in blks:
            evs.extend(b.get("events") or [])
        sh = blks[0].get("start_hands")
        out.append((rn, evs, sh))
    return out


def analyse_round_merged(evs, start_hands):
    """在一局的完整事件序列上重建四家手牌并统计三项指标。

    起点手牌：庄家 14 张（含直抽的第 14 张），其余 13 张 —— 所以第一版用
    `len(hand) == 13 - 3m` 当判据时，庄家整局都被漏掉。这里改成按「刚打完牌」
    的事件驱动：只在 tile_discarded 之后检查，且用**该席位当时的实际张数**
    与该席位应有张数比较。
    """
    hands = [list(h) if h else [] for h in start_hands]
    melds = [[] for _ in range(4)]
    meld_n = [0] * 4
    discards = [[] for _ in range(4)]
    seated = [False] * 4
    first_tp_turn = [None] * 4
    own_disc = [0] * 4
    widths = [[] for _ in range(4)]
    bt = [0] * 4
    winner = None
    draw = False
    for e in evs:
        t = e.get("type")
        s = e.get("seat")
        if not isinstance(s, int) or not (0 <= s < 4):
            continue
        if t == "tile_drawn":
            hands[s].append(e.get("tile"))
        elif t == "tile_discarded":
            tile = e.get("tile") or ""
            if tile in hands[s]:
                hands[s].remove(tile)
            discards[s].append(tile)
            own_disc[s] += 1
            if len(hands[s]) != 13 - 3 * meld_n[s]:
                continue                      # 重建漂移：宁可不算，不编数字
            try:
                c = to_counts(hands[s])
                if S.shanten(c, meld_n[s]) == 0:
                    if not seated[s]:
                        seated[s] = True
                        first_tp_turn[s] = own_disc[s]
                    dead = unseen_from(hands, melds, discards)
                    left = 0
                    for code in winning_tiles(c, meld_n[s]):
                        ti = tile_index(code)
                        left += max(0, 4 - dead[ti] - c[ti])
                    widths[s].append(left)
                    if is_baotou(c, meld_n[s]):
                        bt[s] += 1
            except Exception:                                     # noqa: BLE001
                pass
        elif t in KONG_TYPES or t in ("peng", "chi"):
            tl = e.get("tile") or ""
            if t in KONG_TYPES:
                # ★★ 第 51 轮修 bug（第二个）：服务端**只发一种** `type="gang"`，
                #   它既可能是"新开一个杠"，也可能是**加杠** ——
                #   给**已有的碰**再添第 4 张。
                #   旧代码对所有杠都 `meld_n += 1` 且删 3 张 ⇒ 加杠时手牌长度
                #   恒差 **+3**（`probe_gang_recon.py` 实测：
                #   `peng 西` 紧接 `gang 西`，手里原有该牌 1 张，
                #   meld_n 1->2，期望 7 而手上有 10）。
                #   正确口径：**加杠只从手里删 1 张，meld_n 不变**。
                is_bugang = any(m and len(m) == 3 and all(x == tl for x in m)
                                for m in melds[s])
                if is_bugang:
                    for _ in range(REMOVE.get("bugang", 1)):
                        if tl in hands[s]:
                            hands[s].remove(tl)
                    melds[s].append([tl])        # 死牌计数：该牌现在共 4 张
                else:
                    meld_n[s] += 1
                    for _ in range(REMOVE.get(t, 3)):
                        if tl in hands[s]:
                            hands[s].remove(tl)
                    melds[s].append([tl] * 3)
            elif t == "chi":
                # ★★ 第 51 轮修 bug（第一个，也是影响最大的那个）：
                #   吃的时候必须把**自己出的那两张**从手牌里删掉；
                #   被吃的那张是别人打的，本来就不在自己手里。
                #   旧代码这里**一张都不删**，而 `meld_n[s]` 已经 +1
                #   ⇒ `len(hands[s]) != 13 - 3*meld_n[s]` **此后恒成立**
                #   ⇒ 那道"重建漂移就不算"的闸门会把该席位**这一局剩下的
                #     进听判定全部跳过**。
                #   实测（`probe_chi_recon.py`，6 个房间）：吃过之后长度闸门
                #   **通过 / 跳过 = 0 / 5486**；而碰过之后是 3225 / 1927。
                #   ⇒ 吃得越多的人，进听率被低估得越狠 ——
                #     而**我们正是全场副露/吃最多的那个**。
                meld_n[s] += 1
                tiles = list((e.get("data") or {}).get("tiles") or [])
                melds[s].append(list(tiles))
                rest = list(tiles)
                if tl in rest:
                    rest.remove(tl)
                for code in rest[:2]:          # 最多两张；吃是三张顺子，必定两张
                    if code in hands[s]:
                        hands[s].remove(code)
            else:                                            # 碰：出手里两张
                meld_n[s] += 1
                melds[s].append([tl] * 3)
                for _ in range(2):
                    if tl in hands[s]:
                        hands[s].remove(tl)
        elif t == "round_ended":
            d = e.get("data") or {}
            draw = bool(d.get("draw"))
            winner = None if draw else s
    return {"hands": hands, "melds": melds, "meld_n": meld_n, "seated": seated,
            "first_tp_turn": first_tp_turn, "widths": widths, "bt": bt,
            "winner": winner, "draw": draw, "discards": discards,
            "own_disc": own_disc, "complete": winner is not None or draw}


def analyse_round(blk):
    sh = blk.get("start_hands")
    if not sh or any(h is None for h in sh):
        return None
    evs = blk.get("events") or []
    hands = [list(h) for h in sh]
    melds = [[] for _ in range(4)]          # 牌码列表
    meld_n = [0] * 4
    discards = [[] for _ in range(4)]
    seated = [False] * 4                    # 是否已进听
    first_tp_turn = [None] * 4
    own_disc = [0] * 4
    widths = [[] for _ in range(4)]
    bt = [0] * 4
    winner = None
    draw = False
    for e in evs:
        t = e.get("type")
        s = e.get("seat")
        if not isinstance(s, int) or not (0 <= s < 4):
            continue
        if t == "tile_drawn":
            hands[s].append(e.get("tile"))
        elif t == "tile_discarded":
            tile = e.get("tile") or ""
            if tile in hands[s]:
                hands[s].remove(tile)
            discards[s].append(tile)
            own_disc[s] += 1
            if len(hands[s]) == 13 - 3 * meld_n[s]:
                try:
                    c = to_counts(hands[s])
                    if S.shanten(c, meld_n[s]) == 0:
                        if not seated[s]:
                            seated[s] = True
                            first_tp_turn[s] = own_disc[s]
                        dead = unseen_from(hands, melds, discards)
                        left = 0
                        for code in winning_tiles(c, meld_n[s]):
                            ti = tile_index(code)
                            left += max(0, 4 - dead[ti] - c[ti])
                        widths[s].append(left)
                        if is_baotou(c, meld_n[s]):
                            bt[s] += 1
                except Exception:                                # noqa: BLE001
                    pass
        elif t in KONG_TYPES or t in ("peng", "chi"):
            tl = e.get("tile") or ""
            if t in KONG_TYPES:
                # ★★ 第 51 轮修 bug（第二个）：服务端**只发一种** `type="gang"`，
                #   它既可能是"新开一个杠"，也可能是**加杠** ——
                #   给**已有的碰**再添第 4 张。
                #   旧代码对所有杠都 `meld_n += 1` 且删 3 张 ⇒ 加杠时手牌长度
                #   恒差 **+3**（`probe_gang_recon.py` 实测：
                #   `peng 西` 紧接 `gang 西`，手里原有该牌 1 张，
                #   meld_n 1->2，期望 7 而手上有 10）。
                #   正确口径：**加杠只从手里删 1 张，meld_n 不变**。
                is_bugang = any(m and len(m) == 3 and all(x == tl for x in m)
                                for m in melds[s])
                if is_bugang:
                    for _ in range(REMOVE.get("bugang", 1)):
                        if tl in hands[s]:
                            hands[s].remove(tl)
                    melds[s].append([tl])        # 死牌计数：该牌现在共 4 张
                else:
                    meld_n[s] += 1
                    for _ in range(REMOVE.get(t, 3)):
                        if tl in hands[s]:
                            hands[s].remove(tl)
                    melds[s].append([tl] * 3)
            elif t == "chi":
                # ★★ 第 51 轮修 bug（第一个，也是影响最大的那个）：
                #   吃的时候必须把**自己出的那两张**从手牌里删掉；
                #   被吃的那张是别人打的，本来就不在自己手里。
                #   旧代码这里**一张都不删**，而 `meld_n[s]` 已经 +1
                #   ⇒ `len(hands[s]) != 13 - 3*meld_n[s]` **此后恒成立**
                #   ⇒ 那道"重建漂移就不算"的闸门会把该席位**这一局剩下的
                #     进听判定全部跳过**。
                #   实测（`probe_chi_recon.py`，6 个房间）：吃过之后长度闸门
                #   **通过 / 跳过 = 0 / 5486**；而碰过之后是 3225 / 1927。
                #   ⇒ 吃得越多的人，进听率被低估得越狠 ——
                #     而**我们正是全场副露/吃最多的那个**。
                meld_n[s] += 1
                tiles = list((e.get("data") or {}).get("tiles") or [])
                melds[s].append(list(tiles))
                rest = list(tiles)
                if tl in rest:
                    rest.remove(tl)
                for code in rest[:2]:          # 最多两张；吃是三张顺子，必定两张
                    if code in hands[s]:
                        hands[s].remove(code)
            else:                                            # 碰：出手里两张
                meld_n[s] += 1
                melds[s].append([tl] * 3)
                for _ in range(2):
                    if tl in hands[s]:
                        hands[s].remove(tl)
        elif t == "round_ended":
            d = e.get("data") or {}
            draw = bool(d.get("draw"))
            winner = None if draw else s
    return {"hands": hands, "melds": melds, "meld_n": meld_n, "seated": seated,
            "first_tp_turn": first_tp_turn, "widths": widths, "bt": bt,
            "winner": winner, "draw": draw, "discards": discards,
            "own_disc": own_disc}


def main():
    args = sys.argv[1:]
    if args and args[0] == "--auto":
        n = int(args[1]) if len(args) > 1 else 5
        rows = [json.loads(l) for l in open(__import__("live_files").newest(), encoding="utf-8")
                if l.strip()]
        rooms = [r["room"] for r in rows[-n:]]
    else:
        rooms = args
    if not rooms:
        print("用法: python diag_win_path.py a_room1 ... | --auto 5")
        return 1

    books = {}
    names = {}
    n_rounds = 0
    for room in rooms:
        gl = get("/api/test-rooms/%s/games" % room)
        for gm in ((gl or {}).get("games") or []):
            if gm.get("status") != "finished" or gm.get("batch") is None:
                continue
            ev = get("/api/test-rooms/%s/games/%s/events" % (room, gm.get("batch")))
            if not isinstance(ev, dict) or "_err" in ev:
                print("  %s/%s -> %s" % (room, gm.get("batch"), ev))
                continue
            seats = ev.get("seats") or []
            uids = [(s or {}).get("user_id") or ("seat%d" % i)
                    for i, s in enumerate(seats)]
            for i, s in enumerate(seats):
                names[uids[i]] = (s or {}).get("name") or uids[i]
            for rn, evs, sh in merge_rounds(ev):
                rr = analyse_round_merged(evs, sh) if sh else None
                if rr is None or not rr["complete"]:
                    continue
                n_rounds += 1
                for i in range(4):
                    b = books.setdefault(uids[i], Book())
                    b.rounds += 1
                    b.melds += rr["meld_n"][i]
                    b.discards += rr["own_disc"][i]
                    if rr["seated"][i]:
                        b.tenpai += 1
                        b.speed.append(rr["first_tp_turn"][i])
                        if rr["widths"][i]:
                            b.widths.append(sum(rr["widths"][i]) / len(rr["widths"][i]))
                        if rr["bt"][i]:
                            b.widths_bt.append(1.0)
                    if rr["winner"] == i:
                        b.wins += 1
                        if rr["seated"][i]:
                            b.win_after_tenpai += 1
                        if rr["bt"][i]:
                            b.baotou_win += 1
            time.sleep(0.5)
        print("  房间 %s 已解析" % room)

    print()
    print("=== 共 %d 局 ===" % n_rounds)
    print("%-20s %5s %8s %8s %9s %7s %7s %7s %7s"
          % ("玩家", "局数", "进听率", "自摸率", "胡|进听", "爆头/局", "进听速度",
             "听口宽度", "副露/局"))
    for uid, b in sorted(books.items(), key=lambda kv: -kv[1].wins):
        n = max(1, b.rounds)
        tp = max(1, b.tenpai)
        print("%-20s %5d %7.1f%% %7.1f%% %8.1f%% %6.1f%% %8.2f %8.2f %8.2f"
              % (names.get(uid, uid)[:20], b.rounds, 100 * b.tenpai / n,
                 100 * b.wins / n, 100 * b.win_after_tenpai / tp,
                 100 * b.baotou_win / n,
                 (sum(b.speed) / len(b.speed)) if b.speed else 0,
                 (sum(b.widths) / len(b.widths)) if b.widths else 0,
                 b.melds / n))
    print()
    print("判读：")
    print("  进听率低 -> 手牌展开慢（弃牌/副露选择问题）")
    print("  胡|进听低 -> 听口质量或兑现速度问题（爆头/宽听的比例）")
    print("  进听速度 = 进听前自己打出的张数（越小越快）")
    print("  听口宽度 = 听牌瞬间仍未见过的有效牌枚数（爆头约等于剩余全部）")
    with open("diag_win_path.json", "w", encoding="utf-8") as f:
        json.dump({u: {"name": names.get(u, u), "rounds": b.rounds,
                       "tenpai": b.tenpai, "wins": b.wins,
                       "win_after_tenpai": b.win_after_tenpai,
                       "baotou_win": b.baotou_win, "melds": b.melds,
                       "speed": (sum(b.speed) / len(b.speed)) if b.speed else None,
                       "width": (sum(b.widths) / len(b.widths)) if b.widths else None}
                   for u, b in books.items()}, f, ensure_ascii=False, indent=1)
    print("已写入 diag_win_path.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
