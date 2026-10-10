#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**房间数据层** —— 把"从服务端把一个房间取成结构化数据"这件事只做一遍。

为什么单独一层（第 33 轮的教训）
--------------------------------
`analyze_test_room.py` 里那套"取 games → 逐场取 events → join detail → 定座"
的代码，`review_room.py` 也要用。**同一份逻辑写两遍 = 迟早两边不一致**
（本项目已经因为"同一个逻辑两份实现"踩过好几次）。

更重要的：这一层是第 33 轮**两个 bug** 的现场，把结论钉在这里：
  1. **`batch=0` 是合法值**。`batch` 从 0 编号，`not gm.get("batch")` 会把
     **每个房间的第一场**静默丢掉（全仓 48 处，已由 `codemod_batch_zero.py` 修）。
     ⇒ 本层一律用 `is None`，不用真值判断。
  2. **座位逐场重排**，`events["seats"]` 也不是"按座位排序"的权威表。
     ⇒ 每一场都要**各自**在 `seats` 里按 `user_id` 找自己的那一席。

请求量：**每个对局一次 `/events`**（外加 1 次 games 列表）。
⚠ **线上有客户端在打时，这仍然吃同一个账号的请求预算**（§U）——
   一个 10 场的房间约 11 次 GET，量很小，但批量跑多个房间前请先算一下。
"""
import json
import os
import sys

sys.path.insert(0, ".")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def seat_of_account(events, account):
    """在**这一场**的 `events["seats"]` 里找 `account` 的下标（= 座位号）。

    ⚠ 座位**逐场重排**（第 33 轮实测某房间 10 场是 0,1,1,0,0,3,2,3,2,3），
    所以必须逐场调用，不能拿一场的结果当整房间的座位。
    """
    for i, s in enumerate((events or {}).get("seats") or []):
        if isinstance(s, dict) and s.get("user_id") == account:
            return i
    return None


def _rounds_with_detail(ev):
    """`rounds[]` **没有** `detail`，它在 `blocks[].events[].round_ended` 里。

    第 33 轮踩过：直接读 `rounds[].detail` 会**静默**得到空值，
    打出"爆头 0% / 七对 0%"，而真值是 24% / 5.7%。**扫描器报 0 先怀疑扫描器。**
    """
    det_of = {}
    for blk in (ev.get("blocks") or []):
        for e in (blk.get("events") or []):
            if e.get("type") != "round_ended":
                continue
            dd = e.get("data") or {}
            det_of[dd.get("round_no")] = tuple(dd.get("detail") or ())
    out = []
    for r in (ev.get("rounds") or []):
        if r.get("dealer") is None:
            continue
        d = dict(r)
        d["detail"] = det_of.get(r.get("round_no"), ())
        out.append(d)
    out.sort(key=lambda r: (r.get("round_no") or 0))
    return out


def save_raw_cache(cache_dir, room, evs):
    """把**原始** `/events` 落盘成 `field_cache/<room>.json`（与 `field_ranking` **同格式**）。

    为什么放在这里（第 70 轮）：复盘**本来就要**把一个房间的全部 `/events` 取下来
    （约 11 次 GET，和线上客户端共用同一个账号 16/秒 的预算），但 `fetch_room`
    只留下 `rounds`/`seats` 这些**加工结果**，原始包被丢掉。
    而 `field_ranking.py --fetch` 为了**同一批数据**又会去取一遍 ——
    同一个房间被下载两次，而线上算力正是我们最贵的资源（§Z.53 的成本表）。

    接上之后：复盘值守每复盘一个房间，`field_cache/` 就白得一个房间的
    **服务端真值**，`field_ranking --report` 与逐日 μ 会自己长大，
    **不需要任何额外请求**。这是"买 μ 的证据"（§Z.64(f)）这条路上唯一免费的加速。

    口径（一律保守）：
      * `n_fail != 0` 时**不写** —— 残缺的房间会直接污染 μ（少几局就是错的）；
      * 已有缓存且**局数不少于**新的 ⇒ 不覆盖（幂等，不会把完整缓存改坏）；
      * 原子落盘（tmp + `os.replace`），中断不留半截文件。

    返回 True 表示这次真的写了。
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        dst = os.path.join(cache_dir, room + ".json")
        n_new = sum(len(e.get("rounds") or []) for e in evs)
        if os.path.exists(dst):
            try:
                with open(dst, encoding="utf-8") as fh:
                    old = json.load(fh)
                n_old = sum(len(e.get("rounds") or [])
                            for e in (old.get("evs") or []))
                if n_old >= n_new:
                    return False
            except (OSError, ValueError):
                pass
        tmp = dst + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"room": room, "evs": evs}, fh, ensure_ascii=False)
        os.replace(tmp, dst)
        return True
    except OSError:
        return False


def fetch_room(R, room, account=None, kind_filter=None, verbose=True, delay=0.35,
               raw_cache_dir=None):
    """取一个房间的结构化数据。

    `delay`：**每取完一场对局的 events 就 sleep 这么久**。
    为什么必须有（第 34 轮踩的）：复盘要读一个房间的全部 `/events`，
    而线上客户端同时在打，**两边共用同一个账号的请求预算**（§U，约 16/秒）。
    不 sleep 地连打十来次 GET，会当场把自己顶到限速 ⇒ 部分对局取失败
    ⇒ 复盘少了一部分局 ⇒ **自校验对不上**。
    0.35 秒/场（10 场约 3.5 秒）慢得几乎无感，但足以避开突发。

    `raw_cache_dir`：给了就把**原始** events 顺手落到那里（见 `save_raw_cache`）。
    这样"复盘"与"攒 μ 的服务端真值"变成同一件事，不必取两遍。

    返回 dict：
      room, config, seats(第一场的参与者表), games[ {game_id, batch, seat,
      rounds:[...], seats:[...]} ], n_fail(取不到的对局数)
    """
    import time as _t
    meta = {}
    try:
        st = R.get("/api/test-rooms/%s" % room) or {}
        meta = st if isinstance(st, dict) else {}
    except Exception:                                          # noqa: BLE001
        pass
    gl = R.get("/api/test-rooms/%s/games" % room) or {}
    games = gl.get("games") or []

    out, n_fail, first_seats = [], 0, None
    raw_evs = []                      # ★ 原始包：见 raw_cache_dir / save_raw_cache
    for gm in games:
        k = gm.get("kind") or gm.get("Kind")
        if kind_filter and k and k != kind_filter:
            continue
        if gm.get("status") != "finished":
            continue
        batch = gm.get("batch")
        if batch is None:                    # ★ 不是 `not batch` —— 0 是合法值
            if verbose:
                print("  ⚠ 对局 %s 没有 batch，跳过" % gm.get("game_id"))
            continue
        ev = None
        for attempt in (1, 2):
            try:
                ev = R.get("/api/test-rooms/%s/games/%s/events" % (room, batch))
                break
            except Exception as e:                             # noqa: BLE001
                if attempt == 2:
                    n_fail += 1
                    if verbose:
                        print("  ✗ 对局 %s 的 events 取不到（%s）—— **该场被漏掉**"
                              % (gm.get("game_id"), e))
        if not isinstance(ev, dict):
            n_fail += 1
            if verbose:
                print("  ✗ 对局 %s 的 events 不是 dict —— **该场被漏掉**"
                      % gm.get("game_id"))
            continue
        if first_seats is None:
            first_seats = ev.get("seats")
        raw_evs.append(ev)
        out.append({"game_id": gm.get("game_id"), "batch": batch,
                    "seat": seat_of_account(ev, account) if account else None,
                    "rounds": _rounds_with_detail(ev),
                    "seats": ev.get("seats")})
        if delay:
            _t.sleep(delay)            # ★ 别连打：见 docstring
    # ★ 只有**整房取全**才落盘：残缺房间会让 μ 直接错（`save_raw_cache` 里也再判一次）
    if raw_cache_dir and raw_evs and n_fail == 0:
        save_raw_cache(raw_cache_dir, room, raw_evs)
    return {"room": room, "config": meta.get("config"), "seats": first_seats,
            "games": out, "n_fail": n_fail,
            "n_listed": len(games)}


def room_totals(data, account=None):
    """按**逐场座位**汇总我方得分，并返回 (总表, 逐场明细)。"""
    detail, total, rounds = [], 0.0, 0
    for g in data["games"]:
        seat = g["seat"]
        s = 0.0
        for r in g["rounds"]:
            sc = r.get("scores") or []
            if len(sc) != 4:
                continue
            rounds += 1
            if seat is not None:
                s += float(sc[seat])
        total += s
        detail.append({"game_id": g["game_id"], "batch": g["batch"],
                       "seat": seat, "score": s, "n_rounds": len(g["rounds"]),
                       "seats": g["seats"]})
    return {"total": total, "rounds": rounds, "games": len(data["games"]),
            "detail": detail}
