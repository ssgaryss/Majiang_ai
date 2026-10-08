#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并发版最小 Bot —— 杭州麻将对战平台 AI 玩家（纯标准库、零依赖、无牌技）。

官方模板是「一场一场顺序打」；本版改成「同一账号名下的多场并发对局，各开一个线程并行打」。
原因：auto 房默认 M=10（同开 10 桌），若顺序打只专心第 1 场，其余 9 场会因 3s 出牌超时
被服务端自动代打 —— 等于大半场挂机，看不到 AI 在打。

策略与官方模板一致（故意最弱，等你替换决策分支）：
  · 出牌 = 打手牌第一张（抓打圈只能打刚摸到的那张）
  · 碰/吃窗口一律 pass
  · 不主动判胡/杠 —— 可胡未响应由服务端 3s 超时自动胡兜底（不丢番）

协议要点（快照无 allowed_actions，动作全自判，服务端纯验证）：
  GET  /api/games/{id}/state?seq=N   长轮询状态/事件；seq 落后跨局自动回全量快照(gap)
  POST /api/games/{id}/action        提交 {"action":..., "tile":...}
  动作后 seq=0 重建快照防漂移；409=竞态/动作已失效，重建重试。

用法：
  python my_bot.py <token>             # 报名令牌（/api/me 自带 tournament_id）
  python my_bot.py <token> <room_id>   # 全局令牌自测：显式指定 auto/测试房 id
"""
import argparse
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request

SERVER = "https://10.240.169.190:18080"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows 控制台兜底

TERMINAL = ("finished", "closed", "void", "ended")

_LOCK = threading.Lock()


def say(msg=""):
    """线程安全的一行输出。"""
    with _LOCK:
        print(msg, flush=True)


class ApiError(Exception):
    def __init__(self, status, body):
        self.status = status
        self.body = body

    def __str__(self):
        return "HTTP %s: %s" % (self.status, self.body)


# 撞到 429 时的回调（由 mj_bot 挂上自适应限速）。
# 为什么需要它：撞一次 429 就要退避重试最多 36 秒，而碰/吃窗口只有 1 秒 ——
# 一次 429 等于连丢好几个窗口。所以必须让「撞上」这件事立刻反馈到发号速率上，
# 而不是让每个线程各自闷头重试。
ON_THROTTLE = None


def api(method, path, server=SERVER, token=None, body=None, timeout=35, tries=6):
    """带 Bearer 认证的 JSON 请求；429 / 5xx / MATCH_BUSY 退避重试，其余错误抛 ApiError。

    ★★★ 2026-09-29：新增 `timeout` / `tries` 两个**可选**参数（默认值 = 原行为，向后兼容）。

    为什么必须有它们（这是被线上实测逼出来的，不是口味）：
    ```
      · 默认 `timeout=35` × `tries=6` + 退避(2,4,6,8,8) ⇒ **最坏阻塞约 258 秒** ✗
      · 而线上出牌窗口只有 **3 秒**、碰/吃窗口只有 **1 秒**（服务器 config
        `DiscardTimeoutSec: 3` / `PengTimeoutSec: 1` / `ChiTimeoutSec: 1`）
      · 实测（`live_r16.log` 92,466 行 / 约 240 场）：
          `409 被拒` **1,913** 次、`窗口错过` **7,072** 次、
          ★`放弃动作` **722** 次 —— 类型分布 **100% 都是「丢了胡牌」** ✗
      · 场次结束行里出现过 `p99 2149~3031ms`、**最大 15455ms（15.5 秒）** 的延迟
        ⇒ 正是"若干次重试"的形态 ✓
      · ★ 且它解释了表面矛盾：**阻塞期间根本不发请求** ⇒ 请求率看着很低（2~10/秒），
        却仍错过窗口 ⇒ 所以"降桌数"治不了它 ✗
    ```
    ⇒ **快照 GET**（`mj_bot.py` 里取 `state`）应当用**短超时 + 不重试**：
      快照的"新鲜度"只在窗口内（1~3 秒）有用，**超窗后拿到也没用** ✗
      ⇒ 应【立刻放弃、交给下一轮】✓ 而不是死等 258 秒 ✗
    ⇒ **动作提交**（POST action）**保持**默认（提交值得等）✓
    """
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif method == "POST":            # POST 无 body 也显式发空 body
        data = b""
    if token:
        headers["Authorization"] = "Bearer " + token
    url = server.rstrip("/") + path
    ctx = ssl._create_unverified_context()  # 部署实例为 caddy 自签证书，跳过校验
    for attempt in range(1, max(1, int(tries)) + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                raw = r.read().decode("utf-8", "replace")
                return json.loads(raw) if raw.strip() else None
        except urllib.error.HTTPError as e:
            code = e.code
            err = e.read().decode("utf-8", "replace")
            if code == 429 or code in (500, 502, 503, 504) or \
               (code == 409 and "MATCH_BUSY" in err):
                if code == 429 and ON_THROTTLE is not None:
                    try:
                        ON_THROTTLE()
                    except Exception:                            # noqa: BLE001
                        pass
                time.sleep(min(8, 2 * attempt))
                continue
            raise ApiError(code, err)
    raise ApiError(429, "退避重试耗尽")


class GameWorker(threading.Thread):
    """单场对局循环：长轮询 -> 快照 -> 自研动作判定 -> 提交。"""

    def __init__(self, player, gid):
        super().__init__(daemon=True)
        self.player = player
        self.gid = gid
        self.name = "%s:%s" % (player.label, gid[-6:])

    def _w(self, msg):
        if self.player.trace:
            say("[%s] %s" % (self.gid[-6:], msg))

    def run(self):
        seq = 0
        try:
            while not self.player._stop.is_set():
                try:
                    res = api("GET", "/api/games/%s/state?seq=%d" % (self.gid, seq),
                              self.player.server, self.player.token)
                except ApiError as e:
                    if e.status == 404:      # 房间关停/场次消失：视为打完
                        return
                    raise
                if res is None:
                    continue
                if res.get("pending"):       # 30s 无新事件：继续挂起
                    continue
                if res.get("finished"):      # 本场结束，收积分
                    snap = res.get("snapshot") or {}
                    self.player.results[self.gid] = snap.get("scores") or snap or res
                    say("[%s] 场 %s 结束" % (self.player.label, self.gid[-6:]))
                    return
                snap = res.get("snapshot")
                if snap is None:             # 增量事件：推进 seq 后重建权威快照
                    for ev in res.get("events") or []:
                        seq = ev.get("seq", seq)
                    full = api("GET", "/api/games/%s/state?seq=0" % self.gid,
                               self.player.server, self.player.token) or {}
                    snap = full.get("snapshot") or {}
                    seq = full.get("seq", seq)
                else:
                    seq = res.get("seq", seq)

                seat = snap.get("seat", -1)
                if seat < 0:                 # 观赛视角无动作权
                    continue
                phase, turn = snap.get("phase"), snap.get("turn")
                act = None
                if phase == "draw" and turn == seat:
                    god = snap.get("god") or {}
                    drawn = snap.get("drawn_tile")
                    hand = snap.get("my_hand") or []
                    if god.get("catch_play") and drawn:
                        act = {"action": "discard", "tile": drawn}   # 抓打圈：只出刚摸的
                    elif hand:
                        act = {"action": "discard", "tile": hand[0]}  # 出牌打第一张
                elif phase in ("response_peng", "response_chi"):
                    if seat in (snap.get("responding_seats") or []):
                        act = {"action": "pass", "tile": ""}          # 示例策略：碰/吃一律过
                if act is None:
                    continue
                self._w("seat%d %s -> %s" % (seat, phase, act))
                try:
                    api("POST", "/api/games/%s/action" % self.gid, self.player.server,
                        self.player.token, body=act)
                except ApiError as e:
                    if e.status != 409:      # 409 = 竞态/动作已失效：重建快照后重试
                        raise
                seq = 0                      # 动作后重建快照，避免状态漂移
        except ApiError as e:
            if e.status != 404:
                say("[%s] 场 %s API 错误: %s" % (self.player.label, self.gid, e))
        except Exception as e:
            say("[%s] 场 %s 异常: %r" % (self.player.label, self.gid, e))


class Player(threading.Thread):
    """一个参赛账号：轮询自己名下的活跃场次，每场开一个 GameWorker 并发打。"""

    def __init__(self, server, token, room_id=None, trace=False,
                 max_minutes=40, label="bot"):
        super().__init__()
        self.server = server
        self.token = token
        self.room_id = room_id
        self.trace = trace
        self.max_minutes = max_minutes
        self.label = label
        self._workers = {}           # gid -> GameWorker
        self._wlock = threading.Lock()
        self.results = {}            # gid -> 终局信息
        self._stop = threading.Event()

    def run(self):
        deadline = time.time() + self.max_minutes * 60
        if not self.room_id:         # 报名令牌：/api/me 自带绑定锦标赛
            try:
                me = api("GET", "/api/me", self.server, self.token)
                self.room_id = me.get("tournament_id")
            except ApiError as e:
                say("[%s] 取 /api/me 失败: %s" % (self.label, e))
        if not self.room_id:
            say("[%s] 无 room_id 且令牌未绑定锦标赛，退出" % self.label)
            return

        while not self._stop.is_set():
            if time.time() > deadline:
                say("[%s] 超过 %d 分钟上限，退出" % (self.label, self.max_minutes))
                break
            actives, status = [], None
            try:
                me = api("GET", "/api/me", self.server, self.token)
                for g in me.get("active_games") or []:
                    gid = g.get("game_id") if isinstance(g, dict) else g
                    if gid:
                        actives.append(gid)
                try:
                    t = api("GET", "/api/tournaments/%s" % self.room_id,
                            self.server, self.token)
                    status = t.get("status")
                except ApiError as e:
                    if e.status == 404:      # auto 房关停：一律 404
                        status = "closed"
                    else:
                        raise
            except ApiError as e:
                if e.status == 404:
                    actives, status = [], "closed"
                else:
                    say("[%s] 轮询失败: %s" % (self.label, e))
                    break

            with self._wlock:
                running = set(self._workers)
            for gid in actives:
                if gid not in running:
                    w = GameWorker(self, gid)
                    with self._wlock:
                        self._workers[gid] = w
                    w.start()
                    say("[%s] 发现并开打场次 %s（共 %d 场并行）" % (self.label, gid, len(self._workers)))
            with self._wlock:                # 回收已结束的工作线程
                for gid in [g for g, w in self._workers.items() if not w.is_alive()]:
                    del self._workers[gid]

            if status in TERMINAL and not actives and not self._workers:
                say("[%s] 房间已%s，全部场次打完" % (self.label, status))
                break
            time.sleep(1.0)


def run_player(server, token, room_id, trace=False, max_minutes=40, label="bot"):
    """跑一个玩家直到其名下场次全部结束；返回 player（含 .results）。"""
    pl = Player(server, token, room_id=room_id, trace=trace,
                max_minutes=max_minutes, label=label)
    pl.start()
    pl.join()
    return pl


def main():
    p = argparse.ArgumentParser(description="并发版最小 Bot（单账号玩家）")
    p.add_argument("token")
    p.add_argument("room_id", nargs="?", default=None,
                   help="全局令牌自测时显式给房号；报名令牌可省略")
    p.add_argument("--server", default=SERVER)
    p.add_argument("--trace", action="store_true", help="打印每次提交的动作")
    p.add_argument("--max-minutes", type=int, default=40)
    a = p.parse_args()
    say("玩家启动 room=%s（trace=%s）" % (a.room_id or "<from /api/me>", a.trace))
    pl = run_player(a.server, a.token, a.room_id, a.trace, a.max_minutes)
    say("结束。共打完 %d 场" % len(pl.results))
    for gid in sorted(pl.results):
        say("  %s: %s" % (gid, json.dumps(pl.results[gid], ensure_ascii=False)))


if __name__ == "__main__":
    main()
