#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指定房间并行开打 —— 用已有的 scoped 参赛令牌把一桌真正打起来。

与 run_self_test.py 的分工：
  · run_self_test.py  = 注册全局账号 + POST /api/match 凑 auto 房（走匹配池）
  · 本脚本            = 拿门户「测试房间」/报名派发的参赛令牌，直接进指定房

为什么需要它：my_bot.py 与 run_self_test.py 都**不调** register/ready，
而测试房的开赛条件是「4 个令牌各 register + ready，满 4 席才开赛」——
只挂机不 ready 会永远停在 status=registering。本脚本补齐这一步。

用法：
  python run_room.py --tokens tokens.txt              # 只挂机等开赛（房已 ready 好时用）
  python run_room.py --tokens tokens.txt --ready      # 代提交 register+ready 把房开起来
  python run_room.py --tokens tokens.txt --ready --trace
"""
import argparse
import json
import sys
import time

from my_bot import ApiError, SERVER, api, say
# Player 必须来自 mj_bot：my_bot 里的同名类是最小模板，签名里没有 strategy，
# 直接用它会把 strategy 顶进 room_id 并抛 TypeError（本文件此前正是坏的）。
from mj_bot import Player

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_tokens(path):
    toks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                toks.append(s)
    if not toks:
        raise SystemExit("%s 里没有令牌" % path)
    return toks


def preflight(server, tokens):
    """校验令牌有效、绑定同一房、user_id 互不相同；返回 room_id。

    互不相同这条是硬门槛：同一账号占不了两个席位（/api/match 幂等返原房），
    凑不满 4 席就会无声挂死——所以必须提前拦下。
    """
    rooms, uids = set(), []
    for i, tok in enumerate(tokens, 1):
        try:
            me = api("GET", "/api/me", server, tok)
        except ApiError as e:
            raise SystemExit("令牌 %d 无效：%s" % (i, e))
        uid, tid = me.get("user_id"), me.get("tournament_id")
        uids.append(uid)
        rooms.add(tid)
        say("  [席%d] user=%s room=%s 正在打=%d"
            % (i, uid, tid or "<空>", me.get("active_count", 0)))
    if len(set(uids)) != len(uids):
        raise SystemExit("有重复 user_id —— 一个账号占不了两个席位，永远凑不满，先换令牌")
    if len(rooms) != 1:
        raise SystemExit("令牌没绑定同一个房间：%s" % sorted(rooms))
    rid = next(iter(rooms))
    if not rid:
        raise SystemExit("这是全局令牌（tournament_id 为空）—— 请改用 run_self_test.py 走 /api/match")
    return rid


def ready_all(server, tokens, rid):
    """幂等提交 register + ready（顺序必要：ready 须先报名）。"""
    for i, tok in enumerate(tokens, 1):
        for ep in ("register", "ready"):
            try:
                api("POST", "/api/tournaments/%s/%s" % (rid, ep), server, tok)
                say("  [席%d] %s ok" % (i, ep))
            except ApiError as e:
                say("  [席%d] %s -> %s" % (i, ep, e))
    try:
        t = api("GET", "/api/tournaments/%s" % rid, server, tokens[0])
        say("房间状态：status=%s ready=%s registered=%s"
            % (t.get("status"), t.get("ready_users"), t.get("registered_users")))
    except ApiError:
        pass


def main():
    p = argparse.ArgumentParser(description="指定房间并行开打（scoped 参赛令牌）")
    p.add_argument("--tokens", default="tokens.txt", help="令牌文件，每行一个（默认 tokens.txt）")
    p.add_argument("--server", default=SERVER)
    p.add_argument("--ready", action="store_true", help="代提交 register+ready 开赛")
    p.add_argument("--trace", action="store_true", help="打印每次提交的动作")
    p.add_argument("--max-minutes", type=int, default=90)
    a = p.parse_args()

    say("=== 校验令牌 ===")
    tokens = load_tokens(a.tokens)
    room = preflight(a.server, tokens)
    say("目标房间：%s（%d 个席位）" % (room, len(tokens)))

    players = [Player(a.server, tok, room_id=room, trace=a.trace,
                      max_minutes=a.max_minutes, label="席%d" % (i + 1))
               for i, tok in enumerate(tokens)]
    say("=== 挂机等开赛（%d 线程）===" % len(players))
    for pl in players:
        pl.start()

    if a.ready:
        time.sleep(3)                 # 先让轮询跑起来：开赛时刻 ≤90s 内有请求才算在线
        say("=== 提交 register + ready ===")
        ready_all(a.server, tokens, room)

    t0 = time.time()
    for pl in players:
        pl.join()

    say("=== 全部结束，用时 %.0f 秒 ===" % (time.time() - t0))
    for pl in players:
        say("%s 完成 %d 场：" % (pl.label, len(pl.results)))
        for gid in sorted(pl.results):
            say("  %s: %s" % (gid, json.dumps(pl.results[gid], ensure_ascii=False)))


if __name__ == "__main__":
    main()
