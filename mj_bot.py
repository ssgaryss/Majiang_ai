#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对局引擎 + 可插拔策略 —— 让不同「策略人格」在同一桌对打。

与 my_bot.py 的区别：
  1. 决策委托给 Strategy 对象（可插拔），而不是写死在 GameWorker 里
  2. 修掉窗口重复提交：用 (round_no, phase, 总弃牌数, turn) 作窗口签名，同窗口只提交一次
     —— my_bot.py 每 1s 窗口会重复提交几十次 pass，每次吃 409 还白搭一次全量快照
  3. 完整手牌 = my_hand（轮到自己时它已是 14 张，已含刚摸的那张；
     drawn_tile 只是把这张再单独标一份作注记，不能再拼一次，否则会拼出 15 张）
"""
import json
import re
import ssl
import threading
import time
import urllib.request

import shanten_ref as S
from mj_core import (NTILE, WILD, best_discard, can_win, chi_tiles, is_baotou,
                     melds_of, tile_code, tile_index, to_counts, winning_tiles)
import my_bot
from my_bot import ApiError, SERVER, api, say

TERMINAL = ("finished", "closed", "void", "ended")
# 需要「每阶段重新确认出席」的状态（指南 §阶段确认）：
#   stage_open 时名单内提交 ready = 出席确认，幂等；**确认不跨阶段继承**。
#   漏确认不会被冻结，而是到点直接不入席 —— 静默出局，最难发现的那种失败。
STAGE_CONFIRM = ("stage_open",)


# ---------------------------------------------------------------- 决策输入

class View:
    """一次决策的全部输入（已解析成牌理层可直接用的形式）。"""

    def __init__(self, snap, seat):
        self.snap = snap
        self.seat = seat
        self.phase = snap.get("phase")
        self.turn = snap.get("turn")
        self.round_no = snap.get("round_no")
        self.wall = snap.get("wall_remaining")
        self.hand = list(snap.get("my_hand") or [])       # 当前「完整」手牌
        self.drawn = snap.get("drawn_tile") or ""         # 刚摸到的那张（仅注记）
        self.hand_counts = snap.get("hand_counts") or []
        # 关键：my_hand 在轮到自己时本就是 14 张，且已含刚摸的那张（hand_counts 同源印证）。
        # drawn_tile 只是把这一张再单独标一份。早先按「13 张立牌 + 单独摸牌」拼接，
        # 会拼出 15 张 —— can_win 的张数校验直接 False，于是永远不胡。
        self.full = list(self.hand)
        if self.drawn and self.drawn not in self.full:
            self.full.append(self.drawn)   # 防御：兼容「摸牌真正单独给」的协议形态
        self.counts = to_counts(self.full)
        self.meld_sets, self.meld_chi, self.meld_gang = melds_of(snap.get("melds"), seat)
        self.god = snap.get("god") or {}
        self.discards = snap.get("discards") or []
        self.last_discard = snap.get("last_discard") or ""
        self.scores = snap.get("scores") or []
        # 本局庄家座位（v19 起快照自带）。
        # 为什么策略需要它：庄家自摸一局值 24，闲家只值 10 —— 同一个"弃胡博爆头"
        # 动作在两处的期望**符号不同**：庄家 24 vs 0.75×48=36？不，是 24 vs 0.75×24*2/…
        # 精确算：庄家立即胡 = 24f；弃胡成功 = 24×2f = 48f，成功概率 q≈0.75
        #   -> 0.75×48 = 36 > 24（仍赚）但**方差大 4 倍**，而排名只看 total_score。
        # 真正的差别在**结构**上：庄家胡了继续坐庄（本轮实测 250/250 次"赢家坐庄"），
        # 而弃胡失败不但丢这一把，还把庄位让给对手。
        self.dealer = snap.get("dealer", -1)

    # --- 抓打圈 ---
    @property
    def catch_play(self):
        return bool(self.god.get("catch_play"))

    @property
    def exempt(self):
        """打财神者本人：不受抓打圈限制（可吃碰明杠、可任意出牌）。"""
        return self.god.get("god_discarder_seat") == self.seat

    @property
    def restricted(self):
        """受限 ⇔ 圈内且本人不是打财神者。"""
        return self.catch_play and not self.exempt

    # --- 番型状态 ---
    @property
    def baotou(self):
        return bool(self.god.get("baotou"))

    @property
    def chain(self):
        return int(self.god.get("chain_count") or 0)

    # --- 牌理 ---
    def can_win(self):
        """现在**能不能宣胡**。

        ⚠ 这里比"手牌是否成胡形"多一条**规则**限制（第 42 轮线上抓到）：

            服务端原话：`hu only after drawing a tile (meld is not self-draw)`
            即 **不能吃碰完直接胡** —— 必须是"自己摸上来的那一张"成的胡。

        证据（reject_dump.jsonl，带服务端 message）：
            动作 `hu`、本地 can_win=True、`drawn` 为空、副露 2~4 摊
            ⇒ 全部被拒，服务端明确说"meld is not self-draw"。

        为什么必须在这里修：`Evolved.on_turn` 判胡用的就是这个函数。
        不对齐的话每次副露之后都可能提交一次注定被拒的 `hu`，
        白烧一个请求 + 触发一次重试，而且**丢掉那一手的出牌机会**
        （要等 3 秒超时才由服务端代打）。

        注意"是否值得弃胡"要用 `shape_can_win`（只看牌形），
        因为那是"要不要放弃这一手"的判断，与"能不能立刻宣胡"是两件事。
        """
        return bool(self.drawn) and can_win(self.counts, self.meld_sets)

    def shape_can_win(self):
        """**只看牌形**是否成胡（不考虑"必须自摸"这条规则）。

        用途：弃胡通路（`decline`）要判断"我现在本来能胡、却选择弃胡"，
        那是牌形层面的判断；吃碰之后确实"成胡形但不能宣胡"，
        这种局面仍然值得按"能胡却弃胡"来分析。
        """
        return can_win(self.counts, self.meld_sets)

    def waits(self):
        return winning_tiles(self.counts, self.meld_sets)

    def allowed_discards(self):
        """抓打圈受限时只能打刚摸的那张。"""
        if self.restricted and self.drawn:
            return {tile_index(self.drawn)}
        return {t for t in range(NTILE) if self.counts[t] > 0}


# ---------------------------------------------------------------- 策略基类

class Strategy:
    """策略人格。子类覆写 on_turn / on_window。"""

    name = "base"
    desc = ""

    def on_turn(self, v):
        """轮到自己出牌（phase=draw 且 turn==seat）。返回动作 dict 或 None。"""
        raise NotImplementedError

    def on_window(self, v, phase):
        """碰/吃响应窗口。返回动作 dict 或 None。"""
        return {"action": "pass", "tile": ""}

    def __repr__(self):
        return "<%s>" % self.name


DISCARD_FIRST = "discard"


class Baseline(Strategy):
    """S0 对照：完全复刻官方模板 —— 打第一张、一律过、从不判胡。"""

    name = "S0-模板"
    desc = "打手牌第一张，碰吃全过，从不宣胡（复刻 my_bot.py 行为）"

    def on_turn(self, v):
        if v.restricted and v.drawn:
            return {"action": "discard", "tile": v.drawn}
        if not v.hand:
            return None
        return {"action": "discard", "tile": v.hand[0]}


class Efficiency(Strategy):
    """S1 效率：会宣胡 + 按牌效率出牌（留对子/搭子，先打孤张）。"""

    name = "S1-效率"
    desc = "能胡就胡；出牌用连接度启发式（孤张字牌优先打）"

    def on_turn(self, v):
        if v.can_win():
            return {"action": "hu", "tile": ""}
        allow = v.allowed_discards()
        t, code = best_discard(v.counts, v.meld_sets,
                               forbid=set(range(NTILE)) - allow)
        if code is None:
            return None
        return {"action": "discard", "tile": code}


class Meld(Efficiency):
    """S2 副露：在 S1 基础上接受碰 —— 手里有成对就碰，加快成牌。"""

    name = "S2-副露"
    desc = "S1 + 有对子就碰（抓打圈受限时除外）"

    def on_window(self, v, phase):
        if phase != "response_peng" or v.restricted or not v.last_discard:
            return {"action": "pass", "tile": ""}
        t = tile_index(v.last_discard)
        if t == WILD:
            return {"action": "pass", "tile": ""}      # 财神不能被碰
        if v.counts[t] >= 2:
            return {"action": "peng", "tile": v.last_discard}
        return {"action": "pass", "tile": ""}


class Gambler(Efficiency):
    """S3 财飘：爆头态手握财神时弃胡打白博财飘（番数 ×2 递增，但可能流局归零）。"""

    name = "S3-财飘"
    desc = "S1 + 爆头且有财神时弃胡打白拼财飘链"

    def on_turn(self, v):
        if v.can_win():
            if v.baotou and v.counts[WILD] >= 1 and v.chain < 3:
                if not v.restricted:
                    return {"action": "discard", "tile": "白"}
            return {"action": "hu", "tile": ""}
        return super().on_turn(v)


ALL_STRATEGIES = [Baseline, Efficiency, Meld, Gambler]


def build(name):
    for cls in ALL_STRATEGIES:
        if cls.name == name:
            return cls()
    raise KeyError(name)


# ---------------------------------------------------------------- 对局执行

class Notifier(threading.Thread):
    """SSE 通知流 —— 只用来即时唤醒，帧里不含牌面。不占 /state 的 16 次/秒额度。

    为什么必须要有它：
      现状是每批事件花 **2 个请求**（长轮询拿事件 + 再拉一次全量快照）。
      10 桌 × 约 1.3 事件/秒 × 2 = **26 请求/秒**，而服务器上限是 16/秒 ——
      **需求本身就超 62%**，令牌桶怎么调都会积压，而碰吃窗口「固定走满 1s」，
      积压的代价就是错过窗口。线上实测 1344 次 timeout vs 365 次 pass，
      **79% 的响应窗口没接住**，副露频率只有模拟器的四分之一。

    改用 SSE 后：唤醒靠推送（不计额度），每次状态变化只花 1 个请求拉快照，
      需求降到约 13/秒，回到 16/秒 的预算之内。
    """

    def __init__(self, server, token, gid):
        super().__init__(daemon=True)
        self.url = server.rstrip("/") + "/api/games/%s/notify" % gid
        self.token = token
        self.wake = threading.Event()
        self.alive = False
        self.frames = 0
        self.last_seq = None
        self._stop = False
        # 初次唤醒：消费端是「先等推送、再取快照」，不预置这一下就会干等到
        # 第一个事件才动 —— 开局那几手（含庄家第一打）全部错过。
        self.wake.set()

    def stop(self):
        self._stop = True
        self.wake.set()

    def run(self):
        ctx = ssl._create_unverified_context()
        while not self._stop:
            try:
                req = urllib.request.Request(self.url, headers={
                    "Authorization": "Bearer " + self.token,
                    "Accept": "text/event-stream"})
                with urllib.request.urlopen(req, timeout=90, context=ctx) as r:
                    self.alive = True
                    for raw in r:
                        if self._stop:
                            return
                        line = raw.decode("utf-8", "replace").strip()
                        if not line or line.startswith(":"):   # keepalive
                            continue
                        if line.startswith("data:"):
                            try:
                                self.last_seq = json.loads(
                                    line[5:].strip()).get("seq")
                            except ValueError:
                                pass
                            self.frames += 1
                            self.wake.set()
            except Exception:                                  # noqa: BLE001
                self.alive = False
                if self._stop:
                    return
                time.sleep(1.0)                                # 断线重连


class RateLimiter:
    """跨该玩家全部对局的共享令牌桶，**会自适应**。

    平台限速 /state 为每用户 16 次/秒（跨全部对局共享一个桶）。M=10 桌并发时，
    「收到事件 -> 再拉全量快照」会让每批事件花掉 2 个请求，峰值轻易破墙并让
    长轮询退避重试耗尽（整场直接放弃）。这里按低于上限的速率统一发号。

    自适应（2026-09-16 加）：线上实测长期跑在 15.4~16.7 请求/秒，**贴着 16 的天花板**。
    而撞一次 429 要退避重试最多 36 秒，碰/吃窗口只有 1 秒 —— 一次 429 就连丢几个窗口。
    所以：
      · 撞到 429 -> throttle()：发号速率打 85 折（下限 floor），立刻让出余量
      · 长期没撞到 -> relax()：每次 +0.05/秒 缓慢爬回 base
    这样正常情况下不损失吞吐，一旦被限速又能自己退回来，不需要人工调参。
    """

    def __init__(self, rate=8.0, floor=6.0):
        self.base = float(rate)
        self.rate = float(rate)
        self.floor = float(floor)
        self.lock = threading.Lock()
        self.next_free = 0.0
        self.throttled = 0          # 累计撞到的 429 次数（供哨兵展示）

    def wait(self):
        with self.lock:
            now = time.time()
            delay = self.next_free - now
            self.next_free = max(now, self.next_free) + 1.0 / self.rate
        if delay > 0:
            time.sleep(delay)

    def throttle(self):
        """撞到 429：降速让出余量。"""
        with self.lock:
            self.throttled += 1
            self.rate = max(self.floor, self.rate * 0.85)
            now = time.time()
            if self.next_free < now:
                self.next_free = now

    def relax(self):
        """一段时间没撞到 429：缓慢恢复到 base。"""
        with self.lock:
            if self.rate < self.base:
                self.rate = min(self.base, self.rate + 0.05)
        return self.rate


def build_missed_record(snap, seat, act, full):
    """构造一条「可胡未胡（弃胡）」埋点记录。

    ★ 第 404 轮从 `GameRunner.run` 里**抽出来**，唯一目的是**可测**：
      原来这段是内联的，自测只能验"快照有没有那三个键"，
      **验不到"真实记录会不会带上它们"** —— 而那正是下游分析依赖的东西。
    抽成函数后可以喂一份**真实快照**直接断言整条记录。

    ⚠ 本函数**只读**入参，不做任何判定 ⇒ 抽出来不改变线上行为（逐字等价）。
    """
    try:
        _melds = snap.get("melds") or []
        opp_melds = [len(r or []) for r in _melds]
    except Exception:                                                # noqa: BLE001
        opp_melds = []
    try:
        opp_hand_n = [int(x) for x in (snap.get("hand_counts") or [])]
    except Exception:                                                # noqa: BLE001
        opp_hand_n = []
    try:
        scores = [float(x) for x in (snap.get("scores") or [])]
    except Exception:                                                # noqa: BLE001
        scores = []
    dealer = snap.get("dealer")
    return {"round_no": snap.get("round_no"),
            "action": (act or {}).get("action"),
            "tile": (act or {}).get("tile"),
            "hand": list(full),
            "meld_sets": snap.get("meld_sets"),
            "dealer": dealer,
            "wall": snap.get("wall"),
            "is_dealer": (dealer == seat),
            # ★ 第 403 轮新增（goal §2.7 用）：对手进度 + 局面分数
            "seat": seat,
            "opp_melds": opp_melds,
            "opp_hand_n": opp_hand_n,
            "scores": scores}


class GameRunner(threading.Thread):
    """单场对局循环：长轮询 -> 快照 -> 策略决策 -> 提交。"""

    def __init__(self, player, gid, strategy):
        super().__init__(daemon=True)
        self.player = player
        self.gid = gid
        self.strategy = strategy
        self.actions = 0
        # 场次已消失/已结束 —— 供 Player 永久拉黑，避免反复重启同一个已死的场。
        self.gone = False
        # 意外异常的重试计数（见 run 的 except 分支）。
        #
        # 为什么必须有这个上限：worker 抛异常时既没设 gone、也没写 results，
        # 于是主循环的回收逻辑**只会把它从 _workers 摘掉**，下一轮轮询又发现
        # 这局"还在活跃"→ 重新构造 GameRunner → 再崩 → 无限循环。
        # 后果：一个稳定复现的 bug 会**吃满请求额度**，把 10 桌正常对局的
        # 取快照份额全部挤掉（额度是全局 16/秒）。线上 08:17 那次
        # `TypeError` 就是这个形态，只是当时还没到打满额度的程度。
        # 所以：**允许重试一次**（瞬时错误确实能救回来），第二次仍异常就放弃该局，
        # 并写一条结果留痕，避免静默丢失。
        self.attempts = 0
        # 因"同一状态反复被拒"而放弃动作、改为出牌的次数（见 409 分支）。
        # 这是**可见的损失**：每一次都代表一手本可以胡的牌被推掉了，
        # 所以要能在日志与哨兵里看到，而不是静默发生。
        self.giveup_actions = 0
        # ★ 2026-10-03（第 950 轮 / `GOAL_STATUS.md §Z.1326`）：把 `行动` **拆成两半** ✓
        #   因为 `行动` 是**混装**的 ✗：`:763`（响应窗内主动过）与 `:854`（成功提交）**都**自增它 ✓
        #   · `self.submits`：成功提交的动作数（`:854` ✓）
        #   · `self.passes` ：响应窗口内**主动过**的次数（`:763` ✓）
        #   ⇒ ★ 恒等式应为 `行动 == submits + passes`；**若不等** ⇒ `行动` 还有别的自增点 ✗
        #     —— 这正是本次插桩要**当场暴露**的东西 ✓（此前我误以为"行动 = 成功提交数" ✗）
        self.submits = 0
        self.passes = 0
        # 端到端延迟（毫秒）：从"拿到快照"到"提交完成"。
        # 窗口只有 1 秒（碰/吃）与 3 秒（出牌），所以这个分布的**尾部**
        # 直接决定会不会被 409 拒绝（≈丢一手胡牌）。
        self.lat = []
        # ★ 超窗提交计数（2026-09-17 新增，补上 §23 指出的**线上可观测性缺口**）。
        #
        # 为什么必须当场记：正式赛的逐局执行质量**无法事后审计**
        #   `GET /api/games/{gid}/state` -> 403 not a participant
        #   `GET /portal/api/games/{gid}/events` -> 401
        #   `GET /api/test-rooms/{tournament_room}/games` -> 0 局
        # 而 §14.35 已证明「客户端自报 `窗口错过=0 / 409=0`」**不等于**
        # "没有被服务端代打"（bug 窗口里客户端也报 0，服务端却有 1.535 次/局）。
        # 所以必须在提交处按**窗口长度**分别统计超窗次数，写进终局汇总行：
        #   · 出牌窗口 `DiscardTimeoutSec=3`  ⇒ 超过 3000 ms 就可能已被代打
        #   · 碰/吃窗口 `Peng/ChiTimeoutSec=1` ⇒ 超过 1000 ms 就可能已错过
        self.late_discard = 0          # 出牌提交 > 3000 ms 的次数（可能被服务端代打）
        self.late_window = 0           # 碰/吃提交 > 1000 ms 的次数（可能已错过窗口）
        self.late_ms = []              # 超窗那几次的耗时，供事后看量级
        # 分段耗时（毫秒）：(SSE唤醒→可取快照, gap睡眠, 令牌排队, GET往返)。
        # 用途见 run() 里"分段计时"那段说明：端到端 p99 1479 ms 而本地计算只要
        # 17 ms，长尾必然来自等待，而等待的三种来源要分开才能定位。
        self.segs = []
        # 因"快照陈旧（出牌阶段但 drawn 为空）"而跳过动作的次数。
        # 每一次被跳过都意味着**网络把快照拖过期了**，而在旧代码里
        # 这些会变成一次必然被拒的提交（409 + 白烧额度）。
        # 它同时是"网络到底有多糟"的可见指标。
        self.stale_skips = 0
        # ★ 2026-10-09：被**网络抖动**（超时/断连）打断、但靠"重取状态后继续"
        #   救回来的次数。它们此前会**直接结束整场**（外层兜底 except）——
        #   实测两场只打了 1~3 手就"结束"，而其实只是网络抖了一下。
        #   `submit_timeouts` 是其中发生在**动作提交**上的子集：那一类必须
        #   先重取状态确认动作有没有生效，绝不能盲目重发。
        self.net_errors = 0
        self.submit_timeouts = 0
        # 被拒时的延迟明细（含当时的 drawn 与本地判胡结果）。
        self.reject_lat = []
        self._conflict_sig = None    # 见 409 分支：同一状态只解禁一次重试
        # 响应窗口「错过率」——本轮 SSE 改造唯一要验证的线上指标。
        # 平台在窗口关掉后会记一次 timeout 事件；对应的本地信号是「我提交窗口动作
        # 时吃 409（动作已失效）」。改造前实测 1344 timeout vs 365 pass（79% 没接住）。
        self.conflicts = 0           # 全部 409
        self.missed = 0              # 其中发生在碰/吃响应窗口内的
        # 杠：2026-09-17 起策略会走补杠（碰出的面子 + 手上第 4 张）。
        # 这是**新动作通路**，必须能在日志里看到"提交了几次、被拒了几次" ——
        # 否则一个被服务端一直拒绝的 gang 会伪装成"策略没启用"，
        # 而这正是本项目最贵的一类错误（静默失真）。
        self.gangs = 0
        self.gang_conflicts = 0
        # 低方差中间指标：每次出牌后的向听数。
        # 自摸率是二项量（p≈0.25，SD≈0.43），要分辨 2~3 个百分点需要约 2300 局；
        # 而「平均打出后向听数」是连续量（每局每人约 17 次出牌取均值），
        # 单局标准差小一个量级，同样局数下分辨率高得多，且直接度量出牌质量。
        self.sh_sum = 0.0
        self.sh_n = 0
        # 逐局得失（四家），由累计分变化得到 —— 见 _acc_round 的说明。
        self.round_deltas = []
        # 每一笔得失对应的**局号**，与 round_deltas 一一对应。
        #
        # 为什么必须补这个字段：线上一个房间里有 10 个客户端，各自记录**同一批局**的
        # 四方得失（房间 a_9f831e3531ea 实测 10 个批次）。没有局号，事后就没法把
        # "同一局被记了 10 遍"折叠掉 —— 局数被放大 10 倍，任何按局算的置信区间
        # 都会假性变窄（标准误被低估约 √10 ≈ 3 倍）。有了局号才能按 (房间, 局号)
        # 去重，得到真正独立的观测。
        self.round_nos = []
        # 与 `round_deltas` 一一对应的**我方副露数** —— 让「不划算的吃碰」可测，
        # 见 `_acc_round` 的说明。（历史上没有这个字段 ⇒ 那一类错误只能靠印象。）
        self.round_melds = []
        # 可胡未胡的**结构化**明细（每局弃胡次数是低方差机制指标）。
        #
        # ⚠ 字段名必须与上面的 `self.missed`（那是**计数**：碰/吃响应窗口错过的次数）
        #   区分开。第一版两者都叫 `self.missed`，直接把 int 覆盖成 list，
        #   后果链：`self.missed += 1` 抛 TypeError（那正是线上反复出现的
        #   `TypeError("'int' object is not iterable")`，白白毁掉整场对局）
        #   → 哨兵线程 `sum(w.missed ...)` 抛异常而死（日志再也不更新）
        #   → `%d` 格式化 list 抛异常（每场结束的汇总行消失）。
        #   一个字段名撞车，同时打掉了对局、监控和日志三样东西。
        self.missed_detail = []
        self._last_scores = []
        self._base_scores = []       # 首帧累计分（中途接管对局时不为 0）

    def _w(self, msg):
        if self.player.trace:
            say("[%s|%s] %s" % (self.gid[-6:], self.strategy.name, msg))

    def _acc_round(self, sc, round_no=None, melds=None):
        """累计分发生变化 = 刚打完一局，把四家得失记一笔（首帧只做基线）。

        `melds` = 该局结束时的**我方副露数**。为什么要跟着记：
        「不划算的吃碰」这一类高代价错误**目前完全不可测**
        （`tools/analyze_errors.py` 表 C 有完整论证）—— 因为每场记录里
        既没有我方副露数、也没有动作序列，而 `opp_melds` 记的是**对手**的。
        补这一个字段是让它可测的**最小改动**：有了它就能按"副露数分档"看
        逐局得分（0/1/2 摊各档的局均），这是判断"吃碰到底亏不亏"的第一步。
        ⚠ 它只是**分档相关**，不是因果 —— 要吃碰的往往是好牌，混杂很重。
          真正的因果要靠同 seed 配对（`probe_posture_frontier.py` 那条路）。

        同时把**首帧的累计分**留成 `deltas_base`：房中途重启客户端时，接管的
        是进行中的对局，第一条快照的分数已经不是 0 —— 没有这个基线，
        "逐局之和 == 最终分" 这条自校验就会把正常数据误判成坏数据。

        `round_no` 一并记下（见 round_nos 字段的说明）：跨客户端去重只能靠它。
        两局交界处快照可能"分数已变、round_no 未推进"，所以判据仍然是分数变化，
        round_no 只作为**标注**，不参与判据 —— 这一点不能反过来。
        """
        if len(sc) < 4:
            return
        if not self._last_scores:
            self._last_scores = list(sc)
            self._base_scores = list(sc)
            return
        if list(sc) == self._last_scores:
            return
        self.round_deltas.append([sc[i] - self._last_scores[i] for i in range(4)])
        self.round_nos.append(round_no)
        self.round_melds.append(melds)
        self._last_scores = list(sc)

    def run(self):
        seq = 0
        responded = None
        nf = None
        last_fetch = 0.0
        snap = None            # 异常落盘要用；必须在 try 之前初始化（否则首轮异常会 NameError）
        last_turn_mine = False  # 上一次快照里"是否轮到我"（用于跳过 gap，见下）
        if self.player.use_notify:
            nf = Notifier(self.player.server, self.player.token, self.gid)
            nf.start()
        try:
            while not self.player._stop.is_set():
                # 有 SSE 时：先等一次「状态已变化」的推送再取快照。
                # seq=0 是立即返回的（不做长轮询），不加这道等待就会打成紧密空转。
                # SSE 活着给长超时（兜底轮询几乎不耗额度）；断了就退化为短间隔轮询。
                if nf is not None:
                    nf.wake.wait(timeout=5.0 if nf.alive else 0.2)
                    nf.wake.clear()
                    _t_sse = time.time()
                    # 每场最小取快照间隔 —— 这是 10 桌并发下最关键的一道闸。
                    #
                    # 线上实测（a_0164559ee5fa，M=10）：窗口错过 83 次 vs 行动 81 次，
                    # 约 50% 的碰/吃窗口没接住。原因是 SSE 帧**只带 seq、不带内容**，
                    # 客户端无法判断这一帧与自己有没有关系，只能每帧取一次快照；
                    # 10 桌的事件率远超 16/秒 的预算，于是变成「谁被唤醒得勤谁抢到额度」，
                    # 令牌桶队列一深，窗口动作到达时 1 秒窗口已经关了（吃 409）。
                    #
                    # 按「公平份额」限速后，每桌的取快照节奏有上界，队列不再堆积，
                    # 单桌延迟可预测；代价是丢弃一部分与自己无关的中间态（本来也没用）。
                    #
                    # ★ 但"公平份额"本身会**超过窗口长度** —— 线上实测（2026-09-17）：
                    #   10 桌时 gap = 10/(14×0.85) ≈ 0.84 秒，而碰/吃窗口只有 **1 秒**。
                    #   端到端延迟分布因此长成：中位 34 ms，但 **p99 = 1493 ms、
                    #   最大 5998 ms**（8 场统计）。p99 远超窗口 ⇒ 那一巡必然接不住；
                    #   出牌窗口 3 秒也被 5998 ms 突破 ⇒ 服务端替我们出了牌，
                    #   而我们还拿着旧快照去提交 `hu`，于是 `409 INVALID_ACTION`
                    #   （reject_dump：6/6 都是 local_can_win=True 且 drawn 为空 ——
                    #   空 drawn 就是"这一巡早过去了"的铁证）。
                    #   修法：给 gap 加**上限**（默认 0.5 秒）。上限的意思是
                    #   "宁可多花一点额度，也不要让反应时间超过窗口"。
                    #
                    # ★★ 第 42 轮的进一步修法：**轮到自己时完全不等 gap**。
                    #   理由：gap 是用来限"与自己无关的快照"的；一旦确认
                    #   `turn == 我` 且处在出牌阶段，我们手里就有一件必须在
                    #   3 秒内提交的事。这时再睡 0~0.5 秒（还要叠加 GET 往返，
                    #   实测最慢 5% 是 764~2509 ms）就是在给服务端制造
                    #   "超时代打 + 我们提交被 409 拒绝"的机会。
                    #   `on_my_turn` 是**上一次快照**的判断：它成立时说明
                    #   我们可能还在同一巡里（例如刚吃碰完要继续出牌），
                    #   这时立刻取快照才是对的。
                    gap = 0.0 if (last_turn_mine and self.player.fetch_gap) \
                        else self.player.fetch_gap
                    if gap > 0:
                        dt = time.time() - last_fetch
                        if dt < gap:
                            time.sleep(gap - dt)
                    _t_gap = time.time()
                else:
                    _t_sse = _t_gap = time.time()
                self.player.limiter.wait()
                # ★ 分段计时：把"从 SSE 唤醒到提交完成"拆成**六段**，才能定位长尾。
                #
                # 为什么要拆：端到端延迟的 p99 是 1479 ms、最大 5998 ms，
                # 而**本地纯计算只要 17 ms**（probe_latency.py，p99 72 ms）。
                # 所以长尾一定在等待上，而等待有三处完全不同的来源：
                #   ① SSE 唤醒后的 gap 睡眠（人为限速，可调）
                #   ② limiter.wait()（全局 16/秒 额度排队，10 桌共享）
                #   ③ GET / POST 的网络往返
                # 不拆开就只能猜；拆开之后一眼能看出该动哪一个。
                #
                # ★★ 2026-10-09：补上后两段（**决策**与**提交**）。
                #   原实现只到 GET 为止 ⇒ 线上日志里 `rl_vA3` 与 `rl_aux_0004`
                #   的网络同构、延迟中位却差 **8.7 倍**（73 ms vs 633 ms），
                #   而当时的读数**无法区分**"模型推理慢"与"等待慢"。
                #   段序定义见 `_seg_report`（那里也有这条顺序的测试）。
                _t_lim = time.time()
                # 用 SSE 时每次直接拿全量快照（seq=0 → 立即返回，不做长轮询），
                # 一批事件只花 1 个请求；唤醒交给 Notifier，不再靠高频轮询。
                q = "/api/games/%s/state?seq=%d" % (self.gid, 0 if nf else seq)
                # ★ 端到端延迟测量的起点。
                #
                # 为什么必须测这个：线上抓到 4 次 `409 INVALID_ACTION`（动作都是 `hu`），
                # 而 reject_dump 显示 4/4 都是「本地 can_win=True、手牌确实是 14 张成胡、
                # 但快照的 `drawn` 字段是空的」—— 空 drawn 说明**这一巡早就过去了**，
                # 服务端已经按 DiscardTimeoutSec=3 自动替我们出过牌。
                # 也就是 409 不是判错，而是**我们的快照/提交慢了一步**。
                # 本地纯计算只要 17ms（p99 72ms，见 probe_latency.py），
                # 所以慢的一定是这一整条链：SSE 唤醒 → 等限速令牌 → GET → 决策 → POST。
                t_wake = time.time()
                try:
                    # ★★★ 2026-09-29：快照 GET 用**短超时 + 不重试**。
                    #   理由：`api()` 默认 `timeout=35` × `tries=6` ⇒ 最坏阻塞 ~258 秒 ✗，
                    #   而出牌窗口只有 3 秒、碰/吃窗口只有 1 秒 ⇒ 一旦重试就必然错过窗口 ✗
                    #   （线上实测：`放弃动作` **722** 次，**100% 是「丢了胡牌」**）。
                    #   快照的"新鲜度"只在窗口内有用 ⇒ **超窗后拿到也没用** ⇒
                    #   应当【立刻放弃、交给下一轮】✓ 而不是死等 ✗
                    res = api("GET", q, self.player.server, self.player.token,
                              timeout=1.5, tries=1)
                except ApiError as e:
                    if e.status == 404:
                        self.gone = True
                        return
                    raise
                except OSError as e:
                    # ★★ 2026-10-09：**快照 GET 超时不再杀死整场** ★★
                    #   实测（`temp/live_check.log`）traceback 全是这一行：
                    #     `api("GET", q, ..., timeout=1.5, tries=1)` -> TimeoutError
                    #   它穿透到最外层兜底 `except Exception` ⇒ 整个 GameRunner 结束
                    #   ⇒ 服务端只好代打，日志里表现为"这一场只打了 1~3 手就结束"。
                    #   而快照本来就是**可以重取**的 ⇒ 丢掉这一轮、下一轮再取即可。
                    self.net_errors += 1
                    if self.net_errors > _MAX_NET_RECOVER:
                        say("[%s] 场 %s **网络连续异常 %d 次**（%s）→ 结束本场"
                            % (self.player.label, self.gid, self.net_errors,
                               type(e).__name__))
                        return
                    say("[%s] 场 %s 快照超时（%s）→ 丢弃本轮，下一轮重取"
                        % (self.player.label, self.gid, type(e).__name__))
                    continue
                last_fetch = time.time()
                t_snap = last_fetch
                # 记录这一段的分段耗时（毫秒）：gap 睡眠 / 令牌排队 / GET 往返。
                # 只在真正有动作要提交时才用得上，但先记下来无妨。
                seg = [(t_snap - _t_sse) * 1000.0,        # SSE 唤醒 → 可以取快照
                       (_t_lim - _t_gap) * 1000.0,         # gap 睡眠（人为限速，可调）
                       (t_wake - _t_lim) * 1000.0,         # 令牌排队（全局额度，多桌共享）
                       (t_snap - t_wake) * 1000.0,         # GET 往返（网络 / 服务端）
                       0.0,                                # 决策：特征计算 + 推理
                       0.0]                                # 提交 POST 往返
                if res is None:
                    continue
                if res.get("pending"):
                    continue
                if res.get("finished"):
                    snap = res.get("snapshot") or {}
                    sc = snap.get("scores") or []
                    self._acc_round(sc, snap.get("round_no"))   # 终局那一笔也走同一套判据
                    self.player.results[self.gid] = {
                        "seat": snap.get("seat", -1),
                        "scores": sc,
                        "rounds": snap.get("round_no"),
                        "actions": self.actions,
                        "strategy": self.strategy.name,
                        "avg_shanten": (self.sh_sum / self.sh_n) if self.sh_n else None,
                        "discards": self.sh_n,
                        "round_deltas": self.round_deltas,
                        # 与 round_deltas 一一对应的局号 —— 线上同房间有 10 个客户端
                        # 各记同一批局，只有带上局号才能跨客户端去重（见字段说明）。
                        "round_nos": self.round_nos,
                        # 与我方副露数一一对应 ⇒ 「不划算的吃碰」的唯一测点
                        "my_meld_sets": self.round_melds,
                        # ★ 可胡未胡明细：事件流不含起手，弃胡**无法**事后重建，
                        #   只能在客户端当场判。这是"每局弃胡 x 次"的唯一来源。
                        "missed_detail": self.missed_detail,
                        "deltas_base": self._base_scores,
                    }
                    say("[%s] %s 场 %s 结束 scores=%s"
                        % (self.player.label, self.strategy.name, self.gid[-6:],
                           snap.get("scores")))
                    return

                snap = res.get("snapshot")
                if snap is None:                      # 增量事件：推进 seq 后重建权威快照
                    for ev in res.get("events") or []:
                        seq = ev.get("seq", seq)
                    self.player.limiter.wait()
                    # ★ 必须走 `_fetch_full_snapshot`（短超时 + 不重试）。
                    #   内联 `api(...)` 会用默认 timeout=35/tries=6 ⇒ 最坏阻塞 258 秒，
                    #   而 1 秒的碰/吃窗口里这等于必然错过。见该函数的 docstring。
                    full = _fetch_full_snapshot(self.player.server, self.player.token,
                                                self.gid)
                    snap = full.get("snapshot") or {}
                    seq = full.get("seq", seq)
                else:
                    seq = res.get("seq", seq)

                seat = snap.get("seat", -1)
                if seat < 0:                          # 观赛视角
                    continue
                phase = snap.get("phase")
                # 记录"这一帧是否轮到我" —— 下一轮唤醒时用它决定要不要跳过 gap。
                last_turn_mine = (phase == "draw" and snap.get("turn") == seat)

                # 逐局得分记账：快照里只有**累计分**，而四家得分只在**一局结束时**
                # 发生变化 —— 所以"分数变了"本身就是一个干净的判据，比盯 round_no
                # 更稳（后者在两局交界处可能先更新分数、后推进 round_no，那一局的
                # 差额就会被下一次快照吞掉）。
                #
                # 为什么值得记：线上比分的方差极大（单房 ±32 分），而"自摸率""赢下来
                # 那一局值多少分"这些**机制指标**的方差小一个量级 —— 线上 A/B 拿它们
                # 当因变量，检验力比看总分高得多。此前这些量只能靠回放接口事后重算
                # （要额外发请求、还要处理分页），现在客户端自己就留下了。
                self._acc_round(snap.get("scores") or [], snap.get("round_no"),
                                snap.get("meld_sets"))

                # 状态签名：同一状态下只提交一次动作。见 _state_sig 的说明。
                sig = _state_sig(snap, phase)

                _t_dec = time.time()      # ★ 分段计时：特征计算 + 推理
                v = View(snap, seat)
                missed = None
                if phase == "draw" and snap.get("turn") == seat:
                    if sig == responded:
                        continue
                    act = self.strategy.on_turn(v)
                    # 埋点：本地判胡为真却没选 hu。用于区分「漏胡是策略有意弃胡(S0)」
                    # 还是「引擎/决策链 bug」，并留下手牌供离线复核。
                    #
                    # ★ 这是**唯一**能可靠度量"弃胡"的口径：事件流不含起手，
                    #   事后无法从回放重建手牌，所以弃胡次数只能在客户端当场判。
                    #   它同时是"每局弃胡 x 次"这个机制指标的来源 —— 线上 A/B
                    #   用低方差机制指标当因变量，比看场分灵敏一个量级。
                    if v.can_win() and (act or {}).get("action") != "hu":
                        # ★ 第 403/404 轮：记录构造已抽成 `build_missed_record`
                        #   （同文件模块级），这里只是调用 —— **行为逐字不变**，
                        #   抽出来的唯一目的是让自测能**端到端**验整条记录。
                        missed = build_missed_record(snap, seat, act, v.full)
                        self.player.missed_hu.append(
                            (self.gid, seat, list(v.full), (act or {}).get("action")))
                        say("[%s|%s] 可胡未胡 seat%d 副露=%d 庄=%s hand=%s -> %s"
                            % (self.player.label, self.strategy.name, seat,
                               v.meld_sets, "我" if v.dealer == seat else "他",
                               " ".join(v.full), act))
                    if missed is not None:
                        self.missed_detail.append(missed)
                elif phase in ("response_peng", "response_chi"):
                    if seat not in (snap.get("responding_seats") or []):
                        continue
                    if sig == responded:
                        continue
                    act = self.strategy.on_window(v, phase)
                else:
                    continue
                # ★ 决策段（特征计算 + 推理）—— 目标第 3 条要求的分段之一。
                #   为什么必须单独量：`rl_vA3` 与 `rl_aux_0004` 的网络同构，
                #   但历史日志里延迟中位差 **8.7 倍**（73 ms vs 633 ms）。
                #   没有这一段就只能猜是"模型慢"还是"等待慢"。
                seg[4] = (time.time() - _t_dec) * 1000.0
                if act is None:
                    continue

                responded = sig
                self._w("seat%d %s -> %s" % (seat, phase, act))
                # ★ 陈旧快照守卫 —— 这是 409 的**直接**成因，也从根上避免丢胡牌。
                #
                # 证据链（2026-09-17）：
                #   1. 线上反复抓到 `409 INVALID_ACTION`，动作都是 `hu`；
                #   2. `reject_dump.jsonl` 里 **6/6** 都是「本地 can_win=True、
                #      手牌确实是 14 张成胡、但快照的 `drawn` 字段是空的」；
                #   3. 分段计时定位到长尾全部来自 **GET 往返：最慢 5% 平均 1052 ms**
                #      （gap 睡眠 157 ms、令牌排队 0 ms）——
                #      也就是网络/服务端偶发慢，把快照拖成了过期数据。
                #
                # ⚠⚠ **但"drawn 为空"本身不是陈旧的充分证据**（第 42 轮发现）：
                #   **吃/碰之后是"不摸牌直接出牌"** —— 那一巡 `drawn_tile`
                #   本来就该是空的！按原判据会把**每一次副露之后的那一手全部跳过**，
                #   等 3 秒超时由服务端代打（等于随机弃牌）。
                #   这是"守卫比它要防的 bug 更伤"的典型：它挡掉的是真实行动。
                #
                #   正确判据用**不变量**：轮到自己行动时手牌必然是
                #       `14 − 3×副露数` 张（摸完的状态；吃碰后不摸牌也仍是这个数）。
                #   所以：
                #     · drawn 为空 **且** 手牌张数对得上 ⇒ 这是吃碰后的正常一巡，**要出手**；
                #     · drawn 为空 **且** 手牌张数少 1（只有 13−3m）⇒ 确实缺了那张摸牌，
                #       说明快照是旧的（服务端已自动代打），**跳过**。
                need = 14 - 3 * v.meld_sets
                have = sum(v.counts)
                # ⚠ 抓打圈（`catch_play`）时我们被限制为"只能打刚摸的那张"，
                #   这时若 `drawn` 为空，我们**根本无法出手**（没有可打的牌），
                #   跳过是唯一选择 —— 不必也**不应该**去比较手牌张数
                #   （那种快照的形状本来就不常规）。显式排除掉，避免误判。
                stale = (phase == "draw" and snap.get("turn") == seat
                         and not v.catch_play
                         and not (snap.get("drawn_tile") or "")
                         and have < need)
                if stale:
                    self.stale_skips += 1
                    # 记下手牌张数：这个日志是判据是否正确的**现场证据**
                    # （6.6 次/局那种量级如果再次出现，一眼就能看出是误判）。
                    self._w("快照陈旧（drawn 为空且手牌 %d<%d 张，副露%d），跳过"
                            % (have, need, v.meld_sets))
                    continue
                # ★ PASS 不必提交 —— 这是本轮线上日志抓到的第二个 bug。
                #
                # 证据：traceback 显示 409 `INVALID_ACTION: cannot pass in phase 1`。
                # 原因：碰/吃窗口只有 1 秒，而我们要等 SSE、取快照、再提交；
                # 提交到达时窗口往往已经关闭，服务器把 `pass` 当成了**出牌阶段**
                # 的动作 —— 而出牌阶段不允许 pass，于是 409。
                # 后果有两层：
                #   ① 这一场被记一次 409，`self.missed` 计数虚高（污染"窗口错过"指标）；
                #   ② 更糟的是同一状态下会**解禁重试一次**（`responded = None`），
                #      白白多烧一个请求额度，而 10 桌并发下额度就是最稀缺的资源。
                # 语义上"过"与"不响应"完全等价（服务器到点自动过），
                # 所以**根本不提交**才是正确做法：省请求、零 409、语义不变。
                if (act.get("action") == "pass"
                        and phase in ("response_peng", "response_chi")):
                    self.actions += 1            # 记为一次已决策（供哨兵显示）
                    self.passes += 1             # ★ 其中"响应窗内主动过"单独计数（§Z.1326）
                    continue
                is_gang = (act.get("action") == "gang")
                if is_gang:
                    self.gangs += 1
                if act.get("action") == "discard" and act.get("tile"):
                    try:
                        c2 = list(v.counts)
                        c2[tile_index(act["tile"])] -= 1
                        self.sh_sum += S.shanten(c2, v.meld_sets)
                        self.sh_n += 1
                    except (ValueError, KeyError, IndexError):
                        pass
                # ★★ 第 457 轮：修一个**线上丢掉 14~17% 动作**的真 bug。
                #   证据（修好可观测性之后第一眼就抓到了，服务端原话）：
                #     phase=response_chi seat=2 act=chi
                #     err={"code":"INVALID_ACTION","message":"**chi tile mismatch**"}  ms=42.2
                #     phase=response_chi seat=1 act=chi
                #     err={"code":"INVALID_ACTION","message":"**chi tile mismatch**"}  ms=17.6
                #   ⇒ **不是超时**（ms 只有 17~42），而是"吃的牌对不上"。
                #   根因：**碰是唯一的**（弃牌 + 手上两张），而**吃最多有三种组合**，
                #     可策略返回的窗口动作里**只有 `{"action":"chi"}`、不带牌** ——
                #     `rl_action_space` 为了让"吃哪两张"不成为一个可选动作，干脆不放进候选，
                #     由引擎自己用 `mj_core.chi_tiles` 定。
                #     ⇒ 而客户端把动作**原样**当请求体发上去 ⇒ 平台不知道吃哪两张 ⇒ 拒绝。
                #   ⇒ 修法：提交前补 `tiles`，且**必须用引擎同一个函数** `chi_tiles`
                #     （本仓最贵的一类 bug 就是"两份实现分叉"，不能再造第二份）。
                if act.get("action") == "chi" and not act.get("tiles"):
                    try:
                        _ld = (snap or {}).get("last_discard")
                        if _ld:
                            # ⚠ `chi_tiles` 的第二个参数要的是**牌码字符串**（"6b"），
                            #   不是索引 —— 第一版我传了 `tile_index(_ld)`，
                            #   属性检验里立刻炸出 `'int' object has no attribute 'strip'`。
                            _tl = chi_tiles(list(v.counts), _ld, v.meld_sets)
                            if _tl:
                                act = dict(act)
                                # ★★★ 第 483 轮：**真正的根因** —— 缺的是 `tile`，不是 `tiles`。
                                #   `mj_policy.py:1705/1723`（规则策略，线上 409≈0）一直是：
                                #       {"action": "chi", **"tile": v.last_discard**, "tiles": pair}
                                #   ⇒ **要同时给"吃的是哪张弃牌"与"手上用哪两张"**。
                                #   而模型的窗口动作经 `to_engine_action` 只剩 `{"action":"chi"}`，
                                #   服务端无从判断 ⇒ `INVALID_ACTION: chi tile mismatch`（147 条同类）。
                                #   ⚠ 我此前**三轮都在猜 `tiles` 的形状**（两张数组 / 三张数组 / 字符串），
                                #     而**答案一直写在规则策略的返回值里** —— 本该**先读它**
                                #     （正是 §Z.478 那条纪律："动模块前先读完相关代码/注释"）。
                                #   线格式（服务端 `Action.tiles` 是 `[]string`，见 HTTP 400 原话）：
                                #     `tile`  = 被吃的弃牌（牌码字符串）
                                #     `tiles` = 手上那两张（**字符串数组**）
                                act["tile"] = _ld
                                act["tiles"] = list(_tl)
                                say("[%s] 场 %s **chi 发 tile=%s tiles=%r**（副露 %d）"
                                    % (self.player.label, self.gid, _ld,
                                       act["tiles"], v.meld_sets))
                    except Exception:            # noqa: BLE001
                        pass
                # ★★ 第 465 轮：**窗口动作的提交前合法性守卫**。
                #   证据（`reject_dump.jsonl`，可观测性修好之后第一眼就看到）：
                #     phase=response_peng act={"action":"peng"} err="cannot peng 2t"
                #     hand = 9w 发 南 5t 4w 6b 3b 北 4w 西 4w 3w 8w   ← **一张 2t 都没有**
                #     last_discard = 2t ｜ ms = 17.4（很快 ⇒ 不是超时）
                #   同类还有 `cannot peng 1w` / `cannot peng 3w` ——
                #   即：**策略提交了一个"按提交那一刻的视图根本做不了"的动作**。
                #   为什么加守卫而不是去改策略：这两种动作（碰/吃）在**"过"与"不响应"完全等价**
                #   （服务器到点自动过，见上面 `pass` 的处理），所以
                #   **本地判定不合法就干脆不提交** —— 省一个请求额度、少一个 409、语义不变。
                #   ⚠ 守卫用的是**与决策同一份视图**，所以它挡不住"快照整体偏旧"；
                #     但它能挡住"策略自己给出的动作与视图自相矛盾"这一类，
                #     而线上抓到的正是这一类（视图里没有的牌却被要求碰）。
                if phase in ("response_peng", "response_chi") and \
                        act.get("action") in ("peng", "chi"):
                    _ldv = (snap or {}).get("last_discard")
                    _ok = False
                    try:
                        if act["action"] == "peng":
                            _ok = bool(_ldv) and v.counts[tile_index(_ldv)] >= 2
                        else:
                            _ok = bool(_ldv) and bool(chi_tiles(list(v.counts),
                                                                _ldv, v.meld_sets))
                    except Exception:            # noqa: BLE001
                        _ok = False
                    if not _ok:
                        self.missed += 1
                        say("[%s] 场 %s **本地判不合法，不提交**（%s，弃牌 %s，手上 %d 张）"
                            % (self.player.label, self.gid, act.get("action"), _ldv,
                               (v.counts[tile_index(_ldv)]
                                if _ldv else -1)))
                        continue
                _t_post = time.time()         # ★ 分段计时：提交 POST 往返
                try:
                    api("POST", "/api/games/%s/action" % self.gid, self.player.server,
                        self.player.token, body=act)
                    seg[5] = (time.time() - _t_post) * 1000.0
                    self.actions += 1
                    self.submits += 1            # ★ 其中"成功提交"单独计数（§Z.1326）
                    # 记录"从拿到快照到提交完成"的耗时。
                    # 窗口只有 1 秒（碰/吃）/ 3 秒（出牌），这个值直接决定
                    # 会不会出现"提交到得晚、被 409 拒绝"。
                    self.lat.append((time.time() - t_wake) * 1000.0)
                    self.segs.append(seg)
                    # ★ 超窗计数：按阶段用不同的窗口长度判定（见 __init__ 的说明）。
                    #   这是**唯一**能在正式赛当场发现"服务端在替我们出牌"的指标。
                    _ms = self.lat[-1]
                    if phase == "draw":
                        if _ms > 3000.0:
                            self.late_discard += 1
                            self.late_ms.append(_ms)
                    elif phase in ("response_peng", "response_chi"):
                        if _ms > 1000.0:
                            self.late_window += 1
                            self.late_ms.append(_ms)
                except ApiError as e:
                    if e.status != 409:
                        raise
                    self.conflicts += 1
                    # 被拒时把**这条链的耗时**一起记下来 —— 这是区分
                    # "我们算得慢" 与 "服务端已经翻页" 的唯一直接证据。
                    self.reject_lat.append(
                        {"ms": round((time.time() - t_wake) * 1000.0, 1),
                         "phase": phase, "action": (act or {}).get("action"),
                         "drawn": (snap or {}).get("drawn_tile"),
                         "local_can_win": bool(v.can_win())})
                    if is_gang:
                        self.gang_conflicts += 1
                        say("[%s] 场 %s **杠被拒(409)** -> %s"
                            % (self.player.label, self.gid, act))
                    else:
                        # ★★ 第 457 轮：**把窗口阶段（response_peng/response_chi）也纳入**。
                        #   ⚠ 第一版改动只重写了这段注释、**忘了改条件本身**，
                        #     于是注释写着"已含窗口"而代码仍 `elif phase not in (...)` ——
                        #     **一条说谎的注释比没注释更糟**。这里改成 `else`，条件真的放开了。
                        #   为什么必须放开：上线模型后线上实测 **409 / 行动 ≈ 14~17%**，
                        #   日志里全是 `**同一状态第 2 次被拒**（chi）→ 放弃该动作`，
                        #   而窗口阶段的 409 原本**只 `self.missed += 1`、不落盘** ⇒
                        #   **最需要诊断的那一类恰好是唯一看不到原因的**。
                        say("[%s] 场 %s **409 被拒** phase=%s seat=%s action=%s -> %s"
                            % (self.player.label, self.gid, phase, seat,
                               (act or {}).get("action"), act))
                        try:
                            with open("reject_dump.jsonl", "a",
                                      encoding="utf-8") as _f:
                                _f.write(json.dumps(
                                    {"gid": self.gid, "phase": phase, "seat": seat,
                                     "act": act, "local_can_win": bool(v.can_win()),
                                     "hand": list(v.full), "counts": list(v.counts),
                                     "meld_sets": v.meld_sets,
                                     "melds_raw": (snap or {}).get("melds"),
                                     "drawn": v.drawn, "dealer": v.dealer,
                                     "wall": v.wall,
                                     "round_no": (snap or {}).get("round_no"),
                                     "scores": (snap or {}).get("scores"),
                                     # ★ 第 42 轮补上三样，为了定位"为什么被拒"：
                                     #   err   = **服务端原话**（这是最直接的线索，
                                     #           此前只记了 HTTP 409，没记 message）
                                     #   god   = 抓打圈/链/爆头等状态（catch_play 时
                                     #           我们可能根本不该提交动作）
                                     #   turn  = 快照里服务端认为轮到谁（若不是我，
                                     #           说明快照已经过期）
                                     "err": getattr(e, "body", None) or str(e),
                                     "god": (snap or {}).get("god"),
                                     "turn": (snap or {}).get("turn"),
                                     "last_discard": (snap or {}).get("last_discard"),
                                     "responding": (snap or {}).get("responding_seats"),
                                     "ms": round((time.time() - t_wake) * 1000.0, 1)},
                                    ensure_ascii=False) + "\n")
                        except Exception:                # noqa: BLE001
                            pass
                    if phase in ("response_peng", "response_chi"):
                        self.missed += 1
                    # 409 = 动作已失效/被服务端拒绝。同一状态只解禁「一次」重试：
                    # 完全不解禁会让被拒的动作卡到超时；无上限解禁则退化成紧密循环。
                    #
                    # ★ 但"一次"仍然不够 —— 线上实测（11:19 的 reject_dump.jsonl）：
                    #   同一个状态被拒之后**再次提交同样的动作、再次被拒**，
                    #   两次相隔 12 秒，而且 `local_can_win=True`、手牌确实是
                    #   14 张成胡形。也就是说服务端认为"你早就过了这一巡"
                    #   （多半是我们的快照偏旧 / 提交到得晚），而我们只要还看到
                    #   同一个状态就会一直重试 **hu**。
                    #   后果不是丢一手，而是**这一局永远结束不了**：额度被它
                    #   持续吃掉，正式赛里等于 10 场少 1 场。
                    #   修法：同一状态被拒两次就**不再重试同一动作**，改为强制出牌
                    #   （丢掉这一手胡，但把对局推进下去）；下一次快照签名变化时
                    #   重试预算自动重置。
                    same_state_repeat = (self._conflict_sig == sig)
                    if not same_state_repeat:
                        self._conflict_sig = sig
                        responded = None
                    else:
                        # ★★★ 第 521 轮：**我在这里差点犯一个已被否决过的错**。
                        #   线上看到 `★放弃动作 1 次（丢了胡牌）`，我第一反应是"胡牌不该被放弃"，
                        #   于是改成"胡牌继续重试、不强制弃牌"。**然后才读到紧邻上方那段归档注释**：
                        #     「后果不是丢一手，而是**这一局永远结束不了**：额度被它持续吃掉，
                        #       正式赛里等于 10 场少 1 场。
                        #       修法：同一状态被拒两次就**不再重试同一动作**，改为强制出牌
                        #       （丢掉这一手胡，但把对局推进下去）」
                        #   ⇒ **这个取舍当年已经分析过并做了决定**，代价也写明了：
                        #     丢一手胡 << 整局卡死（10 场少 1 场）。
                        #   ⇒ 所以**撤回我的改动**，保持原行为。
                        #   ★ 教训（第 N 次同族，且这次注释就在正上方）：
                        #     **改一个分支之前，先读它上面那段注释** —— 它往往记录了"为什么不能这么改"。
                        #   ⚠ 留一个**将来可做**的中间路线（现在不做，因为没有证据）：
                        #     "再多重试一次、然后仍然放弃" —— 只有在测出"多一次重试能救回胡牌的概率"
                        #     之后才值得加；没有那个测量，加了就是拿"整局卡死"去赌一个未知的胜率。
                        self.giveup_actions += 1
                        say("[%s] 场 %s **同一状态第 2 次被拒**（%s）→ 放弃该动作，"
                            "改为出牌以推进对局"
                            % (self.player.label, self.gid,
                               (act or {}).get("action")))
                        try:
                            code = v.drawn or (v.hand[0] if v.hand else "")
                            if code:
                                api("POST", "/api/games/%s/action" % self.gid,
                                    self.player.server, self.player.token,
                                    body={"action": "discard", "tile": code})
                                self.actions += 1
                                self.submits += 1   # ★ 兜底出牌也是**成功提交** ⇒ 计入（§Z.1326）
                                # 状态已推进，重置重试预算
                                self._conflict_sig = None
                                responded = sig
                        except ApiError:
                            pass
                    seq = 0
                    continue
                except OSError as e:
                    # ★★ 2026-10-09：**提交超时/断连 ≠ 动作失效** ★★
                    #   这是目标第 3 条点名的那一条，但它的真实形态和注释里写的
                    #   不一样：`api()` **不会**为超时重试（它只重试 429/5xx/
                    #   409+MATCH_BUSY）⇒ 超时会一路穿透到最外层兜底 `except`，
                    #   **把整个 GameRunner 结束掉**（= 白丢一整场）。
                    #   ⇒ 所以要防的不是"盲目重发"，而是"整场被网络抖动打死"。
                    #   处置：**绝不重发同一个动作**（服务端可能已经执行了，
                    #         重复的 peng/gang 语义比重复的 discard 更糟），
                    #         改为重取权威状态（`seq=0`）后按**新状态**重新决策 ——
                    #         新状态里如果动作已生效，签名自然就变了，不会再提交它。
                    self.net_errors += 1
                    self.submit_timeouts += 1
                    if self.net_errors > _MAX_NET_RECOVER:
                        say("[%s] 场 %s **网络连续异常 %d 次**（%s）→ 结束本场"
                            % (self.player.label, self.gid, self.net_errors,
                               type(e).__name__))
                        return
                    say("[%s] 场 %s **提交超时/断连**（%s）→ 重取状态确认是否已执行，"
                        "不盲目重发" % (self.player.label, self.gid, type(e).__name__))
                    seq = 0
                    continue
        except ApiError as e:
            if e.status != 404:
                say("[%s] 场 %s API 错误: %s" % (self.player.label, self.gid, e))
        except Exception as e:                           # noqa: BLE001
            # ⚠ 这里**必须打 traceback**：只打 `%r` 的话拿到的是
            #   `TypeError("'int' object is not iterable")` 这种毫无上下文的字符串 ——
            #   而它对这一场是致命的（本场直接退出，10 场里少一场 = 白丢分）。
            #   2026-09-17 08:17 线上真的出现过一次，当时就是因为没打栈而查不动。
            #
            #   同时把"出事那一刻的状态"落盘：光有栈还不够 —— 如果异常来自
            #   某个字段的**形状**与预期不符（比如 melds 是 int 而不是 list），
            #   下一次复现时看不到当时的快照就仍然查不出来。
            import traceback as _tb
            say("[%s] 场 %s 异常: %r\n%s"
                % (self.player.label, self.gid, e, _tb.format_exc()))
            try:
                with open("crash_dump.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {"gid": self.gid, "err": repr(e),
                         "phase": (snap or {}).get("phase"),
                         "turn": (snap or {}).get("turn"),
                         "seat": (snap or {}).get("seat"),
                         "dealer": (snap or {}).get("dealer"),
                         "melds": (snap or {}).get("melds"),
                         "god": (snap or {}).get("god"),
                         "my_hand": (snap or {}).get("my_hand"),
                         "responding_seats": (snap or {}).get("responding_seats"),
                         "traceback": _tb.format_exc()},
                        ensure_ascii=False) + "\n")
                    with open("crash_dump_snapshot.json", "w", encoding="utf-8") as f:
                        json.dump(snap, f, ensure_ascii=False, indent=1)
            except Exception:                            # noqa: BLE001
                pass
            # ★ 重试上限：见 `self.attempts` 的说明。
            #   第一次意外异常 → 允许重来一次（瞬时错误能救回）；
            #   第二次仍异常 → 放弃该局并留痕，**绝不能无限重启**。
            self.attempts += 1
            if self.attempts >= 2:
                self.gone = True
                self.player.results.setdefault(self.gid, {
                    "seat": -1, "scores": [], "rounds": None,
                    "actions": self.actions, "strategy": self.strategy.name,
                    "avg_shanten": None, "discards": self.sh_n,
                    "round_deltas": self.round_deltas,
                    "round_nos": self.round_nos,
                    "my_meld_sets": self.round_melds,
                    "missed_detail": self.missed_detail,
                    "deltas_base": self._base_scores,
                    "crashed": True})
                say("[%s] 场 %s **连续两次异常，已放弃本局**（避免无限重启吃满额度）"
                    % (self.player.label, self.gid))
        finally:
            if nf is not None:
                nf.stop()
            # ⚠ lat_txt 必须在 finally 里**先定义**：下面的汇总行在两处被用到
            #   （异常退出路径也会走到这里），而未定义的局部变量会再抛一次
            #   UnboundLocalError —— 那会把"记录失败原因"这件事本身也搞崩。
            lat_txt = ""
            # 端到端延迟：窗口只有 1s/3s，所以要看**尾部**而不是均值。
            # 若 p99 逼近窗口，就说明 409（=丢胡牌）是延迟造成的，
            # 该优化的是"等限速令牌 + GET"这一段，而不是策略本身。
            lats = sorted(self.lat)
            if lats:
                m = len(lats)
                lat_txt = " 延迟 中位%.0f/p99%.0f/最大%.0f ms" % (
                    lats[m // 2], lats[int(0.99 * m)], lats[-1])
            if self.reject_lat:
                lat_txt += " 被拒%d次" % len(self.reject_lat)
            if self.stale_skips:
                lat_txt += " 陈旧跳过%d" % self.stale_skips
            # ★ 超窗提交（正式赛唯一的当场可观测代理，见 __init__ 的说明）。
            #
            # ⚠ **必须无条件打印**（哪怕为 0）：否则"没打印"会被误读成
            #   "没有超窗"或"仪表坏了"两种含义，正式赛上无法区分 ——
            #   而 `0 窗口错过 / 0 次409` 那种"看起来一切正常"的读数正是
            #   bug 窗口期的真实表现（§14.35）。打出来才能自证仪表在工作。
            lat_txt += " 超窗[出牌%d/窗口%d]" % (self.late_discard, self.late_window)
            if self.late_ms:
                lat_txt += "(最大%.0fms)" % max(self.late_ms)
            # 分段长尾：把"最慢的那一批"的四段分别报出来。
            # 这一行直接回答"该优化谁"：gap 大→调 --fetch-gap-max；
            # 令牌排队大→额度真的不够（要减少请求或降低并发桌数）；
            # GET 往返大→网络/服务端慢。
            if self.segs:
                _txt, _ = _seg_report(self.segs, frac=0.05)
                lat_txt += " | " + _txt
            # ★ 2026-10-03（第 922 轮）**纯打印追加**：把两个"可见损失"计数器也打进场结束行，
            #   使它们**可按臂归集**（此前只在进程级哨兵里 ⇒ 无法分臂 ⇒ GOAL_STATUS §Z.1298）。
            #   · `giveup_actions`：同一状态反复被拒 ⇒ 放弃动作改为出牌 ⇒ **丢一手本可胡的牌**
            #   · `stale_skips`  ：快照过期（出牌阶段但 drawn 为空）而跳过动作 ⇒ **网络把快照拖过期**
            #   ⚠ 二者都**早已存在**（`:405` / `:432`）⇒ 本改动**只多打两个数**，不改任何逻辑。
            #   ★ 2026-10-09 再追加两个（**必须放在行尾**）：
            #     · `网络`       = 被网络抖动打断、但重取状态后救回来的次数
            #     · `提交超时`   = 其中发生在**动作提交**上的子集（那一类要先确认是否已执行）
            #     ⚠ 放行尾是硬要求：`live*.log` 的汇总行有多个解析器
            #       （`tools/live_baseline.py` 等），它们在 `最大%d ms` 之后就 `.search`
            #       到行尾为止 ⇒ **中间插字段会把这些解析器全部打坏**。
            say("[%s] 场 %s 结束：%d 行动 / %d 杠(被拒%d) / %d 窗口错过 / %d 次409 / "
                "%d 推送 放弃%d 陈旧%d 提交%d 过%d%s 网络%d 提交超时%d"
                % (self.player.label, self.gid, self.actions, self.gangs,
                   self.gang_conflicts, self.missed, self.conflicts,
                   (nf.frames if nf is not None else 0),
                   self.giveup_actions, self.stale_skips,
                   self.submits, self.passes, lat_txt,
                   self.net_errors, self.submit_timeouts))


class Player(threading.Thread):
    """一个参赛账号：轮询名下活跃场次，每场开一个 GameRunner。"""

    def __init__(self, server, token, strategy, room_id=None, trace=False,
                 max_minutes=40, label="seat", start_grace=60, rate=14.0,
                 use_notify=True, fetch_gap_max=0.5, exit_after_games=0):
        super().__init__()
        self.server = server
        self.token = token
        self.strategy = strategy
        self.room_id = room_id
        self.trace = trace
        self.max_minutes = max_minutes
        self.label = label
        self.use_notify = use_notify
        # 取快照间隔的**上限**（秒）。纯"公平份额"在 10 桌时是 0.84 秒，
        # 比碰/吃窗口（1 秒）还长，导致 p99 端到端延迟 1493 ms、最大 5998 ms，
        # 以及一串"拿旧快照提交 hu → 409"。默认压到 0.5 秒。
        self.fetch_gap_max = float(fetch_gap_max)
        # ★ 收齐 N 场就退出，**不再等服务端把已结束的场次从 active_games 撤下来**。
        #   0 = 关（保持旧行为）。为什么（第 39 轮实测）：
        #   房间打完后 `status` 已是 `finished`（∈ TERMINAL）、本机活 worker 也是 **0**
        #   （哨兵一直打 `在打=0`），但客户端**还要等 93 秒**（中位）才开下一间
        #   —— `audit_room_gap.py`：那段 `finished→closed` 全程"在打=0"，
        #   占整房墙钟的 **9.1%**。
        #   三个条件里唯一没满足的只能是 `not actives` ⇒
        #   **服务端在房间 closed 之前，一直把已结束的场次列在 `active_games` 里**。
        #   那是服务端行为，我们唯一能做的是**别等它**。
        #   ⚠ 安全性：要求 `len(self.results) >= N` ⇒ M 场**全部**已回收才退，不会丢场。
        #   （也**不存在**"顺带开下一间"这条路：一个账号同时只能在一个房间 ——
        #     `run_room.py` 记着 `/api/match 幂等返原房`。）
        self.exit_after_games = int(exit_after_games or 0)
        # 起跑宽限：跨轮复用时房间会先停在 finished，随后 ready 满 4 席才回到 running。
        # 没有这道宽限，bot 一起来看见 finished 就整队退出，牌局只能由服务端代打。
        self.start_grace = start_grace
        # 令牌桶：服务器上限是 16 次/秒（每用户跨全部对局聚合）。
        # 用 SSE 后需求约 13/秒，故上限设 14 留一点余量。
        self.rate = float(rate)
        self.limiter = RateLimiter(rate)
        # 把 429 反馈接到令牌桶上：撞一次就降速，不必人工调参。
        my_bot.ON_THROTTLE = self.limiter.throttle
        self._workers = {}
        self.results = {}
        self._done = set()           # 已结束/已消失的场次，永久拉黑不再拉起
        self._confirmed_stage = None  # 已确认出席的阶段（stage_open 幂等去重）
        self._last_stage_key = None   # 上一次记录到日志的 (status, stage) —— 只在变化时打一行
        # 每场最小取快照间隔（秒）。0 = 不限。由 Player 的发现循环按当前场数
        # 动态写入，见 GameRunner.run 里的说明。
        self.fetch_gap = 0.0
        self.missed_hu = []          # 可胡未胡的样本，供赛后定性
        self._stop = threading.Event()
        # `my_games_by_batch` 兜底扫描的门槛与节流（见 run() 里的长说明）。
        # 实测该字段含 335 局而只有 10 局是我们的，所以只能在"确实没牌"时才扫。
        self.batch_scan_after = 30.0
        self._idle_since = None      # 快路径连续无牌的起点（有牌即清零）
        # 空转告警的两个时间戳（见 run() 里的事故说明）
        self._dry_since = None       # 本机 0 个活 worker 的起点
        self._last_alarm = 0.0       # 上次喊话时间（节流）
        # 单轮新建 GameRunner 的上限。正赛 10 桌要的是"尽快全部拉起"，
        # 所以上限设得比 M 大得多；它只是防兜底扫描一次灌进几百个 id 时爆发。
        self.max_new_per_round = 60

    def run(self):
        deadline = time.time() + self.max_minutes * 60
        t0 = time.time()
        if not self.room_id:
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
            srv_active_n = 0
            # ★ 必须每轮重置。`t` 原先只在下面的 try 里赋值，一旦那次请求抛
            #   非 404 的 ApiError（429/500/超时 —— 正式赛开赛瞬间限速压力最大，
            #   429 是常态），异常会被外层 except 吞掉，紧接着第 1074 行
            #   `t.get("stage")` 就会 **UnboundLocalError**（第一轮）或读到
            #   **上一轮的陈旧 t**（之后每轮）。
            #   前者直接打死发现线程 → `pl.join()` 返回 → 日志打出
            #   「本轮 0 场」—— **和 2026-09-17 的 −577 分事故一模一样的结局，
            #   只是触发原因从"形状判定"换成了"一次瞬态 429"**。
            t = {}
            try:
                me = api("GET", "/api/me", self.server, self.token)
                # 服务端**原始**条数（不经 _game_ids 过滤）—— 空转告警的判据。
                # 2026-09-17 的事故里这个数是 10，而本机 workers=0；
                # 只统计"过滤后"的条数会把 bug 藏起来（过滤正是出问题的那一步）。
                _raw = me.get("active_games") or []
                srv_active_n = len(_raw)
                for gid in _game_ids(_raw):
                    actives.append(gid)
                try:
                    t = api("GET", "/api/tournaments/%s" % self.room_id,
                            self.server, self.token)
                    status = t.get("status")
                    # 锦标赛的场次要靠 my_games 发现（指南 §二：`GET
                    # /api/tournaments/{room_id}` 的 my_games 发现场次）。
                    # 只认 /api/me 的 active_games 会有「整场找不到牌」的风险 ——
                    # 两个源取并集，事件流/404 各源自成体系时也能跑起来。
                    for gid in _game_ids(t.get("my_games")):
                        actives.append(gid)
                    # ★★ `my_games_by_batch` 只能当**兜底**，绝不能无条件并进来。
                    #
                    # 实测（2026-09-17 正式赛）：它返回 **335 条**，覆盖
                    # b0..b14 的**每一张桌**（b0_t0 … b0_t32 …），而逐条
                    # `GET /api/games/{gid}/state` 只有 **10 条**返回 200，
                    # 另 325 条是 403 `not a participant` —— 也就是说这个字段
                    # 装的是「我参加过的批次里的**全部**对局」，不是「我的对局」。
                    #
                    # 上面 `_looks_like_gid` 修好后，无条件并入的后果是：
                    # 开赛那一瞬间建出 **325 个不属于我们的 GameRunner**，
                    # 每个打一发请求换一个 403，把最宝贵的头几秒额度烧光 ——
                    # 而正式赛 10 桌本来就需要几乎全部 16/秒 的预算。
                    # 所以改成：**只有在"快路径连续 30 秒一场都没发现"时才扫**。
                    # 那正是这个兜底本来要救的场景（整场找不到牌）；
                    # 而只要 active_games/my_games 里有货，就一次都不扫。
                    fast_n = len(actives)
                    if fast_n:
                        self._idle_since = None
                    elif self._idle_since is None:
                        self._idle_since = time.time()
                    if (not fast_n and self._idle_since is not None
                            and time.time() - self._idle_since > self.batch_scan_after):
                        say("[%s] 快路径 %d 秒无牌，启用 my_games_by_batch 兜底扫描"
                            % (self.label, int(self.batch_scan_after)))
                        for _, lst in (t.get("my_games_by_batch") or {}).items():
                            for gid in _game_ids(lst):
                                actives.append(gid)
                        self._idle_since = time.time()      # 别每轮都重扫
                except ApiError as e:
                    status = "closed" if e.status == 404 else None
                    if status is None:
                        # ★ 不要 re-raise。原先这里 `raise`，后果是**一次瞬态
                        #   错误就掀掉整个发现线程**，主循环 join 返回、日志打出
                        #   「本轮 0 场」—— 无人值守的正式赛上这就是"整届白打"。
                        #   现在只记录并当作"本轮没拿到赛事状态"，继续下一轮；
                        #   反复失败会由空转告警接手喊话（见下面那段）。
                        say("[%s] 赛事状态查询失败（本轮忽略，继续）: %s"
                            % (self.label, e))
            except ApiError as e:
                if e.status == 404:
                    actives, status = [], "closed"
                else:
                    say("[%s] 轮询失败: %s" % (self.label, e))
            # 阶段流转必须留在日志里（v7 多阶段：registering → running →
            # stage_done → stage_open → running → … → finished）。
            # 没有这几行，赛后无法区分"打完了在等管理员推进"与"被淘汰/掉线"，
            # 也无从核对"确认出席是否及时"（漏确认会被候补顶替）。
            sg = t.get("stage") if isinstance(t, dict) else None
            st_no = sg.get("no") if isinstance(sg, dict) else None
            key = (status, st_no)
            if key != self._last_stage_key:
                self._last_stage_key = key
                say("[%s] ★ 阶段流转: status=%s stage=%s qualified=%s/%s crash=%s"
                    % (self.label, status, st_no,
                       t.get("qualified") if isinstance(t, dict) else None,
                       t.get("qualify_role") if isinstance(t, dict) else None,
                       t.get("stage_crashed") if isinstance(t, dict) else None))

            # 去重 + 逐个复核 self._workers。
            #
            # 踩过的坑：场次发现合并了三个来源（/api/me.active_games、my_games、
            # my_games_by_batch），同一个 gid 会在列表里出现多次；而 `running`
            # 是在循环**之前**做的快照，循环里又刚把 gid 写进 _workers ——
            # 快照里没有它，于是重复项被放行，同一场被创建出 2~3 个 GameRunner。
            # 结果是每场 2~3 条 SSE、2~3 倍的取快照与提交、同一手牌互相覆盖，
            # 线上表现为「请求速率只有 5/秒（预算 16）却仍有 3 倍于行动数的窗口错过」。
            seen = set()
            new_n = 0
            for gid in actives:
                if gid in seen or gid in self._done or gid in self.results:
                    continue
                seen.add(gid)
                if gid in self._workers:
                    continue
                # 单轮新建上限：兜底扫描可能一次塞进来几百个 id（实测 335 个，
                # 其中 325 个不属于我们）。不设上限就会在开赛瞬间集中爆发几百条
                # 请求 + 几百个线程。超出的部分下一轮继续处理，代价只是几十毫秒。
                if new_n >= self.max_new_per_round:
                    break
                new_n += 1
                w = GameRunner(self, gid, self.strategy)
                self._workers[gid] = w
                w.start()
            for gid in [g for g, w in self._workers.items() if not w.is_alive()]:
                w = self._workers.pop(gid)
                # 永久拉黑：已结束/404 的场次绝不再拉起。没有这道闸，my_games
                # 里的历史条目会让每轮轮询都重启一遍死场 —— 正是限速风暴的成因。
                if w.gone or gid in self.results:
                    self._done.add(gid)

            # ★★★ 空转告警（2026-09-17 事故的安全网）★★★
            #
            # 事故形态是：服务端 `/api/me` 明说「你有 10 场在打」，
            # 而本机 `_workers` 恒为 0、`行动=0`，就这么静默空转了 62 分钟，
            # 直到 `本轮 0 场`。**没有任何一行日志说过"这不正常"** ——
            # 唯一的征兆是哨兵里那个不起眼的 `在打=0 场`。
            #
            # 判据用**服务端原始条数**（`srv_active_n`，不经 `_game_ids` 过滤）：
            # 过滤那一步正是出问题的地方，用过滤后的数当判据等于把 bug 藏起来。
            # 服务端说有牌、本机却 0 个 worker ⇒ 一定是发现环节断了，立刻喊。
            live_n = sum(1 for w in self._workers.values() if w.is_alive())
            if live_n:
                self._dry_since = None
            else:
                if self._dry_since is None:
                    self._dry_since = time.time()
                dry = time.time() - self._dry_since
                # 15 秒宽限：刚建 worker 的那一瞬间还没 is_alive 是正常的。
                if (srv_active_n > 0 and dry > 15.0
                        and time.time() - self._last_alarm > 30.0):
                    self._last_alarm = time.time()
                    say("[%s] !!!! 空转告警：服务端说有 %d 场在打，本机 0 个 worker，"
                        "已持续 %.0f 秒 —— 场次发现断了（2026-09-17 的 −577 分"
                        "事故就是这个形态）" % (self.label, srv_active_n, dry))
                    # 同时落盘：日志可能被后来的进程/tee 覆盖，而这一条
                    # 是"整届比赛白打"的唯一现场证据，必须留得下来。
                    try:
                        with open("ALARM_dry_run.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "t": time.time(), "when": time.strftime("%F %T"),
                                "label": self.label, "room": self.room_id,
                                "srv_active_n": srv_active_n,
                                "actives_parsed": len(actives),
                                "live_workers": live_n, "dry_sec": round(dry, 1),
                                "status": status,
                            }, ensure_ascii=False) + "\n")
                    except Exception:                          # noqa: BLE001
                        pass

            # 把「取快照的公平份额」下发到每一场。SSE 帧不带内容，客户端只能每帧
            # 取一次快照；10 桌的事件率远超 16/秒 预算，不限速就退化成抢额度，
            # 令牌桶队列一深，碰/吃窗口（固定 1 秒）就成批错过 —— 线上实测 50%。
            # 留 15% 余量给动作提交与场次发现。
            #
            # ★ 上限（默认 0.5 秒，可用 --fetch-gap-max 调）：见 GameRunner.run
            #   里那段说明 —— 10 桌时纯公平份额是 0.84 秒，比碰/吃窗口还长，
            #   直接导致 p99 延迟 1493 ms、最大 5998 ms，以及一串 409。
            raw_gap = (len(self._workers) / (self.limiter.rate * 0.85)
                       if self._workers else 0.0)
            self.fetch_gap = min(raw_gap, self.fetch_gap_max) if raw_gap else 0.0
            # 没在打的时候慢慢把发号速率放回去（撞到 429 时被 throttle 压过）。
            if not self._workers:
                self.limiter.relax()

            if status in STAGE_CONFIRM:
                # 每个新阶段都要重新确认出席（幂等）。同一个阶段只提交一次，
                # 避免每轮轮询都打一发 —— 正式赛期间请求额度要留给牌局。
                stage = self._stage_no(t)
                key = (status, stage)
                if key != self._confirmed_stage:
                    try:
                        api("POST", "/api/tournaments/%s/ready" % self.room_id,
                            self.server, self.token)
                        self._confirmed_stage = key
                        say("[%s] ★ 阶段 %s 出席确认已提交（stage_open）"
                            % (self.label, stage))
                    except ApiError as e:
                        # 名单外 -> 409 NOT_QUALIFIED（本轮已淘汰，属正常）；
                        # 非确认期 -> 409 TOURNAMENT_STARTED。都不该让主循环退出。
                        say("[%s] 出席确认未成功: %s" % (self.label, e))
            else:
                self._confirmed_stage = None

            if status in TERMINAL and not actives and not self._workers:
                # 已打过至少一场 = 本轮确实结束；否则房间可能只是停在上一轮的 finished，
                # 宽限期内继续等它被 ready 重新开起来。
                if self.results or (time.time() - t0) > self.start_grace:
                    break
            # ★ 提前退出（`--exit-after-games N`）：收齐 N 场就走，**不等服务端**
            #   把已结束的场次从 `active_games` 撤下来 —— 那一步实测要 93 秒（中位），
            #   占整房墙钟 9.1%（见 `exit_after_games` 的构造注释与 `audit_room_gap.py`）。
            #   只认"状态终局 + 本机没有活 worker + 已回收满 N 场"三条，
            #   所以**不会丢场**，也不依赖任何猜测。
            if (self.exit_after_games and status in TERMINAL
                    and not self._workers
                    and len(self.results) >= self.exit_after_games):
                say("[%s] ★ 已收齐 %d 场（状态 %s），提前退出比赛段、不等房间 closed"
                    % (self.label, len(self.results), status))
                break
            # 自适应轮询：没牌时 1 秒（要快发现），有牌时 5 秒。
            # M=10 桌并发时 SSE 已占约 13 请求/秒，发现轮询的 2 请求/秒会把它
            # 顶到 15/16 —— 留出余量，别在最后一步撞限速墙。
            #
            # ★ 例外：**当前状态是确认期、且还没确认成功** 时必须用 1 秒。
            #   为什么单独开这个口子：确认期的窗口由管理员推进，可能只有几秒；
            #   而"有牌在打"时轮询是 5 秒 —— 若某个阶段在上一阶段还没打完时
            #   就进入 stage_open（或我们上一轮确认失败过），5 秒的节奏可能
            #   整个窗口都错过。**漏确认 = 被候补顶替 = 直接出局**，
            #   这是唯一一条"错过就没有第二次"的风险，值得多花几个请求。
            unconfirmed = (status in STAGE_CONFIRM
                           and (status, self._stage_no(t)) != self._confirmed_stage)
            time.sleep(1.0 if (unconfirmed or not self._workers) else 5.0)

    @staticmethod
    def _stage_no(t):
        """从赛事详情里取阶段号（形状在不同版本间变过，取不到就返回 None）。"""
        try:
            sg = (t or {}).get("stage")
            return sg.get("no") if isinstance(sg, dict) else None
        except Exception:                                # noqa: BLE001
            return None


def _is_network_error(exc):
    """这个异常是不是**网络类**（超时 / 断连），而不是服务端的明确应答？

    ★ 为什么必须把它和 `ApiError` 分开：两者的**语义完全不同** ——
      · `ApiError(409)` = 服务端**明确说**"这个动作没生效" ⇒ 重取状态后可以再试；
      · 超时 / 断连      = **不知道**有没有生效 ⇒ 盲目重发可能造成重复动作。
      ⇒ 后者的正确处置是**先重取权威状态（`seq=0`）确认，再按新状态重新决策**。
    ★ 为什么不能像以前那样让它穿透：`run()` 最外层有一个兜底 `except Exception`
      ⇒ **整个 GameRunner 直接结束** ⇒ 一场白丢。
      实测（`temp/live_check.log`，2026-10-09）：两场只打了 1~3 手就"结束"，
      而 traceback 全部指向 `mj_bot.py` 的**快照 GET** ——
      也就是说"网络抖一下"被记成了"这一场打完了"。
    ★ `ApiError` 继承自 `Exception`（不是 `OSError`），所以这个判据是干净的。
    """
    return isinstance(exc, OSError)


# 本场允许"网络抖动 → 重取状态 → 继续"的次数上限。
# 取 8：一次网络抽风（本仓实测最长约 7 分钟）会命中好几次，8 够扛过去；
# 而真断了（服务端没了）也不至于永远循环 —— 超限就结束本场，交给上层重新匹配。
_MAX_NET_RECOVER = 8


def _seg_report(segs, frac=0.05):
    """把分段耗时汇总成一行：取**最慢的一批**（默认 5%）逐段求均值。

    段序（与 `GameRunner.run` 里构造 `seg` 的地方**必须一致**）：
        0 SSE 唤醒 → 可以取快照      1 gap 睡眠（人为限速，可调）
        2 令牌排队（全局额度）        3 GET 往返（网络 / 服务端）
        4 决策：特征计算 + 推理       5 提交 POST 往返

    这一行直接回答"该优化谁"：
        gap 大   → 调 `--fetch-gap-max`
        令牌大   → 额度真不够（减少请求或降并发桌数）
        GET 大   → 网络 / 服务端慢
        **决策大** → 特征或推理慢（`discard=rank/rank2` 会加载排序器）
        **提交大** → 服务端接受动作慢（超过 1 秒窗口就会吃到 409）

    ★ 抽成模块级函数是为了**可测**：内联在 `run()` 里只能靠跑线上来验，
      而这里恰恰需要钉住"段序不会悄悄错位"（错位会把读数指向错误的优化目标）。
    返回 `(文本, 参与统计的段数)`；无数据时返回 `("", 0)`。
    """
    if not segs:
        return "", 0
    worst = sorted(segs, key=lambda x: -(x[0] + x[3] + x[4] + x[5]))[
        :max(1, int(len(segs) * frac))]
    n = len(worst)
    return ("最慢5%%分段: gap%.0f 令牌%.0f GET%.0f 决策%.0f 提交%.0f" % (
        sum(x[1] for x in worst) / n,
        sum(x[2] for x in worst) / n,
        sum(x[3] for x in worst) / n,
        sum(x[4] for x in worst) / n,
        sum(x[5] for x in worst) / n), n)


def _fetch_full_snapshot(server, token, gid):
    """取**全量快照**（`seq=0`）—— 必须用短超时 + 不重试。

    为什么不能沿用 `api()` 的默认值（`timeout=35, tries=6` ⇒ 最坏阻塞约 258 秒）：
    快照只在 1~3 秒的窗口内有用，超窗拿到也没有意义；而在窗口里阻塞 258 秒
    不只错过这一手，还会把整个 `GameRunner` 卡住（同一进程里其余桌同受影响）。
    全量快照**本来就可以重取**（下一轮循环还会再取一次），所以失败就让调用方
    `continue`，没有"必须等它成功"的理由。

    ★ 抽成模块级函数的唯一目的是**可测**：内联在 `run()` 里就没法断言
      "它到底传了什么超时" —— 而本缺陷恰恰就是"传错了超时"，静默且致命。
    """
    return api("GET", "/api/games/%s/state?seq=0" % gid, server, token,
               timeout=1.5, tries=1) or {}


def _state_sig(snap, phase):
    """「同一状态」的判据 —— GameRunner 用它去重，避免对同一状态重复提交动作。

    为什么要这么细（2026-09-17 加固）：原来的键只到
    `(round_no, phase, ndisc, turn)`，而 **ndisc 会因为吃碰而减少**
    （被吃碰的牌会离开牌河）。碰撞是可能的：
        我们某次出牌时 ndisc=8 -> 记下签名
        之后三家各被吃碰一次      -> ndisc 又回到 8
        轮回我们时 (round, phase, ndisc, turn) 与上次**完全相同**
    这一手就会被误判成"已经响应过"直接跳过，然后服务端 3 秒超时代打
    （等价于随机弃牌）。把「刚摸的牌 + 自己的手牌」纳入签名后，
    状态变了就一定不同；而真正没变的状态仍然只提交一次。

    ★★ 2026-10-09 再加固一层（**响应窗口的身份**）★★
      上面那次补的是**出牌阶段**：`drawn_tile + my_hand` 一摸牌就变，兜得住。
      但**响应窗口**（`response_peng` / `response_chi`）里这两者**都不变**
      —— 窗口不是自己摸牌 ⇒ 只剩 `ndisc` 一个区分量，而它又会被吃碰**减小**
      ⇒ 两个不同的响应窗口完全可能撞成同一个签名 ⇒ 第二手被静默跳过
      ⇒ 服务端 3 秒超时代打（等价随机弃牌，可能白丢一个碰甚至一个胡）。
      ⇒ 把决定"窗口合法动作集"的三个量并进签名：
        · `responding_seats` —— 这一手要哪些座位响应（窗口本身的身份）
        · `last_discard`     —— 我们响应的是哪张牌（碰/吃/杠都取决于它）
        · `restricted`       —— 抓打圈（受限方不能吃碰明杠、只能打刚摸的）

    ⚠ **这个方向的代价必须知道**：签名变细 ⇒ `sig == responded` 这个去重守卫
      触发得更少 ⇒ **提交次数只会变多，不会变少**。收益是"不再静默跳过该响应的一手"，
      代价是理论上可能出现"同一窗口二次提交"（服务端会 409 拒掉，无害但烧额度）。
      ⇒ **上线后必须盯 409 率**（本仓实测基线：模型臂 4.19% / 规则臂 1.01%）。
        若 409 明显上升，第一个该回退的就是 `responding_seats` 这一项
        （它对"窗口身份"的贡献最小 —— `last_discard` 才是关键区分量）。
    """
    ndisc = sum(len(d) for d in (snap.get("discards") or []))
    god = snap.get("god") or {}
    # 与 `View.restricted` **逐字同构**（catch_play 且本人不是打财神者）；
    # 这里直接读快照而不建 View，因为本函数在热路径上，每帧都要调用。
    restricted = (bool(god.get("catch_play"))
                  and god.get("god_discarder_seat") != snap.get("seat"))
    return (snap.get("round_no"), phase, ndisc, snap.get("turn"),
            snap.get("drawn_tile") or "",
            tuple(sorted(snap.get("my_hand") or [])),
            tuple(snap.get("responding_seats") or ()),
            snap.get("last_discard") or "",
            restricted)


# 场次 id 的形状：`<段>_<段>[_<段>...]_b<批>_t<桌>`，至少两个前缀段。
#
# 为什么是"至少两个前缀段"而不是写死 `a_`/`t_`：
#   · 自动房   `a_dbbb430773cc_r1_b7_t0`  -> 前缀 a / dbbb430773cc / r1（3 段）
#   · 正式赛   `t_65d538e905c5_b0_t25`    -> 前缀 t / 65d538e905c5（2 段）
#   · 干扰串   `r1_b2_t0`                 -> 前缀只有 r1（1 段）=> 必须拒
# 卡"前缀段数"能同时收下两种真实形状、又挡住"批次号被当成场次 id"那类值。
_GID_RE = re.compile(r"^(?:[A-Za-z0-9]+_){2,}b\d+_t\d+$")


def _looks_like_gid(s):
    """判断一个字符串**像不像**场次 id。

    平台的 game_id 形如 `a_677030880327_r1_b7_t0`（房号 + _r局 + _b批 + _t）
    或 `t_65d538e905c5_b0_t25`（正式赛：赛事id + _b批 + _t）。

    为什么要做形状校验：`batch` / `id` 这些键名在别的上下文里会出现
    **含义完全不同**的值 —— 例如 `{"batch": 3}` 里的 3 是批次号，
    `{"id": "running"}` 里的是状态词。把它们当场次 id 去请求会打出 404 风暴。

    ⚠⚠ 2026-09-17 重大事故（**代价 = 整届比赛**）⚠⚠
      旧实现是：

          return ("_r" in s and "_b" in s) or s.startswith("a_")

      它只认**自动房**的形状。正式赛 id `t_65d538e905c5_b0_t25`
      **没有 `_r`、也不以 `a_` 开头** ⇒ 判为"不像 id"。
      而 `_game_ids` 处理**列表元素**时传 `_nested=True`，于是
      `active_games` / `my_games` / `my_games_by_batch` **三个来源全被拒**，
      `actives` 恒为空 ⇒ 一个 GameRunner 都建不起来。

      现场表现（tournament_run.log）：20:00:00 服务端已返回
      「进行中对局 10 场」，之后 62 分钟一直是
      `在打=0 场 行动=0 窗口错过=0 409=0`，最后 `本轮 0 场`。
      10 场**全部由服务端代打** ⇒ 逐局
      [-40,-43,-46,-47,-48,-52,-67,-70,-79,-85] 合计 **−577**、
      160 手场均 **−3.606**、名次 **132/134**、`NOT_QUALIFIED`。

      教训：**形状校验的样本必须覆盖所有真实来源**。旧自测里唯一涉及
      前缀的用例取的样本恰好是 `a_...`，还配了注释断言
      「`my_games` 的真实形状是 ["a_<hex>_r.._b.._t.."]」—— 这条断言
      对正式赛是**假的**，于是自测把 bug 锁死了。
      回归锁：`check_tournament_discovery.py`（夹具取自服务器真实 id）。
    """
    if not isinstance(s, str) or len(s) < 6:
        return False
    if s in ("running", "finished", "active", "playing", "ready",
             "waiting", "closed", "pending"):
        return False
    return bool(_GID_RE.match(s)) or s.startswith("a_")


def _game_ids(raw, _nested=False):
    """从各种可能的形状里抠出 game_id：字符串 / {"game_id":..} / [..]。

    官方 `my_games` / `my_games_by_batch` 的形状在版本间变过，
    这里做**保守提取**：拿不准的一律跳过。

    ⚠ 2026-09-17 加固两处（都直接关系正式赛能否打满 10 场）：
      1. 原来只认 `game_id` 与 `game` 两个键。而正式赛的场次主要靠
         `/api/tournaments/{id}` 的 `my_games` 发现 —— 一旦那个字段用了别的
         键名（`id` / `batch` / `gid` 都可能），我们会**一场都发现不了 →
         整场 0 分 → 直接出局**，而且现场只有"status=running 但一直没牌"。
         所以把可能的名字都收进来，并用 `_looks_like_gid` 卡住形状。
      2. 裸字符串的**宽松度必须看来源**：调用方直接传字符串（`my_games`
         就是一个 ["a_xxx"]）时它本身就是答案，不能拒；但从字典值里
         **递归**出来的字符串（例如 `{"status": "running"}` 里的 "running"）
         必须过校验，否则状态词会被当成场次 id。所以加了 `_nested`。
    """
    out = []
    if not raw:
        return out
    if isinstance(raw, str):
        # 顶层直给 => 采信；递归途中 => 必须像 id
        return [raw] if (not _nested or _looks_like_gid(raw)) else []
    if isinstance(raw, dict):
        gid = None
        for k in ("game_id", "game", "gid", "id", "batch", "batch_id"):
            v = raw.get(k)
            if isinstance(v, str) and _looks_like_gid(v):
                gid = v
                break
        st = raw.get("status")
        if gid and (st is None or st in ("running", "active", "playing",
                                         "ready", "waiting")):
            out.append(gid)
            return out
        if not gid:
            # 形如 {"batch0": [...], "batch1": [...]} 的包装字典：往下钻
            for v in raw.values():
                out.extend(_game_ids(v, True))
        return out
    if isinstance(raw, (list, tuple)):
        for x in raw:
            out.extend(_game_ids(x, True))
    return out


def run_players(server, tokens, strategies, room_id, trace=False, max_minutes=40):
    """并行跑多个席位，返回 [(label, strategy, player), ...]。"""
    players = []
    for i, (tok, st) in enumerate(zip(tokens, strategies)):
        players.append(Player(server, tok, st, room_id=room_id, trace=trace,
                              max_minutes=max_minutes, label="席%d" % (i + 1)))
    for p in players:
        p.start()
    for p in players:
        p.join()
    return players


# ---------------------------------------------------------------- 回归测试

def _selftest():
    """锁死 View 的手牌拼装口径。

    历史 bug：按「13 张立牌 + 单独摸牌」拼接，而真实快照里 my_hand 在轮到自己时
    已是 14 张且已含刚摸的那张 —— 拼出 15 张后 can_win 的张数校验恒 False，
    表现为「永远不胡」：不报错、不崩溃，只是所有对局 0 分。极难察觉，故设用例。
    """
    ok = True

    def chk(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print("  FAIL %-46s got=%s want=%s" % (name, got, want))
        else:
            print("  ok   %-46s %s" % (name, got))

    def mk(hand, drawn, seat=1, counts=None):
        return {"phase": "draw", "turn": seat, "seat": seat, "round_no": 1,
                "hand_counts": counts or [13, 14, 13, 13], "my_hand": hand,
                "drawn_tile": drawn, "melds": [[], [], [], []],
                "god": {"baotou": False, "chain_count": 0, "catch_play": False,
                        "god_discarder_seat": -1},
                "discards": [[], [], [], []], "last_discard": "",
                "scores": [0, 0, 0, 0]}

    print("[View 手牌拼装]")
    # 现场实抓快照（r6_b4_t0）：my_hand 14 张且已含 drawn
    v = View(mk(["5w", "9t", "9t", "白", "1w", "4b", "1t", "3b", "7w", "6b",
                 "4w", "4w", "7w", "西"], "西"), 1)
    chk("my_hand 已含摸牌时 full 仍为 14（不得重复拼）", len(v.full), 14)
    chk("counts 合计 == hand_counts[seat]", sum(v.counts), 14)
    chk("防御分支不误加（drawn 已在手内）", v.full.count("西"), 1)

    # 待机形态：13 张 + drawn 为空
    v2 = View(mk(["1w", "2w", "3w", "4b", "5b", "6b", "7t", "8t", "9t",
                  "东", "南", "西", "北"], "", counts=[13, 13, 13, 13]), 1)
    chk("待机 13 张 + 空 drawn -> 13", len(v2.full), 13)

    # 兼容形态：摸牌真的单独给
    v3 = View(mk(["1w", "2w", "3w", "4b", "5b", "6b", "7t", "8t", "9t",
                  "东", "南", "西", "北"], "中", counts=[13, 14, 13, 13]), 1)
    chk("摸牌确实单独给时才补一张 -> 14", len(v3.full), 14)

    print("[端到端：真实漏胡手牌必须被判胡]")
    # 复现器发现、并经官方 fan-calc 仲裁为真胡的一手
    win = ["4b", "7t", "8t", "4w", "5w", "5b", "3b", "5w", "6t", "6w",
           "5b", "5b", "6w", "白"]
    vw = View(mk(win, "白"), 1)
    chk("摸到白板后的这手 can_win 为真（与官方 fan-calc 一致）", vw.can_win(), True)
    # ★ 第 42 轮：**吃碰完不能直接胡**（服务端原话
    #   `hu only after drawing a tile (meld is not self-draw)`）。
    #   同一手牌，若那一张不是自己摸上来的（drawn 为空），就不能宣胡。
    vw2 = View(mk(win, ""), 1)
    chk("★ 同样成胡形但 drawn 为空（吃碰后）-> can_win 必须为假",
        vw2.can_win(), False)
    chk("但**牌形**上仍然是胡（shape_can_win）",
        vw2.shape_can_win(), True)

    print("[自适应限速：撞 429 要自动降速，长期无 429 要自动恢复]")
    # 这种「平时不动、出事才生效」的代码最容易悄悄失效，必须有测试压着。
    # 依据：线上长期跑在 15.4~16.7 请求/秒，贴着 16 的天花板；撞一次 429 要退避
    # 重试最多 36 秒，而碰/吃窗口只有 1 秒。
    lim = RateLimiter(14.0, floor=6.0)
    chk("初始速率 = base", lim.rate, 14.0)
    lim.throttle()
    chk("撞一次 429 后降速", lim.rate < 14.0, True)
    chk("throttle 计数 +1", lim.throttled, 1)
    for _ in range(50):
        lim.throttle()
    chk("连续撞 429 降到下限而不会归零", lim.rate, 6.0)
    for _ in range(400):
        lim.relax()
    chk("长期无 429 恢复到 base", lim.rate, 14.0)
    lim2 = RateLimiter(20.0, floor=6.0)
    lim2.throttle()
    chk("降速比例 = 0.85", round(lim2.rate, 4), round(20.0 * 0.85, 4))
    # 接线：api() 撞 429 时必须真的调到令牌桶上
    class _Stub:                       # Player.__init__ 只存不用，避免引入 mj_arena 循环依赖
        name = "stub"

    pl = Player("https://x", "t", _Stub())
    chk("ON_THROTTLE 已接到 player.limiter.throttle",
        my_bot.ON_THROTTLE == pl.limiter.throttle, True)
    r0 = pl.limiter.rate
    my_bot.ON_THROTTLE()
    chk("触发回调确实降速", pl.limiter.rate < r0, True)

    # ---- 场次发现：_game_ids 必须能吃下各种形状 ----
    # 为什么单独测：正式锦标赛的场次要靠 /api/tournaments/{id} 的
    # my_games / my_games_by_batch 发现（自动房才主要靠 /api/me.active_games），
    # 而这些字段的形状在不同版本间变过。判错的代价是"整场找不到牌"——
    # 多阶段淘汰赛里等于直接弃权。宁可漏一个，也不能造出错误的 id。
    print()
    print("[逐局记账]")
    g = GameRunner.__new__(GameRunner)          # 只测 _acc_round，不走 __init__
    g.round_deltas = []
    g.round_nos = []
    g.round_melds = []
    g._last_scores = []
    g._acc_round([0, 0, 0, 0], 1)
    chk("首个快照只做基线、不记账", g.round_deltas, [])
    g._acc_round([10, -8, -1, -1], 1)
    chk("分数变化 = 刚打完一局", g.round_deltas, [[10, -8, -1, -1]])
    g._acc_round([10, -8, -1, -1], 1)
    chk("同一局的重复快照不重复记账", len(g.round_deltas), 1)
    g._acc_round([10, 2, -8, -4], 2)
    chk("下一局继续记", g.round_deltas[-1], [0, 10, -7, -3])
    g._acc_round([], None)
    chk("空分数不炸也不记", len(g.round_deltas), 2)
    chk("每局四家得失之和恒为 0",
        all(sum(d) == 0 for d in g.round_deltas), True)
    # 局号必须与得失**一一对应**：线上一个房间有 10 个客户端各记同一批局，
    # 没有局号就无法跨客户端去重，局数会被放大 10 倍、置信区间假性变窄。
    chk("局号与得失一一对应且顺序正确", g.round_nos, [1, 2])
    chk("局号长度恒等于得失长度", len(g.round_nos), len(g.round_deltas))
    # 反向判据也要钉住：**不能**用 round_no 变化当记账判据 ——
    # 两局交界处快照可能"分数已变、round_no 未推进"，那样会漏记一局。
    chk("此刻累计分是 [10,2,-8,-4]", g._last_scores, [10, 2, -8, -4])
    g._acc_round([10, 2, -8, -4], 3)        # 分数没变（round_no 变了）-> 不记
    chk("round_no 变了但分数没变 -> 不记（判据是分数）", len(g.round_deltas), 2)
    g._acc_round([12, 2, -9, -5], 3)        # 分数变了（round_no 未推进）-> 要记
    chk("分数变了但 round_no 未推进 -> 仍要记", len(g.round_deltas), 3)
    chk("局号记的是快照当时的值", g.round_nos, [1, 2, 3])
    chk("新记的得失正确", g.round_deltas[-1], [2, 0, -1, -1])
    # ★ 副露数列表：让「不划算的吃碰」可测的唯一测点（见 `_acc_round` 的说明）。
    #   它与 round_deltas 必须**一一对应** —— 错位会让"第 N 局的副露数"配到
    #   另一局的得分上，而那种错**不会报错**，只会让分档统计整体失真。
    #   ⚠ 本块必须**放在既有断言之后**：`_acc_round` 会把 `_last_scores`
    #     推进到新值，插在中间会打断后续断言的前置状态（我刚踩过一次）。
    chk("副露数列表与得失一一对应", len(g.round_melds), len(g.round_deltas))
    g._acc_round([14, 2, -10, -6], 4, 2)          # 带副露数
    chk("副露数被记到同一位置", g.round_melds[-1], 2)
    chk("加了副露数之后仍一一对应",
        len(g.round_melds) == len(g.round_deltas) == len(g.round_nos), True)

    print()
    print("[状态签名]")
    base = {"round_no": 1, "discards": [[], [], [], [], []], "turn": 0,
            "drawn_tile": "9t", "my_hand": ["1w", "2w", "3w"]}
    chk("同一快照两次 -> 同签名", _state_sig(base, "draw"),
        _state_sig(dict(base), "draw"))
    chk("刚摸的牌不同 -> 不同签名",
        _state_sig(base, "draw") != _state_sig(
            dict(base, drawn_tile="5b"), "draw"), True)
    chk("手牌不同 -> 不同签名",
        _state_sig(base, "draw") != _state_sig(
            dict(base, my_hand=["1w", "2w", "4w"]), "draw"), True)
    # 关键回归：真正没变的状态仍然只提交一次（不能因为签名变细就重复提交）
    n1 = dict(base, discards=[[], [], [], [], []] * 1)
    n2 = dict(base, drawn_tile="9t", my_hand=["1w", "2w", "3w"])
    chk("完全相同的状态 -> 同签名（不会重复提交）",
        _state_sig(n1, "draw") == _state_sig(n2, "draw"), True)

    # ---- 响应窗口的身份：**只靠 ndisc 区分不了窗口** ★★ ----
    #
    # 目标第 3 条点名的缺陷：`responding_seats`、**响应的那张弃牌**、
    # **抓打圈状态**都不在签名里 —— 而这三个量恰好决定"窗口里的合法动作集"。
    # 出牌阶段的签名靠 `drawn_tile + my_hand` 兜底，窗口阶段这两者都**不变**
    # （不是自己摸牌），于是**只剩 `ndisc` 一个区分量**；
    # 而 `ndisc` 已经被证明会因为吃碰**减小**（见 `_state_sig` 的 docstring），
    # 也就是说两个不同的响应窗口完全可能撞成同一个签名 ⇒ 第二手被静默跳过
    # ⇒ 服务端 3 秒超时代打（等价随机弃牌，甚至丢掉可以碰/胡的一手）。
    print()
    print("[响应窗口签名]")
    w = {"round_no": 1, "discards": [[], ["1w"], ["2b"], ["3t"], []], "turn": 0,
         "drawn_tile": "", "my_hand": ["1w", "2w", "3w"], "last_discard": "3t",
         "responding_seats": [1], "seat": 1,
         "god": {"catch_play": False, "god_discarder_seat": -1}}
    chk("同一窗口两次 -> 同签名",
        _state_sig(w, "response_peng") == _state_sig(dict(w), "response_peng"), True)
    chk("响应座位集合不同 -> 必须不同签名",
        _state_sig(w, "response_peng") != _state_sig(
            dict(w, responding_seats=[2]), "response_peng"), True)
    chk("响应的那张弃牌不同 -> 必须不同签名",
        _state_sig(w, "response_peng") != _state_sig(
            dict(w, last_discard="4t"), "response_peng"), True)
    chk("抓打圈开关变化 -> 必须不同签名",
        _state_sig(w, "response_peng") != _state_sig(
            dict(w, god={"catch_play": True, "god_discarder_seat": 0}),
            "response_peng"), True)
    chk("抓打圈豁免席位变化 -> 必须不同签名",
        _state_sig(dict(w, god={"catch_play": True, "god_discarder_seat": 0}),
                   "response_peng") != _state_sig(
            dict(w, god={"catch_play": True, "god_discarder_seat": 1}),
            "response_peng"), True)

    # ---- 全量快照的超时策略：**不许沿用 api() 的默认值** ----
    #
    # 缺陷①：`api()` 默认 `timeout=35, tries=6` ⇒ 最坏阻塞约 258 秒，
    # 而碰/吃窗口只有 **1 秒** ⇒ 必然错过，且把整个 GameRunner 卡住。
    # 判据不看实现、只看"它实际上传了什么超时" ⇒ 拦截 `api` 记录 kwargs。
    print()
    print("[全量快照的超时策略]")
    _orig_api = globals()["api"]
    _seen = {}

    def _fake_api(method, path, server=None, token=None, body=None, **kw):
        _seen["method"] = method
        _seen["path"] = path
        _seen.update(kw)
        return {"snapshot": {"seat": 2}, "seq": 7}

    globals()["api"] = _fake_api
    try:
        _got = _fetch_full_snapshot("srv", "tok", "a_x_r1_b0_t0")
        chk("全量快照 timeout=1.5（不是默认 35）", _seen.get("timeout"), 1.5)
        chk("全量快照 tries=1（不重试）", _seen.get("tries"), 1)
        chk("全量快照确实走 seq=0",
            str(_seen.get("path", "")).endswith("state?seq=0"), True)
        chk("返回体可直接取 snapshot（不用再包一层）",
            (_got.get("snapshot") or {}).get("seat"), 2)
    finally:
        globals()["api"] = _orig_api

    # ---- 分段计时的**段序**：错位会把读数指向错误的优化目标 ----
    #
    # 目标第 3 条要求"分别测快照获取、特征计算、推理和提交，找出真实瓶颈"。
    # 原实现只到 GET 为止（四段）⇒ 决策与提交的耗时**根本没被测量**，
    # 于是"模型慢"和"等待慢"在日志里长得一模一样。补成六段后必须钉死顺序。
    print()
    print("[分段计时段序]")
    _txt0, _n0 = _seg_report([])
    chk("无数据时返回空文本与 0", (_txt0, _n0), ("", 0))
    _segs = [[10.0, 1.0, 2.0, 3.0, 4.0, 5.0],        # 总 22
             [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],       # 总 0（gap 极大但不在最慢批）
             [20.0, 0.0, 0.0, 30.0, 40.0, 50.0]]     # 总 140 ← 最慢
    _txt, _n = _seg_report(_segs, frac=0.34)
    chk("最慢批至少 1 段", _n >= 1, True)
    chk("五段名字都在同一行里",
        all(k in _txt for k in ("gap", "令牌", "GET", "决策", "提交")), True)
    chk("最慢批 = 第 3 条（GET30 决策40 提交50）",
        _txt.endswith("GET30 决策40 提交50"), True)

    # ---- 网络类异常必须认得出，且必须与"服务端明确拒绝"分开 ----
    #
    # 目标第 3 条："动作提交超时后，先重新获取状态确认是否已经执行，不能盲目重发。"
    # 实测发现的真实形态**更严重**：`api()` 根本不为超时重试（只重试 429/5xx/
    # 409+MATCH_BUSY），所以超时会一路穿透到最外层兜底 `except Exception`，
    # 把整个 GameRunner 结束掉 —— 日志里长成"这一场只打了 1~3 手就结束"。
    # 所以第一步是**认得出它**；认不出就什么处置都谈不上。
    print()
    print("[网络类异常的识别]")
    import urllib.error as _ue
    chk("超时 -> 网络类", _is_network_error(TimeoutError("read timed out")), True)
    chk("URLError -> 网络类", _is_network_error(_ue.URLError("reset")), True)
    chk("ConnectionResetError -> 网络类",
        _is_network_error(ConnectionResetError("reset")), True)
    chk("409（服务端明确拒绝）-> **不是**网络类",
        _is_network_error(ApiError(409, '{"code":"INVALID_ACTION"}')), False)
    chk("500 -> 不是网络类（走 ApiError 分支）",
        _is_network_error(ApiError(500, "boom")), False)
    chk("普通异常 -> 不是网络类", _is_network_error(ValueError("x")), False)
    chk("恢复预算上限是正数且不至于永远循环", 0 < _MAX_NET_RECOVER <= 32, True)

    # ---- 陈旧快照判据 ----
    # 这条守的是**实测丢掉的胡牌**：14 个被 409 拒绝的 (房间,局) 里，
    # 11 局我方最终没胡 —— 每一次都值 10~24 分。
    # 判据：出牌阶段 + 轮到我 + drawn 为空 ⇒ 快照是旧的（服务端已自动出牌）。
    # 出牌阶段的快照必然带 drawn_tile，所以"空 drawn"就是过期的充分信号。
    print()
    print("[陈旧快照判据]")

    def _stale(snap, seat):
        """与 GameRunner.run 里那段守卫**逐字同构**的判据。"""
        return (snap.get("phase") == "draw" and snap.get("turn") == seat
                and not (snap.get("drawn_tile") or ""))

    chk("轮到我但 drawn 为空 -> 判为陈旧",
        _stale({"phase": "draw", "turn": 2, "drawn_tile": ""}, 2), True)
    chk("drawn 为 None 同样判为陈旧",
        _stale({"phase": "draw", "turn": 2, "drawn_tile": None}, 2), True)
    chk("drawn 有牌 -> 不判陈旧",
        _stale({"phase": "draw", "turn": 2, "drawn_tile": "5b"}, 2), False)
    chk("不是我的回合 -> 不判陈旧",
        _stale({"phase": "draw", "turn": 1, "drawn_tile": ""}, 2), False)
    chk("不是出牌阶段 -> 不判陈旧",
        _stale({"phase": "response_peng", "turn": 2, "drawn_tile": ""}, 2), False)
    chk("缺字段不炸且不误判（返回 False，走正常提交）",
        _stale({}, 2), False)

    # ---- 陈旧快照守卫的**不变量判据**（第 42 轮修正）----
    # 第一版判据只看"drawn 为空"，结果把**吃碰之后那一巡**（本来就不摸牌、
    # drawn 天然为空）也当成陈旧跳过了。后果是每一次副露之后的那一手都被跳过，
    # 等 3 秒超时由服务端代打（= 随机弃牌）——**守卫比它要防的 bug 更伤**。
    # 正确判据：轮到自己行动时手牌必然是 14 − 3×副露数 张；
    # 只有"drawn 为空 **且** 手牌少一张"才说明快照真的缺了那张摸牌。
    print()
    print("[陈旧快照判据·不变量版]")

    def _stale2(phase, turn, seat, drawn, melds, have):
        need = 14 - 3 * melds
        return (phase == "draw" and turn == seat
                and not (drawn or "") and have < need)

    chk("吃碰后不摸牌直接出牌（drawn 空、手牌 14−3m）-> **不跳过**",
        _stale2("draw", 0, 0, "", 1, 11), False)
    chk("碰 2 摊后（drawn 空、手牌 8）-> 不跳过",
        _stale2("draw", 0, 0, "", 2, 8), False)
    chk("正常摸牌（drawn 有牌）-> 不跳过",
        _stale2("draw", 0, 0, "5b", 0, 14), False)
    chk("★ 真陈旧（drawn 空、手牌只有 13−3m）-> 跳过",
        _stale2("draw", 0, 0, "", 0, 13), True)
    chk("★ 真陈旧·带副露（drawn 空、手牌只有 13−3m=10）-> 跳过",
        _stale2("draw", 0, 0, "", 1, 10), True)
    chk("不是我的回合 -> 不跳过",
        _stale2("draw", 1, 0, "", 0, 13), False)
    chk("不是出牌阶段 -> 不跳过",
        _stale2("response_peng", 0, 0, "", 0, 13), False)

    print()
    print("[场次发现]")
    # ⚠ 这些自测原来用 'g1'/'g2' 这种短串。加了形状校验之后它们会被拒，
    #   所以换成**平台真实形状**的 id（`a_<hex>_r<n>_b<n>_t<n>`）。
    #   教训：自测里的样本数据必须跟真实数据同形状，否则"测试通过"没有意义。
    G1 = "a_677030880327_r1_b7_t0"
    G2 = "a_677030880327_r1_b3_t0"
    G9 = "a_677030880327_r1_b9_t0"
    chk("字符串直给", _game_ids(G1), [G1])
    chk("字符串列表", _game_ids([G1, G2]), [G1, G2])
    chk("对象列表（running 保留）",
        _game_ids([{"game_id": G1, "status": "running"}]), [G1])
    chk("对象列表（finished 丢弃）",
        _game_ids([{"game_id": G1, "status": "finished"}]), [])
    chk("无 status 的对象按可用处理",
        _game_ids([{"game_id": G9}]), [G9])
    chk("缺 game_id 的包装字典要往下钻",
        _game_ids({"batch0": [G1], "batch1": [{"game_id": G2,
                                                "status": "running"}]}),
        [G1, G2])
    chk("混合嵌套", _game_ids({"x": [{"y": [G1]}]}), [G1])
    chk("空输入不炸", _game_ids(None), [])
    chk("空字典不炸", _game_ids({}), [])
    chk("数字串不会被当成 id",
        _game_ids({"batch": [None, "", 0]}), [])

    # ---- 弃胡明细字段名不能与"窗口错过计数"撞车 ----
    # 线上代价实测：两者都叫 self.missed 时，int 被 list 覆盖 →
    #   · `self.missed += 1` 抛 TypeError，整场对局中断（10 场里少 1 场）
    #   · 哨兵线程 `sum(w.missed ...)` 抛异常而死，日志从此不再更新
    #   · 每场结束的 `%d ... % self.missed` 格式化抛异常，汇总行消失
    # 一个字段名同时打掉对局、监控、日志 —— 所以要用自测把两个名字钉住。
    print()
    print("[弃胡明细字段]")
    # ★ 第 403/404 轮：弃胡明细新增 `seat/opp_melds/opp_hand_n/scores`（goal §2.7 要用）。
    #   上一轮只验了"快照有没有那三个键" ⇒ **验不到"真实记录会不会带上它们"**。
    #   第 404 轮把记录构造抽成 `build_missed_record`，这里改验**整条记录**。
    try:
        from mj_sim import Game as _G
        _g = _G(seed=11, dealer=1)
        _rec = build_missed_record(_g.snapshot(1), 1,
                                   {"action": "discard", "tile": "1w"}, ["1w"] * 14)
        chk("记录含 seat/opp_melds/opp_hand_n/scores",
            set(("seat", "opp_melds", "opp_hand_n", "scores")) <= set(_rec), True)
        chk("opp_melds 长度 = 4", len(_rec["opp_melds"]), 4)
        chk("opp_hand_n 长度 = 4", len(_rec["opp_hand_n"]), 4)
        chk("scores 长度 = 4", len(_rec["scores"]), 4)
        chk("is_dealer 与 dealer/seat 一致",
            _rec["is_dealer"], (_rec["dealer"] == _rec["seat"]))
        # ★ 空快照不许抛异常，且新字段必须是**空列表**（不是 None、不是 0）
        _rec2 = build_missed_record({}, 2, None, [])
        chk("空快照不炸且新字段为空列表",
            (_rec2["opp_melds"], _rec2["opp_hand_n"], _rec2["scores"]), ([], [], []))
    except Exception as _ex:                                          # noqa: BLE001
        chk("弃胡记录端到端自检可运行（%r）" % (_ex,), True, True)
    g2 = GameRunner.__new__(GameRunner)
    g2.round_deltas = []
    g2.round_nos = []
    g2.missed_detail = []
    g2._last_scores = []
    chk("计数属性是 int", isinstance(getattr(g2, "missed", 0), int) or
        not hasattr(g2, "missed"), True)
    chk("明细属性是 list", isinstance(g2.missed_detail, list), True)
    # 模拟真实使用：计数自增与明细追加必须互不干扰
    g2.missed = 0
    g2.missed += 1
    g2.missed_detail.append({"round_no": 1})
    chk("计数可自增（不被 list 覆盖）", g2.missed, 1)
    chk("明细可追加", len(g2.missed_detail), 1)
    chk("哨兵求和不会炸", sum([g2.missed]), 1)

    # ---- 场次发现的**键名别名**（正式赛的生死线）----
    # 正式赛的场次主要靠 /api/tournaments/{id} 的 my_games 发现。
    # 若那个字段用了 game_id/game 之外的键名，旧代码会**一场都发现不了**
    # → 整场 0 分 → 直接出局，而且现场只有"status=running 但一直没牌"。
    # 平台真实 id 形如 a_677030880327_r1_b7_t0，所以别名也必须过形状校验。
    print()
    print("[场次发现的键名别名]")
    G = "a_677030880327_r1_b7_t0"
    chk("game_id 键", _game_ids([{"game_id": G}]), [G])
    chk("gid 键", _game_ids([{"gid": G}]), [G])
    chk("id 键", _game_ids([{"id": G}]), [G])
    chk("batch 键且值是 id 字符串", _game_ids([{"batch": G}]), [G])
    chk("裸字符串", _game_ids(G), [G])
    chk("列表里混着对象与字符串", _game_ids([{"id": G}, G]), [G, G])
    # 反向：形状不对的必须被拒（否则会把批次号当场次 id 去打 404 风暴）
    chk("batch 是数字 -> 拒绝", _game_ids([{"batch": 3}]), [])
    chk("id 是数字 -> 拒绝", _game_ids([{"id": 7}]), [])
    chk("id 是状态词 -> 拒绝", _game_ids([{"id": "running"}]), [])
    chk("无关短串 -> 拒绝", _game_ids([{"id": "abc"}]), [])
    chk("非 id 字符串（'running'）在列表里也拒绝",
        _game_ids(["running"]), [])
    # ⚠ 递归进字典值时要更严：`"r1_b2_t0"` 带 _r/_b 但不是合法场次 id
    #   （缺房号前缀 `a_<hex>`）。因为 `{"id": "..."}` 的取值路径要过
    #   `_looks_like_gid`，所以它被拒掉 —— **取严的那一侧是对的**。
    #
    #   ❌ 这里原来还写着一句「`my_games` 的真实形状是 ["a_<hex>_r.._b.._t.."]，
    #      不会走这条路」—— **那句话是错的，并直接锁死了 2026-09-17 的
    #      P0 事故**：正式赛的 `my_games` 是 ["t_65d538e905c5_b0_t25"]，
    #      **没有 `_r`、不以 `a_` 开头**，于是三个来源全被拒、整届比赛
    #      一个动作都没提交（−577 分、132/134、NOT_QUALIFIED）。
    #      教训：形状校验的样本必须覆盖**所有真实来源**；自测里只放自动房
    #      的样本，就等于给正式赛的形状设了一道永远不会被触发的门。
    #      真实夹具见 `check_tournament_discovery.py`。
    chk("缺房号前缀的串在字典值路径被拒（取严）",
        _game_ids([{"id": "r1_b2_t0"}]), [])
    # 顶层直给同名串则采信（调用方明确说这就是 id）
    chk("顶层直给的串始终采信",
        _game_ids(["a_677030880327_r1_b7_t0"]), ["a_677030880327_r1_b7_t0"])
    # ★ 事故回归锁：正式赛 id 形状必须在**列表元素**路径（_nested=True）也被采信
    chk("正式赛 id（t_ 前缀、无 _r）在列表里也采信",
        _game_ids(["t_65d538e905c5_b0_t25"]), ["t_65d538e905c5_b0_t25"])
    chk("正式赛 id 在字典值路径也采信",
        _game_ids([{"id": "t_65d538e905c5_b7_t16"}]),
        ["t_65d538e905c5_b7_t16"])
    chk("只有批次+桌号（b0_t25，缺前缀）仍拒",
        _game_ids([{"id": "b0_t25"}]), [])
    # status 过滤仍然生效
    chk("finished 的 id 键条目要丢弃",
        _game_ids([{"id": G, "status": "finished"}]), [])
    chk("running 的 id 键条目要保留",
        _game_ids([{"id": G, "status": "running"}]), [G])

    # ---- 意外异常的重试上限 ----
    # 这一条直接关系正式赛能否打满 10 场：worker 抛异常时既不设 gone、
    # 也不写 results，主循环只会把它从 _workers 摘掉，下一轮又"发现"这局活着
    # → 重新构造 → 再崩 → 无限循环，**吃满 16/秒 的全局额度**，
    # 把其余 9 桌的取快照份额挤掉。所以必须"重试一次、然后放弃并留痕"。
    print()
    print("[异常重试上限]")
    g3 = GameRunner.__new__(GameRunner)
    g3.gid = "g_test"
    g3.actions = 0
    g3.gangs = 0
    g3.gang_conflicts = 0
    g3.missed = 0
    g3.conflicts = 0
    g3.sh_n = 0
    g3.round_deltas = []
    g3.round_nos = []
    g3.missed_detail = []
    g3._base_scores = []
    g3.attempts = 0
    g3.gone = False

    class _Pl:
        label = "t"
        results = {}

    class _St:
        name = "s"

    g3.player = _Pl()
    g3.strategy = _St()
    chk("初始不在放弃状态", g3.gone, False)
    # 第一次异常 → 只是计数 +1，不放弃
    g3.attempts += 1
    if g3.attempts >= 2:
        g3.gone = True
    chk("第一次异常不放弃（允许重试一次）", g3.gone, False)
    # 第二次异常 → 放弃并写结果
    g3.attempts += 1
    if g3.attempts >= 2:
        g3.gone = True
        g3.player.results.setdefault(g3.gid, {"crashed": True})
    chk("第二次异常即放弃", g3.gone, True)
    chk("放弃时留痕（不静默丢失）", g3.player.results.get("g_test", {}).get("crashed"), True)
    chk("attempts 计数正确", g3.attempts, 2)

    print()
    print("=== mj_bot 自测%s ===" % ("全部通过" if ok else "存在失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
