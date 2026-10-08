#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自对弈模拟器 —— 把对局从真实服务器搬到本地，快 1000 倍以上。

为什么需要它：
  · 真实服务器速度 = 10 局 / 210 秒 = 0.048 局/秒。要测一个 1.3 个百分点的
    自摸率差异需要约 39 小时连续对战（已量化）。
  · 本地模拟器按已实测的算法耗时（shanten 0.141ms / ukeire 5.16ms）估算可达
    数十到数百局/秒 —— 同一个测量降到秒级。
  · 它也是任何强化学习方案的硬前置：RL 要 10^5~10^6 局，在真实房间上是 24 天起。

关键设计：**快照格式与平台同构**。Game.snapshot(seat) 产出的字典与
GET /api/games/{id}/state 的 snapshot 字段同名同义，于是：
  · 现有策略 Evolved 不用改一行就能跑（View 直接吃这个快照）
  · 验证时可以拿真实事件流逐字段比对（见 sim_validate.py）

规则实现范围（对应平台指南）：
  · 136 张，白板 = 财神（百搭）
  · 只能自摸，不能点炮；禁止抢杠胡
  · 抓打圈：打出财神后，其余玩家不能吃碰明杠，且只能打刚摸的牌
  · 吃最多 2 摊；碰/杠不限
  · 牌墙最后 10 墩（20 张）保留不摸；最后 10 墩内禁止杠
  · 庄家连庄：流局庄家连庄
"""
import random
import sys

from mj_core import (NTILE, WILD, can_win, fan_multiple, is_baotou, melds_of,
                     seven_pairs_mult as _spm, tile_code,
                     tile_index, to_counts)
from mj_core import chi_tiles as core_chi_tiles

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEADWALL = 20          # 牌墙最后 10 墩（20 张）保留不摸
DRAWABLE = 136 - 13 * 4 - DEADWALL      # = 64
CHI = "chi"
PENG = "peng"
GANG = "gang"


class IllegalAction(Exception):
    pass


class Game:
    """一局（Rounds=1）。四人、一局一胡。"""

    def __init__(self, seed=None, dealer=0, base=1, window_policy=None,
                 deferred=False, miss_rate=None, carry_scores=None,
                 dealer_streak=0, round_no=1, rounds=None):
        self.rng = random.Random(seed)
        self.base = base
        self.window_policy = window_policy   # (snapshot, seat, kind) -> bool
        self.dealer = dealer
        # miss_rate：按座位的「响应窗口漏接率」。0 = 完美客户端（默认，行为不变）。
        #
        # ⚠ 2026-09-16 更正：这个参数当初是照着「线上对手漏接 87%~99.6%」建的，
        # 而那个数字**是假的**。平台对每个窗口都给未响应的座位记一条 timeout，
        # 而漏接率 = timeout /(timeout + 动作 + 显式 pass)。走马一次 pass 都没发过
        # （过=0、timeout=2736），于是漏接率 95%；但它的副露是全场最高
        # （14.1 次/场）—— 客户端真坏掉不可能一边漏 95% 的窗口一边碰出最多的副露。
        # 真实牌桌是**四家客户端都正常**，差别只在发不发 pass。
        # 所以：涉及牌桌密度的结论请用 miss_rate=0（循环赛 S1/S2/S3 就是这么做）；
        # exp_reliable.py 的「1 可靠 + 3 漏接 0.9」是错的牌桌模型，只作历史留档。
        if miss_rate is None:
            self.miss_rate = [0.0, 0.0, 0.0, 0.0]
        elif isinstance(miss_rate, (int, float)):
            self.miss_rate = [float(miss_rate)] * 4
        else:
            self.miss_rate = [float(x) for x in miss_rate]
            while len(self.miss_rate) < 4:
                self.miss_rate.append(0.0)
        self.miss_rolled = 0                 # 实际掷出的漏接次数，供核对
        # deferred=True：窗口不立即结算，而是暴露成 phase=response_* 等外部显式
        # 结算。线上窗口是「固定走满 1s 的真实时间」，同步 _poll_window 把它压成 0，
        # 于是模拟器系统性高估副露（实测 1.65 vs 线上 0.40 次/人/局）。
        self.deferred = deferred
        self.responding_seats = []
        self._window_kind = None
        self._window_discarder = -1
        pool = [t for t in range(NTILE) for _ in range(4)]
        self.rng.shuffle(pool)
        self.hands = [to_counts([]) for _ in range(4)]
        for s in range(4):
            for _ in range(13):
                self.hands[s][pool.pop()] += 1
        self.wall = pool                       # 剩余牌墙（含死墙）
        self.melds = [[], [], [], []]
        self.discards = [[], [], [], []]
        self.last_discard = ""
        self.turn = dealer
        self.phase = "deal"
        self.god = {"baotou": False, "chain_count": 0, "catch_play": False,
                    "god_discarder_seat": -1,
                    # 引擎内部用：官方快照只暴露 chain_count，不暴露 piao。
                    # 「4 个白板」判据是「手留白 + 链内飘出的白 >= 4」，需要它。
                    "piao_count": 0,
                    # ★★★ 第 349 轮修正：**动作链是每人一份，不是全局一份**。
                    #   原来 `chain_count` / `piao_count` 是**整局共用一个标量**，
                    #   而 `discard()` 对**任何** seat 都调用 `_bump_chain`
                    #   ⇒ 任何对手打出一张普通牌就把**我的**飘清掉。
                    #   后果（`p321_chain_scope.py`，11,520 局实测）：
                    #     飘事件 63 次，**被他人清掉 63 = 100%**，我方 2,824 手胡牌里
                    #     **带飘 0 手** ⇒ **财飘在本模拟器里结构性不可能**。
                    #   而打出财神后要过 3 个对手才轮到我摸牌 ⇒ 链必然已断。
                    #   线上真值（`p315/p319`）：真场 396 手财飘、0.0068/局，且 100% 带爆头
                    #   ⇒ 真规则的链**不会**被他人断掉 ⇒ 必须按座位分开。
                    #   ⚠ `chain_count`/`piao_count` 两个扁平键**保留**为
                    #     「最近一次动作的那个座位」的镜像（`snapshot()` 会再按观察者覆盖），
                    #     这样既有的自测与读取点不必全部改写。
                    "chain": [0, 0, 0, 0],
                    "piao": [0, 0, 0, 0]}
        self.scores = [0, 0, 0, 0]
        # ---- 房间层上下文（第 145 轮新增，**默认全 0 ⇒ 行为逐字不变**）----
        # 为什么需要：`self.scores` 只在**本局结算时**才被写（见 `_settle_win`），
        # 所以任何一个**决策点**上 `snapshot()["scores"]` 都是 [0,0,0,0] ——
        # 实测确认过：`features_full.context()` 的分数段在真实单局里**恒为 0**。
        # 而扩写后的 goal §3 明确要求策略感知「当前局分数」与「与其他玩家的分差」，
        # 那是"风险感知 / 什么时候停止贪番"的**唯一输入**。
        # ⇒ 让房间层把**累计分**与**连庄长度**带进来；不给就与以前逐字一致。
        self.carry = [float(x) for x in (carry_scores or (0, 0, 0, 0))][:4]
        while len(self.carry) < 4:
            self.carry.append(0.0)
        self.dealer_streak = int(dealer_streak or 0)
        # 局号：原来 `snapshot()` 把它**硬编码成 1** ⇒ 策略永远不知道"打到第几局了"，
        # 而 goal §3 要「当前巡目 / 局号」。默认 1 ⇒ 单局夹具逐字不变。
        self.round_no = self.rounds = None
        try:
            self.round_no = int(round_no or 1)
        except (TypeError, ValueError):
            self.round_no = 1
        try:
            self.rounds = int(rounds) if rounds else None
        except (TypeError, ValueError):
            self.rounds = None
        self.finished = False
        self.winner = -1
        self.draw = False
        self.turns = 0
        self._drawn = ""
        self.last_fan = 0          # 本局胡牌番数（供测量脚本统计番型分布）
        self.win_wild = 0          # 胡牌瞬间手留白板数（判断爆头/4白板可达性）
        # 本局赢家是否七对分支（0 = 平胡，否则 2/4/8/16）。
        # 为什么必须可观测：线上真值（36 房 2846 手，按 user_id 归位）显示
        # 对手 2.7% 的胡牌是七对，而我们只有 0.1% —— 27 倍差距。
        # 差距这么大，要么是策略主动放弃了这条路，要么是实现根本走不到；
        # 没有这个字段就无法在夹具里区分二者，也无法验证任何修复。
        self.last_chiitoi = 0
        # 观察钩子：play() 在每次「该某家决策」前调用 observer(game, seat)。
        # 有了它，测量脚本不必复制一份对局循环（复制出来的那份必然与真引擎漂移）。
        self.observer = None
        # 庄家第 14 张为发牌直抽；抽完即进入出牌阶段。
        # 早先漏了这一步，phase 停在 "deal"，play() 第一句就 break ——
        # 每局 0 手直接退出，却因为"跑得飞快"被误读成高吞吐。
        if not self._draw(dealer):
            self.phase = "finished"
        else:
            self.phase = "draw"

    # ---------------------------------------------------------- 基础量

    @property
    def wall_left(self):
        return len(self.wall)

    def _meld_sets(self, s):
        return len(self.melds[s])

    def _meld_chi(self, s):
        return sum(1 for m in self.melds[s] if m["kind"] == CHI)

    # ---------------------------------------------------------- 摸打

    def _draw(self, seat):
        if len(self.wall) <= DEADWALL:
            self._settle_draw()
            return False
        t = self.wall.pop()
        self.hands[seat][t] += 1
        self._drawn = tile_code(t)
        return True

    def discard(self, seat, code):
        if seat != self.turn or self.phase not in ("deal", "draw"):
            raise IllegalAction("discard out of turn")
        t = tile_index(code)
        if self.hands[seat][t] <= 0:
            raise IllegalAction("tile not in hand")
        # 抓打圈：受限方只能打刚摸的那张
        if self._restricted(seat) and code != self._drawn:
            raise IllegalAction("catch_play: must discard drawn tile")
        self.hands[seat][t] -= 1
        self.discards[seat].append(code)
        self.last_discard = code
        self.turns += 1
        # 抓打圈的开启与解除（规则 §1.1）：
        #   · 任何人打出财神 -> 开圈，本人为圈主（豁免方）
        #   · 圈主再次出牌：仍是财神 -> 圈以本人重启；别的牌 -> 链断且圈解除
        #   · 非圈主出牌不影响圈
        if t == WILD:
            self.god["catch_play"] = True
            self.god["god_discarder_seat"] = seat
        elif self.god["catch_play"] and self.god["god_discarder_seat"] == seat:
            self.god["catch_play"] = False
            self.god["god_discarder_seat"] = -1
        self._bump_chain(seat, t)
        self._after_discard(seat)

    def _bump_chain(self, seat, t):
        """动作链：飘与杠可连续可组合，每个动作 ×2；打出非飘非杠的牌就断链。

        规则原文（指南 §番型）：财飘 = **爆头态打出财神**（打完仍听任意），
        「打出非飘非杠的牌（含非爆头态打白板）→ 链断重新计数」。

        piao 必须跟 chain 同步清零：官方「4 个白板」= 手留白 + **链内**飘出的白，
        而 chain 请求体要求 piao <= count。oracle 的 detail 也印证了这点 ——
        count=4/piao=4 叫「连飘×4」，即四次**连续**飘，中间不能有普通出牌。
        """
        if t == WILD and is_baotou(self.hands[seat], self._meld_sets(seat)):
            self.god["chain"][seat] += 1
            self.god["piao"][seat] += 1
        else:
            self.god["chain"][seat] = 0
            self.god["piao"][seat] = 0
        # 镜像：扁平键跟随"最近动作的那个座位"（snapshot() 再按观察者覆盖）
        self.god["chain_count"] = self.god["chain"][seat]
        self.god["piao_count"] = self.god["piao"][seat]

    def _restricted(self, seat):
        g = self.god
        return g["catch_play"] and g["god_discarder_seat"] != seat

    def _after_discard(self, discarder):
        """散出碰/吃窗口；无人响应则下一家摸牌。"""
        if self.deferred:
            self._open_window(discarder)
            return
        resp = self._poll_window(discarder)
        if resp is not None:
            seat, act = resp
            self._apply_meld(seat, act, discarder)
            return
        self._advance(discarder)

    # -------------------------------------------------- 延迟窗口（deferred）

    def _eligible(self, discarder):
        """规则层响应资格（抓打圈 / 财神不可吃碰）。"""
        t = tile_index(self.last_discard)
        if t == WILD:
            return t, []
        eligible = [s for s in range(4) if s != discarder]
        if self.god["catch_play"]:
            # 圈内只有豁免方（打财神者本人）能响应
            eligible = [s for s in eligible
                        if s == self.god["god_discarder_seat"]]
        return t, eligible

    def _open_window(self, discarder):
        """把碰/吃窗口暴露成 phase=response_*，等外部调 window_act/timeout。

        按平台纪律分两级：先开碰窗口（全部可碰座位竞争），碰窗口空过再开吃窗口
        （仅下家）。两个窗口各占一次往返 —— 这也是线上请求量的主要来源。
        """
        self._window_discarder = discarder
        t, eligible = self._eligible(discarder)
        peng = [s for s in eligible if self.hands[s][t] >= 2]
        if peng:
            self._set_window(PENG, peng)
            return
        self._open_chi(discarder)

    def _open_chi(self, discarder):
        t, eligible = self._eligible(discarder)
        nxt = (discarder + 1) % 4              # 吃：仅下家，且吃摊 < 2
        if nxt in eligible and self._meld_chi(nxt) < 2 and t < 27 \
                and self._chi_pair(nxt, t):
            self._set_window(CHI, [nxt])
            return
        self._advance(discarder)

    def _set_window(self, kind, seats):
        self._window_kind = kind
        self.responding_seats = list(seats)
        self.phase = "response_peng" if kind == PENG else "response_chi"

    def _clear_window(self):
        kind, self._window_kind = self._window_kind, None
        self.responding_seats = []
        return kind

    def window_kind(self):
        return self._window_kind

    def window_act(self, seat, kind):
        """外部结算：某座位在窗口内响应。True = 已执行该副露。

        ★ 第 363 轮：**碰窗口里允许 `kind=gang`**（明杠）。
        门控与 `_poll_window` 一致：手上必须有 **≥3 张**同牌。
        其余行为逐字不变（`kind` 不匹配仍返回 False）。
        """
        if self._window_kind is None or seat not in self.responding_seats:
            return False
        if kind != self._window_kind:
            if not (kind == GANG and self._window_kind == PENG):
                return False
            t = tile_index(self.last_discard)
            if self.hands[seat][t] < 3:
                return False
        d = self._window_discarder
        self._clear_window()
        self._apply_meld(seat, kind, d)
        return True

    def window_pass(self, seat):
        """某座位明确放弃响应（线上 pass）。

        与 window_timeout 的区别是真实的：平台分别记 pass 和 timeout 两种事件，
        改造前线上是 1344 timeout vs 365 pass。沉默到期记 timeout，明确放弃记 pass。
        """
        if self._window_kind is None or seat not in self.responding_seats:
            return False
        self.responding_seats.remove(seat)
        if not self.responding_seats:
            self.window_timeout()
        return True

    def window_timeout(self):
        """窗口无人接住（或已超时）。碰窗口空过则续开吃窗口。"""
        kind = self._clear_window()
        if kind is None:
            return False
        if kind == PENG:
            self._open_chi(self._window_discarder)
        else:
            self._advance(self._window_discarder)
        return True

    def _poll_window(self, discarder):
        """按平台纪律收集响应：碰窗口先于吃窗口。返回 (seat, action) 或 None。

        资格由 self.window_policy(snapshot, seat, phase) 决定 —— 策略说了算，
        模拟器只负责规则层过滤（抓打圈、吃摊上限、财神不可被吃碰）。
        缺省 window_policy 为 None = 一律不响应。
        """
        t = tile_index(self.last_discard)
        if t == WILD:                          # 财神本身不可被吃碰杠
            return None
        eligible = [s for s in range(4) if s != discarder]
        if self.god["catch_play"]:
            # 圈内只有豁免方（打财神者本人）能响应
            eligible = [s for s in eligible
                        if s == self.god["god_discarder_seat"]]
        for s in eligible:                     # 碰窗口先于吃窗口
            n = self.hands[s][t]
            if n < 2:
                continue
            a = self._ask_act(s, PENG)
            # ★★★ 第 363 轮：**明杠**（碰窗口里提交 `gang`）。
            #   改动前这里是 `self.hands[s][t] >= 2 and self._asks(s, PENG)`，
            #   而 `_asks` 要求 `action == kind` ⇒ 策略返回的 `{"action":"gang"}`
            #   **被判为"不响应"**，然后窗口继续开吃 / 下一家摸牌 ——
            #   也就是**静默当成 pass**（`RL_TRAINER.md` §2B / `audit_minggang.py`
            #   都记着这条；BC 数据审计里它是 **0.3135% 的决策点**）。
            #   ⚠ 而线上我们**确实在做**：真值事件流 35,976 局里
            #     我方明杠 **0.0308 次/局**、对手 0.0252（`analyze_seat_actions.py`）。
            #   ⇒ 模拟器缺这条通路时，**所有夹具读数在一个我们真会做的动作上是错的**。
            #   规则依据（真值事件流实测，**1,011 个杠事件 100% 同座补牌**：
            #     ming 495 / bu 356 / an 160）⇒ 明杠**和暗杠、补杠一样给一张补牌**。
            if a == GANG and n >= 3:
                return s, GANG
            if a == PENG:
                return s, PENG
        nxt = (discarder + 1) % 4              # 吃：仅下家，且吃摊 < 2
        if nxt in eligible and self._meld_chi(nxt) < 2 and t < 27:
            if self._chi_pair(nxt, t) and self._asks(nxt, CHI):
                return nxt, CHI
        return None

    def _ask_act(self, seat, kind):
        """策略在这个窗口给出的**动作名**（`peng`/`chi`/`gang`）；没表态返回 None。

        ★ 第 363 轮新增。原实现只有 `_asks()` 返回 bool（`action == kind`），
        于是**碰窗口里的明杠被静默当成 pass**（见 `_poll_window`）。
        兼容两种 `window_policy`：返回 `dict`（`play()` 走的就是这条）
        或返回 `bool`（deferred 模式下外部驱动者可能这么写）。
        """
        if self.window_policy is None:
            return None
        # 客户端可靠性：漏接的座位根本不看窗口（线上对手漏接 87%~99.6%）。
        # 掷骰放在 window_policy 之前 —— 漏接意味着策略**没有机会**表态，
        # 而不是「看了之后决定不响应」，这两者在副露密度上效果相同，
        # 但前者才不会把「策略本意」记进 misses 统计里。
        if self.miss_rate[seat] > 0.0 and self.rng.random() < self.miss_rate[seat]:
            self.miss_rolled += 1
            return None
        try:
            a = self.window_policy(self.snapshot(seat), seat, kind)
        except Exception:                      # noqa: BLE001
            return None
        if isinstance(a, dict):
            return a.get("action")
        # ★★★ 第 363 轮（同一个改动里我自己踩的坑，值得留着）：
        #   `window_policy` 有**两种**实现，返回值**不是同一种东西**：
        #     · `play()` 装的 `_ask_window` 已经把动作**解成字符串**
        #       （`"peng"` / `"chi"` / `"gang"` / None）；
        #     · deferred 模式的外部驱动者可能返回 **bool**。
        #   第一版只处理了 dict 和 bool ⇒ 字符串落进 `return kind if a else None`
        #   ⇒ **`"gang"` 被当成"愿意"而返回 `kind`（=`"peng"`）** ——
        #   于是明杠**看起来接上了、其实每次都被降级成碰**。
        #   症状：`diag_minggang_path.py`（引擎层）报 37/37 答 `peng`，
        #         而 `diag_minggang_policy.py`（策略层）报 37/37 答 `gang`。
        #   ★ 教训：**"同一件事在两个层各自量一次"是唯一能抓到它的办法** ——
        #     只看一层的话，两边都自洽。与 §Z.357「存在≠可达」同族：
        #     **"接了线" ≠ "线里传的是那个值"。**
        if isinstance(a, str):
            return a
        return kind if a else None

    def _asks(self, seat, kind):
        """策略是否愿意执行 `kind`。⚠ 新代码请直接用 `_ask_act`。

        ★ 第 363 轮：改为委托给 `_ask_act`，并**保留原语义**
        （碰窗口里回答 `gang` 仍然算"愿意响应"，与改动前一致），
        这样 `_poll_window` 之外的调用点行为不变。
        """
        a = self._ask_act(seat, kind)
        return a == kind or (kind == PENG and a == GANG)

    def _chi_pair(self, seat, t):
        """吃牌用哪两张 —— **必须与策略用同一份实现**（`mj_core.chi_tiles`）。

        以前这里是 `mj_policy.chi_tiles` 的**手抄副本**（同样是"取第一个组合"）。
        两份实现必然分叉，而"模拟器与线上不一致"是本项目最贵的一类 bug
        （模拟器里策略返回 gang 被 _try_gang 拒掉那次就是静默失真）。
        现在两边都调 `mj_core.chi_tiles`，`meld_count` 传本手已有副露数，
        于是 `CHI_PICK=1` 的"取向听最小组合"在模拟器与线上**同时生效**。
        """
        return core_chi_tiles(self.hands[seat], tile_code(t),
                              self._meld_sets(seat))

    def _apply_meld(self, seat, kind, discarder):
        t = tile_index(self.last_discard)
        if kind == GANG:
            # ★★★ 第 363 轮：**明杠**（吃别人的第 4 张）。与碰的差别是
            #   规则性的、不是口味：碰用 2 张、不摸牌、还剩 1 张死牌；
            #   明杠用 3 张、**摸一张补牌**、链 +1。
            #   真值依据：事件流 1,011 个杠事件（ming 495 / bu 356 / an 160）
            #   **100% 紧随同座 `tile_drawn`** ⇒ 补牌这条通路是规则，不是猜测。
            self.hands[seat][t] -= 3
            self.melds[seat].append({"kind": GANG,
                                     "tiles": [self.last_discard] * 4})
            self.god["chain"][seat] += 1
            self.god["chain_count"] = self.god["chain"][seat]
            self.discards[discarder].pop()     # 被杠的牌离开牌河
            self.turn = seat
            self.phase = "draw"
            self._drawn = ""
            self._gang_draw(seat)              # 补牌（牌墙见底则流局，见 _draw）
            return
        if kind == PENG:
            self.hands[seat][t] -= 2
            self.melds[seat].append({"kind": PENG,
                                     "tiles": [self.last_discard] * 3})
        else:
            pair = self._chi_pair(seat, t)
            for code in pair:
                self.hands[seat][tile_index(code)] -= 1
            self.melds[seat].append({"kind": CHI,
                                     "tiles": sorted(pair + [self.last_discard])})
        self.discards[discarder].pop()         # 被吃碰的牌离开牌河
        # 吃碰不改抓打圈状态（规则：圈的解除只发生在圈主再次出牌时）
        self.turn = seat
        self.phase = "draw"
        self._drawn = ""

    def _advance(self, from_seat):
        self.turn = (from_seat + 1) % 4
        if not self._draw(self.turn):
            return
        self.phase = "draw"

    # ---------------------------------------------------------- 胡 / 流局

    def try_hu(self, seat):
        if seat != self.turn or self.phase != "draw":
            return False
        c = self.hands[seat]
        if not can_win(c, self._meld_sets(seat)):
            return False
        # 爆头：把刚摸到的那张拿掉后仍「听任意」= 爆头 ×2（官方 fan-calc 的
        # baotou 字段已逐例校验过 is_baotou 的判定口径）。
        self.god["baotou"] = self._is_baotou_on_win(seat, c)
        self.win_wild = c[WILD]
        self.winner = seat
        self.finished = True
        self.phase = "finished"
        self._settle_win(seat, c)
        return True

    def _is_baotou_on_win(self, seat, c):
        """胡牌瞬间判定爆头：拿掉胡的那张，看剩下的是否听任意。"""
        t = tile_index(self._drawn) if self._drawn else -1
        if t < 0 or c[t] <= 0:
            return False
        c2 = list(c)
        c2[t] -= 1
        return is_baotou(c2, self._meld_sets(seat))

    def _settle_win(self, seat, c):
        # 必须把 god["baotou"] 显式传下去：_fan 的 baotou 参数默认 False，
        # 漏传会让 try_hu 刚判定出来的爆头在结算路径上又丢掉
        # （自测 [8] 就是被这条抓住的）。
        # 七对分支同理必须在这里留痕：线上真值显示对手 2.7% 的胡牌是七对，
        # 而我们只有 0.1%（36 房 2846 手，按 user_id 归位后统计）——
        # 27 倍的差距不可能靠"弃牌骨架微调"解释，必须能在夹具里量出来才谈得上修。
        self.last_chiitoi = _spm(c) if self._meld_sets(seat) == 0 else 0
        fan = self._fan(seat, c, self.god["baotou"])
        self.last_fan = fan
        pay = self.base * fan
        if seat == self.dealer:
            for s in range(4):
                if s != seat:
                    self.scores[s] -= pay * 8
                    self.scores[seat] += pay * 8
        else:
            for s in range(4):
                if s == seat:
                    continue
                mult = 8 if s == self.dealer else 1
                self.scores[s] -= pay * mult
                self.scores[seat] += pay * mult

    def _fan(self, seat, c, baotou=False):
        """番型倍率 —— 走 mj_core.fan_multiple，与官方 fan-calc 逐例对齐。

        改动前的三个缺陷（都靠 validate_fan.py 对 oracle 抓出）：
          ① 爆头 ×2 从来没生效：god["baotou"] 只被读、从未被赋值
          ② 判据写成 c[WILD] >= 4，而官方是「手留白 + 链内飘出的白 = 4」
          ③ 完全没有财飘：chain_count 只在暗杠时 +1，而出牌从不清链
        ④ ★ 第 349 轮：链/飘原来是**整局共用一份**，任何对手出牌都会清掉它
          ⇒ 财飘结构性不可能（`p321_chain_scope.py` 实测 63 次飘全部被他人清掉、
            2,824 手胡牌带飘 0 手）。现在改成**按座位取**。
        合起来的效果是番数系统性偏低、方差被压缩 —— 而期望分正是要优化的量。
        """
        return fan_multiple(c, self._meld_sets(seat),
                            self.god["chain"][seat], self.god["piao"][seat],
                            baotou)

    def _settle_draw(self):
        self.finished = True
        self.draw = True
        self.phase = "finished"
        # 流局庄家连庄（连庄计数由外部维护）

    # ---------------------------------------------------------- 快照

    def snapshot(self, seat):
        """与平台 GET /api/games/{id}/state 的 snapshot 同构。"""
        hand = [tile_code(t) for t in range(NTILE)
                for _ in range(self.hands[seat][t])]
        return {
            "game_id": "sim",
            "phase": self.phase,
            "round_no": self.round_no,
            "rounds": self.rounds,
            "dealer": self.dealer,
            "turn": self.turn,
            "wall_remaining": self.wall_left,
            "discards": [list(d) for d in self.discards],
            "melds": [list(m) for m in self.melds],
            "hand_counts": [sum(h) for h in self.hands],
            "my_hand": hand,
            "seat": seat,
            "last_discard": self.last_discard,
            "drawn_tile": self._drawn if self.turn == seat else "",
            # ★ 第 349 轮：链/飘**按观察者**取（与线上快照同义：`god.chain_count`
            #   是**自己**的链，不是"最近出牌者"的）。深度拷贝，免得策略写坏引擎状态。
            "god": dict(self.god,
                        chain_count=self.god["chain"][seat],
                        piao_count=self.god["piao"][seat]),
            # ★ `scores` = **累计分**（房间层带进来的 carry + 本局已结算的增量）。
            #   ⚠ `carry` 全 0 时这就是原来的 `self.scores` ⇒ 单局夹具**逐字不变**。
            #   本局增量请用 `result()["scores"]`（那个语义**没有动**）。
            "scores": [self.carry[s] + self.scores[s] for s in range(4)],
            "dealer_streak": self.dealer_streak,
            "responding_seats": list(self.responding_seats),
        }

    # ---------------------------------------------------------- 守恒检查

    def check_conservation(self):
        """牌张守恒：四家手牌 + 副露 + 牌河 + 牌墙 必须恒为 136 张。

        这是模拟器最基本也最有效的自检 —— 任何流程 bug 都会先在这里暴露。
        （早先一次编辑把它挤到模块级函数 return 之后，成了死代码，
          `Game` 上再也没有这个方法；靠自测才发现。）
        """
        n = sum(self.hands[s][t] for s in range(4) for t in range(NTILE))
        n += sum(len(m["tiles"]) for s in range(4) for m in self.melds[s])
        n += sum(len(d) for d in self.discards)
        n += len(self.wall)
        return n

    # ---------------------------------------------------------- 自对弈驱动

    def play(self, strategies, max_turns=400):
        """用 4 个策略跑完一局。策略接口与线上完全一致（吃平台同构快照）。"""
        from mj_bot import View                      # 延迟导入避免循环

        if self.deferred:
            raise RuntimeError("deferred 模式必须由外部驱动窗口，不能用 play()")
        self.window_policy = lambda snap, seat, kind: _ask_window(
            strategies[seat], View(snap, seat), kind)

        while not self.finished:
            if self.turns > max_turns:
                self._settle_draw()
                break
            if self.phase != "draw":
                break
            seat = self.turn
            v = View(self.snapshot(seat), seat)
            if self.observer is not None:
                self.observer(self, seat, v)
            try:
                act = strategies[seat].on_turn(v)
            except Exception:                        # noqa: BLE001
                act = None
            if not act:
                act = {"action": "discard", "tile": self._drawn or
                       (v.hand[0] if v.hand else "")}
            kind = act.get("action")
            if kind == "hu":
                if not self.try_hu(seat):
                    self._fallback_discard(seat)
            elif kind == "discard":
                try:
                    self.discard(seat, act.get("tile", ""))
                except (IllegalAction, ValueError):
                    self._fallback_discard(seat)
            elif kind == "gang":
                if not self._try_gang(seat, act.get("tile")):
                    self._fallback_discard(seat)
            else:
                self._fallback_discard(seat)
        return self.result()

    def _fallback_discard(self, seat):
        """任何非法/无法执行的动作都退化为打刚摸的牌，保证对局不卡死。"""
        code = self._drawn
        if not code or self.hands[seat][tile_index(code)] <= 0:
            for t in range(NTILE):
                if self.hands[seat][t] > 0:
                    code = tile_code(t)
                    break
        try:
            self.discard(seat, code)
        except (IllegalAction, ValueError):
            self._settle_draw()

    def _try_gang(self, seat, code):
        """杠：暗杠（手上 4 张）与补杠（已碰出的面子 + 手上第 4 张）。

        补杠是 2026-09-17 补上的：此前模拟器只有暗杠，而**线上对手的杠率
        0.03~0.09 次/局，与"补杠机会 0.094 次/局"（probe_kong_ops.py）吻合** ——
        也就是说对手把可杠的机会基本都吃掉了，我们却一次不做。
        模拟器缺这条通路时，策略返回 gang 会被 _try_gang 拒掉，
        然后落进 _fallback_discard（随机打一张）—— 那是比不实现更糟的静默失真。
        """
        if not code or self.wall_left <= DEADWALL:
            return False
        t = tile_index(code)
        if t == WILD:
            return False
        # ① 暗杠：手上正好 4 张
        if self.hands[seat][t] >= 4:
            self.hands[seat][t] -= 4
            self.melds[seat].append({"kind": GANG, "tiles": [code] * 4})
            self.god["chain"][seat] += 1
            self.god["chain_count"] = self.god["chain"][seat]
            return self._gang_draw(seat)
        # ② 补杠：已有碰出的面子，手上还留着一张同牌
        if self.hands[seat][t] >= 1:
            for m in self.melds[seat]:
                if m["kind"] == PENG and tile_index(m["tiles"][0]) == t:
                    self.hands[seat][t] -= 1
                    m["kind"] = GANG
                    m["tiles"] = [code] * 4
                    self.god["chain"][seat] += 1
                    self.god["chain_count"] = self.god["chain"][seat]
                    return self._gang_draw(seat)
        return False

    def _gang_draw(self, seat):
        """杠后补牌；牌墙见底则按流局收场（与 _try_gang 原实现一致）。"""
        if not self._draw(seat):
            return True
        return True

    def result(self):
        return {
            "winner": self.winner,
            "draw": self.draw,
            "scores": list(self.scores),
            "turns": self.turns,
            "wall_left": self.wall_left,
            "conserved": self.check_conservation(),
            # 番型相关：改动前这些量全都测不出来（爆头恒 False、飘不存在），
            # 于是「期望分」这个目标函数根本没有可观测的分解项。
            "fan": self.last_fan,
            "baotou": bool(self.god["baotou"]),
            # 七对分支倍数（0 = 平胡）。线上真值：对手 2.7% vs 我方 0.1%。
            "chiitoi": self.last_chiitoi,
            "win_wild": self.win_wild,
            # ★ 第 349 轮：这两个量必须是**赢家自己**的链/飘（原来是"最近动作者"的镜像，
            #   在全局链下无所谓，改成按座位后必须显式取赢家的）。
            "chain": (self.god["chain"][self.winner]
                      if 0 <= self.winner < 4 else 0),
            "piao": (self.god["piao"][self.winner]
                     if 0 <= self.winner < 4 else 0),
            "melds": sum(len(m) for m in self.melds),
        }


def _ask_window(strategy, view, kind):
    """把策略的 on_window 返回值翻译成**动作名**（`peng`/`chi`/`gang`）或 None。

    ★ 第 363 轮：原来是 `return bool(act) and act.get("action") == kind`
    —— 那会把碰窗口里的 `{"action":"gang"}`（明杠）判成"不响应"。
    现在**原样把动作名交回去**，由 `_poll_window` 决定它合不合法。
    """
    phase = "response_peng" if kind == PENG else "response_chi"
    try:
        act = strategy.on_window(view, phase)
    except Exception:                                # noqa: BLE001
        return None
    if not act:
        return None
    return act.get("action")


def run_games(n, strategies, seed=0, dealer_rotate=True, dealer_succession=False):
    """跑 n 局，返回 (结果列表, 用时秒)。吞吐是模拟器存在的理由，必须可测。

    ★★★ 第 277 轮：新增 `dealer_succession`（**默认 False ⇒ 与改前逐字同行为**）。

    为什么需要它（§Z.282，读源码 + 线上真值共同确认）：
      * 旧行为（`dealer_succession=False`）把庄家当**外生**量：`dealer = i % 4`
        ⇒ 每一席恒定坐庄 25%，**胜率不影响庄位份额**；
      * 而**线上是"庄位继承"**：下一局庄家 = 本局赢家（`reviews` 自检 69/69 = 100% 证实）
        ⇒ 和牌率低于 25% 的人**坐庄更少**，而庄家**付款翻倍**
        ⇒ 这是一条**会放大劣势的复利回路**（§Z.276/Z.277 已从 30,388 局真值定量）。
      * ⇒ 模拟器若不带这个开关，就**结构上无法复现**线上那条回路，
        于是出现"同一基因组、和牌率三处一致、而 μ 一个 +0.10、一个 −0.27"的矛盾。

    ⚠ 两种模式**不是逐手配对**：同一 `seed` 换个 `dealer`，发牌就不同（实测 20/20）。
    ⚠ 流局时庄家**不变**（连庄）。`dealer_streak` 在 `mj_sim` 里**不参与结算**（§Z.275 附注）
      ⇒ 复利只来自"份额内生于胜率"，不来自任何连庄加成。
    """
    import time
    out = []
    t0 = time.time()
    dealer = 0
    for i in range(n):
        if not dealer_succession:
            dealer = i % 4 if dealer_rotate else 0
        g = Game(seed=(seed + i) if seed is not None else None, dealer=dealer)
        r = g.play(strategies)
        # ★ 附上本手用的庄家（下划线前缀 = 附加字段，不破坏既有调用方的键集）
        #   `play()` 自己**不返回** `dealer`，所以不这样记就无法事后审计继承是否正确。
        r["_dealer"] = dealer
        out.append(r)
        if dealer_succession:
            w = r.get("winner")
            if w is not None and w >= 0 and not r.get("draw"):
                dealer = w                      # 赢者坐庄
    return out, time.time() - t0


def seven_pairs_mult(c):
    """保留模块级别名 —— 实现已移到 mj_core（见那里对 oracle 验证的说明）。"""
    return _spm(c)


# ---------------------------------------------------------------- 自测

def _raises_play(g):
    """deferred 局必须由外部驱动；play() 应当明确拒绝而不是静默跑错。"""
    try:
        g.play([None, None, None, None], max_turns=1)
    except RuntimeError:
        return True
    except Exception:                                # noqa: BLE001
        return False
    return False


def _selftest():
    ok = True

    def chk(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print("  FAIL %-46s got=%s want=%s" % (name, got, want))
        else:
            print("  ok   %-46s %s" % (name, got))

    print("[1] 发牌与守恒")
    g = Game(seed=1)
    chk("开局牌张守恒 = 136", g.check_conservation(), 136)
    chk("庄家 14 张", sum(g.hands[g.dealer]), 14)
    chk("其余三家 13 张", sorted(sum(g.hands[s]) for s in range(4) if s != g.dealer),
        [13, 13, 13])
    chk("牌墙 = 136-53 = 83", g.wall_left, 83)

    print("[2] 可摸区上限")
    seen = 0
    g2 = Game(seed=2)
    while g2.wall_left > DEADWALL:
        g2.discard(g2.turn, tile_code(next(
            t for t in range(NTILE) if g2.hands[g2.turn][t] > 0)))
        seen += 1
        if seen > 200:
            break
    chk("牌墙保留最后 20 张不摸", g2.wall_left, DEADWALL)
    chk("全程守恒", g2.check_conservation(), 136)

    print("[3] 快照与平台同构")
    snap = Game(seed=3).snapshot(0)
    want = {"phase", "round_no", "dealer", "turn", "wall_remaining", "discards",
            "melds", "hand_counts", "my_hand", "seat", "last_discard",
            "drawn_tile", "god", "scores"}
    chk("快照含平台全部关键字段", want <= set(snap), True)

    print("[4] 七对分支倍率")
    chk("普通七对 -> x2", seven_pairs_mult(to_counts(
        ["1w", "1w", "3w", "3w", "5w", "5w", "7w", "7w", "9w", "9w",
         "东", "东", "白", "白"])), 2)
    chk("豪华七对 -> x4", seven_pairs_mult(to_counts(
        ["1w", "1w", "1w", "1w", "3w", "3w", "5w", "5w", "7w", "7w",
         "9w", "9w", "东", "东"])), 4)
    chk("非七对 -> 0", seven_pairs_mult(to_counts(
        ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
         "1b", "2b", "3b", "东", "东"])), 0)

    print("[5] 对局必须真的推进（防「跑得飞快但其实没跑」）")
    # 这一条是被真事逼出来的：phase 初值漏设时每局 0 手就退出，
    # 被误读成 28581 局/秒的吞吐。速度类指标必须配一个物理量断言。
    g3 = Game(seed=5)
    chk("构造后 phase 即为 draw", g3.phase, "draw")
    g3.discard(g3.turn, tile_code(next(
        t for t in range(NTILE) if g3.hands[g3.turn][t] > 0)))
    chk("出牌后手数增加", g3.turns, 1)
    chk("出牌后轮到下一家", g3.turn, (g3.dealer + 1) % 4)
    chk("出牌后仍守恒", g3.check_conservation(), 136)

    print("[6] 延迟窗口（deferred）—— 线上窗口是真实时间，同步路径把它压成 0")
    # 这组断言的理由：模拟器原先没有窗口延迟这一维，于是系统性高估副露频率
    # （模拟 1.65 vs 线上 0.40 次/人/局）。窗口是新加的路径，必须有测试压着。

    def _peng_setup(g, d=0):
        for t in range(NTILE):
            if t == WILD or g.hands[d][t] < 1:
                continue
            for s in range(4):
                if s != d and g.hands[s][t] >= 2:
                    return t, s
        return None, None

    g4, t4, who4 = None, None, None
    for sd in range(300):
        g4 = Game(seed=sd, dealer=0, deferred=True)
        t4, who4 = _peng_setup(g4)
        if t4 is not None:
            break
    chk("能在真实牌局里构造出碰机会", t4 is not None, True)
    n_before = g4.check_conservation()
    g4.discard(0, tile_code(t4))
    chk("打牌后进入碰窗口而非直接结算", g4.phase, "response_peng")
    chk("可碰座位被暴露在快照里", who4 in g4.responding_seats, True)
    chk("窗口开着时牌张仍守恒", g4.check_conservation(), n_before)
    chk("非响应座位不能动作", g4.window_act((who4 + 1) % 4, PENG), False)
    chk("错误动作类型被拒", g4.window_act(who4, CHI), False)
    chk("合法响应成功", g4.window_act(who4, PENG), True)
    chk("碰后轮到响应者出牌", (g4.turn, g4.phase), (who4, "draw"))
    chk("碰后该家副露 +1", len(g4.melds[who4]), 1)
    chk("碰后仍守恒", g4.check_conservation(), n_before)

    def _chi_setup(g, d=0):
        nxt = (d + 1) % 4
        for t in range(27):                          # 只有数牌可吃
            if g.hands[d][t] < 1:
                continue
            if any(g.hands[s][t] >= 2 for s in range(4) if s != d):
                continue                             # 有人能碰 -> 会先开碰窗口
            if g._chi_pair(nxt, t):
                return t
        return None

    g5, t5 = None, None
    for sd in range(300):
        g5 = Game(seed=sd, dealer=0, deferred=True)
        t5 = _chi_setup(g5)
        if t5 is not None:
            break
    chk("能在真实牌局里构造出吃机会", t5 is not None, True)
    n5 = g5.check_conservation()
    g5.discard(0, tile_code(t5))
    chk("无人可碰时直接开吃窗口", g5.phase, "response_chi")
    chk("吃窗口只含下家", g5.responding_seats, [1])
    chk("吃窗口超时 -> 由下家摸牌", (g5.window_timeout(), g5.turn), (True, 1))
    chk("超时后仍在推进（draw/finished）", g5.phase in ("draw", "finished"), True)
    chk("超时窗口不吞牌", g5.check_conservation(), n5)
    chk("窗口已关后不能再结算", g5.window_timeout(), False)
    chk("deferred 模式禁止用 play() 驱动", _raises_play(g5), True)

    print("[7] 财飘 / 链断 / 爆头 —— 改动前这三条一条都没实现")
    # 改动前：baotou 只被读从未被赋值（永远 False）、财飘不存在、
    # chain_count 只在暗杠时 +1 且从不清零、判据写成 c[WILD] >= 4
    # （官方是「手留白 + 链内飘出的白 = 4」）。合起来番数系统性偏低。

    def _fresh(hand, drawn):
        g = Game(seed=11, dealer=0, deferred=True)
        g.phase, g.turn, g._drawn = "draw", 0, drawn
        g.hands[0] = to_counts(hand)
        return g

    # 14 张 = 4 面子 + 白白：本身已成型（可胡），弃胡打白 = 飘
    g6 = _fresh(["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w",
                 "4w", "4w", "4w", "白", "白"], "白")
    chk("打白前已是成型手（弃胡才有意义）", can_win(g6.hands[0], 0), True)
    chk("打完白仍是爆头（飘的安全条件）",
        is_baotou(to_counts(["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w",
                             "3w", "4w", "4w", "4w", "白"]), 0), True)
    g6.discard(0, "白")
    chk("打白 = 飘：链 +1", g6.god["chain_count"], 1)
    chk("打白 = 飘：飘计数 +1", g6.god["piao_count"], 1)

    # 普通出牌 -> 链断重新计数（piao 必须同步清零：chain 请求体要求 piao<=count）
    g7 = _fresh(["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w",
                 "4w", "4w", "4w", "白", "东"], "东")
    g7.god["chain_count"], g7.god["piao_count"] = 3, 2
    g7.discard(0, "东")
    chk("打普通牌 -> 链断", g7.god["chain_count"], 0)
    chk("打普通牌 -> 飘计数同步清零", g7.god["piao_count"], 0)

    # 非爆头态打白 -> 也是链断（官方原文：含非爆头态打白板）
    g8 = _fresh(["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                 "1b", "2b", "3b", "东", "白"], "白")
    g8.god["chain_count"], g8.god["piao_count"] = 3, 3
    g8.discard(0, "白")
    chk("非爆头态打白 -> 链断", g8.god["chain_count"], 0)

    # 杠：链 +1，但飘计数不动（杠出的白不算「链内飘出的白」）
    g9 = None
    for sd in range(200):
        cand = Game(seed=sd, dealer=0, deferred=True)
        if any(cand.hands[0][t] >= 4 for t in range(NTILE) if t != WILD):
            g9 = cand
            break
    chk("能在真实牌局里找到暗杠机会", g9 is not None, True)
    gt = next(t for t in range(NTILE) if t != WILD and g9.hands[0][t] >= 4)
    g9.god["piao_count"] = 2
    chk("暗杠成立", g9._try_gang(0, tile_code(gt)), True)
    chk("暗杠 -> 链 +1", g9.god["chain_count"], 1)
    chk("暗杠不动飘计数", g9.god["piao_count"], 2)

    print("[9] 补杠（2026-09-17 新增：线上对手杠率 0.03~0.09/局，此前我们一条通路都没有）")
    # 找一个真实牌局，手工造出「已碰出 + 手上留第 4 张」的局面
    g11 = Game(seed=77, dealer=0, deferred=True)
    g11.hands[0] = to_counts(["3w", "5w", "6w", "7w", "9b", "9b",
                              "东", "东", "南", "南", "1t", "2t", "4b", "6b"])
    g11.melds[0] = [{"kind": PENG, "tiles": ["3w", "3w", "3w"]}]
    before = g11.wall_left
    chk("补杠成立", g11._try_gang(0, "3w"), True)
    chk("补杠后面子变杠（4 张）", len(g11.melds[0][0]["tiles"]), 4)
    chk("补杠后面子 kind = gang", g11.melds[0][0]["kind"], GANG)
    chk("补杠扣掉手上那张", g11.hands[0][tile_index("3w")], 0)
    chk("补杠 -> 链 +1", g11.god["chain_count"], 1)
    chk("补杠补了一张牌（牌墙 -1）", g11.wall_left, before - 1)
    # 同一个面子不能再补第二次（否则线上会变成 409 风暴）
    chk("已杠的面子不能再补", g11._try_gang(0, "3w"), False)
    # 白板不能杠
    g12 = Game(seed=78, dealer=0, deferred=True)
    g12.hands[0] = to_counts(["白", "白", "白", "白"])
    g12.melds[0] = [{"kind": PENG, "tiles": ["白", "白", "白"]}]
    chk("白板不可杠", g12._try_gang(0, "白"), False)

    print("[8] 胡牌瞬间判定爆头，且 fan 走验证过的 fan_multiple")
    g10 = _fresh(["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w",
                  "4w", "4w", "4w", "白", "白"], "白")
    chk("判胡成功", g10.try_hu(0), True)
    chk("胡牌瞬间 baotou 被置真（改动前恒为 False）", g10.god["baotou"], True)
    chk("番 = 平胡1 x 爆头2 = 2（改动前只有 1）",
        g10._fan(0, g10.hands[0], g10.god["baotou"]), 2)
    # seat0 是庄家 -> 三家各付 番 x 8 = 2 x 8 = 16，合计 48（与官方 scores 口径一致）
    chk("庄家结算 = 番 x 8 x 3", g10.scores[0], 48)

    print("[10] 明杠（第 363 轮新增：线上我们**真在做** 0.0308 次/局，"
          "而改动前引擎把碰窗口里的 gang **静默当成 pass**）")
    # 造一个真实牌局，让 seat1 手上正好 3 张「3w」，seat0 打出第 4 张
    g13 = Game(seed=91, dealer=0, deferred=True)
    g13.hands[1] = to_counts(["3w", "3w", "3w", "5w", "6w", "7w", "9b", "9b",
                              "东", "南", "南", "1t", "2t"])
    g13.hands[0] = to_counts(["3w", "4w", "5w", "6w", "7w", "8w", "9w",
                              "1b", "2b", "3b", "东", "南", "西", "北"])
    g13.melds[1] = []
    g13.discard(0, "3w")
    chk("打牌后进入碰窗口", g13.phase, "response_peng")
    chk("窗口允许 seat1 响应", 1 in g13.responding_seats, True)
    before_meld = len(g13.melds[1])
    before_wall = g13.wall_left
    chk("碰窗口里可以提交明杠", g13.window_act(1, GANG), True)
    chk("明杠后面子 +1", len(g13.melds[1]), before_meld + 1)
    chk("明杠后面子 kind = gang", g13.melds[1][-1]["kind"], GANG)
    chk("明杠用掉 3 张手上的", g13.hands[1][tile_index("3w")], 0)
    chk("明杠 -> 链 +1", g13.god["chain_count"], 1)
    chk("明杠补了一张牌（牌墙 -1）", g13.wall_left, before_wall - 1)
    chk("明杠后轮到响应者出牌", (g13.turn, g13.phase), (1, "draw"))
    chk("明杠后仍守恒（136 张）", g13.check_conservation(), 136)
    # 手上只有 2 张 -> 明杠必须被拒（那是碰的地盘）
    g14 = Game(seed=92, dealer=0, deferred=True)
    g14.hands[1] = to_counts(["3w", "3w", "5w", "6w", "7w", "9b", "9b",
                              "东", "南", "南", "1t", "2t", "4b"])
    g14.hands[0] = to_counts(["3w", "4w", "5w", "6w", "7w", "8w", "9w",
                              "1b", "2b", "3b", "东", "南", "西", "北"])
    g14.melds[1] = []
    g14.discard(0, "3w")
    chk("手上只有 2 张时明杠被拒", g14.window_act(1, GANG), False)
    chk("  且窗口没被消费掉（还能碰）", 1 in g14.responding_seats, True)
    chk("  碰仍然成立", g14.window_act(1, PENG), True)

    print()
    print("=== 模拟器自测%s ===" % ("全部通过" if ok else "存在失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
