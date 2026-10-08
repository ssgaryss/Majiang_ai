#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实战启动器 —— 一个令牌，从「等开赛」到「打完」全程托管。

为什么不能直接用 run_room.py / mj_bot.py：
  · run_room.py 要求令牌已绑定房间（tournament_id 非空），自动匹配房用的是
    **全局令牌**（tournament_id 为空串），会被它直接拒掉。
  · mj_bot.Player 默认 max_minutes=40，而且一上来就 1 秒一轮询 —— 正式赛要等
    26 小时，那样等于 19 万次无效请求。
  · 官方规则（指南 §二）：**新建赛事分桌只认开赛时刻 <=90s 内有已认证请求者**
    （连上又断开会被剔除）。所以「提前挂上」不够，必须在开赛窗口内持续有请求。

因此本脚本分两段：
  ① 等待段：慢轮询（默认 20s 一次），只做 /api/me 探活，几乎不耗额度
  ② 临赛段：进入 T-90s 后切到 1 秒轮询，满足「已认证请求者」条件
  ③ 比赛段：出现 active_games 后交给 mj_bot.Player 正常打（含 SSE 唤醒）

用法：
  # 自动匹配房（全局令牌）
  python run_live.py --token-file global_token.txt --match --minutes 120
  # 正式赛（参赛令牌），等到开赛
  python run_live.py --token-file tournament_token.txt --wait-until 1789646400 --minutes 300
  # 只等开赛再打，且提前 10 分钟就切到快轮询
  python run_live.py --token-file tournament_token.txt --wait-until auto --lead 600
"""
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime

from my_bot import ApiError, SERVER, api, say
# 注意：Player 必须从 mj_bot 导入。my_bot 里也有个同名类，但它是「最小可跑模板」
# 的 Player(server, token, room_id=...) —— 没有 strategy 参数。从那边导入会把
# strategy 顶进 room_id，报 TypeError: got multiple values for argument 'room_id'。
from mj_bot import Player, STAGE_CONFIRM
from mj_arena import make_strategy

# ★ 第 769 轮：把本进程 PID 写到 live.pid —— 供沙箱/其它会话**确认线上客户端身份** ✓
#   为何需要：客户端运行在**另一个 Windows 会话**里，沙箱内 `Get-Process -Name python`
#   看不到它 ✗（本会话实测三次）⇒ 有了 `live.pid` 才能查它的 CPU/亲和性、为延迟归因 ✓
#   ⚠ 必须**绝不**因为写 pid 失败而影响线上 ⇒ 整段包在 try/except 里 ✓
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "live.pid"),
              "w", encoding="utf-8") as _pf:
        _pf.write("%d %s\n" % (os.getpid(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
except Exception:
    pass

# 开赛**之前**的状态集合。正式赛：registering(=报名期，外部已报名) → stage_open(确认期)。
# 测试房（Kind=test）：registering → running/void，**从不经过 stage_open** ——
# 所以测试房必须在 registering 期就把 register + ready 提交掉（见 --register）。
PRE_START = ("registering",) + tuple(STAGE_CONFIRM)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 正式赛基因组。注意：**必须写满 8 位**（discard:peng:chi:gang:decline:ukeire:route:feed）。
# 写成 7 位会被 parse_genome 按位补齐、`route` 静默保持 0 —— 看上去"改了基因"
# 其实没改。调度任务读的就是这一行，改这一行 = 改正式赛策略。
GENOME = "shanten:1:1:2:2:1:1:0"
# ★ 第 150 轮：线上 A/B 的**热切换**文件（每房间读一次，见主循环里的说明）。
#   写成逗号分隔的基因组串即可零中断切换；文件不存在 = 行为逐字不变。
# ★ 2026-10-03（§Z.1336）：允许**每个实例用自己的臂文件** ✓ —— 这是 GOAL_STAGE2 §4.2(b)
#   「同一房内两臂并存 ⇒ 房间级配对」的**唯一前提** ✓（配对后 σ_Δ 25.31 → 16.63 ⇒ 省样本 **2.32×** ✓）。
#   为什么需要：臂文件原本是**全局硬编码**的 ✗ ⇒ 两个客户端读同一份、各自按轮轮换
#   ⇒ 无法钉住"谁用哪条臂" ✗ ⇒ 同房配对无从谈起 ✓。
#   用法 ✓：实例 A 跑 `LIVE_ARMS_FILE=arms_model.txt`（文件里只写模型臂 ✓）
#           实例 B 跑 `LIVE_ARMS_FILE=arms_rule.txt` （文件里只写规则臂 ✓）
#   ⚠ 不设该变量时**行为逐字不变** ✓（默认仍是 `live_ab_arms.txt` ✓ ⇒ 现有线上零影响 ✓）
#   ⚠ 证据基础 ✓：已实测 match API **会复用开放房** ✓（§Z.1332：模型客户端被匹配进规则轮
#     16:42 创建的同一个房 `a_88cacd61da78` ✓）⇒ 两个实例确实有机会进同一房 ✓（待正式验证 ✓）
ARMS_FILE = os.environ.get("LIVE_ARMS_FILE", "live_ab_arms.txt")
# 晋级/回退路径（每一跳都必须是在**同一张牌桌**上赢下来的配对比较，见 MODEL.md §6）：
#   冠军 shanten:1:1:0:0:1:0   线上 40 局 −12.8 分/局（6.4% 爆头、0 次杠）
#     → decline=2 弃胡换爆头    模拟 12000 局 +0.39 [+0.08,+0.70] p=0.0135（首次显著）
#     → gang=1 暗杠             同批 +0.09
#     → discard=baotou 塑形      模拟 12000 局 +0.52 [+0.10,+0.94]，但**与向听骨架
#                               对拍只有 +0.13/+0.09、两条 lineup 互相矛盾**（S3）。
#   2026-09-17 回退塑形，并给出机制证据（probe_baotou_state / probe_skeleton_diverge）：
#     · 真引擎里「爆头态到达率」：向听骨架 6.8%、塑形骨架 6.8% —— **逐位相同**
#     · 随机局面里「弃牌后成爆头形」：向听骨架 54/1500 = 3.6%，塑形骨架 6/1500 = 0.4%
#       ← 塑形的两道额外过滤（留白板 + 自然牌向听）挡在 ukeire 之前，
#         反而把 ukeire 本来会选中的爆头形候选挤掉了
#     · baotou2（允许用 1 向听换自然牌）相对 baotou 的分歧率 = 0.00%，是纯空转
#     结论：**塑形这条基因在动作空间里是空的（甚至有害）**，爆头态全部来自
#     decline=2 的兑现，而不是来自出牌塑形。于是回到向听骨架 —— 它同时把
#     单次决策从 38ms 降到 18.8ms（线上 3 秒出牌超时的风险直接减半）。
#     → gang=2 补杠            模拟 12000 局 +0.10/+0.20（两条 lineup 同向、未分辨）。
#       代价不对称，且线上有「杠提交几次/被拒几次」的日志，故上线并观察。
#   2026-09-17 第二轮（**线上真值普查推翻了旧结论**，这一轮的关键）：
#     · 用回放接口的服务器自报 fan/detail 普查 36 房 2846 手（座位按 user_id 归位）：
#         全场 爆头 22.6%、均番 1.276、杠开 0.9%、财飘 0.3%
#         ⇒ 番型路线是**全场共识**（杠/飘在经济上可忽略），
#           而**七对**：对手 2.7% vs 我方 0.1% —— 27 倍差距。
#     · 夹具复核（mj_sim 新增 chiitoi 留痕，400 局同质）：
#         不承诺七对 0/400 = 0.0%；承诺七对 6/400 = 1.5%
#         ⇒ 差距是**策略主动放弃**，不是实现走不到。
#     · 历史配对证据（tournament_big/log/new 三次独立 lineup，同局同牌配对）：
#         七对承诺 route=1 的「相对冠军」配对差 = +0.39 / +0.45 / +0.06
#         ⇒ 三次都与冠军**同向或未分辨，从不为负**；自身绝对分为正且显著。
#     → route=1 七对承诺       上线。期望收益小（约 +0.02 分/局，
#       来自 2.7%×2 档换掉 1 档），但**下行风险经三次配对检验为零**，
#       且能补上相对全场的 27 倍机制缺口。已进入 live 轮换复核。


class Tee:
    """同时写 stdout 与日志文件 —— 正式赛跑在后台，没有日志等于没跑。"""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8") if path else None

    def __call__(self, msg):
        line = "[%s] %s" % (datetime.now().strftime("%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()


def load_token(path, inline):
    if inline:
        return inline.strip()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                return s
    raise SystemExit("%s 里没有令牌" % path)


def install_log_tee(log):
    """把 my_bot/mj_bot 的 say() 也接到日志文件。

    为什么必须做：GameRunner 每场结束都会 say 一行
    「N 行动 / M 窗口错过 / K 次409 / P 推送」—— 那正是验证 SSE 改造有没有
    生效的**唯一线上指标**。第一轮线上实测就是因为 say 只打 stdout、
    作业输出又只留了尾部 12 行，把这批数据整批丢了。
    注意 mj_bot 里 `from my_bot import say` 绑的是另一个名字，两个都要换。
    """
    import my_bot
    import mj_bot
    orig = my_bot.say

    def tee(msg=""):
        orig(msg)
        log(msg)

    my_bot.say = tee
    mj_bot.say = tee
    return orig


def resolve_room(server, token, do_match, log, m=10, rounds=16):
    """拿到房间 id。全局令牌需要先走 /api/match。

    ⚠⚠ **这个默认值 `m=10` 实际上不起作用** —— 唯一调用点
       （`run_live.py:520`）显式传 `m=a.m`，即由 **`--m`**（`argparse` 默认 10）决定。
       ⇒ 要改"一次匹配几个房间"，**改 `--m`，不要改这里**。
       （2026-09-29：我一度改了这个默认值，重启后实测**毫无效果** ——
         仍 `POST /api/match {"M": 10}`、仍"在打=10 场"。留下这条以免后来者重蹈。）

    ★★★ 2026-09-29：**为什么要调小 `--m`**（有实测证据，不是口味）：
      · 全量线上日志统计（`live_r16.log`，92,466 行 / 约 240 场）：
          `409 被拒` **1,913** 次、`窗口错过` **7,072** 次、
          ★`放弃动作` **722** 次 —— 类型分布 **100% 都是「丢了胡牌」** ✗
      · 直接原因：**请求长期超过服务器限速**
          `[哨兵] 请求 12.8/秒（令牌桶 11.9/秒）…在打=10 场
           ★★ 实际请求 12.8 > 限速 11.9`
      · 因果链：**请求超速 ⇒ 服务器拒绝/延迟 ⇒ 错过碰/吃/胡窗口 ⇒ 丢胡牌 ⇒ 每场直接扣分**
      · 取实测比例：10 桌 ⇒ 约 12~13 请求/秒，限速约 10~12/秒
        ⇒ **6 桌 ⇒ 约 7~8 请求/秒** ⇒ **落入限速以内** ⇒ 不再丢牌 ✅
      ★ 取舍：**不丢牌的 6 桌 > 丢牌的 10 桌** ——
        丢胡牌是"**每场直接扣分**"，而场数只影响"多久能分辨"。

    ★ 原注释（仍有效）：`rounds` 必须与**正式赛的 config** 对齐，否则我们所有线上验证
    都是在另一个赛制上做的。

    实测两份 config（2026-09-18）：
        自动房（我们请求什么就回什么）：{"M": 10, "Rounds": 8,  "Kind": "auto"}
        正式赛（服务器定的）        ：{"M": 10, "Rounds": 16, "Kind": ""}
    也就是说**历史自动房一直是 8 局一场，而正式赛是 16 局一场** ——
    每阶段 160 局 = 10 场 × 16 局，而我们的房间只有 10 × 8 = 80 局。
    所以默认值改成 16：同样的开房开销换两倍的局数，且与正式赛逐字段一致。
    （`rounds` 只是记账标签，客户端不假设局数，见 STATE §"结构对账"。）
    """
    me = api("GET", "/api/me", server, token)
    rid = (me or {}).get("tournament_id") or ""
    log("令牌 user=%s 绑定房间=%s 正在打=%s"
        % (me.get("user_id"), rid or "<空>", me.get("active_count")))
    if not rid:
        if not do_match:
            raise SystemExit("这是全局令牌且未加 --match，无法确定房间")
        body = {"M": int(m), "Rounds": int(rounds)}
        r = api("POST", "/api/match", server, token, body=body)
        rid = (r or {}).get("room_id") or ""
        cfg = (r or {}).get("config") or {}
        log("POST /api/match %s -> 房号 %s（config=%s）"
            % (json.dumps(body), rid, json.dumps(cfg, ensure_ascii=False)))
        # 服务器会回显 config；如果不一致必须当场喊出来，否则"以为改了赛制"
        # 而其实还在打旧赛制 —— 这是本项目最贵的一类静默失真。
        if cfg.get("Rounds") != body["Rounds"] or cfg.get("M") != body["M"]:
            log("⚠ 服务器返回的赛制与请求不一致：请求 %s，实得 M=%s Rounds=%s"
                % (json.dumps(body), cfg.get("M"), cfg.get("Rounds")))
    return rid, me


def countdown(sec):
    if sec <= 0:
        return "已到"
    return "%d 小时 %d 分" % (sec // 3600, (sec % 3600) // 60)


def wait_until(server, token, rid, target, lead, log):
    """慢轮询等到 target-lead，然后切快轮询。返回是否成功进入临赛段。"""
    slow, fast = 20.0, 1.0
    last = 0.0
    wins = 0
    # ★★ 第 298 轮：**瞬时 404 曾经把 9 小时的等待直接打断**。
    #   实测（`live_tour.stdout.log`）：11:09:18 与 11:14:31 各出现一次
    #   「房间 404（已关闭），退出等待」⇒ `run_live.py` 收工退出，
    #   而**同一时刻的只读探针证明房间是好的**：
    #       status=registering ｜ ready_users=19 / registered_users=87 ｜
    #       voided_reason 空 ｜ StartAt=1790164800 → 2026-09-23 20:00:00
    #   ⇒ 这是**瞬时** 404（服务器抖动 / 限流），不能一次就判死。
    #   改为：**连续 NEED_404 次**（默认 5 次 × 20 秒 ≈ 100 秒）才放弃；
    #   中途任何一次成功都清零。行为对"房间真的关了"仍成立（只是慢 100 秒）。
    NEED_404 = 5
    # ★ 第 346 轮：网络类瞬时故障（TimeoutError / URLError / SSLError）**不是 ApiError**，
    #   会直接穿透 wait_until 把客户端打退出。实测 15:23~15:30 连续 4 次
    #   TimeoutError: The read operation timed out ⇒ 4 次进程重启（restart #4~#7）。
    #   阈值取 20（20 × 约 20 秒 ≈ 6~7 分钟）：实测那次网络抽风正好持续约 7 分钟。
    NEED_NET = 20
    misses = 0
    while True:
        now = time.time()
        left = target - now
        if left <= lead:
            log("进入临赛段（距开赛 %s），切到 %.0f 秒轮询" % (countdown(left), fast))
            return True
        interval = slow
        if now - last >= interval:
            last = now
            try:
                me = api("GET", "/api/me", server, token)
                t = api("GET", "/api/tournaments/%s" % rid, server, token)
                misses = 0
                n = len((me or {}).get("active_games") or [])
                if n:
                    log("★ 出现 %d 场进行中对局，直接开打" % n)
                    return True
                wins += 1
                if wins % 20 == 1:
                    log("等待中：距开赛 %s  status=%s  已就位=%s/%s"
                        % (countdown(left), (t or {}).get("status"),
                           (t or {}).get("ready_users"),
                           (t or {}).get("registered_users")))
            except ApiError as e:
                if e.status == 404:
                    misses += 1
                    if misses >= NEED_404:
                        log("连续 %d 次 404（累计 ~%d 秒）⇒ 判定房间已关闭，退出等待"
                            % (misses, int(misses * slow)))
                        return False
                    log("轮询 404（第 %d/%d 次，距开赛 %s）—— **瞬时容忍，不退出**"
                        % (misses, NEED_404, countdown(left)))
                    time.sleep(2.0)
                    continue
                misses = 0
                log("轮询异常（继续）: %s" % e)
            except Exception as e:                                       # noqa: BLE001
                # ★★★ 第 346 轮：**上一轮的修复只覆盖了 404 这一类瞬时故障。**
                #   实测（`live_tour.stdout.log`）15:23–15:30 出现**连续 4 次**
                #   `本轮异常: TimeoutError('The read operation timed out')`
                #   ⇒ 每次都把 `run_live.py` 打退出（见 `=== tournament client exited rc=0 ===`），
                #   靠看护进程 30 秒一轮重拉才活下来（restart #4~#7）。
                #   根因：`TimeoutError` / `URLError` / `ssl.SSLError` 都**不是** `ApiError`，
                #   直接穿透 `wait_until` 抛到调用方 ⇒ 整个客户端退出。
                #   ⇒ 与 404 同一处置：**计入同一个连续计数**，超阈值才放弃。
                #   阈值放宽到 NEED_NET=20（20×约 20 秒 ≈ 6~7 分钟）——
                #   实测那次网络抽风持续约 7 分钟，若按 5 次早就判死了。
                misses += 1
                if misses >= NEED_NET:
                    log("连续 %d 次网络异常（累计 ~%d 秒）⇒ 判定网络/服务不可用，退出等待"
                        % (misses, int(misses * slow)))
                    return False
                log("轮询网络异常（第 %d/%d 次，距开赛 %s）—— **瞬时容忍，不退出**：%s"
                    % (misses, NEED_NET, countdown(left), e))
                time.sleep(2.0)
                continue
        time.sleep(2.0)


def run_fast_poll_until(server, token, rid, target, log, window=600.0,
                        do_register=False):
    """临赛段：保持 1 秒一次已认证请求，直到出现对局或超时。

    官方规则是「新建赛事分桌只认开赛时刻 <=90s 内有已认证请求者」，
    所以这一段是**必须**的：慢轮询（20 秒一次）在开赛瞬间可能刚好没有请求。
    window 默认 600 秒（开赛后 10 分钟仍无对局就放弃）。

    2026-09-17 补：这一段里还要**顺带做出席确认**。
    正式锦标赛是 OnlineConfirm=true，且「确认不跨阶段继承」——
    也就是说进入新阶段后必须重新 POST /ready。原先只有 Player 的游戏主循环
    会做这件事，而 Player 是在「出现对局之后」才启动的：如果平台要求先确认、
    再给它分桌，就成了死锁（我们在等对局，平台在等我们确认）。
    这里每 5 秒查一次赛事状态，看到 stage_open 就确认一次（按阶段去重，幂等）。
    """
    end = target + window
    n = 0
    confirmed = None
    while time.time() < end:
        n += 1
        try:
            me = api("GET", "/api/me", server, token)
            actives = me.get("active_games") or []
            if actives:
                log("★ 开赛！进行中对局 %d 场（临赛轮询第 %d 次）" % (len(actives), n))
                return True
            if n % 5 == 1:
                try:
                    t = api("GET", "/api/tournaments/%s" % rid, server, token)
                    st = (t or {}).get("status")
                    # ── ★ 测试房（Kind=test）的入场是 register + ready ──────────
                    #
                    # 2026-09-18 事故：用户给的测试房 `t_ca2289974237`
                    # （Kind=test, M=10, Rounds=16）在开赛时刻被判
                    # `status=void / voided_reason=not_ready / ready_users=3/4`，
                    # 而我们的客户端从 11:49 就在等，**一行"出席确认已提交"都没有**。
                    # 根因：这段代码只在 `status == "stage_open"` 时发 `/ready`，
                    # 而测试房的状态流转是 `registering → running/void`，
                    # **从不经过 stage_open**；而且**从来没发过 `/register`**。
                    # 实测（void 后仍可探测端点是否存在）：
                    #   POST /register -> 409 TOURNAMENT_STARTED   ← 端点存在
                    #   POST /ready    -> 409 TOURNAMENT_STARTED   ← 端点存在
                    #   POST /join|seat|sit -> 404 bad path        ← 不是这些
                    # `run_room.py` 的注释早就写明：「测试房的开赛条件是
                    # 4 个令牌各 register + ready，满 4 席才开赛；只挂机不 ready
                    # 会永远停在 status=registering」。
                    #
                    # ⇒ 默认**不改**正式赛路径（正式赛走 stage_open + /ready，
                    #   由下面那支处理，行为与以前完全一致）；测试房要用 `--register`。
                    if do_register and st in PRE_START:
                        stage = None
                        sg = (t or {}).get("stage")
                        if isinstance(sg, dict):
                            stage = sg.get("no")
                        key = ("reg", st, stage)
                        if key != confirmed:
                            # ★ 只有**确实完成**才写 `confirmed`。
                            #
                            # 原来的写法是把 `confirmed = key` 放在 for 循环**之后无条件**
                            # 执行 ⇒ register/ready 只要首次尝试失败（任何 `ApiError`），
                            # 同一个 (status, stage) 下就**永远不会重试**。
                            # 而 registering 期可以很长（本例 28 分钟），
                            # 一次瞬时失败 = 整个房间没有 ready ⇒ **直接复现
                            # `t_ca2289974237` 那种 `not_ready` 流局**。
                            # （2026-09-18 由 subagent 审阅指出，见 STATE.md §G 那条根因。）
                            #
                            # 409 是**正常回执**（已报名 / 已 ready / 已开赛），
                            # 必须算"已完成"，否则会因为"已经做过"而无限重试。
                            done = True
                            for ep in ("register", "ready"):
                                try:
                                    api("POST", "/api/tournaments/%s/%s" % (rid, ep),
                                        server, token)
                                    log("★ %s 已提交（临赛段 status=%s，第 %d 次轮询）"
                                        % (ep, st, n))
                                except ApiError as e:
                                    code = getattr(e, "status", None)
                                    try:
                                        code = int(code)
                                    except (TypeError, ValueError):
                                        code = 0
                                    if code == 409:
                                        log("  %s -> %s（409，视为已完成）" % (ep, e))
                                    else:
                                        log("  %s -> %s（**下次轮询会重试**）" % (ep, e))
                                        done = False
                            if done:
                                confirmed = key
                    elif st in STAGE_CONFIRM:
                        stage = None
                        sg = (t or {}).get("stage")
                        if isinstance(sg, dict):
                            stage = sg.get("no")
                        key = (st, stage)
                        if key != confirmed:
                            api("POST", "/api/tournaments/%s/ready" % rid,
                                server, token)
                            confirmed = key
                            log("★ 阶段 %s 出席确认已提交（临赛段，第 %d 次轮询）"
                                % (stage, n))
                    elif st:
                        confirmed = None
                except ApiError as e:
                    log("  赛事状态查询失败（继续）: %s" % e)
            if n % 15 == 1:
                log("  临赛轮询第 %d 次：距开赛 %.0f 秒，暂无对局"
                    % (n, target - time.time()))
        except ApiError as e:
            log("临赛轮询异常: %s" % e)
        time.sleep(1.0)
    log("临赛段结束（经过 %d 次轮询）" % n)
    return False


def main():
    p = argparse.ArgumentParser(description="实战启动器：等开赛 -> 打完整场")
    p.add_argument("--token-file", default="tournament_token.txt")
    p.add_argument("--token", default="")
    p.add_argument("--server", default=SERVER)
    p.add_argument("--match", action="store_true",
                   help="全局令牌：先调 /api/match 入席")
    p.add_argument("--wait-until", default="",
                   help="unix 秒数，或 auto=从房间 config.StartAt 读")
    p.add_argument("--lead", type=float, default=90.0,
                   help="提前多少秒切到 1 秒轮询（官方要求开赛 <=90s 内有请求）")
    p.add_argument("--register", action="store_true",
                   help="★ 测试房（Kind=test）必加：临赛段提交 register + ready。"
                        "测试房的开赛条件是「N 个令牌各 register+ready，满席才开赛」，"
                        "而它**从不经过 stage_open** ⇒ 不加这个参数会一直等到流局"
                        "（实测 t_ca2289974237：ready_users=3/4 → void not_ready）。"
                        "正式赛**不要**加：正式赛走 stage_open + /ready，行为不变。")
    p.add_argument("--minutes", type=int, default=300, help="比赛段时长上限")
    p.add_argument("--exit-after-games", type=int, default=0,
                   help="收齐 N 场就退出比赛段，**不等服务端把已结束的场次从 "
                        "active_games 撤下来**。0=关（旧行为）。"
                        "★ 实测那段等待中位 **93 秒**、占整房墙钟 **9.1%%**"
                        "（`audit_room_gap.py`：finished→closed 全程在打=0），"
                        "而积分总榜里**场数是线性因子** ⇒ 这是最直接的排名杠杆。"
                        "自动房设 10（M=10）。安全性：要求已回收满 N 场才退，不丢场。")
    # 取快照间隔上限（秒）。线上实测：不限上限时 10 桌的"公平份额"是 0.84 秒，
    # 比碰/吃窗口（1 秒）还长，端到端延迟 p99 冲到 1493 ms、最大 5998 ms，
    # 于是出现"拿旧快照提交 hu → 409"。压到 0.5 秒（默认）能把它收进窗口内。
    p.add_argument("--fetch-gap-max", type=float, default=0.5,
                   help="取快照间隔上限（秒），默认 0.5；0 表示不限制")
    p.add_argument("--genome", default=GENOME)
    # 自动房赛制必须与正式赛一致，否则线上验证是在另一个赛制上做的。
    # 正式赛实测 config = {"M": 10, "Rounds": 16}（每阶段 10 场 × 16 局 = 160 局），
    # 而历史自动房请求的是 {"M": 10, "Rounds": 8}（只有 80 局）。
    p.add_argument("--m", type=int, default=10,
                   help="每房场数（正式赛 config.M，默认 10）")
    # ★★★★★ 2026-09-30：`--m` 的**事实与推断分开写**（`GOAL_STATUS.md §Z.894`；只写事实 + 标注推断 ✓）
    #   · **事实**（读码 ✓）：`resolve_room(..., m=a.m)` ⇒ `POST /api/match {"M": m}` ✓
    #   · **事实**（运行观测 ✓）：把它从 10 改成 6 后，哨兵的 `在打=N 场` 随之下降（10 → 7→6→5 ✓）
    #   · **事实**（本仓其它注释 ✓）：顶部写「每阶段 160 局 = **10 场 × 16 局**」；
    #     `--exit-after-games` 的帮助写「自动房设 10（M=10）」 ⇒ M 与"场"直接对应 ✓
    #   · ⚠ **推断**（未独立确认 ✗）：本仓的"场"与"房"是同一个东西（房 == 场 ✓）
    #     ⇒ 那么上面这条 help 的"每房场数"就**容易误导**（它实为"一次请求开几个场/房"✓）
    #   ⚠ 结论：**在做有代价的决策前，先做一次最小实验确认它的语义** ✓（别信任何一方的注释 ✗）
    #     （`GOAL_STATUS.md §Z.894` 记录了我在 2026-09-30 因此白改一次又回退 ✓）
    p.add_argument("--rounds", type=int, default=16,
                   help="每场局数（正式赛 config.Rounds，默认 16；"
                        "历史自动房用的是 8，会让线上验证与正式赛赛制不一致）")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--log", default="")
    p.add_argument("--result-json", default="",
                   help="赛果落盘路径（循环模式下按 JSONL 追加）")
    p.add_argument("--no-review", action="store_true",
                   help="房间打完后**不**生成复盘。默认生成（落 reviews/，"
                        "原始输出追加到 reviews.log）—— 见第 34 轮的工作约定："
                        "每次打完十盘都要出统计与复盘并落到本地")
    p.add_argument("--loop", type=int, default=1,
                   help="连打多少个房间。自动房整场打完自动关停，再调 /api/match "
                        "就开新房；>1 时结果按 JSONL 追加，用来把线上样本量堆到"
                        "功效分析要求的量级（确认 +3.4 分/场需要 215~300 场）。")
    p.add_argument("--loop-hours", type=float, default=0.0,
                   help="循环的总时长上限（小时）；0 = 只看 --loop 次数")
    p.add_argument("--wait-rooms", type=float, default=8.0,
                   help="循环模式下单个房间等满多少分钟还没开打就弃房换新")
    p.add_argument("--fast-window", type=float, default=600.0,
                   help="开赛后临赛轮询还继续多少秒（默认 600）")
    p.add_argument("--dry-run", action="store_true",
                   help="只跑到「临赛段」为止就退出，不进比赛段。"
                        "用来在正式赛之前演练等待路径 —— 定时任务跑的就是它，"
                        "那条路从没执行过，不演练等于赌。")
    a = p.parse_args()

    log = Tee(a.log)
    install_log_tee(log)
    token = load_token(a.token_file, a.token)
    log("=== 实战启动器 ===  策略=%s  循环 %d 个房间%s"
        % (a.genome, a.loop,
           ("  总时长上限 %.1f 小时" % a.loop_hours) if a.loop_hours else ""))

    grand = []
    t_end = time.time() + a.loop_hours * 3600 if a.loop_hours else None
    # 线上 A/B：--genome 可以给多个（逗号分隔），**按房间轮换**。
    #
    # 为什么必须按房间轮换、而不能"早上跑 A、下午跑 B"：房级方差 ±30 分，
    # 而对手池随时间段变化（深夜与傍晚不是同一批机器人）。分时段比较会把
    # 时段差异记到基因组头上 —— 我们已经因此看到过一个 1σ 的假信号。
    # 轮换之后两个基因组分到的是**同一段时间、同一批对手池**的房间。
    genomes = [g.strip() for g in str(a.genome).split(",") if g.strip()]
    if len(genomes) > 1:
        log("=== 线上 A/B 轮换：%s（每房间换一个）===" % " | ".join(genomes))
    for k in range(max(1, a.loop)):
        if t_end and time.time() >= t_end:
            log("=== 达到总时长上限，停止循环（已完成 %d 轮）===" % k)
            break
        # ★★ 第 150 轮：**臂列表热切换**（每房间重读一次）。
        #   为什么需要它：`--genome` 在**进程启动时**就固定了，而一个
        #   `run_r16.bat` 进程要跑 **8 小时的 400 轮**⇒ 想换策略只能重启客户端，
        #   而那会**把正在打的 ~10 个房间一起丢掉**（一个账号一个座位，
        #   `keep_live.ps1` 的规矩是"等自然退出、绝不 kill"）。
        #   ⇒ 让它在**每个房间边界**重读一个小文件，就能**零中断**切换。
        #   ⚠ 文件不存在时行为**逐字不变**（默认路径就是空的）。
        try:
            if os.path.exists(ARMS_FILE):
                hot = [g.strip() for g in
                       open(ARMS_FILE, encoding="utf-8").read().strip().split(",")
                       if g.strip()]
                if hot and hot != genomes:
                    log("=== ★ 线上 A/B 臂热切换：%s  ->  %s ==="
                        % (" | ".join(genomes) or "(空)", " | ".join(hot)))
                    genomes = hot
        except Exception as _e:                                      # noqa: BLE001
            log("  ⚠ 读 %s 失败（沿用当前臂）：%r" % (ARMS_FILE, _e))
        if genomes:
            a.genome = genomes[k % len(genomes)]
        log("")
        log("########## 第 %d / %d 轮（策略 %s）##########"
            % (k + 1, max(1, a.loop), a.genome))
        try:
            rows = _run_one(a, token, log, first=(k == 0))
        except SystemExit as e:
            log("本轮中止: %s" % e)
            rows = []
        except Exception as e:                                   # noqa: BLE001
            log("本轮异常: %r" % e)
            rows = []
        grand.extend(rows or [])
        if a.dry_run:
            log("=== dry-run 模式：演练结束，不继续循环 ===")
            break
        if not rows:
            log("本轮 0 场（房间没开起来或没人来），等 60 秒再试。")
            time.sleep(60)
    if grand:
        tot = sum(r["mine"] or 0 for r in grand)
        n = len(grand)
        log("")
        log("=== 合计 %d 场，总得分 %+d，场均 %+.2f ===" % (n, tot, tot / n))
    return 0


def _run_one(a, token, log, first):
    """跑一个房间，返回该房间的 per-game 记录列表。"""
    t_start = time.time()
    rid, me = resolve_room(a.server, token, a.match, log, m=a.m, rounds=a.rounds)
    if not rid:
        raise SystemExit("未拿到房间")

    # 开赛时间只在第一轮解析：循环模式后续都是自动房，随时可能开赛。
    if first and a.wait_until:
        target = None
        if a.wait_until == "auto":
            t = api("GET", "/api/tournaments/%s" % rid, a.server, token)
            target = (t or {}).get("config", {}).get("StartAt") or 0
            log("从房间读到的 StartAt = %s" % target)
        else:
            target = float(a.wait_until)
        if target:
            log("开赛时间 = %s（距现在 %s）"
                % (datetime.fromtimestamp(target),
                   countdown(target - time.time())))
            if not wait_until(a.server, token, rid, target, a.lead, log):
                return []
            run_fast_poll_until(a.server, token, rid, target, log,
                                window=a.fast_window, do_register=a.register)
            if a.dry_run:
                log("=== dry-run：等待路径已演练完毕，按计划在此退出（不进比赛段）===")
                log("    定时任务执行的正是这条路径；正式赛时它会继续进比赛段。")
                return []
    else:
        log("未指定开赛时间（自动房随时可能开赛），直接进入比赛段")

    log("=== 进入比赛段：策略 %s，上限 %d 分钟 ===" % (a.genome, a.minutes))
    # ★★ 第 150 轮：**进比赛段之前先把模型加载好**。
    #   为什么必须预热：`discard=rank/rank2` 第一次调用时才会 `torch.load` +
    #   `sys.path.insert("_pkgs")`，实测要**几秒** ⇒ 如果第一次调用正好落在
    #   某个弃牌窗口里，那一手会超时（`DiscardTimeoutSec: 3`）被记成漏窗。
    #   预热之后每次推理只有毫秒级。
    try:
        if "rank" in str(a.genome):
            import mj_policy as _P
            _P.load_cand_ranker()
            log("  已预热候选排序器（discard=rank/rank2）")
    except Exception as _e:                                          # noqa: BLE001
        log("  ⚠ 候选排序器预热失败（%s: %s）—— 策略内部会退回向听规则"
            % (type(_e).__name__, _e))
    pl = Player(a.server, token, make_strategy(a.genome), room_id=rid,
                trace=a.trace, max_minutes=a.minutes, label="主",
                fetch_gap_max=a.fetch_gap_max,
                exit_after_games=getattr(a, "exit_after_games", 0))

    stop, reqs, restore = _install_counters(pl, log)
    threading.Thread(target=_sentinel, args=(pl, stop, reqs, log),
                     daemon=True).start()
    threading.Thread(target=_watch_room_dead, args=(pl, a.wait_rooms, log),
                     daemon=True).start()
    pl.start()
    pl.join()
    stop["v"] = True
    restore()

    rows = []
    tot = 0
    for gid in sorted(pl.results):
        r = pl.results[gid]
        sc = r.get("scores") or []
        seat = r.get("seat", -1)
        mine = sc[seat] if 0 <= seat < len(sc) else None
        tot += mine or 0
        rows.append({"game": gid, "seat": seat, "scores": sc, "mine": mine,
                     "rounds": r.get("rounds"),
                     "avg_shanten": r.get("avg_shanten"),
                     "discards": r.get("discards"),
                     # 逐局四家得失：线上 A/B 的**低方差因变量**。
                     # 单房总分 ±32，而"自摸率/赢一局值多少分"的方差小一个量级 ——
                     # 同样 10 小时的实验，用它们当因变量检验力高得多。
                     # deltas_base = 首帧累计分（中途接管对局时不为 0），
                     # 自校验要用：Σ deltas + base == 最终分。
                     "round_deltas": r.get("round_deltas") or [],
                     # 与 round_deltas 一一对应的局号。**跨客户端去重只能靠它**：
                     # 一个房间里有 10 个客户端各记同一批局，没有局号就无法折叠，
                     # 局数会被放大 ~10 倍、所有按局算的置信区间假性变窄约 3 倍。
                     "round_nos": r.get("round_nos") or [],
                     # ★ 可胡未胡明细（弃胡次数）。事件流不含起手，
                     #   弃胡**无法**事后从回放重建，只能客户端当场判 ——
                     #   这是"每局弃胡 x 次"这个低方差机制指标的唯来源。
                     #   字段名必须是 missed_detail：曾与"窗口错过计数 missed"撞名，
                     #   见 mj_bot 的说明（代价是整场对局 + 哨兵 + 汇总行三样全废）。
                     "missed_detail": r.get("missed_detail") or [],
                     "deltas_base": r.get("deltas_base") or []})
    # 顺带把机制指标打进日志（一行），便于肉眼跟踪而不必事后解析 jsonl。
    wins = sum(1 for g in rows for d in (g.get("round_deltas") or [])
               if g.get("seat") is not None and 0 <= g["seat"] < 4
               and d[g["seat"]] > 0)
    rounds = sum(len(g.get("round_deltas") or []) for g in rows)
    winvals = [d[g["seat"]] for g in rows for d in (g.get("round_deltas") or [])
               if g.get("seat") is not None and 0 <= g["seat"] < 4
               and d[g["seat"]] > 0]
    # ★ 弃胡次数：唯一能在客户端当场判定、事后无法从回放重建的机制量
    #   （事件流不含起手）。它同时是"我们的弃胡机制到底在做什么"的审计入口：
    #   `audit_missed.py` 会把每一条重放一遍，判定这次弃胡有没有依据。
    n_miss = sum(len(g.get("missed_detail") or []) for g in rows)
    n_miss_dealer = sum(1 for g in rows for m in (g.get("missed_detail") or [])
                        if m.get("is_dealer"))
    log("本轮 %d 场，本席 %+d（场均 %+.2f），429 节流 %d 次，当前限速 %.1f/秒%s%s"
        % (len(rows), tot, tot / max(1, len(rows)), pl.limiter.throttled,
           pl.limiter.rate,
           ("  |  自摸 %d/%d=%.1f%%  平均赢下 %.1f 分"
            % (wins, rounds, 100.0 * wins / rounds,
               (sum(winvals) / len(winvals)) if winvals else 0))
           if rounds else "",
           ("  |  弃胡 %d 次(%.2f/局，其中庄位 %d)"
            % (n_miss, n_miss / max(1, rounds), n_miss_dealer)) if rounds else ""))

    if a.result_json:
        rec = {"genome": a.genome, "room": rid, "started": t_start,
               "ended": time.time(), "games": len(rows), "total": tot,
               "rows": rows, "throttled": pl.limiter.throttled,
               "missed_hu": len(pl.missed_hu)}
        with open(a.result_json, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ★ 工作约定（第 34 轮，用户要求）：
    #   **每次打完十盘需要复盘，生成对局统计数据并落到本地 `reviews/`。**
    #   放在这里是因为这正是"一个房间刚打完"的时刻：比赛段已结束、
    #   下一次开房还没开始 ⇒ 这十来次 GET **不会和任何窗口抢预算**。
    #
    #   ⚠ 两个沙箱细节（都踩过）：
    #     1. 子进程 stdout **不能**用 `capture_output`/PIPE（沙箱禁命名管道），
    #        必须重定向到**文件**；
    #     2. 复盘**绝不能**影响比赛 —— 任何异常都只记日志，不往上抛。
    if not a.no_review:
        try:
            # ★★ 第 357 轮：这里原来是
            #      `with open("reviews.log","a") as rl: Popen(cmd, stdout=rl, ...)`
            #    —— **两个**已实测的故障叠在一起：
            #     ① `Popen(stdout=<普通文件对象>)` 被沙箱拒绝（PermissionError 13），
            #        `watch_reviews.py` 的注释里早就记着这张实测表，但**没同步到这里**；
            #     ② `reviews.log` 会被别的进程以「拒绝写共享」占住 ⇒ 连 `open` 都失败。
            #   症状：每打完一房就写一行
            #     `⚠ 复盘生成失败（不影响比赛）：PermissionError(13, ...)`
            #   而复盘**一份都没生成**。⇒ 改用 `proc_log` 的唯一安全通路
            #   （只 dup2 + 继承；日志写不进去就自动轮转，绝不把复盘带走）。
            from proc_log import open_log, spawn_logged
            here = os.path.dirname(os.path.abspath(__file__))
            cmd = [sys.executable, os.path.join(here, "review_room.py"),
                   "--room", rid]
            if a.result_json:
                cmd += ["--jsonl", a.result_json]
            logpath = getattr(a, "log", None)
            if logpath and os.path.exists(logpath):
                cmd += ["--log", logpath]
            rlog = os.path.join(here, "reviews.log")
            fh, eff = open_log(rlog)
            try:
                fh.write("\n===== %s  复盘房间 %s =====\n"
                         % (time.strftime("%Y-%m-%d %H:%M:%S"), rid))
                fh.flush()
            finally:
                fh.close()
            spawn_logged(cmd, rlog, cwd=here)
            log("★ 复盘已生成（见 reviews/ 与 %s）" % os.path.basename(eff))
        except Exception as e:                                 # noqa: BLE001
            log("⚠ 复盘生成失败（不影响比赛）：%r" % e)
    return rows


def _install_counters(pl, log):
    """把请求速率计数的补丁装上，并返回还原函数。

    my_bot.api 与 mj_bot.api 是两个独立的名字绑定（mj_bot 是 from..import 绑的），
    只换一个等于没换 —— 第一版哨兵就是这么打出 0.0/秒 的。
    """
    import my_bot as _mb
    import mj_bot as _mjb
    stop = {"v": False}
    reqs = {"n": 0, "get": 0, "post": 0, "throttled": 0}
    orig = api

    def counted(method, path, server=None, token=None, body=None, **kw):
        # ★★★ 2026-09-29：**必须透传 `**kw`**。
        #   事故：我给 `my_bot.api` 加了 `timeout=` / `tries=` 两个可选参数
        #   （快照 GET 要短超时，见 `my_bot.api` 的 docstring），
        #   但这个包装只声明了 5 个位置参数 ⇒ 调用方一传 `timeout=1.5` 就
        #     `TypeError: counted() got an unexpected keyword argument 'timeout'`
        #   ⇒ **每一局都异常结束（0 行动）**，表现为"在打=0 场"、看起来像匹配没开 ✗
        #   ⇒ 教训：**包装函数必须透传 `**kw`**，否则它会静默成为"参数黑名单"。
        reqs["n"] += 1
        reqs["get" if method == "GET" else "post"] += 1
        try:
            return orig(method, path, server, token, body, **kw)
        except ApiError as e:
            if getattr(e, "status", None) == 429:
                reqs["throttled"] += 1
            raise

    _mb.api = counted
    _mjb.api = counted

    def restore():
        _mb.api = orig
        _mjb.api = orig
    return stop, reqs, restore


def _sentinel(pl, stop, reqs, log):
    """每 30 秒报一次真实请求速率 —— 判断「策略不行」还是「撞限速墙」的唯一依据。"""
    last_n = last_g = last_p = last_th = 0
    last_t = time.time()
    while not stop["v"]:
        time.sleep(30)
        if stop["v"]:
            return
        now = time.time()
        dt = max(0.1, now - last_t)
        n, g, p = reqs["n"], reqs["get"], reqs["post"]
        th = reqs["throttled"]
        live = [w for w in pl._workers.values() if w.is_alive()]
        acts = sum(w.actions for w in pl._workers.values())
        miss = sum(w.missed for w in pl._workers.values())
        conf = sum(w.conflicts for w in pl._workers.values())
        giveup = sum(getattr(w, "giveup_actions", 0) for w in pl._workers.values())
        # ★★★ 第 370 轮：**"预算 16" 是硬编码的标签，不是真实 allowance。**
        #   第 2 阶段（16 人决赛圈）实测：令牌桶被 429 反复压到**下限 6.0/秒**
        #   （`mj_bot.RateLimiter.throttle`: rate = max(floor=6.0, rate*0.85)），
        #   而实际请求仍是 ~7.0/秒 ⇒ 快照比服务端旧 ⇒ **该阶段 409 = 40 次（0.87%）**，
        #   而第 1 阶段只有 ~2 次（0.02%）。旧文案把这件事**美化成了"预算 16"**。
        #   ⇒ 现在改成：① 打印**令牌桶的真实速率**；② 实际请求**超过**它时直接标出来。
        r_lim = float(getattr(pl.limiter, "rate", 0.0) or 0.0)
        r_now = (n - last_n) / dt
        warn = ""
        if th > last_th:
            warn = "  ★429 +%d 次！已自动降速" % (th - last_th)
        if r_lim > 0 and r_now > r_lim * 1.05:
            warn += ("  ★★ 实际请求 %.1f > 限速 %.1f ⇒ **快照会变旧、弃牌会被 409 拒**"
                     % (r_now, r_lim))
        # giveup 单独显示：它每次都代表**一手本可以胡的牌被推掉了**，
        # 是可见的损失，不能混在别的计数里看不见。
        if giveup:
            warn += "  ★放弃动作 %d 次（丢了胡牌）" % giveup
        log("  [哨兵] 请求 %.1f/秒（令牌桶 %.1f/秒；取快照 %.1f + 提交 %.1f）"
            " 在打=%d 场 行动=%d 窗口错过=%d 409=%d%s"
            % (r_now, r_lim, (g - last_g) / dt, (p - last_p) / dt,
               len(live), acts, miss, conf, warn))
        last_n, last_g, last_p, last_t, last_th = n, g, p, now, th


def _watch_room_dead(pl, wait_min, log):
    """弃房闸：等太久还没开打就退出换新房。

    自动房是「满 4 人即开」，若匹配池没人，房间会一直空着；
    没有这道闸，循环会卡在空房间里干等到 max_minutes。
    """
    t0 = time.time()
    while pl.is_alive():
        if time.time() - t0 > wait_min * 60 and not pl.results:
            log("房间等满 %.0f 分钟仍无人来，弃房换新" % wait_min)
            pl._stop.set()
            return
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())

