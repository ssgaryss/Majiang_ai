#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向听数 / 有效牌 参考实现（带财神百搭）—— 与 mj_core.py 的牌码约定一致。

索引 0-33：
  0-8   1w..9w
  9-17  1b..9b
  18-26 1t..9t
  27-30 东南西北
  31-33 中发白   （33 = 白板 = 财神 = WILD）

核心恒等式（替换数 / replacement number 定义）：
    d(h, w) = max(0, M - absorbed(h, w) - J)
    S(h)    = min_w d(h, w) - 1
其中 w 是目标胡牌型（4 面子 + 1 将，副露部分已固定），M = 暗牌需凑齐的张数
= 14 - 3*meld_count，absorbed = 手上真牌能塞进 w 的最大张数，J = 财神张数。

本文件同时给出一个「穷举所有胡牌型」的暴力实现用于交叉验证（_brute_*）。
"""
from itertools import combinations_with_replacement
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NTILE = 34
WILD = 33
SUIT_LEN = 9


# ------------------------------------------------------------------ 工具

def _meld_types():
    """返回所有面子（刻子 + 顺子）的 34-计数元组。"""
    out = []
    for t in range(NTILE):
        c = [0] * NTILE
        c[t] = 3
        out.append(tuple(c))
    for base in (0, 9, 18):
        for s in range(base, base + 7):
            c = [0] * NTILE
            c[s] = c[s + 1] = c[s + 2] = 1
            out.append(tuple(c))
    return out


# ------------------------------------------------------------------ 核心 DFS

def max_absorbed(counts, meld_slots, pair_slots=1):
    """手上真牌最多能有多少张被塞进「meld_slots 个面子槽 + pair_slots 个将槽」。

    每个槽允许的填法（其余空位由财神或后续摸牌补足）：
      面子槽：3 张（完整面子）/ 2 张（对子或 i,i+1 / i,i+2 搭子）/ 1 张（孤张）
      将  槽：2 张（对子）/ 1 张（孤张）
    返回值越大越好：absorbed 每 +1，向听数 -1。

    关于 memo **必须每次调用新建**（试过提到模块级，自测当场挂掉：
    632 处与 can_win 不符，向听普遍算成 -1）：
      键 (i, c[i], c[i+1], c[i+2], ms, ps) 之所以够用，靠的是「更深的位置
      在本层递归中不会被触碰」——于是 c[i+3..] 等于**这一手牌的原始值**，
      相当于一个隐含的、手牌固定的尾部。跨手牌共享后尾部不再固定，
      键就丢信息了。想共享必须把整个后缀放进键，命中率也就没了。
      所以提速只能走「让每次调用本身更快」，或者多进程。

    性能改造（语义不变，靠 shanten_ref 自测的 5.5 万穷举牌型把关）：
      · 数组尾部补两个 0 —— 去掉热路径上每层两次边界判断
      · memo 键用**整数**而不是 6 元组 —— 省掉元组构造与哈希（profile 里
        dict.get 占 11.5%，元组构造混在 dfs 的 tottime 里）
      · **上界剪枝**：位置 i 起最多还能吸收 min(剩余张数, 3*ms + 2*ps) 张。
        一旦 best 触到上界就立即返回，不必展开其余分支。
      · 剩余张数 rem 沿递归传递（它是键的函数：同一键在同一手牌内对应的
        尾部一定相同，所以 rem 不参与键也不会错）
    """
    c = list(counts)
    c[WILD] = 0                                   # 财神不参与「真牌吸收」
    c.append(0)
    c.append(0)
    n = NTILE
    memo = {}

    def dfs(i, ms, ps, rem):
        while i < n and c[i] == 0:
            i += 1
        if i >= n or (ms <= 0 and ps <= 0):
            return 0
        key = ((((i * 5 + c[i]) * 5 + c[i + 1]) * 5 + c[i + 2]) * 5 + ms) * 2 + ps
        hit = memo.get(key)
        if hit is not None:
            return hit

        ub = 3 * ms + 2 * ps
        if rem < ub:
            ub = rem
        best = 0

        # --- 选项 A：把这张 i 当废牌丢掉（丢一张，剩下的继续在 i 处决策）---
        c[i] -= 1
        v = dfs(i, ms, ps, rem - 1)
        c[i] += 1
        if v > best:
            best = v
        if best >= ub:
            memo[key] = best
            return best

        if ms > 0:
            same_suit = i < 27
            rank = i % SUIT_LEN if same_suit else -1
            # 完整顺子 i,i+1,i+2
            if same_suit and rank <= 6 and c[i + 1] > 0 and c[i + 2] > 0:
                c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
                v = 3 + dfs(i, ms - 1, ps, rem - 3)
                c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            # 两面搭子 i,i+1
            if same_suit and rank <= 7 and c[i + 1] > 0:
                c[i] -= 1; c[i + 1] -= 1
                v = 2 + dfs(i, ms - 1, ps, rem - 2)
                c[i] += 1; c[i + 1] += 1
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            # 嵌张搭子 i,i+2
            if same_suit and rank <= 6 and c[i + 2] > 0:
                c[i] -= 1; c[i + 2] -= 1
                v = 2 + dfs(i, ms - 1, ps, rem - 2)
                c[i] += 1; c[i + 2] += 1
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            # 对子当搭子（摸/财神补第 3 张成刻）
            if c[i] >= 2:
                c[i] -= 2
                v = 2 + dfs(i, ms - 1, ps, rem - 2)
                c[i] += 2
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            # 完整刻子
            if c[i] >= 3:
                c[i] -= 3
                v = 3 + dfs(i, ms - 1, ps, rem - 3)
                c[i] += 3
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            # 孤张占一个面子槽
            c[i] -= 1
            v = 1 + dfs(i, ms - 1, ps, rem - 1)
            c[i] += 1
            if v > best:
                best = v
                if best >= ub:
                    memo[key] = best
                    return best

        if ps > 0:
            # 将：真对子
            if c[i] >= 2:
                c[i] -= 2
                v = 2 + dfs(i, ms, 0, rem - 2)
                c[i] += 2
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            # 将：单张 + 财神/摸牌
            c[i] -= 1
            v = 1 + dfs(i, ms, 0, rem - 1)
            c[i] += 1
            if v > best:
                best = v

        memo[key] = best
        return best

    return dfs(0, meld_slots, pair_slots, sum(c) - c[n] - c[n + 1])


# ⚠ 第 132 轮试过「**跨调用共享精确键 memo**」，**已回退** —— 数字记在这里，别再走这条路：
#   · 节点数确实**腰斩**：8,998,891 → 4,484,594（**50%**），结果逐局一致 ✅
#   · **但墙上时间只快 1.11×**（8.0s → 7.2s）
#   ⇒ **DFS 的节点数不是墙上时间的瓶颈**；每个节点的开销（键构造 + dict 操作）才是。
#     共享键省掉一半节点，却因为 `bytes(c[i:])` 比原来的整数键贵得多而几乎抵消。
#   ⇒ 想把 `ukeire` 提快 5×（§Z.114 说那能让"弃胡"变得可学），
#     **必须换一个"每节点更便宜"的算法**（例如把尾部信息压成可增量维护的滚动哈希、
#     或按花色分段 DP），**而不是继续在记忆化上做文章**。
def max_absorbed_variants(counts, meld_slots, pair_slots, tiles):
    """一次算出「给 counts 加上 tiles 中每一张」之后的 max_absorbed。

    这是 ukeire 的性能核心：ukeire 要对 34 张候选牌各求一次向听，而
    **这 34 手牌彼此只差一个位置**。位置 i 之后的子树在它们之间是同构的，
    所以只要记忆化键能区分「被改动的那张牌落在窗口内还是窗口外」，就能安全共享。

    键 = (i, c[i], c[i+1], c[i+2], ms, ps, t_eff)
      t_eff = t        当 t >= i+3（改动落在窗口之外，必须进键才不丢信息）
      t_eff = 34       否则（改动在窗口内，已被 c[i..i+2] 捕获）

    为什么这是**安全**的共享（和之前被自测否决的那次不同）：
      之前把 memo 提到模块级、跨**不同手牌**复用，键里没有任何尾部信息，
      于是不同尾部的状态撞在一起。这里的共享家族只有一个自由度 ——
      相对同一副底牌 c 改动一个位置 —— t_eff 恰好把这个自由度编码进键。
    """
    base = list(counts)
    base[WILD] = 0
    base.append(0)
    base.append(0)
    c = base
    n = NTILE
    memo = {}
    total = sum(c)

    def dfs(i, ms, ps, rem, t):
        while i < n and c[i] == 0:
            i += 1
        if i >= n or (ms <= 0 and ps <= 0):
            return 0
        teff = t if t >= i + 3 else 34
        key = (((((i * 5 + c[i]) * 5 + c[i + 1]) * 5 + c[i + 2]) * 5 + ms)
               * 2 + ps) * 35 + teff
        hit = memo.get(key)
        if hit is not None:
            return hit

        ub = 3 * ms + 2 * ps
        if rem < ub:
            ub = rem
        best = 0
        # ⚠ 第 135 轮试过**重排分支顺序**（把"价值 0 的跳过分支"从最前挪到最后，
        #   希望上界早退 `best >= ub` 更早触发）。**测不出加速，已回退**：
        #   同一工作量的三次计时 6.9 / 8.8 / 9.1s vs 基线 8.0s，
        #   而**同一份代码的噪声本身就有 ±30%** ⇒ 分辨不出。
        #   ⇒ 这是**第三次**在 `ukeire` 上尝试廉价优化失败
        #     （前两次：顶层记忆化 §Z.114、跨调用精确键共享 §Z.115）。
        #   ⇒ 结论：**`ukeire` 要提速必须换成结构不同的算法**
        #     （按花色分段 DP / 尾部增量滚动哈希），不是顺序、也不缓存。
        c[i] -= 1
        v = dfs(i, ms, ps, rem - 1, t)
        c[i] += 1
        if v > best:
            best = v
        if best >= ub:
            memo[key] = best
            return best

        if ms > 0:
            same_suit = i < 27
            rank = i % SUIT_LEN if same_suit else -1
            if same_suit and rank <= 6 and c[i + 1] > 0 and c[i + 2] > 0:
                c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
                v = 3 + dfs(i, ms - 1, ps, rem - 3, t)
                c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            if same_suit and rank <= 7 and c[i + 1] > 0:
                c[i] -= 1; c[i + 1] -= 1
                v = 2 + dfs(i, ms - 1, ps, rem - 2, t)
                c[i] += 1; c[i + 1] += 1
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            if same_suit and rank <= 6 and c[i + 2] > 0:
                c[i] -= 1; c[i + 2] -= 1
                v = 2 + dfs(i, ms - 1, ps, rem - 2, t)
                c[i] += 1; c[i + 2] += 1
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            if c[i] >= 2:
                c[i] -= 2
                v = 2 + dfs(i, ms - 1, ps, rem - 2, t)
                c[i] += 2
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            if c[i] >= 3:
                c[i] -= 3
                v = 3 + dfs(i, ms - 1, ps, rem - 3, t)
                c[i] += 3
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            c[i] -= 1
            v = 1 + dfs(i, ms - 1, ps, rem - 1, t)
            c[i] += 1
            if v > best:
                best = v
                if best >= ub:
                    memo[key] = best
                    return best

        if ps > 0:
            if c[i] >= 2:
                c[i] -= 2
                v = 2 + dfs(i, ms, 0, rem - 2, t)
                c[i] += 2
                if v > best:
                    best = v
                    if best >= ub:
                        memo[key] = best
                        return best
            c[i] -= 1
            v = 1 + dfs(i, ms, 0, rem - 1, t)
            c[i] += 1
            if v > best:
                best = v

        memo[key] = best
        return best

    out = {}
    for t in tiles:
        if t == WILD:
            # 财神在 max_absorbed 里恒被清零，所以「加一张财神」不改变吸收数。
            # 必须走「无改动」这条路（t=-1），否则会凭空多算一张。
            out[t] = dfs(0, meld_slots, pair_slots, total, -1)
            continue
        c[t] += 1
        out[t] = dfs(0, meld_slots, pair_slots, total + 1, t)
        c[t] -= 1
    return out


def shanten_regular(counts, meld_count=0):
    """四面子一雀头形的向听数。-1 = 已胡，0 = 听牌。"""
    J = counts[WILD]
    M = 14 - 3 * meld_count
    ab = max_absorbed(counts, 4 - meld_count, 1)
    d = M - ab - J
    if d < 0:
        d = 0
    return d - 1


def shanten_chiitoi(counts, meld_count=0):
    """七对子向听数（副露 > 0 时不可行，返回 99）。

    ★★★ 第 411 轮修 bug（**已用 120,000 手随机牌对拍验证，0 违例**）：
      原实现是
    ```python
    pairs   = sum(1 for x in c if x >= 2)     # ← 3 张/4 张都只算 **1** 对
    singles = sum(1 for x in c if x == 1)     # ← 3 张那张的"多出来的一单张"丢了
    ```
      ⇒ **任何 `count >= 3` 的牌都会丢掉多出来的部分**：
        · 4 张 → 应算 **2 对**，原来只算 1 对；
        · 3 张 → 应算 **1 对 + 1 单张**，原来只算 1 对。
      ⇒ 后果：手牌 `4×1w + 5×对`（14 张）**已经能胡**（`can_win=True`、
        `seven_pairs_mult=4` ＝ **豪华七对 ×4**、`fan_multiple=4.0`），
        而本函数报 **1** —— **和牌被报成"还差一向听"**。
      ⇒ 更远的下游（§Z.410）：`commit_chiitoi` 因此在**握有 4 张同牌时永不成立**
        ⇒ 策略从不把七对当路线 ⇒ `route` 基因在关键情形上失效
        ⇒ 那 4 张又被 `gang=2` 顺手杠掉 ⇒ **豪华七对结构上不可能**
        （§Z.377 实测 35,436 局 **0 次**豪华七对，此前误归因于"自动杠"，
          真正第一颗钉子是这里）。

    **发生率实测**（`probe_chiitoi_shanten_bug.py`）：
    ```
      120,000 手随机 14 张 ｜ 其中七对和牌 223 手
      现版：和牌却报 != −1 的 **26 手 = 11.66%**
      修正版：**0 手 = 0.00%**      ← 判据（跑之前写死）要求 0，已满足
      两版取值不同的手牌：332 / 120,000 = **0.28%**
    ```
    ⚠ 修正只改 pairs/singles 的**拆法**（`x // 2` / `x % 2`），
      其余（财神 J、`ab`/`d` 的算式）**逐字保持不变**。
    ⚠ 它会**全局降低**七对向听（`shanten = min(平胡, 七对)`）⇒ 策略更常考虑七对
      ⇒ **上线前必须走同 seed 配对 A/B**（见 `GOAL_STATUS.md` §Z.411）。
    """
    if meld_count > 0:
        return 99
    J = counts[WILD]
    c = list(counts)
    c[WILD] = 0
    pairs = sum(x // 2 for x in c)
    singles = sum(x % 2 for x in c)
    if pairs > 7:
        pairs = 7
    ab = 2 * pairs
    rest = 7 - pairs
    ab += min(singles, rest)
    d = 14 - ab - J
    if d < 0:
        d = 0
    return d - 1


def shanten(counts, meld_count=0):
    """总向听数 = min(平胡, 七对)。

    ⚠ 第 131 轮试过**顶层记忆化**，**已回退** —— 实测无收益，理由记在这里免得再试一次：
      · 命中率确实高（20 局只产生 ~5,893 个不同手牌 ⇒ 约 88% 命中）；
      · 但**同进程、同 seed 的受控 A/B 测不出加速**：
        组1 有缓存慢 20.9%、组2 有缓存快 7.1%，而**同一份无缓存代码两次**
        跑出 1.86 vs 1.48 局/秒 ⇒ **测量噪声（±25%）大于效应**。
      · 解释：`shanten` 本身很快，而每次调用要构造 `bytes(counts)` 当键
        ⇒ **键的构造成本与省下的计算相当**；再加上无界增长的内存（要设 300 万上限）。
      ⇒ **结论：这个负载上手牌空间没有可利用的复用**，
        想提速只能靠**更快的算法**，不是缓存。
    """
    return min(shanten_regular(counts, meld_count), shanten_chiitoi(counts, meld_count))


def shanten_both(counts, meld_count=0):
    return (shanten_regular(counts, meld_count), shanten_chiitoi(counts, meld_count))


# ------------------------------------------------------------------ 有效牌

def ukeire(counts, meld_count=0, unseen=None, chiitoi=False):
    """13 张手牌的有效牌。

    counts : 13 张（已打完一张）的手牌计数
    unseen : 长度 34 的「剩余可见张数」（不含自己手牌）；None 表示按 4-counts 估算
    返回 (总枚数, {tile_index: 枚数}, 减少的向听数基准)

    性能：早先是逐张调 `shanten(c+t)`（34 次完整 DFS）。现在改成一次
    `max_absorbed_variants` 共享记忆化表算出全部 34 张的吸收数，再自己合成
    平胡向听（七对向听本来就只要 1.9µs，逐张算也无所谓）。
    实测 1.55x，结果与旧实现逐位一致（27,200 组比对 + 120 手端到端）。
    """
    base = shanten_chiitoi(counts, meld_count) if chiitoi else shanten(counts, meld_count)
    tiles = [t for t in range(NTILE) if counts[t] < 4]
    am = None
    if not chiitoi:
        ms = 4 - meld_count
        M = 14 - 3 * meld_count
        J = counts[WILD]
        am = max_absorbed_variants(counts, ms, 1, tiles)
    got = {}
    total = 0
    for t in tiles:
        if chiitoi:
            c = list(counts)
            c[t] += 1
            s = shanten_chiitoi(c, meld_count)
        else:
            J2 = J + (1 if t == WILD else 0)
            d = M - am[t] - J2
            if d < 0:
                d = 0
            s = d - 1
            c = list(counts)
            c[t] += 1
            sc = shanten_chiitoi(c, meld_count)
            if sc < s:
                s = sc
        if s < base:
            left = (4 - counts[t]) if unseen is None else unseen[t]
            if left > 0:
                got[t] = left
                total += left
    return total, got, base


# ------------------------------------------------------------------ 穷举验证

def _brute_shapes(types, meld_count=0):
    """在受限牌种集合 types 上穷举所有胡牌型的「暗牌部分」：
    (4 - meld_count) 个面子 + 1 个将。"""
    melds = []
    for t in types:
        c = [0] * NTILE
        c[t] = 3
        melds.append(tuple(c))
    for base in (0, 9, 18):
        for s in range(base, base + 7):
            if s in types and (s + 1) in types and (s + 2) in types:
                c = [0] * NTILE
                c[s] = c[s + 1] = c[s + 2] = 1
                melds.append(tuple(c))
    need = 4 - meld_count
    shapes = set()
    for combo in combinations_with_replacement(range(len(melds)), need):
        tot = [0] * NTILE
        ok = True
        for k in combo:
            m = melds[k]
            for i in range(NTILE):
                if m[i]:
                    tot[i] += m[i]
                    if tot[i] > 4:
                        ok = False
                        break
            if not ok:
                break
        if not ok:
            continue
        for p in types:
            if tot[p] + 2 > 4:
                continue
            w = list(tot)
            w[p] += 2
            shapes.add(tuple(w))
    return shapes


def _brute_shanten(counts, meld_count, shapes):
    J = counts[WILD]
    M = 14 - 3 * meld_count
    c = list(counts)
    c[WILD] = 0
    cb = bytes(c)
    best = 1 << 30
    for w in shapes:
        matched = sum(map(min, cb, w))
        d = M - matched - J
        if d < 0:
            d = 0
        if d < best:
            best = d
    return best - 1


# ------------------------------------------------------------------ 自测

def _selftest():
    ok = True

    def chk(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print("  FAIL %-52s got=%s want=%s" % (name, got, want))
        else:
            print("  ok   %-52s %s" % (name, got))

    def C(codes):
        # 牌码 -> 计数
        from mj_core import to_counts
        return to_counts(codes)

    print("[1] 无财神：基本向听数")
    chk("胡牌 123w456w789w123b东东", shanten(C(
        ["1w","2w","3w","4w","5w","6w","7w","8w","9w","1b","2b","3b","东","东"])), -1)
    chk("听牌 123w456w789w123b+东（单钓）", shanten(C(
        ["1w","2w","3w","4w","5w","6w","7w","8w","9w","1b","2b","3b","东"])), 0)
    chk("听牌 123w456w789w13b东东（嵌 2b）", shanten(C(
        ["1w","2w","3w","4w","5w","6w","7w","8w","9w","1b","3b","东","东"])), 0)
    chk("一向听 123w456w789w13b东南", shanten(C(
        ["1w","2w","3w","4w","5w","6w","7w","8w","9w","1b","3b","东","南"])), 1)
    chk("七对听牌 6对+1单", shanten(C(
        ["1w","1w","3w","3w","5w","5w","7w","7w","9w","9w","东","东","南"])), 0)
    chk("七对两向听 5对+3单", shanten(C(
        ["1w","1w","3w","3w","5w","5w","7w","7w","9w","9w","东","南","西"])), 1)
    chk("13 张全孤张（走七对路线 = 6 向听）", shanten(C(
        ["1w","4w","7w","1b","4b","7b","1t","4t","7t","东","南","西","北"])), 6)
    chk("同上取平胡分支 = 8 向听", shanten_regular(C(
        ["1w","4w","7w","1b","4b","7b","1t","4t","7t","东","南","西","北"])), 8)

    print("[2] 带财神")
    chk("4 面子 + 财神 = 听任意（爆头）", shanten(C(
        ["1w","2w","3w","4w","5w","6w","7w","8w","9w","1b","2b","3b","白"])), 0)
    chk("财神当将：4 面子 + 财神 已是胡", shanten(C(
        ["1w","2w","3w","4w","5w","6w","7w","8w","9w","1b","2b","3b","白","白"])), -1)
    chk("财神补顺子缺口", shanten(C(
        ["1w","2w","白","4w","5w","6w","7w","8w","9w","1b","2b","3b","东"])), 0)

    print("[3] 副露")
    chk("3 副露 + 暗牌 2w3w4w5w5w 已胡", shanten(C(["2w","3w","4w","5w","5w"]), 3), -1)
    chk("3 副露 + 暗牌 2w3w4w5w 听牌", shanten(C(["2w","3w","4w","5w"]), 3), 0)

    print("[4] 与穷举法交叉验证（独立实现）")
    import random
    for uni, name, trials in (
            (list(range(0, 9)) + [27, 28, 29, 30], "1w-9w + 4 字牌", 40),
            (list(range(0, 13)), "1w-9w + 1b-4b", 20)):
        shapes = {mc: [bytes(w) for w in _brute_shapes(set(uni), mc)]
                  for mc in (0, 1, 2)}
        print("   牌种集合 %-18s 胡牌型 %d 个，%d 手/档" %
              (name, len(shapes[0]), trials))
        random.seed(20240607)
        bad = 0
        for trial in range(trials):
            for meld_count in (0, 1, 2):
                n = 14 - 3 * meld_count
                pool = []
                for t in uni:
                    pool += [t] * 4
                pick = random.sample(pool, n)
                c = [0] * NTILE
                for t in pick:
                    c[t] += 1
                J = random.choice([0, 0, 1, 1, 2])
                for _ in range(J):
                    for t in uni:
                        if c[t] > 0:
                            c[t] -= 1
                            break
                c[WILD] = J
                got = shanten_regular(c, meld_count)
                want = _brute_shanten(c, meld_count, shapes[meld_count])
                if got != want:
                    bad += 1
                    if bad <= 3:
                        print("      MISMATCH meld=%d got=%d want=%d  %s" %
                              (meld_count, got, want,
                               [(i, v) for i, v in enumerate(c) if v]))
        chk("%s：随机 %d 手 × 3 副露档" % (name, trials), bad, 0)

    print("[4b] 与 mj_core.can_win 交叉验证（14 张，副露 0/1/2）")
    from mj_core import can_win, to_counts as _tc, tile_code as _tc2
    random.seed(99)
    pool = []
    for t in range(NTILE):
        pool += [t] * 4
    bad = 0
    for trial in range(3000):
        meld_count = random.choice([0, 0, 1, 2])
        n = 14 - 3 * meld_count
        if meld_count == 0:
            pick = random.sample(pool, n)
        else:
            # 副露时暗牌手数少，用「成型手 + 噪声」更容易命中胡牌
            pick = random.sample(pool, n)
        c = [0] * NTILE
        for t in pick:
            c[t] += 1
        if random.random() < 0.5:      # 掺入财神
            J = random.randint(1, min(4, n))
            for _ in range(J):
                for t in range(NTILE):
                    if t != WILD and c[t] > 0:
                        c[t] -= 1
                        break
            c[WILD] += J
        got = (shanten_regular(c, meld_count) == -1) or \
              (shanten_chiitoi(c, meld_count) == -1)
        want = can_win(c, meld_count)
        if got != want:
            bad += 1
            if bad <= 3:
                print("      MISMATCH meld=%d shanten=-1?%s can_win=%s %s" %
                      (meld_count, got, want,
                       [(t, c[t]) for t in range(NTILE) if c[t]]))
    chk("3000 手随机对照", bad, 0)

    print("[4c] 七对子 vs 穷举（7 个互异对子，12 牌种子宇宙）")
    import random
    from itertools import combinations as _comb
    uni_c = list(range(0, 9)) + [27, 28, 29]
    random.seed(7)
    bad = 0
    for trial in range(400):
        pool = []
        for t in uni_c:
            pool += [t] * 4
        pick = random.sample(pool, 13)
        c = [0] * NTILE
        for t in pick:
            c[t] += 1
        J = random.choice([0, 1, 1, 2, 3])
        for _ in range(J):
            for t in uni_c:
                if c[t] > 0:
                    c[t] -= 1
                    break
        c[WILD] = J
        best = 1 << 30
        for combo in _comb(uni_c, 7):
            w = [0] * NTILE
            for t in combo:
                w[t] = 2
            matched = sum(min(c[i], w[i]) for i in range(NTILE))
            d = 14 - matched - J
            if d < 0:
                d = 0
            if d < best:
                best = d
        want = best - 1
        got = shanten_chiitoi(c, 0)
        if got != want:
            bad += 1
            if bad <= 3:
                print("      MISMATCH got=%d want=%d %s" %
                      (got, want, [(t, c[t]) for t in range(NTILE) if c[t]]))
    chk("七对 400 手随机对照", bad, 0)

    print("[5] 有效牌")
    tot, got, base = ukeire(C(
        ["1w","2w","3w","4w","5w","6w","7w","8w","9w","1b","2b","东","东"]))
    from mj_core import tile_code
    print("   基准向听 %d，有效牌 %d 种 / %d 枚：%s" %
          (base, len(got), tot, sorted(tile_code(t) for t in got)))

    print()
    print("=== %s ===" % ("全部通过" if ok else "存在失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
