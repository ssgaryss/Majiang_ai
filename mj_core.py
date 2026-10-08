#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""杭州麻将牌理核心 —— 牌码、胡牌判定、听牌/爆头检测、出牌评估。

纯标准库、零依赖。全部判定以「白板 = 财神（百搭，可替代任意牌）」为前提。

牌码（与平台 API 一致）：
  1w-9w 万 / 1b-9b 筒 / 1t-9t 条 / 东 南 西 北 中 发 白

内部索引 0-33：
  0-8   1w..9w
  9-17  1b..9b
  18-26 1t..9t
  27-30 东南西北
  31-33 中发白   （33 = 白板 = 财神 = WILD）

关键结论（推导见 _form_sets 注释）：
  · 财神只需填「真实张数为 0」的位置即可，不必枚举「用财神顶替已有真牌」的分支
    —— 因为任何这样的方案都能通过「真牌归位、财神改去顶替它原来占的位」等价改写。
  · 七对带财神的判定简化为 sum(c%2) <= wilds（奇偶性由总张数=14 自动保证）。
"""
import sys

SUITS = ("w", "b", "t")
HONORS = "东南西北中发白"
WILD = 33                      # 白板 = 财神
NTILE = 34

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 牌码转换

def tile_index(code):
    """牌码 -> 0..33 索引。"""
    s = (code or "").strip()
    if not s:
        raise ValueError("空牌码")
    if s in HONORS:
        return 27 + HONORS.index(s)
    suit = s[-1]
    if suit in SUITS:
        n = int(s[:-1])
        if 1 <= n <= 9:
            return SUITS.index(suit) * 9 + (n - 1)
    raise ValueError("无法识别的牌码: %r" % (code,))


def tile_code(idx):
    """0..33 索引 -> 牌码。"""
    if idx < 27:
        return "%d%s" % (idx % 9 + 1, SUITS[idx // 9])
    return HONORS[idx - 27]


def to_counts(codes):
    """牌码列表 -> 34 元素计数数组。"""
    c = [0] * NTILE
    for x in codes:
        c[tile_index(x)] += 1
    return c


def to_codes(counts):
    """计数数组 -> 牌码列表（升序）。"""
    out = []
    for i, n in enumerate(counts):
        out.extend([tile_code(i)] * n)
    return out


def popcount(x):
    return bin(x).count("1")


# ---------------------------------------------------------------- 胡牌判定

def _form_sets(c, k, wilds, memo):
    """c（已剔除财神）能否恰好拆成 k 组面子，且把 wilds 张财神全部用掉。

    只考虑「财神填补真实张数为 0 的位置」的分支：若某方案把财神放在已有真牌的
    位置 p，则把该真牌挪回 p、财神改去顶替这张真牌原本所在组的角色，得到的仍是
    合法分解（财神万能）。故该限制不丢解，却能把分支数从 3*8 降到很小。

    每次只处理「最小非零索引 i」所在的那个组，保证枚举不重复、深度 <= k。
    """
    if k == 0:
        return sum(c) == 0 and wilds == 0
    key = (tuple(c), k, wilds)
    hit = memo.get(key)
    if hit is not None:
        return hit

    i = -1
    for x in range(NTILE):
        if c[x] > 0:
            i = x
            break
    if i < 0:                       # 只剩财神：3 张一组
        res = (wilds == 3 * k)
        memo[key] = res
        return res

    res = False
    # --- 刻子：i 的真牌全用上（最多 3 张），不足的用财神补 ---
    # 归一化后「够 3 张就无需财神」，故只有这一条分支。
    need_w = 0 if c[i] >= 3 else 3 - c[i]
    if wilds >= need_w:
        c[i] -= 3 - need_w
        if _form_sets(c, k - 1, wilds - need_w, memo):
            res = True
        c[i] += 3 - need_w

    # --- 顺子：i 可能落在顺子的第 0/1/2 位 ---
    # 关键：真实张数为 0 的位置「必须」全部由财神补，所以财神用量恰为 popcount(zero_mask)，
    # 没有子集可枚举（早先按子集枚举会让空位既不补牌又不补财神，直接减成负数 → 凭空造面子）。
    if not res and i < 27:
        lo = (i // 9) * 9
        hi = lo + 8
        for s in (i - 2, i - 1, i):
            if s < lo or s + 2 > hi:
                continue
            zero_mask = 0
            for b in range(3):
                if c[s + b] == 0:
                    zero_mask |= (1 << b)
            if zero_mask == 7:      # 三张全空：纯财神组，交给 i<0 分支处理
                continue
            need_w = popcount(zero_mask)
            if need_w > wilds:
                continue
            for b in range(3):
                if not (zero_mask >> b) & 1:
                    c[s + b] -= 1
            if _form_sets(c, k - 1, wilds - need_w, memo):
                res = True
            for b in range(3):
                if not (zero_mask >> b) & 1:
                    c[s + b] += 1
            if res:
                break
    memo[key] = res
    return res


def _seven_pairs(c, wilds):
    """七对：非财神部分的落单张数 <= 财神数即可（奇偶性由总张数 14 保证）。"""
    if sum(c) + wilds != 14:
        return False
    return sum(x % 2 for x in c) <= wilds


def can_win(counts, meld_count=0):
    """counts（34 元素，含财神）是否构成胡牌。

    meld_count = 已副露的面子数（吃/碰/杠各计 1 组），暗牌需补足 4-meld_count 组 + 1 将。
    """
    if sum(counts) != 3 * (4 - meld_count) + 2:
        return False
    if meld_count < 0 or meld_count > 4:
        return False

    wilds = counts[WILD]
    c = list(counts)
    c[WILD] = 0
    need_sets = 4 - meld_count

    if meld_count == 0 and _seven_pairs(c, wilds):
        return True

    # 将牌三形态：真+真 / 真+财神 / 财神+财神
    for i in range(NTILE):
        if c[i] >= 2:
            c[i] -= 2
            ok = _form_sets(c, need_sets, wilds, {})
            c[i] += 2
            if ok:
                return True
    if wilds >= 1:
        for i in range(NTILE):
            if c[i] >= 1:
                c[i] -= 1
                ok = _form_sets(c, need_sets, wilds - 1, {})
                c[i] += 1
                if ok:
                    return True
    if wilds >= 2:
        if _form_sets(c, need_sets, wilds - 2, {}):
            return True
    return False


def winning_tiles(counts, meld_count=0):
    """13 张手牌（或任意张）再摸一张能胡的牌码列表（升序）。"""
    out = []
    for t in range(NTILE):
        if counts[t] >= 4:          # 四张都在自己手里，物理上摸不到第五张
            continue
        c = list(counts)
        c[t] += 1
        if can_win(c, meld_count):
            out.append(tile_code(t))
    return out


def is_baotou(counts, meld_count=0):
    """爆头：听牌态摸「任意」一张都胡。"""
    avail = 0
    for t in range(NTILE):
        if counts[t] >= 4:
            continue
        avail += 1
        c = list(counts)
        c[t] += 1
        if not can_win(c, meld_count):
            return False
    return avail > 0


# ---------------------------------------------------------------- 吃的选牌
#
# 下面这段是**吃牌「用哪两张」的唯一实现**，策略（mj_policy）与模拟器（mj_sim）
# 都必须调它 —— 两份实现必然会分叉，而"模拟器与线上不一致"是本项目最贵的
# 一类 bug（模拟器里策略返回 gang 被 _try_gang 拒掉那次，就是静默失真）。

def chi_pairs(counts, tile):
    """列出吃这张牌的**所有**可行组合（顺序固定，与历史行为一致）。

    顺序：(t-2,t-1) → (t-1,t+1) → (t+1,t+2)。财神不能参与吃。
    返回 [(a, b), ...]，a/b 是**牌码索引**。
    """
    t = tile_index(tile)
    if t >= 27 or t == WILD:
        return []
    lo = (t // 9) * 9
    out = []
    for a, b in ((t - 2, t - 1), (t - 1, t + 1), (t + 1, t + 2)):
        if a < lo or b > lo + 8 or a == WILD or b == WILD:
            continue
        if counts[a] > 0 and counts[b] > 0:
            out.append((a, b))
    return out


# 吃牌「用哪两张」的选择模式。
#   0 = 取第一个可行组合（**历史行为**）      ← **当前默认，见下面的 A/B 结论**
#   1 = 在可行组合里取「吃完后向听最小」的那个；向听并列则仍取第一个
#
# 为什么这是个**从来没人做过的决定**：`chi_tiles` 一直是
# `for a,b in (...): 若手上都有就 return` —— "用哪两张吃"完全由**牌码顺序**
# 决定，从来不是策略选择。而提交动作里确实带 `tiles`
# （`{"action":"chi","tile":..,"tiles":pair}`），所以这个选择**是我们的**，
# 不是服务端替我们定的。
#
# 实测频率（`probe_chi_combo.py`，3000 局同质桌，见 STATE.md §26）：
#   · 吃 动作 ≈1.03 次/人·局；
#   · 其中 **14.5%** 存在 ≥2 个可行组合；
#   · 这些多组合局面里 **11.1%** 的"第一个"向听更差（明确劣）；
#   ⇒ 非最优选择 ≈ **0.017 次/人·局**。
#
# ★★★ 干预实验（`probe_chi_combo_ab.py`，2400 局/臂，同 seed 配对）：
#
#     A 取第一个（历史）  -0.7900 分/局   σ=18.33   自摸率 25.96%
#     B 取向听最小（新）  -0.9167 分/局   σ=18.06   自摸率 25.58%
#     配对差分 B−A = **-0.1267** 分/局   SE=0.1274  t=-0.99
#     95% CI [-0.3763, +0.1230]        B 更好 43 局 / B 更差 47 局（其余 2310 局相同）
#
#   ⇒ **未检出增益，点估计还偏负**。所以默认保持 0（历史行为），
#     这条改动**不上线**。
#
# ★★ 这个负结果本身是本次最有价值的东西，因为它**推翻了一个看起来很硬的先验论证**：
#   我原本的理由是"取向听最小在任何局面下都不会更差，因为向听是主目标"。
#   那个论证是**错的**，错在把"吃完那一瞬间的向听"当成了真实目标：
#     · 吃掉的牌不同 → 之后**可打的牌**不同 → 11 张那一瞬的向听更低，
#       不代表**弃牌之后**仍然更低（贪心局部最优 ≠ 全局最优）；
#     · 它还会破坏**将牌/对子**结构、拆掉**七对**与**爆头**路线 ——
#       而这两条正是本项目里唯一被证明值得追的高番路线。
#   这正是 §14 那条规矩的又一次胜利：
#     **横截面只能"提议"，只有干预才能"定罪"** ——
#     如果我只凭频率测量 + "显然不会更差"的论证就上线，
#     就会把一个（可能）有害的改动当成免费收益写进基因组史。
#
# 保留 mode 1 与其测试，是为了让这条**已被实验否定**的假设留在代码里可复现，
# 而不是删掉后过半年再被重新"发现"一次。
CHI_PICK = 0


def chi_tiles(counts, tile, meld_count=None):
    """用手上哪两张吃这张牌。不可吃返回 None（牌码字符串对）。

    `meld_count` 为本手**当前副露数**。给了它才启用 `CHI_PICK=1` 的最优选择；
    不给（None）则永远返回第一个组合 —— 旧调用点语义**完全不变**。
    """
    pairs = chi_pairs(counts, tile)
    if not pairs:
        return None
    if CHI_PICK != 1 or meld_count is None or len(pairs) == 1:
        return [tile_code(pairs[0][0]), tile_code(pairs[0][1])]
    import shanten_ref as _S          # 延迟导入：mj_core 不依赖 shanten_ref
    best, best_sh = pairs[0], None
    for a, b in pairs:
        c = list(counts)
        c[a] -= 1
        c[b] -= 1
        try:
            sh = _S.shanten(c, meld_count + 1)
        except Exception:                                      # noqa: BLE001
            continue
        if best_sh is None or sh < best_sh:
            best_sh, best = sh, (a, b)
    return [tile_code(best[0]), tile_code(best[1])]


# ---------------------------------------------------------------- 番型倍率
#
# 下面两个函数是**得分倍率**的唯一实现。此前 mj_sim 里那份手写乘法表从来
# 没跟官方对过（只有 can_win 对过），而 exp_ev.py 里还有第二份互相矛盾的
# 实现 —— 两份不可能都对。现在用官方 fan-calc 当基准逐例验（validate_fan.py）。

def seven_pairs_mult(counts):
    """七对分支倍数：0 = 不是七对；否则 2/4/8/16（七对/豪华/双豪华/三豪华）。

    官方口径（指南 §番型 + fan-calc 逐例验证）：
      · 无副露、共 14 张；**财神可补落单张**
      · 非财神四张 = 1 组「豪华」
      · 4 张真白板在其余牌全为自然对、未用于补落单时也算 1 组豪华
      · 白板已作百搭补配其他牌成对时不重复计豪华

    历史 bug（本次靠 oracle 抓出，6 副牌错 5 副）：早先实现遇到**任何奇数张的
    非财神牌就直接 return 0**，等于否定「白板补落单」，并让它后面那句
    `odd > c[WILD]` 变成永不执行的死代码。实测：
      白补1落单 官方 2 / 本地 0；豪华+白补落单 官方 4 / 本地 0；
      4真白+5对 官方 4(豪华×1) / 本地 2
    注意 `rest == 4` 与「4 张真白且 odd == 0」等价：白板总共只有 4 张。
    """
    if sum(counts) != 14:
        return 0
    j = counts[WILD]
    odd = 0
    quads = 0
    for t in range(NTILE):
        if t == WILD:
            continue
        if counts[t] % 2:
            odd += 1
        if counts[t] == 4:
            quads += 1
    if odd > j:                       # 落单张比财神还多，补不满
        return 0
    rest = j - odd
    if rest % 2:                      # 剩下的财神凑不成对
        return 0
    if rest == 4:                     # 4 张真白板全没用去补落单 -> 算 1 组豪华
        quads += 1
    return 2 * (2 ** quads)


def fan_multiple(counts, meld_count=0, chain_count=0, piao_count=0,
                 baotou=False):
    """总番倍率 —— 官方 fan-calc 逐例验证过的口径。

    官方公式：总番 = 1 × 分支因子 × 2^动作链次数 ×（4 白板 ×2）×（爆头 ×2）

      · 分支因子：无副露的七对 2/4/8/16，否则平胡 1
      · 动作链：每个「飘 / 杠」动作 ×2；打出非飘非杠的牌 -> 链断重新计数
      · 4 个白板：**胡牌时手留 + 链内飘出的白板 = 4** 即 ×2
        （普通打出的、杠出的不算 —— 故判据是 counts[WILD] + piao >= 4）
      · 爆头：听任意牌态，最后统一加 ×2

    验证覆盖（validate_fan.py，全部对上 oracle）：chain 0-4 × piao 0-count、
    手留白 0-4 × piao 0-4、七对各档、爆头各档，以及结算分数。
    """
    branch = 1
    if meld_count == 0:
        sp = seven_pairs_mult(counts)
        if sp:
            branch = sp
    mult = branch * (2 ** max(0, chain_count))
    if counts[WILD] + piao_count >= 4:
        mult *= 2
    if baotou:
        mult *= 2
    return max(1, mult)


# ---------------------------------------------------------------- 出牌评估

def tile_value(c, t):
    """单张牌在手上的「连接度」——越大越该留。"""
    if t == WILD:
        return 100.0                    # 财神几乎永远留手
    n = c[t]
    v = 0.0
    if n >= 3:
        v += 9.0
    elif n == 2:
        v += 5.0
    if t < 27:
        lo = (t // 9) * 9
        hi = lo + 8
        for d, w in ((-2, 2.0), (-1, 3.5), (1, 3.5), (2, 2.0)):
            p = t + d
            if lo <= p <= hi and c[p] > 0:
                v += w
    else:
        if n == 1:
            v += 0.5                    # 孤张字牌基本无用
        elif n == 2:
            v += 1.5
    return v


def hand_value(c, meld_count=0):
    """13 张手牌的粗略价值（越大越好）。含财神加成与副露进度。"""
    v = sum(tile_value(c, t) * c[t] for t in range(NTILE))
    # 已成面子数：3 张同牌 / 顺子（粗算，不做完整拆解）
    sets = 0
    cc = list(c)
    for t in range(NTILE):
        while cc[t] >= 3:
            cc[t] -= 3
            sets += 1
    for t in range(27):
        if t % 9 <= 6:
            while cc[t] and cc[t + 1] and cc[t + 2]:
                cc[t] -= 1
                cc[t + 1] -= 1
                cc[t + 2] -= 1
                sets += 1
    return v + sets * 6.0 + meld_count * 6.0


def best_discard(counts, meld_count=0, forbid=()):
    """选一张打出去：返回 (索引, 牌码)。forbid = 不可打的牌索引集合。"""
    best, best_v = None, None
    for t in range(NTILE):
        if counts[t] <= 0 or t in forbid:
            continue
        c = list(counts)
        c[t] -= 1
        v = hand_value(c, meld_count)
        if best is None or v > best_v:
            best, best_v = t, v
    if best is None:
        for t in range(NTILE):          # 全被 forbid 兜住时退化为打第一张有牌的
            if counts[t] > 0:
                return t, tile_code(t)
        return None, None
    return best, tile_code(best)


def melds_of(snap_melds, seat):
    """从快照 melds[seat]（[{kind,tiles},...]）取本人副露，返回 (面子数, 吃摊数, 杠数)。"""
    row = []
    if isinstance(snap_melds, dict):
        row = snap_melds.get(str(seat)) or snap_melds.get(seat) or []
    elif isinstance(snap_melds, list) and 0 <= seat < len(snap_melds):
        row = snap_melds[seat] or []
    sets = chi = gang = 0
    for m in row:
        if not isinstance(m, dict):
            continue
        kind = (m.get("kind") or "").lower()
        sets += 1
        if kind == "chi":
            chi += 1
        elif kind == "gang":
            gang += 1
    return sets, chi, gang


# ---------------------------------------------------------------- 自测

def _selftest():
    ok = True

    def chk(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print("  FAIL %-46s got=%s want=%s" % (name, got, want))
        else:
            print("  ok   %-46s %s" % (name, got))

    print("[1] 基础胡牌")
    chk("平胡 123w456w789w123b+东东",
        can_win(to_counts(["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                           "1b", "2b", "3b", "东", "东"])), True)
    chk("差一张不算胡",
        can_win(to_counts(["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                           "1b", "2b", "4b", "东", "东"])), False)

    print("[2] 财神（白板）替代")
    chk("1 张白补顺子缺口",
        can_win(to_counts(["1w", "2w", "白", "4w", "5w", "6w", "7w", "8w", "9w",
                           "1b", "2b", "3b", "东", "东"])), True)
    chk("2 张白补两组",
        can_win(to_counts(["1w", "2w", "白", "4w", "5w", "白", "7w", "8w", "9w",
                           "1b", "2b", "3b", "东", "东"])), True)
    chk("白自己成刻",
        can_win(to_counts(["白", "白", "白", "4w", "5w", "6w", "7w", "8w", "9w",
                           "1b", "2b", "3b", "东", "东"])), True)
    chk("白当将",
        can_win(to_counts(["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                           "1b", "2b", "3b", "东", "白"])), True)
    chk("白不够就胡不了",
        can_win(to_counts(["1w", "3w", "5w", "7w", "9w", "1b", "4b", "7b",
                           "1t", "4t", "7t", "东", "南", "白"])), False)

    print("[3] 七对")
    chk("七对",
        can_win(to_counts(["1w", "1w", "3w", "3w", "5w", "5w", "7w", "7w",
                           "9w", "9w", "东", "东", "白", "白"])), True)
    chk("单张多于财神则非七对",
        can_win(to_counts(["1w", "1w", "3w", "3w", "5w", "5w", "7w", "7w",
                           "9w", "9w", "东", "南", "西", "白"])), False)
    chk("东/南 各配一张财神 -> 仍是七对",
        can_win(to_counts(["1w", "1w", "3w", "3w", "5w", "5w", "7w", "7w",
                           "9w", "9w", "东", "南", "白", "白"])), True)
    chk("豪华七对(含四张)",
        can_win(to_counts(["1w", "1w", "1w", "1w", "3w", "3w", "5w", "5w",
                           "7w", "7w", "9w", "9w", "东", "东"])), True)

    print("[4] 副露后的胡牌")
    chk("3 副露 + 暗牌 2w3w4w5w 听牌",
        can_win(to_counts(["2w", "3w", "4w", "5w", "5w"]), 3), True)
    chk("4 副露只剩对子（单钓将）",
        can_win(to_counts(["东", "东"]), 4), True)
    chk("副露数不匹配则 False",
        can_win(to_counts(["东", "东"]), 3), False)

    print("[5] 听牌 / 爆头")
    chk("单钓将：123w456w789w111b+东 听东/白（白配东成将）",
        winning_tiles(to_counts(["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w",
                                 "9w", "1b", "1b", "1b", "东"])), ["东", "白"])
    wt = winning_tiles(to_counts(["2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                                  "1b", "2b", "3b", "东", "东"]))
    chk("两面听含 1w 与 4w", ("1w" in wt, "4w" in wt), (True, True))
    chk("9b 不是听牌张", "9b" in wt, False)
    chk("3 对+3 单+4 白 = 爆头（听任意）",
        is_baotou(to_counts(["1w", "1w", "3w", "3w", "5w", "5w", "7w", "9w",
                             "1b", "白", "白", "白", "白"])), True)
    chk("普通两面听不是爆头",
        is_baotou(to_counts(["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w",
                             "9w", "1b", "1b", "东", "南"])), False)

    print("[6] 出牌选择")
    c = to_counts(["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
                   "1b", "2b", "3b", "东", "北"])
    _, code = best_discard(c)
    chk("孤张字牌优先打（东/北 之一）", code in ("东", "北"), True)
    c2 = to_counts(["1w", "1w", "1w", "2w", "3w", "4w", "5w", "6w", "7w",
                    "8w", "9w", "白", "东", "南"])
    _, code2 = best_discard(c2)
    chk("财神绝不打出", code2 != "白", True)

    print("[7] 番型倍率（取值全部来自官方 fan-calc，见 validate_fan.py）")
    # 这一组是「回归锁」：下面每个期望值都是官方 /portal/api/tools/fan-calc
    # 实测返回的 fan，不是推导出来的。旧实现 6 副七对里错 5 副。
    chk("普通七对 -> x2", seven_pairs_mult(to_counts(
        ["1w", "1w", "2w", "2w", "3w", "3w", "4w", "4w", "5w", "5w",
         "6w", "6w", "7w", "7w"])), 2)
    chk("豪华七对 -> x4", seven_pairs_mult(to_counts(
        ["1w", "1w", "1w", "1w", "2w", "2w", "3w", "3w", "4w", "4w",
         "5w", "5w", "6w", "6w"])), 4)
    chk("三豪华七对 -> x16", seven_pairs_mult(to_counts(
        ["1w", "1w", "1w", "1w", "2w", "2w", "2w", "2w",
         "3w", "3w", "3w", "3w", "4w", "4w"])), 16)
    chk("白板补1落单仍是七对（旧实现返回 0）", seven_pairs_mult(to_counts(
        ["白", "1w", "2w", "2w", "3w", "3w", "4w", "4w", "5w", "5w",
         "6w", "6w", "7w", "7w"])), 2)
    chk("双白补2落单仍是七对（旧实现返回 0）", seven_pairs_mult(to_counts(
        ["白", "白", "1w", "8w", "2w", "2w", "3w", "3w", "4w", "4w",
         "5w", "5w", "6w", "6w"])), 2)
    chk("豪华+白补落单 -> x4（旧实现返回 0）", seven_pairs_mult(to_counts(
        ["1w", "1w", "1w", "1w", "2w", "白", "3w", "3w", "4w", "4w",
         "5w", "5w", "6w", "6w"])), 4)
    chk("4真白+5对 -> 豪华x1 = x4（旧实现返回 2）", seven_pairs_mult(to_counts(
        ["白", "白", "白", "白", "1w", "1w", "2w", "2w", "3w", "3w",
         "4w", "4w", "5w", "5w"])), 4)
    chk("4真白但补了2落单 -> 不算豪华", seven_pairs_mult(to_counts(
        ["白", "白", "白", "白", "1w", "2w", "3w", "3w", "4w", "4w",
         "5w", "5w", "6w", "6w"])), 2)
    chk("落单多于白板 -> 非七对", seven_pairs_mult(to_counts(
        ["白", "1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
         "1b", "2b", "3b", "4b"])), 0)
    chk("非七对 -> 0", seven_pairs_mult(to_counts(
        ["1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
         "1b", "2b", "3b", "东", "东"])), 0)

    PANHU = to_counts(["1w", "1w", "1w", "2w", "2w", "2w", "3w", "3w", "3w",
                       "4w", "4w", "4w", "5w", "5w"])
    chk("平胡无链无爆头 -> x1", fan_multiple(PANHU), 1)
    chk("动作链 n 次 -> x2^n", [fan_multiple(PANHU, 0, n) for n in range(5)],
        [1, 2, 4, 8, 16])
    chk("爆头 -> x2", fan_multiple(PANHU, 0, 0, 0, True), 2)
    chk("爆头 x 链2 -> x8", fan_multiple(PANHU, 0, 2, 0, True), 8)
    # 「4个白板」= 手留白 + 链内飘出的白 >= 4（普通打出的、杠出的不算）
    W2 = to_counts(["白", "白", "1w", "1w", "1w", "2w", "2w", "2w",
                    "3w", "3w", "3w", "4w", "4w", "4w"])
    chk("手留白2 -> 无 4白板加成", fan_multiple(W2), 1)
    chk("手留白2+飘2 -> 凑满 4 张 x2", fan_multiple(W2, 0, 2, 2), 8)
    chk("手留白2+纯杠2 -> 不加 4白板", fan_multiple(W2, 0, 2, 0), 4)
    # 七对与链/白板叠加（官方最大牌型可达 x1024）
    QID = to_counts(["1w", "1w", "1w", "1w", "2w", "2w", "2w", "2w",
                     "3w", "3w", "3w", "3w", "4w", "白"])
    chk("三豪华七对+飘1+爆头 -> 16x2x2", fan_multiple(QID, 0, 1, 1, True), 64)

    print()
    print("=== 自测%s ===" % ("全部通过" if ok else "存在失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
