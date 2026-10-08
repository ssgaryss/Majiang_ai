#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_teacher_data.py - 从 field_cache 抽取「高手决策样本」(阶段一：清点+抽原始记录)。

目标（B = 信任域蒸馏）：让模型学**强手的判断**，而不是继续跟自己练。
本脚本只做**阶段一**：把原始决策记录抽出来并清点质量 ✓（不做状态编码，那一步复用仓内编码器）。

★ 关键质量关：`timeout` 是"超时自动打"✗ —— 实测占 35% ✗ ⇒ **必须剔除** ✓
   否则等于把"没思考"当高手判断学 ✗（§⑫ 那类"学了反而更差"的典型来源）

用法: python tools/build_teacher_data.py [--max-files N]
输出: _tmp/teacher_raw.jsonl（每行一个"决策点"✓）＋ 屏幕清点报告
"""
import glob, json, os, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    mx = 10 ** 9
    if "--max-files" in sys.argv:
        mx = int(sys.argv[sys.argv.index("--max-files") + 1])
    files = sorted(glob.glob(os.path.join(ROOT, "field_cache", "*.json")))
    print("field_cache 文件数 =", len(files), " 本次处理 =", min(mx, len(files)))

    os.makedirs(os.path.join(ROOT, "_tmp"), exist_ok=True)
    out_path = os.path.join(ROOT, "_tmp", "teacher_raw.jsonl")
    out = open(out_path, "w", encoding="utf-8")

    n_ev = n_blk = n_dec = n_timeout = 0
    types = Counter(); seats_seen = Counter(); per_seat_dec = defaultdict(int)
    seats_shape = None
    for i, fp in enumerate(files[:mx]):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for ev in d.get("evs", []):
            n_ev += 1
            if seats_shape is None and "seats" in ev:
                s = ev["seats"]
                seats_shape = (type(s).__name__, (list(s)[:4] if isinstance(s, dict)
                                                  else (s[:4] if isinstance(s, list) else str(s)[:60])))
            for blk in ev.get("blocks", []):
                n_blk += 1
                dealer = blk.get("dealer"); rno = blk.get("round_no")
                sh = blk.get("start_hands")
                for e in blk.get("events", []):
                    t = e.get("type"); seats_seen[e.get("seat")] += 1; types[t] += 1
                    if t == "timeout":
                        n_timeout += 1
                        continue                      # ★ 剔除超时 ✗
                    if t in ("tile_drawn",):           # 摸牌不是"选择"✗
                        continue
                    n_dec += 1
                    per_seat_dec[e.get("seat")] += 1
                    out.write(json.dumps({
                        "file": os.path.basename(fp), "game_id": ev.get("game_id"),
                        "round_no": rno, "dealer": dealer, "seq": e.get("seq"),
                        "seat": e.get("seat"), "type": t, "tile": e.get("tile"),
                        "has_start_hands": bool(sh),
                    }, ensure_ascii=False) + "\n")
    out.close()

    print("--- 清点 ---")
    print("  局(evs) = %d ｜ 小局(blocks) = %d" % (n_ev, n_blk))
    print("  事件类型 =", dict(types.most_common(10)))
    print("  ★ timeout = %d（占全部事件 %.1f%%）⇒ **已剔除** ✓" % (n_timeout, 100.0 * n_timeout / max(1, sum(types.values()))))
    print("  ★ 可用决策点（非 timeout / 非摸牌）= **%d**" % n_dec)
    print("  seats 字段形状 =", seats_shape)
    print("  按座位分布 =", dict(sorted(per_seat_dec.items(), key=lambda kv: str(kv[0]))))
    print("  输出 =", out_path, "%.1f MB" % (os.path.getsize(out_path) / 1e6))

if __name__ == "__main__":
    main()