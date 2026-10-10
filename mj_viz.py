#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""麻将 bot 数据可视化（单文件、零第三方依赖、可打包成 exe）。

它做什么
--------
把工作目录里四类数据源扫成一份统一的数据集，然后：
  * 在浏览器里画出来（本地 HTTP 服务 + 内嵌单页，**不依赖任何 CDN**）；
  * 同时把**同一份数字**导出成 JSON/CSV（`viz_export/`），
    这样"人看到的"和"AI 分析的"是**同一个来源**，不会各说各话。

四类数据源
----------
1. `live_*.jsonl` / `room_*.jsonl` —— 线上每一房的逐场结果
   （`genome` 就是**那一房用的策略**，所以能按策略分组）。
2. `reviews/*.json` —— 复盘，最细：逐局有 `winner` / `fan` / `detail`（番型名）。
3. `probe_*_shard*.jsonl` —— **离线姿态实验**，每个 seed 一行，
   带 `g`（姿态=基因组）⇒ 这是"每种策略的效果"最干净的来源（同 seed 可配对）。
4. `field_ranking.json` / `field_profile.json` —— 全场账号排名与行为画像。

用法
----
    python mj_viz.py                 # 起服务并自动开浏览器
    python mj_viz.py --port 8765
    python mj_viz.py --export-only   # 只导出，不起服务
    python mj_viz.py --data-dir <放 live*.jsonl / reviews 的目录>   # 默认就是本目录

口径（每个数字怎么来的，页面上「指标说明」也写了同一份）
--------------------------------------------------------
* 一局 = 一次结算（`rounds[]` 里的一行）；一场 = 一个房间里的 10 局。
* `μ = 总得分 / 局数`；`σ` 是**逐局得分**的标准差。
* 胜率、自摸率、均番、爆头率都按**局**统计。
* 策略对比里 `Δμ` 是**同 seed 配对差**（同 seed ⇒ 同起手牌），
  不是两组均值的差 —— 后者会把牌运算进策略里。
"""
import argparse
import csv
import glob
import http.server
import json
import math
import os
import socketserver
import sys
import threading
import time
import webbrowser
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO_PAT = "live_*.jsonl"
TEST_PAT = "room_*.jsonl"
PROBE_PAT = "probe_*_shard*.jsonl"


# ==========================================================================
# 数据扫描
# ==========================================================================
def _read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except Exception:                                # noqa: BLE001
                    pass
    except Exception:                                            # noqa: BLE001
        pass
    return out


def scan_live(data_dir):
    """线上房间：一行 = 一房。按文件归类（自动房 / 测试房）。"""
    rooms = []
    for pat, kind in ((AUTO_PAT, "auto"), (TEST_PAT, "test")):
        for p in sorted(glob.glob(os.path.join(data_dir, pat))):
            base = os.path.basename(p)
            for d in _read_jsonl(p):
                rows = d.get("rows") or []
                if not rows:
                    continue
                n_rounds = 0
                wins = 0
                deltas_n = 0
                games = []
                for r in rows:
                    nr = int(r.get("rounds") or 0)
                    n_rounds += nr
                    sc = r.get("scores") or []
                    seat = r.get("seat")
                    mine = r.get("mine")
                    my_rounds = r.get("round_deltas") or []
                    w = 0
                    for rd in my_rounds:
                        if not isinstance(rd, list) or len(rd) != 4:
                            continue
                        deltas_n += 1
                        if seat is not None and rd[seat] > 0:
                            w += 1
                    wins += w
                    games.append({
                        "game": r.get("game"), "seat": seat, "mine": mine,
                        "scores": sc, "rounds": nr, "wins": w,
                        "avg_shanten": r.get("avg_shanten"),
                        "discards": r.get("discards"),
                    })
                total = sum(float(g["mine"] or 0) for g in games)
                rooms.append({
                    "room": d.get("room"), "kind": kind, "file": base,
                    "genome": d.get("genome") or "?",
                    "started": d.get("started"), "ended": d.get("ended"),
                    "games": len(games), "n_rounds": n_rounds,
                    "total": total,
                    "mu": (total / n_rounds) if n_rounds else 0.0,
                    "win_rate": (wins / deltas_n) if deltas_n else None,
                    "throttled": d.get("throttled"),
                    "missed_hu": d.get("missed_hu"),
                    "rows": games,
                })
    rooms.sort(key=lambda r: r.get("started") or 0)
    return rooms


def scan_reviews(data_dir):
    """复盘：逐局有 winner/fan/detail，是番型统计的唯一来源。"""
    out = []
    for p in sorted(glob.glob(os.path.join(data_dir, "reviews", "*.json"))):
        if ".reviewed" in os.path.basename(p):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
        if not d.get("room"):
            continue
        rounds = []
        for g in (d.get("per_game") or []):
            for r in (g.get("rounds") or []):
                rounds.append({
                    "game": g.get("game_id"), "batch": g.get("batch"),
                    "seat": g.get("seat"),
                    "round_no": r.get("round_no"), "dealer": r.get("dealer"),
                    "winner": r.get("winner"), "fan": r.get("fan"),
                    "scores": r.get("scores"), "mine": r.get("mine"),
                    "we_dealer": r.get("we_dealer"),
                    "detail": r.get("detail") or [],
                })
        out.append({
            "room": d.get("room"), "at": d.get("at"),
            "games": d.get("games"), "n_rounds": d.get("n_rounds"),
            "total": d.get("total"), "mu": d.get("mu"), "sd": d.get("sd"),
            "wins": d.get("wins"), "self_draw_rate": d.get("self_draw_rate"),
            "mine_fan_avg": d.get("mine_fan_avg"),
            "opp_fan_avg": d.get("opp_fan_avg"),
            "mine_baotou": d.get("mine_baotou"),
            "mine_chiitoi": d.get("mine_chiitoi"),
            "opp_baotou": d.get("opp_baotou"),
            "opp_chiitoi": d.get("opp_chiitoi"),
            "draws": d.get("draws"), "identity_bad": d.get("identity_bad"),
            "seats": [s.get("name") for s in (d.get("seats") or [])],
            "self_check": d.get("self_check"),
            "rounds": rounds,
        })
    out.sort(key=lambda r: r.get("at") or "")
    return out


def scan_probes(data_dir):
    """离线姿态实验：每行 (seed, g) ⇒ 可做同 seed 配对。"""
    per_g = defaultdict(list)
    files = sorted(glob.glob(os.path.join(data_dir, PROBE_PAT)))
    for p in files:
        for d in _read_jsonl(p):
            g = d.get("g")
            if not g or "seed" not in d:
                continue
            per_g[g].append(d)
    strategies = []
    for g, rows in per_g.items():
        sc = [float(r.get("score") or 0) for r in rows]
        n = len(sc)
        mu = sum(sc) / n if n else 0.0
        var = (sum((x - mu) ** 2 for x in sc) / (n - 1)) if n > 1 else 0.0
        wins = [r for r in rows if r.get("win")]
        fans = [float(r.get("fan") or 0) for r in wins]
        strategies.append({
            "genome": g, "n": n, "mu": mu, "sd": math.sqrt(var),
            "win_rate": len(wins) / n if n else 0.0,
            "fan_avg": (sum(fans) / len(fans)) if fans else 0.0,
            "baotou_rate": (sum(1 for r in rows if r.get("baotou")) / n) if n else 0.0,
            "chiitoi_rate": (sum(1 for r in rows if r.get("chiitoi")) / n) if n else 0.0,
            "draw_rate": (sum(1 for r in rows if r.get("draw")) / n) if n else 0.0,
            "melds_avg": (sum(float(r.get("melds_n") or 0) for r in rows) / n) if n else 0.0,
        })
    strategies.sort(key=lambda s: -s["mu"])

    # 同 seed 配对：posture -> {seed: score}
    seedmap = {}
    for g, rows in per_g.items():
        m = {}
        for r in rows:
            m[r["seed"]] = float(r.get("score") or 0)
        seedmap[g] = m
    pairs = {}
    keys = list(seedmap)
    for a in keys:
        for b in keys:
            if a >= b:
                continue
            ma, mb = seedmap[a], seedmap[b]
            common = ma.keys() & mb.keys()
            if len(common) < 30:
                continue
            ds = [ma[s] - mb[s] for s in common]
            k = len(ds)
            mean = sum(ds) / k
            var = (sum((x - mean) ** 2 for x in ds) / (k - 1)) if k > 1 else 0.0
            pairs["%s||%s" % (a, b)] = {
                "a": a, "b": b, "n": k, "dmu": mean,
                "se": math.sqrt(var / k) if k else 0.0,
            }
    return {"strategies": strategies, "pairs": pairs, "n_files": len(files)}


def scan_field(data_dir):
    out = {"ranking": [], "profile": {}, "meta": {}}
    p = os.path.join(data_dir, "field_ranking.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            out["ranking"] = d.get("rows") or []
            out["meta"] = d.get("meta") or {}
        except Exception:                                        # noqa: BLE001
            pass
    p = os.path.join(data_dir, "field_profile.json")
    if os.path.exists(p):
        try:
            out["profile"] = json.load(open(p, encoding="utf-8")) or {}
        except Exception:                                        # noqa: BLE001
            pass
    return out


def build(data_dir):
    t0 = time.time()
    live = scan_live(data_dir)
    reviews = scan_reviews(data_dir)
    probes = scan_probes(data_dir)
    field = scan_field(data_dir)

    # 线上分策略汇总
    by_genome = defaultdict(lambda: {"rooms": 0, "n_rounds": 0, "total": 0.0,
                                     "kinds": Counter()})
    for r in live:
        e = by_genome[r["genome"]]
        e["rooms"] += 1
        e["n_rounds"] += r["n_rounds"]
        e["total"] += r["total"]
        e["kinds"][r["kind"]] += 1
    online = []
    for g, e in by_genome.items():
        n = e["n_rounds"]
        online.append({"genome": g, "rooms": e["rooms"], "n_rounds": n,
                       "total": e["total"], "mu": (e["total"] / n) if n else 0.0,
                       "kinds": dict(e["kinds"])})
    online.sort(key=lambda x: -x["mu"])

    ds = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": data_dir,
        "online": online,
        "live_rooms": live,
        "reviews": reviews,
        "probe_strategies": probes["strategies"],
        "probe_pairs": probes["pairs"],
        "probe_files": probes["n_files"],
        "field": field,
        "scan_seconds": round(time.time() - t0, 2),
    }
    return ds


# ==========================================================================
# 导出（人看到的和 AI 分析的必须是同一份数字）
# ==========================================================================
def export(ds, outdir):
    os.makedirs(outdir, exist_ok=True)
    written = []
    p = os.path.join(outdir, "dataset.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(ds, fh, ensure_ascii=False, indent=1)
    written.append(p)

    def wcsv(name, header, rows):
        q = os.path.join(outdir, name)
        with open(q, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        written.append(q)

    wcsv("online_by_strategy.csv",
         ["genome", "rooms", "n_rounds", "total", "mu"],
         [[s["genome"], s["rooms"], s["n_rounds"], round(s["total"], 1),
           round(s["mu"], 4)] for s in ds["online"]])

    wcsv("live_rooms.csv",
         ["room", "kind", "genome", "started", "games", "n_rounds", "total", "mu",
          "win_rate", "throttled", "missed_hu"],
         [[r["room"], r["kind"], r["genome"],
           time.strftime("%Y-%m-%d %H:%M", time.localtime(r["started"]))
           if r.get("started") else "",
           r["games"], r["n_rounds"], round(r["total"], 1), round(r["mu"], 4),
           "" if r["win_rate"] is None else round(r["win_rate"], 4),
           r.get("throttled"), r.get("missed_hu")] for r in ds["live_rooms"]])

    wcsv("probe_strategies.csv",
         ["genome", "n", "mu", "sd", "win_rate", "fan_avg", "baotou_rate",
          "chiitoi_rate", "draw_rate", "melds_avg"],
         [[s["genome"], s["n"], round(s["mu"], 4), round(s["sd"], 4),
           round(s["win_rate"], 4), round(s["fan_avg"], 4),
           round(s["baotou_rate"], 4), round(s["chiitoi_rate"], 4),
           round(s["draw_rate"], 4), round(s["melds_avg"], 4)]
          for s in ds["probe_strategies"]])

    pl = sorted(ds["probe_pairs"].values(), key=lambda x: -abs(x["dmu"]))
    wcsv("probe_pairs.csv",
         ["a", "b", "n", "dmu_a_minus_b", "se", "t"],
         [[x["a"], x["b"], x["n"], round(x["dmu"], 4), round(x["se"], 4),
           round(x["dmu"] / x["se"], 2) if x["se"] else ""] for x in pl])

    wcsv("review_rooms.csv",
         ["room", "at", "games", "n_rounds", "total", "mu", "sd", "wins",
          "self_draw_rate", "mine_fan_avg", "opp_fan_avg", "mine_baotou",
          "mine_chiitoi", "identity_bad"],
         [[r["room"], r["at"], r["games"], r["n_rounds"], r["total"], r["mu"],
           r["sd"], r["wins"], r["self_draw_rate"], r["mine_fan_avg"],
           r["opp_fan_avg"], r["mine_baotou"], r["mine_chiitoi"],
           r["identity_bad"]] for r in ds["reviews"]])

    wcsv("field_ranking.csv",
         ["name", "uid", "rooms", "n", "mu", "wr", "fan", "lo", "hi"],
         [[r.get("name"), r.get("uid"), r.get("rooms"), r.get("n"),
           round(r.get("mu", 0), 4), round(r.get("wr", 0), 4),
           round(r.get("fan", 0), 4), round(r.get("lo", 0), 4),
           round(r.get("hi", 0), 4)] for r in ds["field"]["ranking"]])

    return written


# ==========================================================================
# 页面（内嵌，无 CDN）
# ==========================================================================
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>麻将 bot 数据看板</title>
<style>
*{box-sizing:border-box}
body{margin:0;font:13px/1.55 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;
     background:#0f1420;color:#dbe3f0}
header{padding:14px 18px;background:#161d2e;border-bottom:1px solid #26314a;
       position:sticky;top:0;z-index:10}
h1{margin:0 0 4px;font-size:16px;font-weight:600}
.sub{color:#8494b0;font-size:12px}
.kpis{display:flex;gap:22px;flex-wrap:wrap;margin-top:10px}
.kpi{background:#1b2438;border:1px solid #26314a;border-radius:8px;padding:7px 12px}
.kpi b{display:block;font-size:18px;color:#eaf1ff}
.kpi span{color:#8494b0;font-size:11px}
nav{display:flex;gap:6px;padding:10px 18px 0;flex-wrap:wrap}
nav button{background:#1b2438;color:#b9c6dd;border:1px solid #26314a;
  border-radius:7px;padding:6px 13px;cursor:pointer;font-size:13px}
nav button.on{background:#2d6cdf;color:#fff;border-color:#2d6cdf}
main{padding:14px 18px 60px}
section{display:none}section.on{display:block}
.card{background:#161d2e;border:1px solid #26314a;border-radius:10px;
  padding:13px 15px;margin-bottom:14px}
.card h3{margin:0 0 10px;font-size:13px;color:#9fb0cc;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:5px 8px;text-align:right;border-bottom:1px solid #222c42;
  white-space:nowrap}
th{color:#8494b0;font-weight:600;position:sticky;top:0;background:#161d2e}
td:first-child,th:first-child,td.l,th.l{text-align:left}
tr:hover td{background:#1b2438}
.pos{color:#57d98a}.neg{color:#ff7b7b}
.scroll{max-height:560px;overflow:auto}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
select,input{background:#0f1420;color:#dbe3f0;border:1px solid #33405e;
  border-radius:6px;padding:5px 8px;font-size:12px}
label{color:#8494b0;font-size:12px}
svg{display:block;width:100%;overflow:visible}
.tip{position:fixed;pointer-events:none;background:#000c;border:1px solid #3b4a6b;
  border-radius:6px;padding:5px 8px;font-size:12px;color:#fff;display:none;z-index:99}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#8494b0;margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.note{color:#8494b0;font-size:12px;line-height:1.7}
code{background:#0f1420;border:1px solid #2a3550;border-radius:4px;padding:1px 5px;
  color:#9fd0ff}
</style></head><body>
<header>
  <h1>麻将 bot 数据看板 <span class="sub" id="gen"></span></h1>
  <div class="sub" id="liveStamp" style="margin:4px 0 8px"></div>
  <div class="kpis" id="kpis"></div>
</header>
<nav id="nav"></nav>
<main>
  <section id="tab-overview" class="on">
    <div class="card"><h3>逐房累计得分（按时间）</h3><div id="c-cum"></div>
      <div class="legend"><span><i style="background:#2d6cdf"></i>累计分</span>
      <span><i style="background:#e0a33a"></i>滚动 μ（每 10 房，分/局）</span></div></div>
    <div class="card"><h3>逐房场均分（每房 10 场）</h3><div id="c-room"></div></div>
  </section>

  <section id="tab-games">
    <div class="card">
      <div class="row"><label>房间</label><select id="roomSel"></select>
        <label>数据源</label><select id="gameSrc">
          <option value="live">线上结果文件（快）</option>
          <option value="review">复盘（含番型逐局）</option></select></div>
      <div id="gameKpi" class="row"></div>
      <div id="c-games"></div>
    </div>
    <div class="card"><h3>逐局明细</h3><div class="scroll" id="roundTbl"></div></div>
  </section>

  <section id="tab-strategy">
    <div class="card"><h3>离线姿态实验 —— 每种策略（基因组）的效果</h3>
      <div class="row"><label>排序</label><select id="sortBy">
        <option value="mu">μ</option><option value="win_rate">胜率</option>
        <option value="fan_avg">均番</option><option value="baotou_rate">爆头率</option>
        <option value="n">样本量</option></select>
        <label>最少样本</label><input id="minN" type="number" value="0" style="width:80px">
      </div>
      <div id="c-strat"></div><div class="scroll" id="stratTbl"></div></div>
    <div class="card"><h3>同 seed 配对差 Δμ（同起手牌，才分得清策略与牌运）</h3>
      <div class="row"><label>基准策略</label><select id="baseSel"></select>
        <label>只看 |t| ≥</label><input id="minT" type="number" value="0" step="0.5" style="width:70px"></div>
      <div class="note" style="margin-bottom:8px">
        口径：<code>Δμ = 对比策略 − 基准策略</code>（**正数 = 这个对比策略比基准好**）。
        同一 seed ⇒ 同一副起手牌，逐局相减再平均 ⇒ **牌运被消掉**，
        剩下的才是策略差异。<code>t = Δμ / SE</code>，<code>|t| ≥ 1.96</code> 标绿/红。
      </div>
      <div class="scroll" id="pairTbl"></div></div>
    <div class="card"><h3>线上房间按策略汇总</h3><div class="scroll" id="onlineTbl"></div></div>
  </section>

  <section id="tab-field">
    <div class="card"><h3>全场账号排名（来自 field_ranking.json）</h3>
      <div class="scroll" id="fieldTbl"></div></div>
    <div class="card"><h3>μ vs 进听率 / 爆头率（每个点是一个账号）</h3>
      <div class="row"><label>横轴</label><select id="axSel">
        <option value="tenpai_rate">进听率</option>
        <option value="baotou_per_round">爆头/局</option>
        <option value="melds_per_round">副露/局</option>
        <option value="width">听口宽度</option>
        <option value="win_given_tenpai">胡|进听</option>
        <option value="speed">进听速度</option></select></div>
      <div id="c-scatter"></div><div id="corr" class="note"></div></div>
  </section>

  <section id="tab-help">
    <div class="card"><h3>指标说明（每个数字怎么来的）</h3><div class="note" id="helpBox"></div></div>
  </section>
</main>
<div class="tip" id="tip"></div>
<script>
const S={data:null};
const fmt=(x,d=2)=>x===null||x===undefined||isNaN(x)?"—":(+x).toFixed(d);
const sgn=x=>(x>0?"pos":x<0?"neg":"");
const sp=x=>(x>0?"+":"")+fmt(x,2);
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function tip(el,html){
  const t=document.getElementById("tip");
  el.addEventListener("mousemove",e=>{t.innerHTML=html;t.style.display="block";
    t.style.left=(e.clientX+14)+"px";t.style.top=(e.clientY+14)+"px";});
  el.addEventListener("mouseleave",()=>t.style.display="none");
}
// --- 极简 SVG 折线/柱状/散点 ---
function lineChart(host,series,opt){
  opt=opt||{};const W=host.clientWidth||900,H=opt.h||220,P={l:52,r:14,t:12,b:26};
  const all=series.flatMap(s=>s.pts.map(p=>p[1]));
  if(!all.length){host.innerHTML='<div class="note">没有数据</div>';return;}
  let y0=Math.min(0,...all),y1=Math.max(0,...all);if(y1===y0){y1=y0+1;}
  const n=Math.max(...series.map(s=>s.pts.length),1);
  const X=i=>P.l+(W-P.l-P.r)*(n<=1?0.5:i/(n-1));
  const Y=v=>H-P.b-(H-P.t-P.b)*(v-y0)/(y1-y0);
  let g="";
  for(let k=0;k<=4;k++){const v=y0+(y1-y0)*k/4,y=Y(v);
    g+=`<line x1="${P.l}" y1="${y}" x2="${W-P.r}" y2="${y}" stroke="#222c42"/>`+
       `<text x="${P.l-6}" y="${y+4}" fill="#6b7c99" font-size="10" text-anchor="end">${fmt(v,0)}</text>`;}
  series.forEach(s=>{
    const d=s.pts.map((p,i)=>`${i?"L":"M"}${X(i)},${Y(p[1])}`).join(" ");
    g+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.8"/>`;});
  if(opt.hover!==false)series.forEach(s=>s.pts.forEach((p,i)=>{
    g+=`<circle class="hv" cx="${X(i)}" cy="${Y(p[1])}" r="7" fill="transparent"
        data-t="${esc(p[0])}｜${esc(s.name)} = ${fmt(p[1],2)}"/>`;}));
  host.innerHTML=`<svg viewBox="0 0 ${W} ${H}" height="${H}">${g}</svg>`;
  host.querySelectorAll(".hv").forEach(c=>tip(c,c.dataset.t));
}
function barChart(host,items,opt){
  opt=opt||{};const W=host.clientWidth||900,H=opt.h||200,P={l:52,r:14,t:12,b:44};
  if(!items.length){host.innerHTML='<div class="note">没有数据</div>';return;}
  const vals=items.map(i=>i.v);let y0=Math.min(0,...vals),y1=Math.max(0,...vals);
  if(y1===y0)y1=y0+1;
  const bw=(W-P.l-P.r)/items.length;
  const Y=v=>H-P.b-(H-P.t-P.b)*(v-y0)/(y1-y0);
  let g="";
  for(let k=0;k<=4;k++){const v=y0+(y1-y0)*k/4,y=Y(v);
    g+=`<line x1="${P.l}" y1="${y}" x2="${W-P.r}" y2="${y}" stroke="#222c42"/>`+
       `<text x="${P.l-6}" y="${y+4}" fill="#6b7c99" font-size="10" text-anchor="end">${fmt(v,0)}</text>`;}
  const yz=Y(0);
  items.forEach((it,i)=>{
    const x=P.l+i*bw+bw*0.12,w=bw*0.76,y=it.v>=0?Y(it.v):yz;
    const h=Math.abs(Y(it.v)-yz);
    g+=`<rect class="hv" x="${x}" y="${y}" width="${w}" height="${Math.max(1,h)}"
        fill="${it.v>=0?"#2d6cdf":"#c0504d"}" data-t="${esc(it.t)}｜${fmt(it.v,2)}"/>`;
    g+=`<text x="${x+w/2}" y="${H-P.b+13}" fill="#6b7c99" font-size="10"
        text-anchor="end" transform="rotate(-40 ${x+w/2} ${H-P.b+13})">${esc(it.l).slice(0,14)}</text>`;});
  host.innerHTML=`<svg viewBox="0 0 ${W} ${H}" height="${H}">${g}</svg>`;
  host.querySelectorAll(".hv").forEach(c=>tip(c,c.dataset.t));
}
function scatter(host,pts,xl,yl){
  const W=host.clientWidth||900,H=300,P={l:56,r:16,t:14,b:40};
  if(!pts.length){host.innerHTML='<div class="note">没有数据</div>';return;}
  const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  if(x1===x0)x1=x0+1;if(y1===y0)y1=y0+1;
  const X=v=>P.l+(W-P.l-P.r)*(v-x0)/(x1-x0),Y=v=>H-P.b-(H-P.t-P.b)*(v-y0)/(y1-y0);
  let g="";
  for(let k=0;k<=4;k++){const y=P.t+(H-P.t-P.b)*k/4,v=y1-(y1-y0)*k/4;
    g+=`<line x1="${P.l}" y1="${y}" x2="${W-P.r}" y2="${y}" stroke="#222c42"/>`+
       `<text x="${P.l-6}" y="${y+4}" fill="#6b7c99" font-size="10" text-anchor="end">${fmt(v,2)}</text>`;}
  for(let k=0;k<=4;k++){const x=P.l+(W-P.l-P.r)*k/4,v=x0+(x1-x0)*k/4;
    g+=`<text x="${x}" y="${H-P.b+15}" fill="#6b7c99" font-size="10" text-anchor="middle">${fmt(v,1)}</text>`;}
  pts.forEach(p=>{g+=`<circle class="hv" cx="${X(p.x)}" cy="${Y(p.y)}" r="4.5"
    fill="${p.me?"#e0a33a":"#4a7fd6"}" opacity="0.9"
    data-t="${esc(p.n)}｜${esc(xl)}=${fmt(p.x,2)}｜μ=${fmt(p.y,3)}｜${p.n2}局"/>`;});
  host.innerHTML=`<svg viewBox="0 0 ${W} ${H}" height="${H}">
    <text x="${P.l}" y="${P.t-2}" fill="#6b7c99" font-size="10">${esc(yl)}</text>
    <text x="${W-P.r}" y="${H-4}" fill="#6b7c99" font-size="10" text-anchor="end">${esc(xl)}</text>
    ${g}</svg>`;
  host.querySelectorAll(".hv").forEach(c=>tip(c,c.dataset.t));
}
function table(host,cols,rows){
  if(!rows.length){host.innerHTML='<div class="note">没有数据</div>';return;}
  let h="<table><thead><tr>"+cols.map(c=>`<th class="${c.l?"l":""}">${esc(c.t)}</th>`).join("")+"</tr></thead><tbody>";
  rows.forEach(r=>{h+="<tr>"+cols.map(c=>{const v=r[c.k];
    return `<td class="${c.l?"l":""} ${c.cls?c.cls(v,r):""}>${c.f?c.f(v,r):esc(v)}</td>`;}).join("")+"</tr>";});
  host.innerHTML=h+"</tbody></table>";
}
function pearson(p){const n=p.length;if(n<3)return NaN;
  const mx=p.reduce((a,b)=>a+b.x,0)/n,my=p.reduce((a,b)=>a+b.y,0)/n;
  let sxy=0,sxx=0,syy=0;p.forEach(q=>{sxy+=(q.x-mx)*(q.y-my);sxx+=(q.x-mx)**2;syy+=(q.y-my)**2;});
  return (sxx&&syy)?sxy/Math.sqrt(sxx*syy):NaN;}

// ---------------- 渲染 ----------------
function renderKpis(){
  const d=S.data,lr=d.live_rooms;
  const nr=lr.reduce((a,b)=>a+b.n_rounds,0);
  const tot=lr.reduce((a,b)=>a+b.total,0);
  const mu=nr?tot/nr:0;
  const st=d.probe_strategies;
  const best=st.length?st.reduce((a,b)=>a.mu>b.mu?a:b):null;
  const mine=d.field.ranking.find(r=>r.uid==="u_cec3bcdb6d93")||
             d.field.ranking.find(r=>r.rooms>=Math.max(...d.field.ranking.map(x=>x.rooms)));
  document.getElementById("kpis").innerHTML=`
   <div class="kpi"><b>${lr.length}</b><span>线上房间</span></div>
   <div class="kpi"><b>${nr}</b><span>局数</span></div>
   <div class="kpi"><b class="${sgn(mu)}">${sp(mu)}</b><span>总体 μ（分/局）</span></div>
   <div class="kpi"><b>${st.length}</b><span>离线策略数</span></div>
   <div class="kpi"><b>${d.reviews.length}</b><span>复盘房间</span></div>
   ${best?`<div class="kpi"><b class="pos">${sp(best.mu)}</b><span>最好策略 μ</span></div>`:""}
   ${mine?`<div class="kpi"><b class="${sgn(mine.mu)}">${sp(mine.mu)}</b><span>我们（全场排名）</span></div>`:""}`;
  document.getElementById("gen").textContent=
    `生成于 ${d.generated_at}　数据目录 ${d.data_dir}　扫描 ${d.scan_seconds}s`;
}
function renderOverview(){
  const lr=S.data.live_rooms;let c=0;const cum=[],roll=[];
  const w=[];
  lr.forEach((r,i)=>{c+=r.total;cum.push([r.room,c]);
    w.push(r.mu);if(w.length>10)w.shift();
    roll.push([r.room,w.reduce((a,b)=>a+b,0)/w.length]);});
  lineChart(document.getElementById("c-cum"),
    [{name:"累计分",color:"#2d6cdf",pts:cum},{name:"滚动μ×100",color:"#e0a33a",
      pts:roll.map(p=>[p[0],p[1]*100])}]);
  barChart(document.getElementById("c-room"),
    lr.map(r=>({l:(r.room||"").slice(2,8),v:r.total,t:`${r.room}｜${r.genome}｜${r.total}分`})));
}
function fillRooms(){
  const sel=document.getElementById("roomSel");
  const rev=new Set(S.data.reviews.map(r=>r.room));
  sel.innerHTML=S.data.live_rooms.slice().reverse().map(r=>
    `<option value="${esc(r.room)}">${esc(r.room)}　${esc(r.genome)}　${r.total>=0?"+":""}${fmt(r.total,0)}分　${r.n_rounds}局${rev.has(r.room)?"　[有复盘]":""}</option>`
  ).join("");
}
function renderGames(){
  const room=document.getElementById("roomSel").value;
  const src=document.getElementById("gameSrc").value;
  const lr=S.data.live_rooms.find(r=>r.room===room);
  const rv=S.data.reviews.find(r=>r.room===room);
  if(!lr){document.getElementById("gameKpi").innerHTML="";return;}
  document.getElementById("gameKpi").innerHTML=`
    <div class="kpi"><b>${esc(lr.genome)}</b><span>策略</span></div>
    <div class="kpi"><b class="${sgn(lr.total)}">${sp(lr.total)}</b><span>本房总分</span></div>
    <div class="kpi"><b class="${sgn(lr.mu)}">${sp(lr.mu)}</b><span>μ 分/局</span></div>
    <div class="kpi"><b>${lr.n_rounds}</b><span>局数</span></div>
    ${lr.win_rate!=null?`<div class="kpi"><b>${fmt(lr.win_rate*100,1)}%</b><span>自摸率</span></div>`:""}
    ${rv?`<div class="kpi"><b>${fmt(rv.mine_fan_avg,2)} / ${fmt(rv.opp_fan_avg,2)}</b><span>我方/对手 均番</span></div>
      <div class="kpi"><b>${fmt(rv.mine_baotou*100,1)}% / ${fmt(rv.opp_baotou*100,1)}%</b><span>爆头率 我/对</span></div>`:""}`;
  barChart(document.getElementById("c-games"),
    lr.rows.map(g=>({l:"b"+g.game.split("_b")[1],v:g.mine,
      t:`第${g.game.split("_b")[1].split("_")[0]}场 座${g.seat}｜我方 ${g.mine}｜四家 ${(g.scores||[]).join(", ")}`})),{h:190});
  const bt=document.getElementById("roundTbl");
  if(src==="review"&&rv&&rv.rounds.length){
    table(bt,[{k:"game",t:"场次",l:1,f:v=>esc(String(v).slice(-6))},
      {k:"round_no",t:"局"},{k:"dealer",t:"庄"},{k:"winner",t:"胡家"},
      {k:"fan",t:"番"},{k:"mine",t:"我方",cls:v=>sgn(v),f:v=>sp(v)},
      {k:"scores",t:"四家",f:v=>esc((v||[]).join(", "))},
      {k:"detail",t:"番型",l:1,f:v=>esc((v||[]).join("+"))}],rv.rounds);
  }else{
    table(bt,[{k:"game",t:"场次",l:1,f:v=>esc(String(v).slice(-6))},
      {k:"seat",t:"座位"},{k:"mine",t:"我方",cls:v=>sgn(v),f:v=>sp(v)},
      {k:"wins",t:"本场胡数"},{k:"rounds",t:"局数"},
      {k:"avg_shanten",t:"平均向听",f:v=>fmt(v,3)},
      {k:"discards",t:"弃牌数"},
      {k:"scores",t:"四家",f:v=>esc((v||[]).join(", "))}],lr.rows);
  }
}
function renderStrategy(){
  const key=document.getElementById("sortBy").value;
  const minN=+document.getElementById("minN").value||0;
  const ss=S.data.probe_strategies.filter(s=>s.n>=minN)
        .slice().sort((a,b)=>b[key]-a[key]);
  barChart(document.getElementById("c-strat"),
    ss.slice(0,26).map(s=>({l:s.genome.replace(/^shanten:/,""),v:s.mu,
      t:`${s.genome}｜μ=${fmt(s.mu,3)}｜n=${s.n}`})),{h:230});
  table(document.getElementById("stratTbl"),
    [{k:"genome",t:"策略（基因组）",l:1},{k:"n",t:"样本(局)"},
     {k:"mu",t:"μ",cls:v=>sgn(v),f:v=>sp(v)},{k:"sd",t:"σ",f:v=>fmt(v,2)},
     {k:"win_rate",t:"自摸率",f:v=>fmt(v*100,2)+"%"},{k:"fan_avg",t:"均番",f:v=>fmt(v,3)},
     {k:"baotou_rate",t:"爆头率",f:v=>fmt(v*100,2)+"%"},
     {k:"chiitoi_rate",t:"七对率",f:v=>fmt(v*100,2)+"%"},
     {k:"draw_rate",t:"流局率",f:v=>fmt(v*100,2)+"%"},
     {k:"melds_avg",t:"副露/局",f:v=>fmt(v,2)}],ss);

  const sel=document.getElementById("baseSel");
  if(!sel.options.length){
    sel.innerHTML=S.data.probe_strategies.map(s=>`<option>${esc(s.genome)}</option>`).join("");
  }
  const base=sel.value,minT=+document.getElementById("minT").value||0;
  // ★ 符号约定（容易写反，第 52 轮就写反过一次）：
  //   后端存的是 `dmu_a_minus_b = mean(score[a] - score[b])`，a/b 按字典序。
  //   表里要显示的是 **对比 − 基准**（正数 = 这个对比策略比基准好）。
  //     · 基准 == a  ⇒ 存的是 (基准 − 对比) ⇒ 要取负
  //     · 基准 == b  ⇒ 存的是 (对比 − 基准) ⇒ 保持
  //   写反的后果：peng5 的历史值是 −1.407（更差），会被显示成 +1.407（更好）——
  //   一个**结论方向完全反了**的错误。所以这里同时打上口径说明。
  const rows=[];
  Object.values(S.data.probe_pairs).forEach(p=>{
    let other=null,sign=1;
    if(p.a===base){other=p.b;sign=-1;}else if(p.b===base){other=p.a;sign=1;}
    if(!other)return;
    const dmu=p.dmu*sign,t=p.se?dmu/p.se:0;
    if(Math.abs(t)<minT)return;
    rows.push({g:other,n:p.n,dmu:dmu,se:p.se,t:t});
  });
  rows.sort((a,b)=>b.dmu-a.dmu);
  table(document.getElementById("pairTbl"),
    [{k:"g",t:"对比策略",l:1},{k:"n",t:"共同 seed"},
     {k:"dmu",t:"Δμ = 对比 − 基准",cls:v=>sgn(v),f:v=>sp(v)},
     {k:"se",t:"SE",f:v=>fmt(v,4)},
     {k:"t",t:"t",cls:v=>v>1.96?"pos":v<-1.96?"neg":"",f:v=>fmt(v,2)}],rows);

  table(document.getElementById("onlineTbl"),
    [{k:"genome",t:"线上策略",l:1},{k:"rooms",t:"房间"},{k:"n_rounds",t:"局数"},
     {k:"total",t:"总分",cls:v=>sgn(v),f:v=>sp(v)},
     {k:"mu",t:"μ",cls:v=>sgn(v),f:v=>sp(v)}],S.data.online);
}
function renderField(){
  const fr=S.data.field.ranking.slice().sort((a,b)=>b.mu-a.mu);
  table(document.getElementById("fieldTbl"),
    [{k:"name",t:"名字",l:1},{k:"rooms",t:"房间"},{k:"n",t:"局数"},
     {k:"mu",t:"μ",cls:v=>sgn(v),f:v=>sp(v)},
     {k:"wr",t:"自摸率",f:v=>fmt(v*100,2)+"%"},
     {k:"fan",t:"均番",f:v=>fmt(v,3)},
     {k:"lo",t:"CI下界",f:v=>fmt(v,3)},{k:"hi",t:"CI上界",f:v=>fmt(v,3)}],fr);
  drawScatter();
}
function drawScatter(){
  const ax=document.getElementById("axSel").value;
  const prof=S.data.field.profile,rank=S.data.field.ranking;
  const pts=[];
  rank.forEach(r=>{const p=prof[r.uid];if(!p||p[ax]==null)return;
    if(r.rooms<3)return;
    pts.push({x:p[ax],y:r.mu,n:r.name,n2:r.n,me:r.uid==="u_cec3bcdb6d93"});});
  scatter(document.getElementById("c-scatter"),pts,ax,"μ（分/局）");
  const r=pearson(pts);
  document.getElementById("corr").innerHTML=
    `参与账号 <b>${pts.length}</b> 个（出场 ≥3 房、画像完整）　`+
    `Pearson <b>r = ${fmt(r,3)}</b>　`+
    `<span class="note">相关只说明"两件事一起变"，不说明因果 —— `+
    `要定罪必须做干预实验（同 seed 配对）。</span>`;
}
function renderHelp(){
  document.getElementById("helpBox").innerHTML=`
  <b>一局 / 一场 / 一房</b><br>
  一局 = 一次结算（`rounds[]` 的一行）；一场 = 房间里的一盘（自动房固定 8 局）；
  一房 = 10 场。<br><br>
  <b>μ（分/局）</b> = 该组的总得分 ÷ 总<u>局</u>数。它是最重要的一个数：
  锦标赛的晋级线约 <code>+1.36</code>。<br><br>
  <b>σ（分/局）</b> = 逐局得分的标准差。μ≈0 时它决定名次的波动，
  但**当对手 μ 各不相同（σ_μ≈1.6）时，μ 的差别远大于自己的 σ**。<br><br>
  <b>自摸率 / 均番 / 爆头率 / 七对率 / 流局率</b> 都按<u>局</u>统计。<br><br>
  <b>Δμ（同 seed 配对差）</b>：同一个 seed ⇒ 同一副起手牌的同一局。
  把两个策略在同一批 seed 上逐局相减再平均，
  <b>牌运被消掉</b>，剩下的才是策略差异。
  <code>t = Δμ / SE</code>；<code>|t| ≥ 1.96</code> 才算 95% 显著（绿色/红色标注）。<br><br>
  <b>进听率</b> = 该账号进过听的局 ÷ 参战局数；
  <b>胡|进听</b> = 胡牌局 ÷ 进听局（"进了听之后兑不兑现"）；
  <b>首听宽度</b> = 第一次听牌时仍未见过的有效胡牌张数。<br><br>
  <b>数据来源</b>：① <code>live_*.jsonl</code> 线上逐房结果；
  ② <code>reviews/*.json</code> 复盘（唯一有逐局番型的来源）；
  ③ <code>probe_*_shard*.jsonl</code> 离线姿态实验（策略对比靠它）；
  ④ <code>field_ranking.json</code> / <code>field_profile.json</code> 全场排名与画像。<br><br>
  <b>导出</b>：页面用到的同一份数字已写到 <code>viz_export/</code>
  （<code>dataset.json</code> + 若干 CSV），AI 与人都读同一份，不会各说各话。`;
}
function switchTab(id){
  document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on",b.dataset.t===id));
  document.querySelectorAll("section").forEach(s=>s.classList.toggle("on",s.id==="tab-"+id));
  if(id==="overview")renderOverview();
  if(id==="games")renderGames();
  if(id==="strategy")renderStrategy();
  if(id==="field")renderField();
  if(id==="help")renderHelp();
}
function boot(d){
  S.data=d;
  document.getElementById("nav").innerHTML=
    [["overview","总览"],["games","每场/逐局"],["strategy","策略对比"],["field","全场"],["help","指标说明"]]
    .map(([k,t],i)=>`<button data-t="${k}" class="${i===0?"on":""}">${t}</button>`).join("");
  document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>switchTab(b.dataset.t));
  ["roomSel","gameSrc"].forEach(id=>document.getElementById(id).onchange=renderGames);
  ["sortBy","minN"].forEach(id=>document.getElementById(id).oninput=renderStrategy);
  ["baseSel","minT"].forEach(id=>document.getElementById(id).onchange=renderStrategy);
  document.getElementById("axSel").onchange=drawScatter;
  renderKpis();fillRooms();renderOverview();renderGames();renderStrategy();renderField();renderHelp();
  window.addEventListener("resize",()=>{renderOverview();renderGames();renderStrategy();renderField();});
  setupRefresh(d);stamp(d);
}
/* ---- 实时刷新（第 144 轮）-------------------------------------------- */
let TIMER=null;
function uiState(){                    // 刷新前把用户的选择记下来
  const g=id=>{const e=document.getElementById(id);return e?e.value:null;};
  const on=document.querySelector("nav button.on");
  return {tab:on?on.dataset.t:null,roomSel:g("roomSel"),gameSrc:g("gameSrc"),
          baseSel:g("baseSel"),sortBy:g("sortBy"),minN:g("minN"),axSel:g("axSel")};
}
function uiRestore(s){                 // 刷新后恢复，别把用户跳回默认页
  if(!s)return;
  const set=(id,v)=>{const e=document.getElementById(id);if(e&&v!=null)e.value=v;};
  set("roomSel",s.roomSel);set("gameSrc",s.gameSrc);set("baseSel",s.baseSel);
  set("sortBy",s.sortBy);set("minN",s.minN);set("axSel",s.axSel);
  if(s.tab)switchTab(s.tab);
}
function stamp(d){
  const el=document.getElementById("liveStamp");if(!el)return;
  const r=d.refresh_sec||0;
  el.innerHTML=`<span class="${d.refresh_error?"bad":""}">${
    d.refresh_error?esc(d.refresh_error)+"　":""}</span>`
   +`更新于 <b>${esc(d.generated_at)}</b>　扫描 ${d.scan_seconds}s　`
   +`<button id="fbtn" title="立刻重扫一次">立即刷新</button>　`
   +(r>0?`自动刷新 每 ${r} 秒`:`自动刷新已关`);
  const b=document.getElementById("fbtn");if(b)b.onclick=()=>pull(true);
}
function pull(force){
  fetch("/api/data"+(force?"?force=1":""),{cache:"no-store"})
   .then(r=>r.json()).then(d=>{
      if(d.error)throw new Error(d.error);
      const s=uiState();boot(d);uiRestore(s);
   }).catch(e=>{
      const el=document.getElementById("liveStamp");
      if(el)el.innerHTML=`<span class="bad">拉取失败：${esc(String(e))}</span>`;
   });
}
function setupRefresh(d){
  if(TIMER){clearInterval(TIMER);TIMER=null;}
  const r=d.refresh_sec||0;
  if(r>0)TIMER=setInterval(()=>pull(false),Math.max(2,r)*1000);
}
fetch("/api/data",{cache:"no-store"}).then(r=>r.json()).then(boot);
</script></body></html>
"""


# ==========================================================================
# 服务
# ==========================================================================
class Handler(http.server.BaseHTTPRequestHandler):
    data = None
    # ---- 实时刷新（第 144 轮新增）------------------------------------------
    # 原来 `data` 是**启动时算一次**的快照 ⇒ 页面上的数字会一直停在打开那一刻，
    # 要看到新数据只能关掉重开。现在：`/api/data` 会按 `refresh` 秒的间隔
    # **重新扫描 + 重新导出**，页面自己也定时来拉。
    # ⚠ `refresh=0` ⇒ 完全退回旧行为（只发启动时那一份）。
    data_dir = None
    refresh = 0
    _cache = {"t": 0.0, "ds": None}

    @classmethod
    def current(cls, force=False):
        """取当前数据；需要时重扫。重扫失败**不打断服务**，而是把错误带给页面。"""
        if cls.data_dir is None or not cls.refresh:
            return cls.data
        now = time.time()
        if force or cls._cache["ds"] is None or (now - cls._cache["t"]) >= cls.refresh:
            try:
                ds = build(cls.data_dir)
                ds["refresh_sec"] = cls.refresh
                # ★ 保持"人看到的 = AI 分析的"这条不变量：
                #   页面刷新了而 `viz_export/` 还是旧的，两边就会各说各话。
                try:
                    export(ds, os.path.join(cls.data_dir, "viz_export"))
                except Exception as e:                           # noqa: BLE001
                    ds["refresh_error"] = "导出失败：%s: %s" % (type(e).__name__, e)
                cls._cache["ds"] = ds
                cls._cache["t"] = now
            except Exception as e:                               # noqa: BLE001
                if cls._cache["ds"] is None:
                    raise
                ds = dict(cls._cache["ds"])
                ds["refresh_error"] = "重扫失败（仍在显示上一份）：%s: %s" % (
                    type(e).__name__, e)
                return ds
        return cls._cache["ds"]

    def log_message(self, *a):                                   # noqa: A003
        pass

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:                                        # noqa: BLE001
            pass

    def do_GET(self):                                            # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/data":
            try:
                ds = self.current(force="force=1" in self.path)
            except Exception as e:                               # noqa: BLE001
                return self._send(500, json.dumps(
                    {"error": "%s: %s" % (type(e).__name__, e)}),
                    "application/json; charset=utf-8")
            return self._send(200, json.dumps(ds, ensure_ascii=False),
                              "application/json; charset=utf-8")
        if path == "/api/health":
            return self._send(200, '{"ok":true}', "application/json")
        return self._send(404, "not found", "text/plain; charset=utf-8")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(ds, port, open_browser=True, data_dir=None, refresh=0):
    ds["refresh_sec"] = refresh
    Handler.data = ds
    Handler.data_dir = data_dir
    Handler.refresh = refresh
    Handler._cache = {"t": time.time(), "ds": ds}
    for p in range(port, port + 20):
        try:
            httpd = Server(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    else:
        print("找不到可用端口")
        return 1
    url = "http://127.0.0.1:%d/" % httpd.server_address[1]
    print("=" * 60)
    print("麻将数据看板已启动：%s" % url)
    print("数据目录：%s" % ds["data_dir"])
    if refresh:
        print("实时刷新：每 %d 秒重扫一次（页面也会自己来拉；--refresh 0 可关）"
              % refresh)
    else:
        print("实时刷新：关（--refresh 0）—— 页面显示的是启动那一刻的快照")
    print("（关掉这个窗口即可停止；Ctrl+C 也可以）")
    print("=" * 60)
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)),
                         daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


def _looks_like_data(d):
    return bool(glob.glob(os.path.join(d, AUTO_PAT))
                or os.path.isdir(os.path.join(d, "reviews"))
                or os.path.exists(os.path.join(d, "field_ranking.json")))


def resolve_data_dir(explicit):
    """找数据目录。exe 双击运行时，工作目录往往是别处，所以要多试几个地方。

    顺序：显式参数 → exe 自己所在目录 → 本文件所在目录 → 当前工作目录。
    **第一个"看起来有数据"的赢**（有 live_*.jsonl / reviews/ / field_ranking.json）。
    """
    cands = []
    if explicit:
        cands.append(explicit)
    if getattr(sys, "frozen", False):
        cands.append(os.path.dirname(os.path.abspath(sys.executable)))
    cands.append(HERE)
    try:
        cands.append(os.getcwd())
    except Exception:                                            # noqa: BLE001
        pass
    seen = set()
    uniq = []
    for c in cands:
        if not c:
            continue
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    for c in uniq:
        if os.path.isdir(c) and _looks_like_data(c):
            return c, uniq
    return (uniq[0] if uniq else os.getcwd()), uniq


def _init_console():
    """Windows 控制台默认代码页不是 UTF-8 —— 打包成 exe 之后中文全是乱码。

    （网页本身是 UTF-8 的，不受影响；但那个窗口用户每次都会看到，
      所以还是修掉。`reconfigure` 在 stdout 被重定向成二进制时也可能失败，
      一律吞掉，不能因为"打印"把主流程搞挂。）
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:                                            # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                        # noqa: BLE001
            pass


def main():
    _init_console()
    ap = argparse.ArgumentParser(description="麻将 bot 数据可视化看板")
    ap.add_argument("--data-dir", default=None, help="数据所在目录（默认自动查找）")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--export-only", action="store_true", help="只导出，不起服务")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--refresh", type=int, default=10,
                    help="实时刷新间隔（秒）；0 = 关，退回「启动时快照」的旧行为")
    a = ap.parse_args()

    data_dir, tried = resolve_data_dir(a.data_dir)
    if not os.path.isdir(data_dir):
        print("目录不存在：%s" % data_dir)
        return 1
    if not _looks_like_data(data_dir):
        print("⚠ 这个目录里没找到数据文件（live_*.jsonl / reviews/ / field_ranking.json）。")
        print("  找过这些地方：")
        for c in tried:
            print("    %s" % c)
        print("  用 --data-dir <目录> 指一下（放 live*.jsonl 与 reviews/ 的地方）。")

    print("扫描数据：%s" % data_dir)
    ds = build(data_dir)
    print("  线上房间 %d　复盘 %d　离线策略 %d　配对 %d　全场账号 %d"
          % (len(ds["live_rooms"]), len(ds["reviews"]),
             len(ds["probe_strategies"]), len(ds["probe_pairs"]),
             len(ds["field"]["ranking"])))
    files = export(ds, os.path.join(data_dir, "viz_export"))
    print("  已导出 %d 个文件到 viz_export/" % len(files))
    if a.export_only:
        for f in files:
            print("   ", os.path.relpath(f, data_dir))
        return 0
    return serve(ds, a.port, open_browser=not a.no_browser,
                 data_dir=data_dir, refresh=max(0, a.refresh))


if __name__ == "__main__":
    sys.exit(main())
