# ★★★ 环境重启恢复清单（★ 重启后【一条条】照做 ✓）

> 写于第 1533 轮（2026-10-08 ≈17:35）—— ★ **运行时正在反复重启** ✗
> 现象：`pwsh` 工具连续报 **`exit code 3221226091`**（= `STATUS_CONTROL_C_EXIT`）✗
> 依 **硬规则⑩**：★ **环境重启会杀掉全部后台作业** ✓

## ★★★ 最后一次【成功】读到的状态（17:14:32 ✓）

```
✅✓✓✓ **`live_r16g.log` mtime 距今 【14 s】`** ✓✓
      ⇒ ★★★ **线上客户端【活着】** ✓✓ ⇒ **线上房数【未中断】** ✓
      ⇒ ★ 臂 = `ckpt:rl_aux_0004.pt,shanten:1:1:2:5:1:1:0`（★ L2 上最好 ✓）
⚠✗   `rl_self2.log` **MISSING** ✗ ⇒ ★ **⑥ 两次恢复都被 killed** ✗
      （`pwsh-402` killed ✓ ｜ `pwsh-407` killed ✓）
      ⇒ ★★ **⑥ 目前【未在跑】** ✗ ⇒ ★ **重启后必须走 §2** ✓

★ 之后 `pwsh` 又报 `3221226091` ✗ ⇒ ★ **环境又重启了一次** ⇒ ★ **必须重新判断线上** ✓
```

★ **所以重启后第一步永远是 §0（判断谁死了）** ✓ —— ★ **不要假设线上还活着** ✗

## 0. 先判断谁死了（★ 只读 mtime ✓）

```powershell
cd E:\MyGithub\majiang_bot
'now = ' + (Get-Date -Format 'HH:mm:ss')
foreach ($lg in @('live_r16g.log','rl_self2.log')) {
  if (Test-Path $lg) { $f=Get-Item $lg; $lg + ' mtime 距今 ' + [math]::Round(((Get-Date)-$f.LastWriteTime).TotalSeconds) + ' s' }
  else { $lg + ' 不存在' }
}
'live_r16g.jsonl 行数 = ' + (@(Get-Content live_r16g.jsonl -Encoding UTF8 -ErrorAction SilentlyContinue).Count)
```

判读：
- `live_r16g.log` mtime **< 60 s** ⇒ ★ 线上活着 ✓（**上次检查时它是 `running`** ✓）
- mtime **> 5 分钟** ⇒ ★ 线上也断了 ✗ ⇒ 走 §1
- `rl_self2.log` 不存在 ⇒ ★ ⑥ 恢复失败 ✗ ⇒ 走 §2

## 1. 起线上客户端（★ §Z.1701 模板，逐字 ✓）

```powershell
cd E:\MyGithub\majiang_bot
.\_venv\Scripts\python.exe run_live.py --token-file global_token.txt --match --loop 400 --loop-hours 72 --wait-rooms 8 --minutes 90 --rounds 16 --m 10 --exit-after-games 10 --fetch-gap-max 1.2 --log live_r16g.log --result-json live_r16g.jsonl
```
★ **必须**用 `pwsh` 后台作业直起 ✓（`run_in_background: true`）✓
★ `--token-file` **默认是 `tournament_token.txt`** ⇒ ★ **漏了就废** ✗
★ 臂自动跟随 `live_ab_arms.txt` ✓（当前 = `ckpt:rl_aux_0004.pt,shanten:1:1:2:5:1:1:0` ✓）

## 2. 起 ⑥ 训练（★ 补跑 iter 4 ✓）

```powershell
cd E:\MyGithub\majiang_bot
.\_venv\Scripts\python.exe tools\run_train.py rl_self2.log train_selfplay_rl.py --init rl_self_0003.pt --iters 1 --games 250 --nproc 3 --epochs 20 --per-hand-return 1 --nstep 5 --val-n 300 --ent-coef 0.1 --ret-norm running --live-seats 1 --out-prefix rl_self2_ --log rl_self2.log --history rl_self2_history.jsonl
```
★ **为什么 `--init rl_self_0003.pt`** ✓：★ **iter 4 未落盘** ✓ ⇒ ★ **`rl_self_0001/0002/0003` 是全部已有结果** ✓ ⇒ **只补 iter 4，无损失** ✓

## 3. 状态（★ 重启前最后确认 ✓）

```
臂        = ckpt:rl_aux_0004.pt,shanten:1:1:2:5:1:1:0     ← ★ L2 上最好（房均 −1.077 ✓）
⑥         = rl_self 已跑完 iter 1/2/3 ✓，iter 4 丢失 ✗
⑦         = rl_sil 已跑完 4 iters ✓ ⇒ ★ 被 L2 否定 ✗（房均 −2.884 差于规则 −2.781）
台账      = GOAL_STATUS.md 写到 §Z.1921 ✓（§Z.1922 未落盘 ✗）
```

## 4. 关键结论（★ 不要丢 ✓）

```
★★★ **L2 首次给出【杠杆间对照】** ✓（同窗口 10-07～10-09 ✓）：
   · `rl_aux_0004`（⑤）：房均 −1.077 ／ 局均 −0.389 ⇒ **好于规则 −2.781** ✓✓
   · `rl_sil_0004`（⑦）：房均 −2.884 ／ 局均 −3.407 ⇒ **差于规则** ✗
   ⇒ ★★★ **与 L1 的 `kl` 完全一致** ✓✓：
     · ⑦ `kl` = 0.0003 ⇒ **自模仿几乎不改策略** ⇒ L2 无收益 ✗
     · ⑤ `kl` = 0.008 ／ ⑥ `kl` = 0.009 ⇒ **真改策略** ⇒ ⑤ 在 L2 上最好 ✓
   ⇒ ★★ **"自模仿"杠杆【无效】** ✓ ⇒ 依 §3 **不再投入 ⑦ 类** ✓

★★★ **⑥（纯自对弈）是当前最有希望的路径** ✓（同族 ⑤ 在 L2 上最好 ✓）
   · L1 三点：−0.102 → **+0.136 → +0.136** ✓
   · CI 下界：−0.715 → −0.523 → **−0.503** ✓（收窄 ✓）
   · `kl` 稳定 **0.009** ✓

★★★ **M1 现状** ✓：分辨 +1 分/局 = **非配对 44.4 天** ✗ ／ **小时配对 25.9 天** ✓（桶数 4 < 30 ✗）
   · 吞吐 **3.84 房/小时** ✓
   · ★ **杀僵尸 ⇒ 桶回 14.0 ⇒ 吞吐 +≈50% ⇒ ≈29 天** ✓
```

## 5. 仍然【只有你能做】的两件事

```
★★★ **① 杀掉僵尸 `live_r16`**（命令行含 `--log live_r16.log`）
   ⇒ 桶 8.6 → 14.0 ⇒ 吞吐 +≈50% ⇒ 分辨 1 分/局：44.4 天 → ≈29 天
   ⇒ 兼释放 CPU ⇒ 消除训练变慢（427–559 s → ≈170 s）
★★ **② 第二账号令牌** ⇒ 同房配对 ⇒ 25.9 天
```
