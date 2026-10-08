# 参赛交付物 —— 杭州麻将 AI（核心代码）

> 本目录是**从开发仓 `majiang_bot` 自动抽取的导入闭包**：从入口脚本出发，
> 用 `ast` 递归解析 `import`，只保留**代码真正依赖的模块**。
> 抽取工具：`tools/extract_core.py`（同仓可复现）。
> 校验工具：`tools/smoke_core.py`（在**本目录内**跑，验证能 import、能载权重、能建策略）。

---

## 1. 里面有什么

| 文件 | 作用 |
|---|---|
| `run_live.py` | **实战启动器**（主入口）：等开赛 → 进临赛段 → 打完整房 → 可循环 |
| `my_bot.py` | 平台协议层（`api` / `say` / `ApiError` / `SERVER`；长轮询、409 重建、令牌桶反馈） |
| `mj_bot.py` | 对局引擎：`Player` / `GameWorker`、状态签名去重、自适应限速、向听埋点 |
| `mj_arena.py` | 策略工厂 `make_strategy()`：`shanten:…` 基因组 / `ckpt:<file>.pt` 神经臂 |
| `mj_core.py` | 牌理核心：带财神（百搭）的胡牌判定、爆头检测、出牌评估 |
| `shanten_ref.py` | 向听数 / 有效牌（穷举胡牌型 + `can_win` 双向交叉验证） |
| `mj_policy.py` | 策略池 + **8 位**基因组 + 向听数出牌 + 可选 value / ranker / decline 头 |
| `mj_sim.py` | 离线对局模拟（训练与自测用） |
| `features.py` `features_full.py` | 特征（含全量特征） |
| `rl_policy.py` `rl_action_space.py` `rl_boot.py` | 神经策略网络、动作空间、torch/numpy 启动层 |
| `cand_rulevars.py` `q_peng_policy.py` | 候选规则变量 / 碰的 Q 策略（被策略池引用） |
| `mj_room.py` `run_room.py` `proc_log.py` `stats.py` | 房间驱动、日志、配对自助法 CI 与显著性 |
| `train_selfplay_rl.py` | **RL 主训练器**（PPO + 自对弈）；M4 证据 |
| `train_value.py` `train_cand_ranker.py` `train_cand_ranker_seq.py` `train_decline.py` `train_q_peng.py` | 各监督头训练器 |
| `mj_evolve.py` `prescreen.py` | 锚点制进化 / 分歧率预筛 |
| `mj_viz.py` | 数据看板（可选） |
| `rl_aux_0004.pt` | ★ **部署权重**（3.8 MB，RL 自对弈产物，`state_dim=370`，iter=4） |
| `live_ab_arms.txt` | ★ **臂配置**（逗号分隔，每房重读 ⇒ 热切换零中断） |
| `requirements.txt` | 运行期第三方依赖 |

**规模：代码 32 个文件 ≈ 912 KB ＋ 权重 3.8 MB ⇒ 合计 ≈ 4.87 MB（预算 20 MB）。**

> ★ **与仓内权威工具一致**：本包是 `tools/core_manifest.py`（仓里既有的核心闭包工具，
> 其入口集为 `run_live, run_room, train_selfplay_rl, train_value, train_cand_ranker,
> analyze_live_paired, mj_bot`）清单的**超集** —— 权威清单 30 项**逐项都在**，
> 另多 3 个有用工具（`mj_evolve.py` / `mj_viz.py` / `prescreen.py`）。
> 两个工具用同一套 AST 解析，结论互相印证。

---

## 2. 依赖

```powershell
pip install -r requirements.txt      # 只有 numpy + torch
```

其余全部是 Python 标准库（`urllib` / `ssl` / `threading` / `json` / `argparse` …）。
打牌只需要 **CPU 版 torch**；GPU 只有训练才用。

> `rl_boot.bootstrap()` 会先尝试把仓库自带的 `_pkgs/`（离线轮子）挂到 `sys.path`。
> 本交付物**不包含** `_pkgs/`（3.2 GB），所以它会静默跳过、直接用**已安装的** torch ——
> 这是设计内的回退，不需要改代码。

---

## 3. 怎么跑

### ★ 先做冒烟测试（10 秒，不联网）

```powershell
python ..\tools\smoke_core.py .        # 在交付物目录内执行
```

预期看到：

```
[ckpt] 已加载 rl_aux_0004.pt（iter=4，cfg.state_dim=370）-> rl:rl_aux_0004.pt#4
OK   ckpt:rl_aux_0004.pt  -> NetPolicy
OK   shanten:1:1:2:5:1:1:0  -> Evolved
==== 结论: 通过 ✓ ====
```

### 正式赛（参赛令牌）

```powershell
python run_live.py --token-file tournament_token.txt --wait-until auto --minutes 300 --rounds 16 --m 10
```

`--wait-until auto` 会从房间 `config.StartAt` 读开赛时刻，并在**开赛前 90 秒**切到 1 秒轮询
（官方规则：新建分桌只认开赛 ≤90s 内有已认证请求者）。

### 自动匹配房（全局令牌，自测/攒样本）

```powershell
python run_live.py --token-file global_token.txt --match --loop 400 --loop-hours 72 `
  --wait-rooms 8 --minutes 90 --rounds 16 --m 10 --exit-after-games 10 `
  --fetch-gap-max 1.2 --log live.log --result-json live.jsonl
```

### 只演练等待路径（不进比赛段）

```powershell
python run_live.py --token-file tournament_token.txt --wait-until auto --dry-run
```

---

## 4. ★ 令牌（**本包故意不含**）

`global_token.txt` / `tournament_token.txt` / `room_token*.txt` 是**账号凭据**，
已从交付物中**排除**。请自己放置，或显式传参。

> ⚠ `--token-file` 的默认值是 `tournament_token.txt`。
> **忘了传 ⇒ 直接读不到令牌**（这是本项目实测踩过的坑）。

---

## 5. ★ 臂配置（决定用哪条策略上场）

`live_ab_arms.txt` 是一行**逗号分隔**的臂列表，当前内容：

```
ckpt:rl_aux_0004.pt,shanten:1:1:2:5:1:1:0
```

- `ckpt:<file>.pt` ⇒ 神经网络臂（一次前向，RL 权重）
- `shanten:1:1:2:5:1:1:0` ⇒ 规则臂（**8 位**基因组：
  `discard:peng:chi:gang:decline:ukeire:route:feed`）

机制：
- 客户端**每房重读一次**该文件 ⇒ 改它 = **零中断热切换**。
- 环境变量 `LIVE_ARMS_FILE` 可让**每个实例读自己的臂文件** —— 这是"同房两臂并存 ⇒
  房间级配对"的前提。
- ⚠ 基因组**必须写满 8 位**；写 7 位会被按位补齐，`route`/`feed` **静默保持 0**。
- ⚠ 文件**不能带 BOM**（带 BOM 会让 `startswith("ckpt:")` 判假 ⇒ 模型静默不上场）。

---

## 6. 本包**不含**什么（以及为什么）

| 不含 | 原因 |
|---|---|
| `_pkgs/` `_pkgs_gpu/`（3.2 GB） | 离线 torch 轮子；用 `pip install` 即可 |
| `field_cache/`（2.7 GB）`_tmp/`（889 MB） | 抓取缓存与临时产物，非代码 |
| `reviews/`（37 MB）`*.jsonl` / `*.log` | 对局复盘与运行日志 |
| `*.npz`（9–143 MB） | 训练数据集 |
| `.git/`（878 MB） | 版本历史 |
| 其余 400+ 个顶层 `*.py` | 探针 / 诊断 / 一次性分析脚本，**不在导入闭包内** |
| 令牌文件 | **账号凭据** |

---

## 7. 实测状态（抽取时的线上读数）

```
[哨兵] 请求 8.2/秒（令牌桶 14.0/秒）在打=10 场 行动=154 窗口错过=0 409=0
```

- 出牌窗口 3 秒、碰/吃窗口 1 秒；`409` 与「窗口错过」均已为 **0**。
- 延迟 p99 约 **295 ms**（并发定标 `--nproc 3`；`--nproc 4` 会把线上 p99 打到 2873 ms）。

---

## 8. 复现抽取

在**开发仓根目录**执行：

```powershell
python tools\extract_core.py --repo . --out mahjong_ai_submit `
  --roots run_live.py my_bot.py mj_bot.py mj_arena.py mj_core.py mj_policy.py shanten_ref.py `
          train_selfplay_rl.py train_value.py mj_evolve.py mj_viz.py prescreen.py stats.py `
  --assets rl_aux_0004.pt live_ab_arms.txt requirements.txt SUBMIT.md `
  --dirs tools
```

再来一遍冒烟测试（★ 必须在交付物目录**内**跑，且本机 torch 在 `_pkgs/` 时需
`$env:PYTHONPATH="<repo>\_pkgs"`）：

```powershell
python tools\smoke_core.py mahjong_ai_submit
```

抽取是**确定性**的：入口清单 + `ast` 导入解析 ⇒ 同样的入口必然得到同样的文件集。

---

## 9. 已知的无关紧要的副产物

- `run_live.py` 的**模块级代码**会往自己所在目录写一个 `live.pid`（27 B，
  用于在沙箱里确认线上客户端身份）。`smoke_core.py` 会在测试后自动清理它。
- 跑测试会产生 `__pycache__/`。用 `$env:PYTHONDONTWRITEBYTECODE=1` 可避免。
