# 麻将 Bot —— 使用说明书

> **规范位置**：本包位于 `E:\MyGithub\mahjong_ai_submit`
> （开发仓在 `E:\MyGithub\majiang_bot`，本包是从它**自动抽取**出来的）。
>
> 面向「杭州麻将」对战平台的可参赛 Bot。✅ **已实测能入席并打完一整场**（真实日志见 §12.1）。
>
> 抽取工具：[`tools/extract_core.py`](tools/extract_core.py)｜自检工具：[`tools/smoke_core.py`](tools/smoke_core.py)

## 目录

1. [这是什么](#1-这是什么)
2. [环境与依赖](#2-环境与依赖)
3. [快速开始](#3-快速开始)
4. [令牌](#4-令牌)
5. [`run_live.py` 全部参数](#5-run_livepy-全部参数)
6. [臂与策略（基因组 / 神经权重）](#6-臂与策略基因组--神经权重)
7. [杭州麻将牌理要点](#7-杭州麻将牌理要点)
8. [模块地图](#8-模块地图)
9. [看护与运维](#9-看护与运维)
10. [数据分析（判据）](#10-数据分析判据)
11. [训练（RL）](#11-训练rl)
12. [自检与验证](#12-自检与验证)
13. [故障排查（都是踩过的坑）](#13-故障排查都是踩过的坑)
14. [本目录不含什么](#14-本目录不含什么)

---

## 1. 这是什么

一个**完整闭环**的麻将 AI：

```
run_live.py  ── 实战启动器（等开赛 / 自动匹配 / 循环攒样本）
   │
   ├─ my_bot.py    平台协议层（HTTP 长轮询、令牌桶反馈、409 重建）
   ├─ mj_bot.py    对局引擎（Player / GameWorker / 状态签名去重 / 自适应限速）
   ├─ mj_arena.py  策略工厂 make_strategy()（基因组 / 具名策略 / ckpt: 神经臂）
   │     ├─ mj_policy.py      策略池 + 8 位基因组
   │     ├─ rl_policy.py      神经策略网络（一次前向出牌）
   │     └─ shanten_ref.py    向听数 / 有效牌
   ├─ mj_core.py   牌理核心（财神百搭胡牌判定、爆头、番数）
   └─ feature*     特征（供神经臂使用）
```

对局平台：`https://10.240.169.190:18080`（自签证书，客户端跳过校验）。

---

## 2. 环境与依赖

### 依赖只有两个

```powershell
pip install -r requirements.txt      # numpy>=2.0, torch>=2.4
```

其余全是 Python 标准库（`urllib` / `ssl` / `threading` / `json` / `argparse` …）。
**打牌只需要 CPU 版 torch**；GPU 只有训练才用。

### 本机（开发机）已配好，零配置可跑

本机 venv 原本 `import numpy` / `import torch` **都失败**：仓库自带的 `_pkgs/`
是**解包目录**（没有 `.whl` ⇒ 不能 `pip install`），而 `numpy` 只装在基础解释器里。
已在 venv 里放了一个路径文件：

```
_venv\Lib\site-packages\_repo_deps.pth
    E:\MyGithub\majiang_bot\_pkgs
    C:\Program Files\Python314\Lib\site-packages
```

效果：用 `_venv\Scripts\python.exe` 从**任意目录**（含本目录）都能直接
`import numpy 2.5.3` / `import torch 2.14.0+cpu`，**不需要设 `PYTHONPATH`**。
删掉该 `.pth` 即回退。

> `rl_boot.bootstrap()` 会先找**自己所在目录**的 `_pkgs/`。本目录里没有 ⇒ 它
> **静默跳过**、直接用已安装的 torch。这是设计内的回退，**不要改代码**。

### 解释器（下文所有命令统一用 `$PY`）

本机的 venv 在**开发仓里**，不在本目录里 —— 所以下面所有示例都用**绝对路径变量**，
**从任何工作目录都能跑**（不要用 `..\_venv\...` 这种相对写法：换个位置就失效）：

```powershell
$PY = 'E:\MyGithub\majiang_bot\_venv\Scripts\python.exe'
& $PY -c "import numpy, torch; print(numpy.__version__, torch.__version__)"   # 自检
```

预期打印 `2.5.3 2.14.0+cpu`。

---

## 3. 快速开始

### 先做自检（10 秒，不联网）

```powershell
cd E:\MyGithub\mahjong_ai_submit
& $PY tools\smoke_core.py .
```

预期输出末行：`==== 结论: 通过 ✓ ====`，且能看到
`[ckpt] 已加载 rl_aux_0004.pt（iter=4，cfg.state_dim=370）-> rl:rl_aux_0004.pt#4`。

### 自动匹配房（攒样本 / 自测）

```powershell
& $PY run_live.py --token-file global_token.txt --match `
  --loop 400 --loop-hours 72 --wait-rooms 8 --minutes 90 `
  --rounds 16 --m 10 --exit-after-games 10 --fetch-gap-max 1.2 `
  --log ..\live.log --result-json ..\live.jsonl
```

### 正式赛（等到开赛再打）

```powershell
& $PY run_live.py --token-file tournament_token.txt `
  --wait-until auto --minutes 300 --rounds 16 --m 10
```

`--wait-until auto` 会从房间 `config.StartAt` 读开赛时刻，并在**开赛前 90 秒**
切到 1 秒轮询（官方规则：新建分桌只认开赛 ≤90s 内有已认证请求者）。

### 只演练等待路径（不进比赛段）

```powershell
& $PY run_live.py --token-file tournament_token.txt --wait-until auto --dry-run
```

---

## 4. 令牌

| 文件 | `/api/me` 的绑定 | 用途 |
|---|---|---|
| `global_token.txt` | `tournament_id = ""`（全局） | `--match` 自动匹配房 |
| `tournament_token.txt` | `tournament_id = t_…`（绑定赛事） | 正式赛 `--wait-until auto` |

- 两者是**不同凭据** ⇒ 不会共抢同一个令牌桶。
- 也可以用 `--token <串>` 直接给，不必落地成文件。

### ★ 安全提醒（它们是账号凭据）

本包为了"开箱即跑"**包含了**这两个令牌文件。开发仓的 `.gitignore` 已忽略本目录
（不会进版本库），但：

- **不要**把本目录上传到任何公开位置（GitHub / 网盘 / 聊天工具）；
- **打包或转发前建议先删掉这两个文件**，改用 `--token <串>` 传入：

```powershell
Remove-Item global_token.txt, tournament_token.txt      # 转发前
& $PY run_live.py --token "<你的令牌>" --match --m 10 --rounds 16
```

### ⚠ 一个必踩的坑

`--token-file` 的**默认值是 `tournament_token.txt`**（`run_live.py:412`）。
跑自动匹配时**必须显式传 `--token-file global_token.txt`**，否则会用错凭据。

---

## 5. `run_live.py` 全部参数

> 参数与默认值取自 `run_live.py --help` 与源码，已逐条核对。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--token-file` | `tournament_token.txt` | 令牌文件路径。**默认值是个坑**，见 §4 |
| `--token` | `""` | 直接给令牌串（覆盖 `--token-file`） |
| `--server` | `https://10.240.169.190:18080` | 平台地址 |
| `--match` | 关 | 全局令牌：先调 `POST /api/match` 入席 |
| `--wait-until` | `""` | unix 秒，或 `auto`＝从房间 `config.StartAt` 读 |
| `--lead` | `90.0` | 提前多少秒切到 1 秒轮询（官方要求开赛 ≤90s 内有请求） |
| `--register` | 关 | **测试房（`Kind=test`）必加**：临赛段提交 `register` + `ready`。测试房「N 个令牌各 register+ready，满席才开赛」且**从不经过 `stage_open`** ⇒ 不加会一直等到流局。**正式赛不要加** |
| `--minutes` | `300` | 比赛段时长上限（分钟） |
| `--exit-after-games` | `0`（关） | 收齐 N 场就退出比赛段，**不等服务端把已结束的场次从 `active_games` 撤下来**。实测那段等待中位 **93 秒**、占整房墙钟 **9.1%** ⇒ 这是最直接的排名杠杆。自动房设 10。安全性：要求已回收满 N 场才退，不丢场 |
| `--fetch-gap-max` | `0.5` | 取快照间隔上限（秒）；`0`＝不限制 |
| `--genome` | `shanten:1:1:2:2:1:1:0` | 默认基因组。**可给多个（逗号分隔）⇒ 按房间轮换**（见 §6） |
| `--m` | `10` | ⚠ 语义注意：`--help` 写"每房场数"，但它实际是 **「一次 `/api/match` 请求开几个场/房」**（源码注释 `§Z.894` 明确把它标为**推断**，并记录曾因误判白改一次又回退）。**做有代价的决策前先做最小实验确认** |
| `--rounds` | `16` | 每场局数（正式赛 `config.Rounds`）。历史自动房是 8 |
| `--trace` | 关 | 打印每次提交的动作 |
| `--log` | `""` | 日志文件路径 |
| `--result-json` | `""` | 赛果落盘路径（循环模式下按 JSONL 追加） |
| `--no-review` | 关 | 默认**会**生成复盘（落 `reviews/`，原始输出追加 `reviews.log`） |
| `--loop` | `1` | 连打多少个房间。`>1` 时结果按 JSONL 追加 |
| `--loop-hours` | `0.0` | 循环总时长上限（小时）；`0`＝只看 `--loop` 次数 |
| `--wait-rooms` | `8.0` | 循环模式下单个房间等满多少分钟还没开打就弃房换新 |
| `--fast-window` | `600.0` | 开赛后临赛轮询还继续多少秒 |
| `--dry-run` | 关 | 只跑到「临赛段」为止就退出，不进比赛段。**正式赛前演练等待路径用** |

### 一个已知的服务器侧现象

请求 `--rounds 16`，**自动房实际返回 `config.Rounds = 8`**（80 局/房）。
脚本会打印 `⚠ 服务器返回的赛制与请求不一致`。自动房历来是 8 局/场，不是本包的缺陷。

---

## 6. 臂与策略（基因组 / 神经权重）

### 6.1 臂配置文件

`live_ab_arms.txt` 是**一行逗号分隔**的臂列表：

```
ckpt:rl_aux_0004.pt,shanten:1:1:2:5:1:1:0
```

机制（源码 `run_live.py:504-521`）：

- **每开一个房间就重读一次**该文件 ⇒ 改它 = **零中断热切换**。
- 文件不存在或读失败 ⇒ **沿用当前臂**（不崩）。
- 环境变量 `LIVE_ARMS_FILE` 可让**每个实例读自己的臂文件**（默认 `live_ab_arms.txt`）
  ⇒ 这是"同房两臂并存 ⇒ 房间级配对"的前提。
- ⚠ 文件**不要带 BOM**：带 BOM 时 `startswith("ckpt:")` 判假 ⇒
  **模型静默不上场**（历史事故）。
- 若该文件不存在，则退回 `--genome`（可给多个，**按房间轮换**，这样两臂分到的是同一段时间、同一批对手池）。

### 6.2 三种臂的写法

| 写法 | 例子 | 说明 |
|---|---|---|
| 基因组 | `shanten:1:1:2:5:1:1:0` | 规则策略，见 §6.3 |
| 具名策略 | `S0-模板` / `S1-效率` / `S2-副露` / `S3-财飘` | 内置 4 个，`mj_arena.py --list` 可看 |
| 神经权重 | `ckpt:rl_aux_0004.pt` | 一次前向出牌；**文件必须在工作目录**（`ckpt:` 后面就是路径） |

### 6.3 ★★ 基因组：权威定义（附过时文档警告）

**8 位，顺序固定**：

```
discard : peng : chi : gang : decline : ukeire : route : feed
```

| 基因 | 合法取值 | 语义 |
|---|---|---|
| `discard` | `first` `heuristic` `shanten` `shanten2` `shanten_fan` `value` `baotou` `net` `baotou2` `rank` `rank2` | 出牌准则。`shanten` = 向听最小 → 有效牌最多（字典序） |
| `peng` | **0–9** | 0 不碰；1 见碰就碰；2/6 碰后向听不变差才碰；3 见碰就碰但手上有白板不碰；4 已有 ≥2 摊不再碰；5 已有 ≥1 摊不再碰；7 最激进（无速率闸）；8 已听牌不再碰（时机门控） |
| `chi` | **0–8** | 同 `peng`（无 9）。另有规则硬上限：**最多吃 2 摊** |
| `gang` | **0–2** | 0 不杠；1 只暗杠；2 暗杠 + 补杠 |
| `decline` | **0–7, 9**（**没有 8**） | 弃胡博财飘/爆头。0/1 不弃胡；2 弃胡；3 +威胁门控；4 = 3 +解除庄位禁令；5 = 2 +解除庄位禁令；6/7 是 `pre_baotou` 守卫的两档（**已按上界关闭**，勿跑配对） |
| `ukeire` | `0`/`1` | 有效牌是否做「未见牌修正」 |
| `route` | `0`/`1` | 弃牌时是否承诺七对路线 |
| `feed` | `0`/`1` | 弃牌时是否做「喂牌风险」修正（本游戏没有点炮，但弃牌仍会喂对手碰/吃） |

> ### ⚠⚠ 两处**过时**的基因组说明，不要信
>
> 1. `mj_policy.py` 的**模块 docstring**（开头那几行）写的是
>    `peng 0/1`、`chi 0/1`、`gang 0/1`、`decline 0/1`、例子 `"shanten:1:1:0:0"`（5 位）
>    —— 这是早期版本，**与真正的基因空间不符**。
> 2. `mj_arena.py --list` 打印的说明也写「其余 0|1」并给 5 位例子 —— **同样过时**。
>
> **唯一权威**：`mj_policy.py` 里的 `GENES` 表（上表就是它）＋
> `run_live.py:63` 的 `GENOME` 常量。`CHAMPION_GENOME` 是**历史残留**，源码里
> 明确标注"别用它"。

### 6.4 基因组解析的两个硬规则

- **必须写满 8 位**。写 7 位会被按位补齐，`route`/`feed` **静默保持 0** ——
  看起来"改了基因"其实没改。
- **多写一位会被拒绝解析**（不是截断）：`parse_genome` 会抛
  `基因组字段过多` —— 因为多写会让后面所有基因**整体错位一位且不报错**。

### 6.5 本目录带的部署权重

| | |
|---|---|
| 文件 | `rl_aux_0004.pt` |
| 大小 | 3,988,617 B（3.81 MB） |
| 出身 | **RL 自对弈训练产物**（`train_selfplay_rl.py`，辅助奖励档）—— 满足"部署权重必须来自 RL" |
| 结构 | `hidden=512` ｜ `state_dim=370` ｜ **14 个参数张量 / 990,723 个参数** |
| 迭代 | `iter = 4` |
| 体检证据 | 顶层键含 `args, cfg, eval, iter, sd, train_scores` ⇒ **带训练分数，不是空壳** |

复核用 [`tools/ckpt_info.py`](tools/ckpt_info.py)（见 §12.3）。

> 开发仓里另有 **88 个历史检查点（共 338 MB）**，本包只带**当前部署**的这一个。

### 6.6 为什么是这一条臂

选择依据是**线上读数**，不是离线指标（离线夹具在本项目被证明会给出相反方向）：

| 臂 | 线上房均（分/局） | 备注 |
|---|---|---|
| **`ckpt:rl_aux_0004`** | **−1.077** | 同窗口下**好于规则臂 −2.781** ⇒ 当时部署的就是它 |
| `ckpt:rl_sil_0004`（自模仿） | −2.884 | **差于**规则臂 ⇒ 该杠杆被线上否定 |
| `shanten:1:1:2:5:1:1:0`（规则臂） | −2.781 | 参照系，**永不下场** |

同窗口的三个统计口径（非配对 / 小时固定效应 / 房间配对）**方向全为正**，
但 **CI 含 0** ⇒ 按纪律记为「**未分辨**」，**不**宣称模型更强。
当前臂配置见 [`live_ab_arms.txt`](live_ab_arms.txt)。

---

## 7. 杭州麻将牌理要点

（来自 `mj_core.py` 与 `mj_policy.py` 的注释，均已核对）

- **牌码**：`1w-9w` 万 / `1b-9b` 筒 / `1t-9t` 条 / `东 南 西 北 中 发 白`。
  内部索引 `0–33`：`0-8` 万、`9-17` 筒、`18-26` 条、`27-30` 风、`31-33` 箭。
- **白板 = 财神 = 百搭 = 索引 33（`WILD`）** —— 可替代任意牌。全线以它为前提。
- **爆头**：听牌态**摸任意一张都胡** ⇒ 番数 **×2**，**最后统一乘**。
- **番数公式**：`总番 = 1 × 分支因子 × 2^动作链次数 ×（4 白板 ×2）×（爆头 ×2）`，
  七对有各档倍率（普通 ×2 / 豪华 ×4 / 双豪华 ×8 …）。
  引擎自测钉住了若干组合，例如 `三豪华七对+飘1+爆头 = 16×2×2 = 64`。
- **没有点炮** ⇒ **只能自摸**。所以传统"安全牌"概念不适用；但弃牌仍会**喂对手碰/吃**。
- **副露与胜场呈倒 U**：几乎不副露和过度副露都亏。机制是**爆头 = 4 个自然面子 + 一张余白**，
  白板正是那条路线的钥匙 ⇒ **手里有白板时碰吃等于亲手把 ×2 变成 ×1**。
- 庄家结算在引擎自测里是 `番 × 8 × 3`（`mj_sim.py` 自测断言）。

---

## 8. 模块地图

| 文件 | 作用 |
|---|---|
| `run_live.py` | **实战启动器**（主入口）：等开赛 → 临赛段 → 比赛段 → 循环 |
| `my_bot.py` | 平台协议层：`api()` / `say()` / `ApiError` / `SERVER`。长轮询、429 退避、409 重建、令牌桶反馈（`ON_THROTTLE`）。也含一个**最小可跑模板** `Player`（故意最弱，别用） |
| `mj_bot.py` | 对局引擎：`Player`（真引擎，带 `strategy` 参数）、`GameWorker`、状态签名去重、自适应限速、向听埋点 |
| `mj_arena.py` | `make_strategy(name)`：基因组 / 具名 / `ckpt:` 三路；`ckpt:` 惰性导入 torch |
| `mj_policy.py` | 策略池 + `GENES` 表 + `Evolved`（基因组驱动）+ 可选 value / ranker / decline 头 |
| `mj_core.py` | 牌理核心：`can_win` / `winning_tiles` / `best_discard` / `is_baotou` / `fan_multiple` |
| `shanten_ref.py` | 向听数 / 有效牌（穷举胡牌型 + `can_win` 双向交叉验证） |
| `mj_sim.py` | 对局模拟器（含**大量自测**，`python mj_sim.py` 直接跑） |
| `mj_room.py` / `run_room.py` | 房间驱动 / 单房入口 |
| `proc_log.py` | 日志处理 |
| `stats.py` | 配对自助法 CI、基因级边际效应、自摸率、一致性护栏 |
| `features.py` / `features_full.py` | 特征 |
| `rl_policy.py` | 神经策略网络（`NetPolicy`），一次前向 |
| `rl_action_space.py` | 动作空间 |
| `rl_boot.py` | torch/numpy 启动层 + 设备选择（`MJ_USE_GPU=1` 才用 GPU） |
| `rl_env.py` / `aux_reward.py` | RL 环境 / 辅助奖励 |
| `q_peng_policy.py` / `cand_rulevars.py` | 碰的 Q 策略 / 候选规则变量 |
| `train_selfplay_rl.py` | **RL 主训练器**（PPO + 自对弈） |
| `train_value.py` / `train_cand_ranker.py` / `train_cand_ranker_seq.py` / `train_decline.py` / `train_q_peng.py` | 各监督头训练器 |
| `mj_evolve.py` | 锚点制进化搜索 |
| `prescreen.py` | 分歧率预筛（跑对局前零成本筛掉不可测的改动） |
| `mj_viz.py` | 数据可视化看板 |

### 协议要点（来自 `my_bot.py` 的模块 docstring）

```
GET  /api/games/{id}/state?seq=N   长轮询状态/事件；seq 落后跨局自动回全量快照(gap)
POST /api/games/{id}/action        提交 {"action":..., "tile":...}
GET  /api/me                       身份 / 绑定赛事
POST /api/match                    入席开房 {"M": m, "Rounds": r}
```

- 快照**不含** `allowed_actions` ⇒ 动作全自判，服务端纯验证。
- 动作后 `seq=0` 重建快照防漂移；**`409` = 竞态/动作已失效** ⇒ 重建重试。
- 服务器窗口：**出牌 3 秒、碰 1 秒、吃 1 秒**（`DiscardTimeoutSec` / `PengTimeoutSec` / `ChiTimeoutSec`）。
- `api()` 默认 `timeout=35, tries=6` ⇒ **最坏阻塞约 258 秒**。所以**快照 GET 应当用短超时 + 不重试**
  （快照只在窗口内 1–3 秒有用，超窗拿到也没用），而**动作提交保持默认**（提交值得等）。

---

## 9. 看护与运维

### 谁还活着

```powershell
& $PY tools\who_alive.py            # 默认 300 秒内算活
& $PY tools\who_alive.py --stale 120
```

### 总览（房数 / 局数 / 各臂 / 哨兵）

```powershell
& $PY tools\status_all.py --hours 24
```

### 后台看护（建议交给任务计划程序）

```powershell
& $PY tools\liveness_monitor.py --log ..\live.log --watch
```

`--stale` 默认 900 秒（15 分钟）：I1 要求空转 ≤10 分钟，15 分是宽松余量。

### 旧客户端死了就换新（一条命令）

```powershell
& $PY tools\restart_live.py --old-log ..\live.log `
  --token-file global_token.txt --loop-hours 72 --m 10 --rounds 16 --yes
```

⚠ `--token-file` **必须与旧客户端一致**（脚本默认 `tournament_token.txt`，是个坑）。
`--old-log` 静默多久算已停：默认 420 秒（7 分钟，> 一局的间隔）。

### ★★ 并发定标：`--nproc 3`

训练/评测的并发数**必须 3**。实测：`--nproc 4` ⇒ 线上 p99 **2873 ms**；
`--nproc 3` ⇒ 线上 p99 **295 ms**。评测期间线上延迟会爆，别贪并发。

### ★★ 环境重启后的恢复（硬规则⑩）

**运行环境重启会杀掉全部后台作业**（本会话实测连续三次）。现象可能是
`Job runner exited with exit code …` 或 `pwsh` 工具直接报错。

恢复清单见本目录 [`RECOVER_NOW.md`](RECOVER_NOW.md)（含只读判断法 + 两条自启命令）。
核心是**先查谁死了**（只读 mtime），**不要假设**：

```powershell
foreach ($lg in @('..\live.log','..\rl_self2.log')) {
  if (Test-Path $lg) { $f=Get-Item $lg; $lg + ' mtime 距今 ' + [math]::Round(((Get-Date)-$f.LastWriteTime).TotalSeconds) + ' s' }
  else { $lg + ' 不存在' }
}
```

- **只许一条客户端**：两个客户端用**同一个令牌**会共抢令牌桶 ⇒ 实测 **409 占 9%**、
  等于把动作成批丢掉。要并行必须用**不同账号的令牌**。
- 线上战斗误杀：**不许 `Stop-Process`**（身份不明就不动）。用作业管理（`job_kill`）而不是杀进程。

---

## 10. 数据分析（判据）

### ★ L2 唯一判据：`analyze_live_paired.py`

```powershell
& $PY analyze_live_paired.py --jsonl ..\_tmp\live_clean.jsonl `
  --window-hours 200 --model-arm ckpt:rl_aux_0004.pt --bucket-hours 6
```

- `--model-arm` 默认从 `live_ab_arms.txt` 自动取（单一真相源）；**显式传更稳**。
- 输出含：每臂**房均 / 局均 + CI**、两臂差 + CI、房/小时、`missed_hu`。
- ★ **口径纪律**：只有本工具的**小时固定效应 β（`--bucket-hours`）**是可引用的判据；
  朴素 Welch 会被时段混淆。**桶数 < 5 ⇒ 不给判定**；< 30 ⇒ 不稳定。
- 判定门槛：**只有 CI 下界 > 0 才算模型真的更强**。

### 先合并 / 清洗线上数据

```powershell
& $PY tools\merge_live.py
```

它会扫**全部** `live_r16*.jsonl`（不要只看一个文件 —— 曾因硬编码文件列表漏掉 14–17 房，
导致读数虚高，见仓根台账 `§Z.1755`），写出 `_tmp\live_clean.jsonl`。

### 逐小时跟踪某条臂

```powershell
& $PY tools\l2_track.py --model-arm ckpt:rl_aux_0004.pt --hours 48
```

### 校准台账（L1 预测方向 vs L2 实测方向）

```powershell
& $PY tools\calibration_ledger.py
& $PY tools\calibration_ledger.py --add "run,arm,l1,l2,note"
```

当前状态：已记录候选 3 ｜ 可判定 2 ｜ 一致 1 ⇒ **一致率 0.500**。
组数 2 < 10 ⇒ 按纪律 **L1 只能否决、不能排序候选**。

### 换臂

```powershell
& $PY tools\promote_arm.py --ckpt rl_aux_0004.pt        # 干跑
& $PY tools\promote_arm.py --ckpt rl_aux_0004.pt --yes  # 真换
```

### 看板

```powershell
& $PY mj_viz.py --port 8770        # 实时刷新看板
& $PY mj_viz.py --export-only      # 只导出
```

---

## 11. 训练（RL）

主训练器是 `train_selfplay_rl.py`（PPO + 自对弈）。用 `tools/run_train.py` 包一层
（它负责日志与心跳）：

```powershell
& $PY tools\run_train.py <日志名> train_selfplay_rl.py `
  --init <起点.pt> --iters 4 --games 250 --nproc 3 --epochs 20 `
  --per-hand-return 1 --nstep 5 --val-n 300 --ent-coef 0.1 --ret-norm running `
  --live-seats 1 --out-prefix <前缀>_ --log <日志名> --history <前缀>_history.jsonl
```

⚠ `tools/run_train.py` 是**位置参数**包装器（`run_train.py <日志> <脚本> [脚本参数…]`），
**没有 `--help`**（传 `--help` 会 `IndexError`）。

要点：

- `--nproc 3`：见 §9 的并发定标。
- `--self-imit`：自模仿权重（本会话实测它把 `kl` **压小约 24 倍**，且该杠杆在线上被否定）。
- `--aux`：辅助奖励。
- `--value-init`：价值热启动。
- 迭代耗时量级：`iter_secs` ≈ **16800–21000 秒**（约 4.7–5.8 小时/迭代）。
- **等待期不要加负载**（硬规则⑥）：连自己的分析脚本都不要跑，否则训练会从 ~170 s/epoch
  慢到 600+ s/epoch（实测）。
- 检查点出厂信息可用 `tools/ckpt_info.py` 查（见 §12）。

---

## 12. 自检与验证

### 12.1 ★★ 真实参与线上匹配（最强证据）

客户端曾成功入席并**打完一整场**（2026-10-08 21:00:53 起）。★ **说明**：那次是从
一份**与本目录逐字节相同**的副本（`E:\MyGithub\majiang_bot\mahjong_ai_submit`）跑的
（代码与权重经 SHA256 校验一致，见 §12.2）；本目录自身的可用性由 §12.4 的冒烟
在本目录内直接验证过。

```
POST /api/match {"M":10,"Rounds":16} -> 房号 a_8d365f43de51
[ckpt] 已加载 rl_aux_0004.pt（iter=4，cfg.state_dim=370）      ← 权重真的在用
[主] ★ 阶段流转: status=running                                ← 比赛进行中
[主] 场 a_8d365f43de51_r1_b2_t0 结束 scores=[24, -22, -33, 31]  ← 打完并结算（本席 +24）
       10 行动 / 0 杠(被拒0) / 0 窗口错过 / 0 次409
       延迟 中位 173 / p99 250 / 最大 250 ms
[哨兵] 请求 13.4/秒（令牌桶 14.0/秒）在打=9 场 行动=107 窗口错过=0 409=0
```

### 12.2 一致性校验

| 校验 | 结果 |
|---|---|
| 32 个核心 `.py` 与开发仓源文件 **SHA256 逐字节比对** | ✓ 全部相同 |
| `rl_aux_0004.pt` SHA256 比对 | ✓ 与源相同 |
| `mj_sim.py` 模拟器自测（见 §12.5） | ✓ 全部通过 |
| `tools/smoke_core.py`（零 `PYTHONPATH`，在目录内，见 §12.4） | ✓ 通过 |
| 与开发仓权威工具 `tools/core_manifest.py` 交叉核对 | ✓ 其清单 30 项**逐项都在**（本包是它的超集） |

### 12.3 检查点体检

```powershell
& $PY tools\ckpt_info.py                  # 列全部
& $PY tools\ckpt_info.py rl_aux_0004.pt   # 看某一个
```

打印 `iter` / `state_dim` / `hidden` / **参数张量数** / **参数总量** / 顶层键
（`train_scores` 存在 ⇒ 是训练产物而非空壳）。本目录带的那个见 §6.5。

### 12.4 交付物冒烟（**必须在交付物目录内跑**）

```powershell
& $PY tools\smoke_core.py .
```

检查：核心模块逐个 import → 入口 `run_live` import → 读 `live_ab_arms.txt`
并 `make_strategy` **每一条臂** → 神经臂权重体检 → 清理 `run_live` 导入产生的
`live.pid` 副产物。

### 12.5 模拟器自测（牌理/引擎全量断言）

```powershell
& $PY mj_sim.py
```

覆盖：136 张牌守恒、碰/吃窗口与超时、财飘/链断/爆头、暗杠/明杠/补杠、七对倍率、
番数与结算。末行 `=== 模拟器自测全部通过 ===`。

### 12.6 交付物与源码是否一致（防止"改过源码但没重抽"）

```powershell
Get-ChildItem . -File -Filter *.py | ForEach-Object {
  $src = Join-Path 'E:\MyGithub\majiang_bot' $_.Name
  if ((Get-FileHash $_.FullName).Hash -ne (Get-FileHash $src).Hash) { 'DIFF ' + $_.Name }
}
```

### 12.7 静态核对：抽取器 vs 仓内权威清单

```powershell
& $PY tools\core_manifest.py     # 仓内既有的核心闭包工具
```

### 12.8 复现抽取

在**开发仓根目录**执行：

```powershell
python tools\extract_core.py --repo . --out E:\MyGithub\mahjong_ai_submit `
  --roots run_live.py run_room.py train_selfplay_rl.py train_value.py `
          train_cand_ranker.py analyze_live_paired.py mj_bot.py my_bot.py `
          mj_arena.py mj_core.py mj_policy.py shanten_ref.py `
          mj_evolve.py mj_viz.py prescreen.py stats.py `
  --assets rl_aux_0004.pt live_ab_arms.txt requirements.txt `
           global_token.txt tournament_token.txt `
  --dirs tools `
  --map README_BOT.md=README.md --map RECOVER_NOW.md=RECOVER_NOW.md
```

抽取是**确定性**的：入口清单 + `ast` 导入解析 ⇒ 同样的入口必然得到同样的文件集。
`--map SRC=DST` 用于把仓根的手写文档改名搬进包内（仓根已有历史 `README.md`，
不能覆盖）。`--assets` / `--dirs` / `--map` **可重复给且会累加**。

---

## 13. 故障排查（都是踩过的坑）

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | venv 里没装 torch，且本目录没有 `_pkgs/` | 见 §2：`pip install -r requirements.txt`，或恢复 `_venv\Lib\site-packages\_repo_deps.pth` |
| 跑起来但**模型没上场**、行为像规则 | `live_ab_arms.txt` 带了 **BOM** ⇒ `startswith("ckpt:")` 判假 | 用无 BOM 的 UTF-8 重写该文件 |
| 改了基因组但**行为没变** | 写了 7 位 ⇒ `route`/`feed` 静默保持 0 | 写满 8 位；多写会被显式拒绝 |
| 一直连不上 / 立刻退出 | `--token-file` 默认是 `tournament_token.txt`，你忘了传 | 显式 `--token-file <你的文件>` |
| 测试房一直等到流局 | 测试房**从不经过 `stage_open`**，且要 N 个令牌满席 | 加 `--register` |
| 大量 `409`、`窗口错过` 不为 0 | **两个客户端共用同一个令牌** ⇒ 抢令牌桶 | 只跑一个；要并行用不同账号令牌 |
| 线上 p99 飙到 2000+ ms | 训练并发 > 3 | `--nproc 3` |
| 训练突然慢 3–4 倍 | 等待期跑了分析脚本 / 有僵尸客户端抢 CPU | 等待期不加负载（硬规则⑥）；清掉多余客户端 |
| 后台作业全没了 | **运行环境重启**会杀掉全部后台作业 | 按 `RECOVER_NOW.md` 恢复；**先查谁死了，别假设** |
| `Job runner exited with exit code …` | 同上（环境重启） | 同上 |
| 赛制不一致告警 | 自动房固定 `Rounds=8`，你请求了 16 | 正常现象，非缺陷 |
| 多文件统计工具读数虚高 | 工具**硬编码**了文件列表，漏了新日志 | 一律用 glob 扫 `live_r16*.jsonl`（`tools/merge_live.py` 已修） |
| 跑完之后目录里多了 `reviews.log` / `reject_dump.jsonl` / `_tmp/` / `live.pid` | 这些是**客户端在本目录运行时的产物**（复盘默认开启；`live.pid` 是模块级写入） | 想让目录保持纯净：加 `--no-review`，并把 `--log` / `--result-json` 指到目录外 |

---

## 14. 本目录不含什么

| 不含 | 原因 |
|---|---|
| `_pkgs/` `_pkgs_gpu/`（3.2 GB） | 离线 torch 轮子；用 `pip install` 即可 |
| `field_cache/`（2.7 GB）、`_tmp/`（889 MB） | 抓取缓存与临时产物，非代码 |
| `reviews/`（37 MB）、`*.jsonl`、`*.log` | 对局复盘与运行日志 |
| `*.npz`（9–143 MB） | 训练数据集 |
| `.git/`（878 MB） | 版本历史 |
| 其余 400+ 个顶层 `*.py` | 探针 / 诊断 / 一次性分析脚本，**不在导入闭包内** |
| 其它 88 个 `*.pt`（共 338 MB） | 历史检查点；本包只带**当前部署**的那一个 |

### 规模

```
代码 32 个 .py            ≈ 912 KB
权重 rl_aux_0004.pt        3.81 MB（14 个张量 / 990,723 个参数）
tools/  36 个 .py + 1 个 .jsonl
文档    README.md / RECOVER_NOW.md
合计 ≈ 4.90 MB（预算 20 MB，余量 ≈15.1 MB）
```

> 若在本目录里长期跑客户端，它会写入 `reviews.log` / `reject_dump.jsonl` /
> `_tmp/` / `live.pid`（运行产物，不是代码）。想让目录保持纯净，见 §13 最后一行。
