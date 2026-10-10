<#
  start_live.ps1 -- 线上实战启动器（一个窗口）

  ★ 为什么需要它
    原来拉起线上循环的是**另一个目录**里的 `run_r16.bat`，而它写死了
        cd /d <那个目录>
    ⇒ 代码复制到别处（比如本目录）也仍然只从那个老目录跑，
      连带 token / 臂文件 / 日志 / 权重 / `_pkgs` 全部相对 CWD 解析到老目录。
    本脚本**只用 `$PSScriptRoot`**（自己所在的目录）⇒ 整个目录搬到哪就跟到哪，
    **不依赖任何绝对路径，也不依赖任何其它目录**。

  ★ 解释器
    本仓库不自带解释器（只有纯 stdlib + 依赖）。`numpy` / `torch` 由 `rl_boot.py`
    从**本目录的 `_pkgs/`** 挂上（`bootstrap()` 会 `_add(REPO/_pkgs)`，
    而 `REPO` = `rl_boot.py` 自己所在目录）⇒ 用系统 Python 也能跑。
    解析顺序见下面 Find-Python，**不碰其它项目目录**。

  用法：
    正常起一窗：   powershell -NoProfile -ExecutionPolicy Bypass -File .\start_live.ps1
    只做自检：     ... -File .\start_live.ps1 -Check      # 不联网、不开房
    演练等待路径：  ... -File .\start_live.ps1 -DryRun     # 跑到临赛段就退（会调 /api/match）
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$DryRun,
    [int]$Loop = 400,
    [double]$LoopHours = 24.0,
    [int]$WaitRooms = 8,
    [int]$Minutes = 90,
    [int]$Rounds = 16,
    [int]$M = 10,
    [int]$ExitAfterGames = 10,
    [double]$FetchGapMax = 1.2,
    [string]$Log = "live.log",
    [string]$ResultJson = "live.jsonl",
    [switch]$NoReview,
    [string]$Extra = ""
)

$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
if (-not $Here) { $Here = (Get-Location).Path }
Set-Location $Here

# ---- 输出用 UTF-8，避免中文在 GBK 控制台上变成乱码（本仓踩过这个坑）----
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Say($m) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m)
}

# ---- 1. 找解释器（只在本目录和系统里找）------------------------------------
function Find-Python {
    $cands = @()
    if ($env:MJ_PYTHON) { $cands += [pscustomobject]@{ Exe = $env:MJ_PYTHON; Pre = @() } }
    foreach ($v in @("_venv", ".venv")) {
        $p = Join-Path $Here "$v\Scripts\python.exe"
        if (Test-Path $p) { $cands += [pscustomobject]@{ Exe = $p; Pre = @() } }
    }
    foreach ($n in @("python", "python3")) {
        $c = Get-Command $n -ErrorAction SilentlyContinue
        if ($c) { $cands += [pscustomobject]@{ Exe = $c.Source; Pre = @() } }
    }
    # 机器上常见的安装位置（新版本优先）。★ 这是**发现**，不是写死某个项目目录 ——
    # 老 run_r16.bat 写死 `...\majiang_bot\_venv\...`，那个项目一删就全废。
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)},
               (Join-Path $env:LOCALAPPDATA "Programs\Python")) | Where-Object { $_ }
    foreach ($r in $roots) {
        Get-ChildItem (Join-Path $r "Python3*\python.exe") -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            ForEach-Object { $cands += [pscustomobject]@{ Exe = $_.FullName; Pre = @() } }
    }
    $pyL = Get-Command py -ErrorAction SilentlyContinue
    if ($pyL) { $cands += [pscustomobject]@{ Exe = $pyL.Source; Pre = @("-3") } }
    # ★ 只做"存在性"粗筛，**不去探测它能不能跑**。
    #   为什么不探测：在本沙箱下，函数作用域里的 `& $exe -c "import sys"`
    #   连 `$LASTEXITCODE` 都不落（实测 rc 为空、也不抛异常）⇒ 会把**所有**候选
    #   都误判成"不可用"，而解释器明明就在那儿（脚本外单独测是 rc=0）。
    #   真正的门槛是**下面那两步**：能不能 import numpy / torch ——
    #   那才决定成败，而且失败时会打印用的是哪个解释器，可诊断。
    foreach ($c in $cands) {
        if (Test-Path $c.Exe) {
            if ($env:MJ_DEBUG) { Say ("    选中 {0}" -f $c.Exe) }
            return $c
        }
        if ($env:MJ_DEBUG) { Say ("    跳过（不存在）{0}" -f $c.Exe) }
    }
    if ($env:MJ_DEBUG) { Say ("试过的候选：" + (($cands | ForEach-Object { $_.Exe }) -join " | ")) }
    return $null
}

$pyc = Find-Python
if (-not $pyc) {
    Say "✗ 找不到可用的 Python。请装 Python 3.10+，或设 MJ_PYTHON=<python.exe 绝对路径>"
    Say "  （本目录里没有 _venv/.venv，PATH 上也没有 python/py）"
    exit 2
}
$PYEXE = $pyc.Exe
$PYARG = @($pyc.Pre)
Say ("解释器 = {0} {1}" -f $PYEXE, ($PYARG -join " "))
Say ("工作目录 = {0}" -f $Here)

# ---- 2. 依赖自检 ----------------------------------------------------------
# ★★ 为什么**不看退出码、也不接管道**，而让 Python 写文件 ★★
#   实测（本沙箱）：PowerShell **函数作用域内拿不到原生程序的退出码** ——
#   `& $exe -c ...` 之后 `$LASTEXITCODE` 为空、也不抛异常，于是第一版把
#   **所有** Python 候选都判成"不可用"（而 `python --version` 在别处明明是好的）。
#   同一件事的第二个面：`return $LASTEXITCODE` 会把子进程的 **stdout 一起返回**
#   ⇒ 调用方 `$rc` 变成数组，`-ne 0` 必然为真 ⇒ 又误判一次。
#   而 `| Out-Null` / `2>$null` 会让子进程**根本起不来**（Access is denied）。
#   ⇒ 结论：这三个机制在这台机器上都不可靠。**改用文件当信道**：
#     tools/preflight.py 自己把结论写成 `_preflight.txt`，这里只读文件。
$pf = Join-Path $Here "tools\preflight.py"
$pfOut = Join-Path $Here "_preflight.txt"
if (-not (Test-Path $pf)) { Say "✗ 找不到 tools\preflight.py"; exit 2 }
Remove-Item $pfOut -Force -ErrorAction SilentlyContinue
& $PYEXE @PYARG $pf
if (-not (Test-Path $pfOut)) {
    Say ("✗ 自检没产出 _preflight.txt ⇒ 解释器根本没能启动：" + $PYEXE)
    Say "  换一个：设 MJ_PYTHON=<python.exe 绝对路径>"
    exit 2
}
$pre = Get-Content $pfOut -Raw -Encoding UTF8 | ConvertFrom-Json
Say ("  python {0} ｜ numpy {1} ｜ torch {2} ｜ run_live {3}" -f `
     $pre.python, $pre.numpy, $pre.torch, $pre.run_live)
if (-not $pre.ok) {
    Say "✗ 自检未通过："
    if ($pre.err) { Say ("    " + $pre.err) }
    # ★ 只提示**真正缺**的那一项 —— 全都提示会让部署者以为 numpy 也没装（实测踩过）
    if (-not $pre.numpy) { Say "  · 缺 numpy：`"$PYEXE`" -m pip install numpy" }
    if (-not $pre.torch) {
        Say "  · 缺 torch：`"$PYEXE`" -m pip install torch"
        Say "      （或用离线包：把 torch 解到本目录 `_pkgs\`，rl_boot 会自动挂上）"
    }
    if ($pre.numpy -and $pre.torch -and "$($pre.run_live)" -ne "ok") {
        Say "  · run_live 导入链失败：$($pre.run_live)"
    }
    Say "  一次装齐：`"$PYEXE`" -m pip install -r requirements.txt"
    exit 2
}

if ($Check) {
    Say "✓ 自检通过：解释器、numpy、torch、run_live.py 导入链都正常"
    Say "✓ 臂来自本目录 live_ab_arms.txt（不再由命令行 --genome 决定）"
    exit 0
}

# ---- 3. 前置检查：token + 臂文件 ------------------------------------------
$tok = Join-Path $Here "global_token.txt"
if (-not (Test-Path $tok) -or ((Get-Item $tok).Length -lt 8)) {
    Say "✗ 找不到或为空：global_token.txt（本目录下）⇒ 不启动，避免空转打服务端"
    exit 2
}
$arms = Join-Path $Here "live_ab_arms.txt"
if (Test-Path $arms) {
    Say ("臂文件 = live_ab_arms.txt -> " + ((Get-Content $arms -Raw -Encoding UTF8).Trim()))
} else {
    Say "⚠ 没有 live_ab_arms.txt ⇒ 将使用 run_live.py 的内置默认臂"
}

# ---- 4. 组参数并启动 ------------------------------------------------------
# ★ 故意**不传 --genome**：臂由本目录的 live_ab_arms.txt 决定（run_live.py 里有热切换，
#   文件存在时会覆盖 --genome）。老 run_r16.bat 传的是 `--genome ckpt:rl_bc6.pt`，
#   一旦臂文件丢失就会**静默退回那个旧模型** —— 那种双份真相源是坑，这里去掉。
$a = @(
    (Join-Path $Here "run_live.py"),
    "--token-file", "global_token.txt",
    "--match",
    "--loop", "$Loop",
    "--loop-hours", "$LoopHours",
    "--wait-rooms", "$WaitRooms",
    "--minutes", "$Minutes",
    "--rounds", "$Rounds",
    "--m", "$M",
    "--exit-after-games", "$ExitAfterGames",
    "--fetch-gap-max", "$FetchGapMax",
    "--log", $Log,
    "--result-json", $ResultJson
)
if ($NoReview) { $a += "--no-review" }
if ($DryRun)   { $a += "--dry-run" }
if ($Extra)    { $a += ($Extra -split "\s+") }

Say ("启动：run_live.py " + (($a[1..($a.Count - 1)]) -join " "))
$t0 = Get-Date
& $PYEXE @PYARG @a
$rc = $LASTEXITCODE
$ran = ((Get-Date) - $t0).TotalSeconds
Say ("退出 rc={0}，用时 {1:N0}s" -f $rc, $ran)
exit $rc
