<#
  keep_live.ps1 -- 常驻监督：把一个又一个窗口接起来（本目录自足，不依赖别的目录）

  ★ 为什么需要它
    `start_live.ps1` 起的是一个**窗口**（`--loop-hours 24` 到点就退）。
    没有监督的话，窗口一结束就**再也没人把它拉起来** ⇒ 剩下的时间全是白丢的分数。
    （老开发仓的 `keep_live.ps1` 头注释里写的就是这件事。）

  ★ 两条不可违反的规则（沿用老脚本的设计，理由见下）
    1. **同时只许一个客户端。** 一个账号 = 一个座位；两个客户端会互相覆盖手牌。
       所以本脚本**等它自然退出，从不 kill**。
    2. **连续两次"秒退"就停。** 秒退通常意味着令牌失效/被限速；
       再重试只会打服务端。停下来留痕，比继续撞好。

  ★ 判活口径
    看 `live.log`（`start_live.ps1` 写的那份）的**修改时间**：
    比赛段里客户端每 30 秒会写一行哨兵，所以 >StaleSec 没动静就当作它没了。

  用法：
    powershell -NoProfile -ExecutionPolicy Bypass -File .\keep_live.ps1
    powershell ... -File .\keep_live.ps1 -MaxRounds 5 -MaxHours 24
#>
[CmdletBinding()]
param(
    [int]$MaxRounds = 3,
    [double]$MaxHours = 24.0,
    [int]$StaleSec = 180,
    [int]$GraceSec = 120,
    [int]$MinGapSec = 600,
    [string]$LiveLog = "live.log"
)

$Here = $PSScriptRoot
if (-not $Here) { $Here = (Get-Location).Path }
Set-Location $Here
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$log = Join-Path $Here $LiveLog
$sup = Join-Path $Here "keep_live.log"
$starter = Join-Path $Here "start_live.ps1"

function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m
    Write-Host $line
    # 直接用 .NET 追加：`Add-Content -Encoding UTF8` 在 Windows PowerShell 5.1 上会写 BOM，
    # 每次追加都插一个 BOM 会把日志搞坏（老脚本踩过）。
    try {
        [System.IO.File]::AppendAllText($sup, $line + "`r`n",
            (New-Object System.Text.UTF8Encoding($false)))
    } catch { }
}

function ClientAlive {
    if (-not (Test-Path $log)) { return $false }
    $age = ((Get-Date) - (Get-Item $log).LastWriteTime).TotalSeconds
    return ($age -lt $StaleSec)
}

if (-not (Test-Path $starter)) { Say "✗ 找不到 start_live.ps1（应与本脚本同目录）"; exit 2 }

$tok = Join-Path $Here "global_token.txt"
if (-not (Test-Path $tok) -or ((Get-Item $tok).Length -lt 8)) {
    Say "TOKEN 缺失或为空 -> 不启动（不去打服务端）"
    exit 2
}

Say ("监督启动：maxRounds={0} maxHours={1} stale={2}s grace={3}s minGap={4}s" -f `
     $MaxRounds, $MaxHours, $StaleSec, $GraceSec, $MinGapSec)

$t0 = Get-Date
$round = 0
$fastFail = 0

while ($round -lt $MaxRounds) {
    if (((Get-Date) - $t0).TotalHours -gt $MaxHours) { Say "到 MAX_HOURS，退出"; break }

    # --- 1. 等前一个客户端自然退出（**绝不 kill**）---
    $waited = 0
    while (ClientAlive) {
        Start-Sleep -Seconds 30
        $waited += 30
        if (((Get-Date) - $t0).TotalHours -gt $MaxHours) { break }
    }
    if (((Get-Date) - $t0).TotalHours -gt $MaxHours) { Say "等的时候到 MAX_HOURS，退出"; break }
    Say ("客户端不在了（等了 {0}s）-> 缓冲 {1}s 再起" -f $waited, $GraceSec)
    Start-Sleep -Seconds $GraceSec

    # --- 2. 令牌体检：缺失就根本不试 ---
    if (-not (Test-Path $tok) -or ((Get-Item $tok).Length -lt 8)) {
        Say "TOKEN 缺失或为空 -> 停（不去打服务端）"
        break
    }

    # --- 3. 起一个窗口 ---
    $round += 1
    $tLaunch = Get-Date
    Say ("第 {0}/{1} 轮：启动 start_live.ps1" -f $round, $MaxRounds)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $starter
    $rc = $LASTEXITCODE
    $ran = ((Get-Date) - $tLaunch).TotalSeconds
    Say ("第 {0} 轮结束 rc={1}，用时 {2:N0}s" -f $round, $rc, $ran)

    # --- 4. 秒退刹车 ---
    if ($ran -lt $MinGapSec) {
        $fastFail += 1
        Say ("秒退 {0}/2（只跑了 {1:N0}s < {2}s）" -f $fastFail, $ran, $MinGapSec)
        if ($fastFail -ge 2) { Say "连续两次秒退 -> 停（多半是令牌失效/被限速）"; break }
        Start-Sleep -Seconds 60
    } else {
        $fastFail = 0
    }
}

Say "监督退出"
