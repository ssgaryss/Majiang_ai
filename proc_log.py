#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""子进程 + 日志的**唯一安全通路**（第 357 轮抽出，供 `run_live.py` / `watch_reviews.py` 共用）。

为什么要有这个模块
==================
同一条故障在本仓出现过至少三次，每次都被单独"修"在一个调用点上：

1. ★★★ **第 357 轮的更正：仓里流传的"沙箱禁传文件句柄"是错的。**
   `watch_reviews.py`（以及 `probe_spawn_perm.py`）记着一张三行结论的表：
   ```
     Popen(cmd, stdout=None)             -> OK
     Popen(cmd, stdout=DEVNULL)          -> OK
     Popen(cmd, stdout=open(p, "a"))     -> PermissionError(13)   ← 这条是错的
     Popen(cmd, stdout=PIPE)             -> PermissionError(13) WinError 5
   ```
   而 `run_shards.py:67` 同时写着「沙箱禁的是命名管道，**普通文件重定向不受影响**」，
   并且一直用 `Popen(cmd, stdout=open(logpath, "w"))` 跑分片。**两者不可能同时对。**

   `probe_spawn_forms2.py`（第 357 轮，子进程写哨兵文件判定"是否真的跑起来"）实测：
   ```
     ✅ stdout=None（继承）                    rc=0  真的跑了
     ✅ stdout=DEVNULL                        rc=0  真的跑了
     ❌ stdout=PIPE                           PermissionError(WinError 5)
     ✅ stdout=open(p,'w')                    rc=0  真的跑了
     ✅ stdout=open(p,'a')                    rc=0  真的跑了
   ```
   ⇒ **唯一被禁的是 `PIPE`**；普通文件对象**可以**交给子进程。
   ⇒ 于是"`run_live.py:643` 的 `Popen(stdout=rl)` 被沙箱拒绝"这个解释**不成立** ——
     它那行 `PermissionError(13)` 的真实来源是**下一段的文件锁**：`open("reviews.log","a")`
     本身就没成功（`Popen` 根本没走到）。★ 教训：**异常类型相同 ≠ 根因相同**；
     把"我看到的报错"归给"我刚好知道的那条沙箱限制"，会同时掩盖真正的锁问题。
     （本模块的实现**保持不变**：继承形式已被两份实测独立验证，且它顺带把父进程
       在子进程运行期间的输出也留在同一份日志里。变的只是**理由**。）

2. **`reviews.log` 会被别的进程以「拒绝写」的共享方式占住**，此后
   `open("reviews.log", "a")` 一律 `PermissionError(13)`，而**读**还是好的。
   实测（2026-09-24 11:10）：
   ```
     reviews.log            -> PermissionError: 另一个程序正在使用此文件
     live_r16.stdout.log    -> 同上（那是 run_r16.bat 里 `>>` 重定向持有的，属正常）
     live_r16.log           -> OK（Python 自己 open 的，共享方式是 allowed）
     _t.py / arena_log.json -> OK
     用 CreateFileW(GENERIC_WRITE, FILE_SHARE_READ) 打开 → 成功
       （★ 这一步**不足以**指认占用者是谁 —— 见下）
     ACL 检查：Authenticated Users = Modify，**不是权限问题**，是共享冲突
   ```
   ★★ **占用者的身份：我没有证出来，就不要写成证出来了**（第 357 轮自查）。
      本模块的第一版在这里写了「占用者正是 cmd.exe `>>` 那种『只共享读』的句柄形态」
      —— 那是**从一次 CreateFileW 的成功反推出来的**，而那次成功恰恰**不能**
      唯一确定占用者的 access/share 组合。**唯一站得住的证据是后来出现的**：
   ```
     11:14  旧值守（PID 24100，跑的是修复前的代码）被终止
     11:5x  [proc_log] reviews.log 恢复可写 ⇒ 切回主路径
   ```
      ⇒ **文件在旧值守活着时写不进去、在它被终止后立刻恢复可写。**
      这就是全部结论：**占用者是那个进程持有的某个句柄**（具体是它自己的
      `spawn_logged` 里 `dup2` 出去的 fd 1、还是它子进程继承的句柄，
      我没有分辨）。★ 与 §「异常类型相同 ≠ 根因相同」同族：
      **「我能打开它」≠「我知道谁锁着它」。**
      处置不依赖身份：`open_log` 在**任何**占用下都返回可写句柄。
   ⇒ 这才是 `run_live.py` 每房一行 `⚠ 复盘生成失败（PermissionError 13）`
     和 `watch_reviews.py` **连续失败 89 次**、最后一份复盘停在 09-23 18:01 的
     **唯一**根因。`watch_reviews.py` 的注释（第 175~178 行）早就把它认对了，
     但**没有兜底** —— 认对根因 ≠ 恢复功能。

3. 于是"日志文件写不进去"这条故障，每次都把**整条复盘流水线**带走，
   因为下游把"写日志失败"当成了"复盘失败"。

本模块的两条不变式
==================
* **日志永远写得进去**：主路径被占住时自动轮转到带时间戳的同名文件，
  并在 stderr 说明；调用方拿到的一定是**可写的**句柄。
* **子进程输出永远有去处**：`spawn_logged` 只用实测过的继承形式。

⚠ 轮转只改**日志文件名**，不改任何对局/训练逻辑；也不删除、不移动被占住的原文件
（占住它的进程可能还在写，动它会丢数据）。
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# 已判定"主路径不可写"的缓存：path -> (decision_ts, effective_path)
_ROT = {}
_REPROBE_SEC = 60.0


def _try_open(path):
    """能不能以追加方式打开。返回句柄或 None（**不抛**）。"""
    try:
        return open(path, "a", encoding="utf-8")
    except OSError:
        return None


# ---- 只在自测里用：造一个「拒绝写共享」的占用者（复现 cmd.exe `>>`）------------
def _hold_deny_write(path):
    r"""以 `GENERIC_WRITE` + `FILE_SHARE_READ` 打开 `path`，返回句柄。

    这一对参数就是 **cmd.exe `>>` 重定向**用的那对：它**不**允许别人再写
    （share mode 里没有 `FILE_SHARE_WRITE`）⇒ 别人 `open(path, "a")` 会拿到
    `PermissionError(13, '另一个程序正在使用此文件')`。
    """
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                wintypes.HANDLE]
    GENERIC_WRITE, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ = \
        0x40000000, 4, 0x80, 1
    h = k32.CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, None,
                        OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
    if h == wintypes.HANDLE(-1).value or h is None:
        raise OSError("CreateFileW failed: %d" % ctypes.get_last_error())
    return h


def _close_handle(h):
    import ctypes
    ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(
        ctypes.c_void_p(h))


def resolve_log(path):
    """返回**当前可用**的日志路径。主路径被占用时返回轮转名（同名 + 时间戳）。"""
    now = time.time()
    hit = _ROT.get(path)
    if hit and now - hit[0] < _REPROBE_SEC:
        return hit[1]
    fh = _try_open(path)
    if fh is not None:
        fh.close()
        if hit:
            sys.stderr.write("[proc_log] %s 恢复可写 ⇒ 切回主路径\n" % path)
        _ROT.pop(path, None)
        return path
    base, ext = os.path.splitext(path)
    # ★ 轮转名**必须确定**（不带时间戳）：`run_live.py` 与 `watch_reviews.py`
    #   是两个进程，各自解析一次；带时间戳会让它们各开一个新文件，把同一条
    #   日志撕成两半。用固定名 ⇒ 所有写者收敛到同一个文件。
    #   "什么时候开始被占"由 stderr 那行提示记录，不靠文件名。
    cand = "%s.rot%s" % (base, ext or ".log")
    if not hit or hit[1] != cand:
        sys.stderr.write(
            "[proc_log] ⚠ %s 不可写（被其他进程占用）⇒ 本轮日志改用 %s\n"
            "            （原文件既不删除也不改名：占用者可能仍在写它）\n"
            % (path, cand))
    _ROT[path] = (now, cand)
    return cand


def open_log(path):
    """打开日志用于追加，**保证成功**。返回 (句柄, 实际路径)。"""
    p = resolve_log(path)
    try:
        return open(p, "a", encoding="utf-8"), p
    except OSError:
        # 连轮转名都开不了（几乎不可能）⇒ 退到临时目录，绝不抛给调用方
        import tempfile
        p2 = os.path.join(tempfile.gettempdir(),
                          "dsh_%s.log" % os.path.basename(path))
        return open(p2, "a", encoding="utf-8"), p2


def spawn_logged(argv, logpath, cwd=HERE):
    r"""跑 `argv`，把它**和它的输出**一起写进 `logpath`，返回 returncode。

    实现：先 `dup2` 把父进程 fd1/fd2 指向日志，再 `Popen(..., stdout=None)` 继承。

    ⚠ **这不是唯一可行的形式**（第 357 轮更正）：`Popen(cmd, stdout=<普通文件对象>)`
      实测**可用**，被禁的只有 `PIPE`。选继承形式有两个实际理由：
      ① 父进程在子进程运行期间的输出也留在同一份日志里（一次复盘是一段完整记录）；
      ② 两份独立实测（`probe_spawn_perm.py` / `probe_spawn_forms2.py`）都验过它。
      详见模块头部第 1 条 —— 那里记着仓里那条**错的**沙箱结论。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:                                        # noqa: BLE001
            pass
    saved = (os.dup(1), os.dup(2))
    try:
        lg, _eff = open_log(logpath)
        try:
            os.dup2(lg.fileno(), 1)
            os.dup2(lg.fileno(), 2)
            try:
                p = subprocess.Popen(argv, stdout=None, stderr=None, cwd=cwd)
                p.wait()
            finally:
                for stream in (sys.stdout, sys.stderr):
                    try:
                        stream.flush()
                    except Exception:                            # noqa: BLE001
                        pass
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
        finally:
            lg.close()
        return p.returncode
    finally:
        os.close(saved[0])
        os.close(saved[1])


def selftest():
    """自测：① 被占住的日志能自动轮转；② 子进程输出确实进了日志。

    ⚠ 自测**必须**落在工作区内。踩过：第一版用 `tempfile.mkdtemp()`，结果
      `open(<mkdtemp>/x.log, "a")` 直接 `PermissionError`（沙箱只放行
      `gettempdir()` 根与工作区，新建的临时子目录不在放行名单里）⇒
      自测把"沙箱不许写"误读成了"文件被占用"，第 ① 条假失败。
    """
    ok = 0
    bad = 0

    def chk(name, cond, extra=""):
        nonlocal ok, bad
        if cond:
            ok += 1
            print("  ok   %-46s %s" % (name, extra))
        else:
            bad += 1
            print("  FAIL %-46s %s" % (name, extra))

    d = os.path.join(HERE, "_proclog_selftest")
    os.makedirs(d, exist_ok=True)
    main = os.path.join(d, "x.log")
    for stale in os.listdir(d):
        os.remove(os.path.join(d, stale))

    # ① 主路径可写 ⇒ 就用主路径
    fh, eff = open_log(main)
    fh.write("a\n")
    fh.close()
    chk("主路径可写时不做轮转", eff == main, eff)

    # ② 用「拒绝写」的共享方式占住它，模拟 cmd.exe `>>` / 别家的句柄
    #
    #   ⚠ 踩过：第一版用 `os.open(main, O_WRONLY|O_APPEND)` 当"占用者"，结果
    #     **第 ② 条假失败** —— CPython 的 `os.open`/`open` 走 `_wopen`，
    #     共享方式是 **`_SH_DENYNO`（全允许）**，所以它根本占不住任何东西。
    #     真正的占用者是 **cmd.exe 的 `>>` 重定向**（access=GENERIC_WRITE、
    #     share=**FILE_SHARE_READ**）⇒ 必须用 `CreateFileW` 显式指定 share mode
    #     才能复现。这正是 `reviews.log` / `live_r16.stdout.log` 的实测状态。
    hold = _hold_deny_write(main)
    _ROT.clear()
    try:
        fh, eff2 = open_log(main)
        chk("被占住时轮转到新文件名", eff2 != main and os.path.exists(eff2), eff2)
        fh.write("b\n")
        fh.close()
        with open(eff2, encoding="utf-8") as f:
            chk("轮转文件真的收下了内容", f.read().strip() == "b")
    finally:
        _close_handle(hold)

    # ③ 主路径恢复 ⇒ 切回去（清掉 60s 缓存）
    _ROT.clear()
    fh, eff3 = open_log(main)
    fh.close()
    chk("占用解除后切回主路径", eff3 == main, eff3)

    # ④ 子进程输出进日志（这就是 run_live.py 缺的那条通路）
    _ROT.clear()
    rc = spawn_logged([sys.executable, "-c", "print('CHILD_OUT_OK')"], main)
    with open(main, encoding="utf-8") as f:
        body = f.read()
    chk("spawn_logged 返回码为 0", rc == 0, rc)
    chk("子进程 stdout 进了日志", "CHILD_OUT_OK" in body)

    print("\n=== proc_log 自测 %s（%d 通过 / %d 失败）==="
          % ("全部通过" if bad == 0 else "有失败", ok, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(selftest())
