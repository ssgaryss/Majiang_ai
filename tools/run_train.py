#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_train.py - 训练启动包装器：把 stdout/stderr **都**写进日志（进程内重定向 ✓）。

为什么需要：本沙箱**禁止**原生程序用管道捕获输出 ✗
  ⇒ PowerShell 的 `2> file`（原生程序）会失败 ✗（实测：python 直接起不来 ✗）
  ⇒ 崩溃时**没有 traceback** ⇒ 死因查不出（本会话已踩过 ✗）
本包装器用 `sys.stdout/sys.stderr = open(log)` 在**进程内**完成 ⇒ 不触发沙箱边界 ✓

用法: python tools/run_train.py <logfile> <train_selfplay_rl.py 及其参数...>
"""
import io, os, runpy, sys

def main():
    log = sys.argv[1]
    script = sys.argv[2]
    argv = sys.argv[2:]
    f = open(log, "a", encoding="utf-8", buffering=1)
    sys.stdout = f
    sys.stderr = f
    print("\n=== [run_train] 启动 %s 参数=%s ===" % (script, " ".join(argv[1:])), flush=True)
    sys.argv = argv
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc(file=f)
        f.flush()
        raise

if __name__ == "__main__":
    main()