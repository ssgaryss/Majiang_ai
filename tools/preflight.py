# -*- coding: utf-8 -*-
"""启动前自检：把结论写进 `_preflight.txt`，供 start_live.ps1 读取。

★ 为什么要"写文件"而不是"看退出码"
  实测：在本沙箱里，**PowerShell 函数作用域内拿不到原生程序的退出码**
  （`& $exe -c ...` 之后 `$LASTEXITCODE` 为空，且不抛异常）。
  于是 `start_live.ps1` 第一版把**所有** Python 候选都判成"不可用"，
  而解释器明明就在 PATH 上（用户在别处 `python --version` 是正常的）。
  同一个坑还有第二个面：`return $LASTEXITCODE` 会把子进程的 **stdout 一起返回**，
  调用方拿到的就变成数组，`$rc -ne 0` 必然为真。
  ⇒ 所以这里改成：**Python 写文件，PowerShell 读文件**。两边都不碰退出码。

结论格式（JSON 单行也认，这里用 indent=1 便于人看）：
  {"ok": true/false, "python": "...", "exe": "...", "numpy": "...", "torch": "...",
   "run_live": "ok"/"<错误>", "err": "..."}
"""
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_preflight.txt")

res = {
    "ok": False, "python": sys.version.split()[0], "exe": sys.executable,
    "numpy": None, "torch": None, "run_live": None, "err": "",
}


def _fail(msg):
    res["err"] = msg
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))
    raise SystemExit(1)


try:
    import numpy
    res["numpy"] = numpy.__version__
except Exception as e:                                              # noqa: BLE001
    _fail("numpy 不可用：%s: %s" % (type(e).__name__, e))

try:
    sys.path.insert(0, ROOT)
    import rl_boot
    torch = rl_boot.bootstrap(verbose=False)
    res["torch"] = torch.__version__
except Exception as e:                                              # noqa: BLE001
    _fail("torch 不可用：%s: %s" % (type(e).__name__, e))

# ★ run_live.py 的导入链：把 argv 换成 `--help`，这样即使它没有 __main__ 守卫，
#   也只会打印帮助并以 SystemExit(0) 结束，而不会真的去联网开房。
try:
    _argv = list(sys.argv)
    sys.argv = ["run_live.py", "--help"]
    try:
        import run_live                                            # noqa: F401
        res["run_live"] = "ok"
    except SystemExit:
        res["run_live"] = "ok"          # --help 正常退出
    finally:
        sys.argv = _argv
except Exception as e:                                              # noqa: BLE001
    res["run_live"] = "%s: %s" % (type(e).__name__, e)
    res["err"] = traceback.format_exc(limit=3)

res["ok"] = bool(res["numpy"] and res["torch"] and res["run_live"] == "ok")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False))
raise SystemExit(0 if res["ok"] else 1)
