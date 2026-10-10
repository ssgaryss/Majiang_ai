#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL 训练器的**启动前修复层**（sys.path / numpy / CUDA torch / 设备）。

为什么单独一个文件
------------------
`train_cand_ranker.py` 的注释里记着两个**实测踩过的坑**，本仓凡是要用 torch 的新脚本
都必须避开它们，所以把"修路径"这件事收进一个函数，供所有 RL 模块在最前面调用：

  (a) `sys.path` 顺序写反 ⇒ 静默用回 CPU 版 torch（会打印 `设备 = cpu`）。
      `_pkgs`（含 CPU torch 2.14.0）与 `_pkgs_gpu`（CUDA torch 2.9.1+cu130）都是
      `insert(0, ...)`，**谁后插谁在前面** ⇒ 必须**先** `_pkgs`、**后** `_pkgs_gpu`。
      本模块额外做一次**硬核对**：`MJ_USE_GPU=1` 时若 `torch.__version__` 里含 "cpu"
      就直接抛错，而不是打印一行容易被忽略的警告。

  (b) numpy 的坑：仓库自带的 `_pkgs/` **只有 torch 及其依赖，没有 numpy**。
      如果运行的解释器里 `import numpy` 失败（例如 venv 是
      `include-system-site-packages=false`、而 numpy 只装在基础解释器里），
      就必须把那个 site-packages **挂进 sys.path**，否则 `import numpy` 直接
      ModuleNotFoundError（而 torch 只会给一句 "Failed to initialize NumPy" 的警告继续跑）。
      `find_numpy_site()` 按 `_venv/pyvenv.cfg` 的 `home =` 反查、并用 `sysconfig`
      与**环境变量**枚举候选 —— **不写死任何盘符或绝对路径**（见该函数注释）。

用法（**必须在 `import torch` / `import numpy` 之前**）：
    import rl_boot
    TORCH = rl_boot.bootstrap()          # 返回 torch 模块
    dev = rl_boot.pick_device(TORCH)     # "cuda" 或 "cpu"
"""
import glob
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))

_DONE = {"paths": False, "numpy": None}


def _add(p, front=True):
    p = os.path.abspath(p)
    if not os.path.isdir(p):
        return False
    if p in sys.path:
        sys.path.remove(p)
    if front:
        sys.path.insert(0, p)
    else:
        sys.path.append(p)
    return True


def find_numpy_site():
    """找出含 numpy 的 site-packages（**不猜**：逐个候选实际探测 `numpy/` 是否存在）。

    ★ 可移植性要求：这里**不许出现写死的盘符/绝对路径**（否则别人 clone 下来就废）。
      所以候选全部由「相对当前解释器的位置」+「标准环境变量」拼出来：
        · `MJ_NUMPY_SITE`（显式指定，最高优先）
        · venv 的 `pyvenv.cfg` 里的 `home =`（反查基础解释器）
        · `sysconfig` 给出的 purelib / platlib（**跨平台**的唯一正确写法）
        · `sys.base_prefix` / `sys.prefix`
        · Windows 上其它 Python 安装（用 `ProgramFiles` 等**环境变量**拼，不写盘符）
    """
    cands = []
    env = os.environ.get("MJ_NUMPY_SITE")
    if env:
        cands.append(env)
    # ① venv 的 pyvenv.cfg 里写着基础解释器的 home —— 最可靠的一条
    cfg = os.path.join(REPO, "_venv", "pyvenv.cfg")
    if os.path.isfile(cfg):
        try:
            with open(cfg, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip().lower().startswith("home"):
                        home = line.split("=", 1)[1].strip()
                        cands.append(os.path.join(home, "Lib", "site-packages"))
        except OSError:
            pass
    # ② 当前解释器自己报的站点目录（跨平台正确写法）
    try:
        import sysconfig
        for key in ("purelib", "platlib"):
            p = sysconfig.get_paths().get(key)
            if p:
                cands.append(p)
    except Exception:                                                # noqa: BLE001
        pass
    # ③ 基础/当前前缀（Windows 布局）
    cands.append(os.path.join(sys.base_prefix, "Lib", "site-packages"))
    cands.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    # ④ 机器上**其它** Python 的安装位置：用环境变量拼，绝不写死盘符
    for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        root = os.environ.get(var)
        if not root:
            continue
        cands += glob.glob(os.path.join(root, "Python*", "Lib", "site-packages"))
        cands += glob.glob(os.path.join(root, "Programs", "Python",
                                        "Python*", "Lib", "site-packages"))
    seen = set()
    for p in cands:
        if not p or p in seen:
            continue
        seen.add(p)
        try:
            if os.path.isdir(os.path.join(p, "numpy")):
                return p
        except OSError:
            continue
    return None


def bootstrap(verbose=True):
    """修好 sys.path（numpy + _pkgs + _pkgs_gpu），返回 `torch` 模块。幂等。"""
    if not _DONE["paths"]:
        _add(REPO)                                     # 让 `import features_full` 等成立
        # numpy：必须**在 import torch 之前**挂好，否则 torch 会退化（无 numpy 桥）
        try:
            import numpy                                  # noqa: F401
            _DONE["numpy"] = "already-importable"
        except ImportError:
            site = find_numpy_site()
            if site is None:
                raise RuntimeError(
                    "找不到 numpy。请设 MJ_NUMPY_SITE=<含 numpy 的 site-packages>。"
                    "（本仓库 `_pkgs/` 不含 numpy，venv 也没装 —— 见本文件头注释 (b)。）")
            _add(site)
            _DONE["numpy"] = site
        # ★ 顺序：先 _pkgs，后 _pkgs_gpu（后者必须排在前面）
        _add(os.path.join(REPO, "_pkgs"))
        if os.environ.get("MJ_USE_GPU") == "1":
            _add(os.path.join(REPO, "_pkgs_gpu"))
        _DONE["paths"] = True

    import torch                                          # noqa: E402

    # 硬核对 (a)：显式要 GPU 却拿到 cpu 版 torch ⇒ 这是"静默退化"，直接报错
    if os.environ.get("MJ_USE_GPU") == "1":
        if "cpu" in torch.__version__ and "+cpu" in torch.__version__:
            raise RuntimeError(
                "MJ_USE_GPU=1 但 import 到的是 CPU 版 torch（%s）⇒ sys.path 顺序被写反了。"
                "必须 `insert(0, '_pkgs')` 之后再 `insert(0, '_pkgs_gpu')`。"
                % torch.__version__)
    if verbose and not _DONE.get("_said"):
        _DONE["_said"] = True
        print("[rl_boot] repo=%s" % REPO)
        print("[rl_boot] numpy 来源 = %s" % _DONE["numpy"])
        print("[rl_boot] torch %s（build=%s, cuda=%s）"
              % (torch.__version__, getattr(torch.version, "cuda", None),
                 torch.cuda.is_available()))
    return torch


def pick_device(torch):
    """`MJ_USE_GPU=1` 且 cuda 可用 ⇒ "cuda"，否则 "cpu"（与 train_cand_ranker 同口径）。"""
    return "cuda" if (os.environ.get("MJ_USE_GPU") == "1"
                      and torch.cuda.is_available()) else "cpu"


def device_report(torch, dev):
    if dev == "cuda":
        return "设备 = cuda（%s，torch %s / cu%s，cap %s）" % (
            torch.cuda.get_device_name(0), torch.__version__,
            getattr(torch.version, "cuda", "?"),
            ".".join(str(x) for x in torch.cuda.get_device_capability(0)))
    return ("设备 = cpu（torch %s；MJ_USE_GPU=%r, cuda.is_available=%r）"
            % (torch.__version__, os.environ.get("MJ_USE_GPU"),
               torch.cuda.is_available()))


if __name__ == "__main__":
    t = bootstrap()
    print(device_report(t, pick_device(t)))
    import numpy as np
    print("numpy", np.__version__, np.__file__)
    a = t.zeros(2, dtype=t.float32)
    d = pick_device(t)
    print("smoke: (a+1) on %s = %s" % (d, (a.to(d) + 1).detach().cpu().numpy()))
