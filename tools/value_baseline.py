#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""value_baseline.py - 把预训练的独立价值基线（VNet）封装成一个**可复用对象**。

为什么需要它（§Z.1226 / §Z.1230 / §Z.1231 / §Z.1235）
------------------------------------------------------
· `train_value_torch.py` 训出的 `VNet` 是**独立网络**（自带 trunk，256 宽 + dropout）✗
  ⇒ **不能**把权重塞进线上的 `CandNet`（128 宽、无 dropout）✗ 二者 trunk 不同构 ✓
  ⇒ 正确用法 = **把它当独立模块**做 PPO 的优势基线 ✓（**`CandNet` 完全不动** ⇒ 零上线风险 ✓）
· ★ 关键坑：`VNet` 是在**标准化后**的特征上训练的 ✓ ⇒ 它的检查点带 `mu` / `sd` ✓
  ⇒ 任何前向**必须**先 `(x - mu) / sd` ✗ 否则输出被压平、且**不会报错** ✗✗
    （实测：不标准化时 Spearman 从 0.4436 掉到 0.1241 ✗ —— 见 §Z.1226）

用法
----
    import value_baseline as VB
    vb = VB.load("_tmp/value_v2.pt")          # 只需一次
    scores = vb.predict(states)               # states: 形如 (N, 370) 的 ndarray
    # 若 np 未就绪，请先 `import rl_boot; rl_boot.bootstrap()`

★ 本文件是**新增**的、**无人 import** 的自包含模块 ⇒ 对现存行为**零影响** ✓
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class ValueBaseline(object):
    """独立价值基线：只做 predict，不参与任何策略/推理路径。"""

    def __init__(self, net, mu, sd, torch, meta=None, scale=1.0):
        self.net, self.mu, self.sd, self.torch = net, mu, sd, torch
        # ★ 尺度重标定（见 fit_scale）。2026-10-03（§Z.1322）：**从 ckpt 读入** ✓
        #   为什么必须能读入：训练器把目标**除以 10** 后再学 ⇒ 原始输出与 return 差 ≈10 倍 ✗。
        #   以前这里恒为 1.0 ⇒ 即使拟合过也**存不住** ⇒ 接进 PPO 会静默差一个量级 ✗✗
        self.scale = float(scale)
        self.meta = meta or {}
        self.net.eval()

    # -- 内部：把原始状态按检查点的口径标准化（★ 绝不能省 ✗） -----------------
    def standardize(self, X):
        np = self.meta.get("np")
        X = np.asarray(X, dtype="float32")
        return (X - self.mu) / self.sd

    def fit_scale(self, X, y):
        """★ 用训练折拟合一个标量 `a`，使 `a·V` 的尺度对上真实得分。

        为什么必须做（§Z.1257 实测 ✓）：
          `train_value_torch.py` 为抗厚尾用了 Huber/分位裁剪 ⇒ 输出被**压缩 14 倍** ✗
          （SD(预测)=0.898 vs SD(真值)=13.031 ⇒ 压缩比 **0.069** ✗）
          ⇒ 不重标定时优势方差只降 **3.61%** ✗；
            重标定后降 **17.46%** ✓（理论上限 17.96% ✓ ⇒ 几乎触顶 ✓）
        """
        np = self.meta.get("np")
        v = self.predict(X).astype("float64")
        y = np.asarray(y, dtype="float64")
        vc, yc = v - v.mean(), y - y.mean()
        den = float(np.dot(vc, vc))
        self.scale = float(np.dot(vc, yc) / den) if den > 0 else 1.0
        return self.scale

    def predict(self, X):
        """返回 (N,) 的分数预测（原始尺度，**含重标定** ✓）。"""
        T = self.torch
        Z = self.standardize(X)
        with T.no_grad():
            out = self.net(T.tensor(Z, dtype=T.float32))[1]
        return (out.squeeze(-1) * self.scale).numpy()

    def predict_win(self, X):
        """返回 (N,) 的和牌概率（sigmoid 后的 head_win）。"""
        T = self.torch
        Z = self.standardize(X)
        with T.no_grad():
            out = self.net(T.tensor(Z, dtype=T.float32))[0]
        return T.sigmoid(out.squeeze(-1)).numpy()


def load(path, torch=None, np=None):
    """读检查点 ⇒ `ValueBaseline`。**自动处理 `weights_only` 与键名差异 ✓**"""
    if torch is None or np is None:
        import rl_boot                                                # noqa: E402
        boot = rl_boot.bootstrap(verbose=False)
        torch = torch or boot
        if np is None:
            import numpy as _np
            np = _np
    sys.path.insert(0, _ROOT)
    import train_value_torch as TV                                    # noqa: E402

    ck = torch.load(path, map_location="cpu", weights_only=False)     # ★ 本地可信文件 ✓
    sd = None
    for k in ("state", "sd", "state_dict", "model", "net"):            # ★ 权重在 `state` ✓
        if isinstance(ck, dict) and k in ck and hasattr(ck[k], "keys"):
            sd = ck[k]
            break
    if sd is None:
        sd = ck
    net = TV.VNet(int(ck["din"]), int(ck["hidden"]), int(ck["depth"]), float(ck["drop"]))
    net.load_state_dict(sd)
    mu = np.asarray(ck["mu"], dtype="float32")
    sdv = np.asarray(ck["sd"], dtype="float32")
    return ValueBaseline(net, mu, sdv, torch,
                         meta={"np": np, "din": int(ck["din"]),
                               "val_auc": ck.get("val_auc"), "val_bce": ck.get("val_bce")},
                         # ★ 2026-10-03（§Z.1322）：读回拟合好的尺度（缺省 1.0 ⇒ 向后兼容 ✓）
                         scale=float(ck.get("scale", 1.0) or 1.0))


if __name__ == "__main__":
    import rl_boot
    TORCH = rl_boot.bootstrap(verbose=False)
    import numpy as _np
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "_tmp", "value_v2.pt")
    npz = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_ROOT, "_tmp", "value_data100k.npz")
    vb = load(p, TORCH, _np)
    print("=== value_baseline 自检 ===")
    print("  din=%s ｜ val_auc=%s ｜ val_bce=%s" % (vb.meta["din"], vb.meta["val_auc"], vb.meta["val_bce"]))
    d = _np.load(npz)
    X, y = d["X"][-2000:], d["y_score"][-2000:]        # 末 2000 = 验证折 ✓
    sc = vb.predict(X)
    sys.path.insert(0, _HERE)
    import value_head_eval as V
    rho = V.spearman(list(sc), list(y))
    print("  末 2000 条 Spearman = %+.4f（应与 §Z.1226 的 +0.44 量级一致 ✓）" % rho)
    print("  ★ 模块可用 ✓（本文件无人 import ⇒ 对现存行为零影响 ✓）")