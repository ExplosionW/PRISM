#!/usr/bin/env python
"""运行元数据留存（§104.7 第 1 项）。

审查要求：「保存 checkpoint、完整配置、序列/靶点 ID、每 epoch 指标与代码 hash」。
此前的运行只留下 `.npz` 预测与一行汇总，无法回溯到底跑的是哪一版代码、
哪一批肽、以及训练过程中验证损失的轨迹——这正是本轮多次误判的根源之一
（例如 §103.9 只能靠参数量反推主干构成）。

本模块被 `gpu_margin_model.py` 与 `gpu_r8_metric.py` 共用，
以保证两边留存的字段一致，结构对照时可逐项核对。
"""
import hashlib
import json
import os
import time

import numpy as np
import torch


def code_hash(*paths):
    """对参与本次运行的源文件取 sha256，按文件名排序后串联。"""
    h = {}
    for p in paths:
        if p and os.path.exists(p):
            with open(p, "rb") as fh:
                h[os.path.basename(p)] = hashlib.sha256(fh.read()).hexdigest()
    return h


def save_run(ckpt_dir, tag, seed, *, state, cfg, hist, ids, code, extra=None):
    """写一份自足的运行记录。

    state — 早停选出的 state_dict（已 detach 到 CPU）
    cfg   — argparse 的完整 vars()，加上派生量
    hist  — [(epoch, val_loss, ...), ...] 逐 epoch 指标
    ids   — {'train': [...], 'val': [...], 'test': [...]} 肽序列与靶点名
    code  — code_hash() 的返回
    """
    if not ckpt_dir:
        return None
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{tag}_seed{seed}.pt")
    torch.save(dict(state={k: v.cpu() for k, v in state.items()},
                    cfg=cfg, hist=hist, ids=ids, code=code,
                    saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    extra=extra or {}), path)
    # 同时写一份纯文本侧车，便于不加载 torch 就能核对配置与代码版本
    side = dict(tag=tag, seed=seed, cfg={k: _plain(v) for k, v in cfg.items()},
                code=code, n_epoch=len(hist),
                n_train=len(ids.get("train", [])), n_val=len(ids.get("val", [])),
                n_test=len(ids.get("test", [])), extra=_plain(extra or {}))
    with open(path.replace(".pt", ".json"), "w") as fh:
        json.dump(side, fh, ensure_ascii=False, indent=1)
    return path


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
