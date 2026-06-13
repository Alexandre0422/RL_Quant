"""
qrt — GLAD Quantized RunTime（非 TRT 量化主线）
══════════════════════════════════════════════════════════════════════════════
路线: torch.compile(max-autotune) 图融合 + CUDA Graph + 选择性 INT8(_int_mm) + LET

与 TRT 路线(bench_let_int8_full_remote.py / bench_refit_train.py / engines/)
完全并行,不修改任何原文件;quant/let.py 的 LETQuantizer 只读复用。

模块:
    runtime.InferenceRunner   rollout 推理加速器(compile / manual-graph 双形态)
    qlinear.QuantLinear       W8A8 静态量化 Linear(LET 前置,训练/推理双路径)
    swap                      模型手术(替换/还原/批量刷新)
    calib                     校准管线(激活采集 → LET 优化 → swap plan)

设计依据与实测数据: GLAD_Quant/PLAN_compile_quant.md
"""
from .runtime import InferenceRunner, TrainStepRunner
from .qlinear import QuantLinear
from .encoder_opt import patch_encoder, unpatch_encoder
from .cocalib import CoCalibrator
from .tap import TapStore, install_taps, TapLinear
from . import swap, calib

# qconv 依赖 triton(Linux/GPU 环境);Windows 本地开发无 triton 时跳过
try:
    from .qconv import QuantCNN, swap_cnn, calib_cnn_scale
    _HAS_QCONV = True
except ImportError:
    _HAS_QCONV = False

__all__ = ["InferenceRunner", "TrainStepRunner", "QuantLinear", "CoCalibrator",
           "TapStore", "install_taps", "TapLinear",
           "swap", "calib", "QuantCNN", "swap_cnn", "calib_cnn_scale",
           "patch_encoder", "unpatch_encoder"]
