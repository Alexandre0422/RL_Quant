"""
quant/ — GLAD 量化工具库

对外核心接口：
    LETQuantizer             : LET（α, β 两参数）+ INT8 fake-quant，折叠后推理零开销
    fold_weights_in_onnx_node: 将折叠后的 W', b' 写入 ONNX GraphSurgeon 节点
    FakeQuantize             : 固定步长 fake-quant（带 STE）
    MinMaxObserver           : 激活范围统计
"""

from .let       import LETQuantizer, fold_weights_in_onnx_node
from .fake_quant import FakeQuantize, ste_round
from .observer  import MinMaxObserver, EMAObserver, PercentileObserver

__all__ = [
    "LETQuantizer",
    "fold_weights_in_onnx_node",
    "FakeQuantize",
    "ste_round",
    "MinMaxObserver",
    "EMAObserver",
    "PercentileObserver",
]
