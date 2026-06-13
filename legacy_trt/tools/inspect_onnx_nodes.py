"""检查 ONNX 图里所有 op 类型和节点名，找出需要 INT8 量化和需要排除的节点"""
import onnx
from collections import Counter

m = onnx.load("engines/glad_actor_base.onnx")
g = m.graph

# 统计 op 类型分布
op_count = Counter(n.op_type for n in g.node)
print("=== Op 类型统计 ===")
for op, cnt in sorted(op_count.items(), key=lambda x: -x[1]):
    print(f"  {op:<25} {cnt}")

# INT8 量化目标（Conv/GEMM）和需要排除的节点
INT8_TARGET  = {"Conv", "Gemm", "MatMul"}
MUST_EXCLUDE = {"Softmax", "Gather", "GatherND", "TopK", "ScatterND",
                "BatchNormalization", "LayerNormalization"}

print("\n=== INT8 目标节点（Conv/GEMM/MatMul） ===")
for n in g.node:
    if n.op_type in INT8_TARGET:
        inputs  = list(n.input)[:2]
        outputs = list(n.output)[:1]
        print(f"  [{n.op_type}] {n.name or '(noname)'}  in={inputs}  out={outputs}")

print("\n=== 需要排除的节点（保留 FP16）===")
for n in g.node:
    if n.op_type in MUST_EXCLUDE:
        print(f"  [{n.op_type}] {n.name or '(noname)'}  in={list(n.input)}")

print("\n=== 所有节点名（供 nodes_to_exclude 使用）===")
exclude_names = [n.name for n in g.node if n.op_type in MUST_EXCLUDE]
print(exclude_names)
