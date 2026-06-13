"""Inspect MHA node weight inputs and trace producers in glad_actor_base.onnx"""
import onnx, onnx_graphsurgeon as gs

g = gs.import_onnx(onnx.load("engines/glad_actor_base.onnx"))

print("=== MHA nodes ===")
for node in g.nodes:
    if any(k in node.name for k in ["MatMul_69", "MatMul_71", "Gemm_139"]):
        print(f"\n[{node.op}] {node.name}")
        for i, inp in enumerate(node.inputs):
            t = "Constant" if isinstance(inp, gs.Constant) else "Variable"
            shape = inp.shape if hasattr(inp, "shape") else "?"
            print(f"  inputs[{i}]: {t:<10}  name={inp.name!r}  shape={shape}")
        for i, out in enumerate(node.outputs):
            print(f"  outputs[{i}]: name={out.name!r}  shape={out.shape}")

for target_name in ["node_MatMul_69", "node_MatMul_71"]:
    print(f"\n=== Trace {target_name} inputs[1] ===")
    w_var = None
    for node in g.nodes:
        if node.name == target_name:
            w_var = node.inputs[1]
            break
    if w_var is None:
        print("  not found")
        continue
    print(f"  weight: {type(w_var).__name__}  name={w_var.name!r}  shape={w_var.shape}")
    for node in g.nodes:
        if w_var in node.outputs:
            print(f"  producer: [{node.op}] {node.name}")
            for i, inp in enumerate(node.inputs):
                t = "Constant" if isinstance(inp, gs.Constant) else "Variable"
                shape = inp.shape if hasattr(inp, "shape") else "?"
                if isinstance(inp, gs.Constant) and inp.values is not None:
                    dtype = str(inp.values.dtype)
                else:
                    dtype = ""
                print(f"    inputs[{i}]: {t:<10}  name={inp.name!r}  shape={shape}  {dtype}")
            # If producer's input is also a Variable, trace one more level
            for inp2 in node.inputs:
                if isinstance(inp2, gs.Variable):
                    for node2 in g.nodes:
                        if inp2 in node2.outputs:
                            print(f"    -> producer of {inp2.name!r}: [{node2.op}] {node2.name}")
                            for i2, inp3 in enumerate(node2.inputs):
                                t2 = "Constant" if isinstance(inp3, gs.Constant) else "Variable"
                                shape2 = inp3.shape if hasattr(inp3, "shape") else "?"
                                print(f"         inputs[{i2}]: {t2:<10}  name={inp3.name!r}  shape={shape2}")
                            break
            break

print("\n=== All Constant tensors with 3D-related shapes (D=64) ===")
D = 64
for name, tensor in g.tensors().items():
    if isinstance(tensor, gs.Constant) and tensor.values is not None:
        shape = tuple(tensor.shape) if tensor.shape else ()
        if any(s in (3*D, D, 2*D) for s in shape) and len(shape) in (1, 2):
            print(f"  {name!r}  shape={shape}  dtype={tensor.values.dtype}")
