import subprocess, sys
py = sys.executable

tools = [
    ("tensorrt",                  "import tensorrt as trt; print(trt.__version__)"),
    ("torch_tensorrt",            "import torch_tensorrt; print(torch_tensorrt.__version__)"),
    ("onnxruntime-gpu",           "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"),
    ("onnxruntime.quantization",  "from onnxruntime.quantization import quantize_static, QuantFormat; print('ok')"),
    ("onnx-graphsurgeon",         "import onnx_graphsurgeon as gs; print(gs.__version__)"),
    ("polygraphy",                "import polygraphy; print(polygraphy.__version__)"),
    ("bitsandbytes",              "import bitsandbytes as bnb; print(bnb.__version__)"),
    ("onnxsim",                   "import onnxsim; print('ok')"),
    ("neural-compressor (Intel)", "import neural_compressor; print(neural_compressor.__version__)"),
    ("auto-gptq",                 "from auto_gptq import AutoGPTQForCausalLM; print('ok')"),
    ("optimum",                   "import optimum; print(optimum.__version__)"),
    ("modelopt",                  "import modelopt; print(modelopt.__version__)"),
]

print(f"{'工具':<30} {'状态':>6}  {'版本/信息'}")
print("-" * 70)
for name, cmd in tools:
    result = subprocess.run(
        [py, "-c", cmd],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0:
        info = result.stdout.strip().split("\n")[-1][:40]
        print(f"  {name:<28} [OK]   {info}")
    else:
        err = result.stderr.strip().split("\n")[-1][:40]
        print(f"  {name:<28} [--]   {err}")

# 额外检查：trtexec 命令行工具
import os, glob
trtexec_paths = (
    glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\trtexec.exe") +
    glob.glob(r"C:\Users\LiZixuan\.conda\envs\pytorch_gpu\Lib\site-packages\tensorrt_bindings\*.exe")
)
print(f"\n  {'trtexec (CLI)':<28} {'[OK]' if trtexec_paths else '[--]'}   {trtexec_paths[0] if trtexec_paths else 'not found'}")
