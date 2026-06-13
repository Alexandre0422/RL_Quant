import sys, torch
print(f"torch       : {torch.__version__}")
print(f"cuda        : {torch.version.cuda}")
print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"SM          : {torch.cuda.get_device_capability()}")
print()

pkgs = {
    "torch_tensorrt":  "torch_tensorrt",
    "tensorrt":        "tensorrt",
    "onnx":            "onnx",
    "onnxruntime_gpu": "onnxruntime",
    "polygraphy":      "polygraphy",
}

for display, import_name in pkgs.items():
    try:
        m = __import__(import_name)
        ver = getattr(m, "__version__", "installed")
        print(f"  {display:<18}: {ver}")
    except ImportError:
        print(f"  {display:<18}: NOT installed")

# 额外检查：pip list 里看 tensorrt 相关
print()
try:
    import pkg_resources
    installed = {d.project_name.lower(): d.version for d in pkg_resources.working_set}
    trt_pkgs = {k:v for k,v in installed.items() if "tensorrt" in k or "trt" in k}
    if trt_pkgs:
        print("TRT-related packages found:")
        for k, v in trt_pkgs.items():
            print(f"  {k}: {v}")
    else:
        print("No TRT packages in pip list.")
except Exception as e:
    print(f"pkg_resources error: {e}")
