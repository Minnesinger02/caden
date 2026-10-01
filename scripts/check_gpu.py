"""CUDA smoke check; never treat CUDA availability alone as a successful kernel test."""
import argparse
import json
import platform
import subprocess
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--out', default='environment-windows.json')
a = p.parse_args()
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Check NVIDIA GPU, driver and CUDA PyTorch wheel.')
x = torch.randn(256, 256, device='cuda')
y = x @ x
assert torch.isfinite(y).all()
torch.cuda.synchronize()
info = {'platform': platform.platform(), 'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'vram_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
        'bf16_supported': torch.cuda.is_bf16_supported(), 'kernel_test_passed': True}
if platform.system() == 'Windows':
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong),
                    ('total_pagefile', ctypes.c_ulonglong), ('available_pagefile', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
                    ('available_extended', ctypes.c_ulonglong)]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        info.update(ram_gib=memory.total_physical / 2**30, available_ram_gib=memory.available_physical / 2**30)
try:
    info['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
                                      capture_output=True, text=True, timeout=10, check=True).stdout.strip()
except (OSError, subprocess.SubprocessError):
    pass
Path(a.out).write_text(json.dumps(info, indent=2), encoding='utf-8')
print(json.dumps(info, indent=2))
