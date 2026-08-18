"""Hardware detection: CPU, RAM, GPU (CUDA / Apple Silicon / None)."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUInfo:
    name: str
    vram_gb: float
    type: str  # "cuda", "apple-silicon", "none"


@dataclass
class HardwareInfo:
    cpu_name: str
    cpu_cores: int
    ram_gb: float
    os: str
    arch: str
    gpu: GPUInfo
    python_version: str
    mlx_available: bool = False
    vllm_available: bool = False
    torch_available: bool = False
    recommended_backend: str = "none"  # "mlx", "vllm", "cpu"
    labels: list[str] = field(default_factory=list)

    @property
    def total_memory_gb(self) -> float:
        """Usable memory for model inference (unified on Apple Silicon, VRAM on CUDA)."""
        if self.gpu.type == "apple-silicon":
            # Apple Silicon: unified memory, but OS reserves ~2-3GB
            return max(1.0, self.ram_gb - 2.5)
        elif self.gpu.type == "cuda":
            return self.gpu.vram_gb
        return min(self.ram_gb * 0.5, 4.0)  # CPU: be conservative


def _detect_apple_silicon() -> Optional[GPUInfo]:
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout
        chip_line = [l for l in output.splitlines() if "Chip:" in l]
        name = chip_line[0].split("Chip:")[1].strip() if chip_line else "Apple Silicon"
        # Apple Silicon uses unified memory — VRAM = total RAM
        mem_line = [l for l in output.splitlines() if "Memory:" in l]
        vram = 0.0
        if mem_line:
            mem_str = mem_line[0].split("Memory:")[1].strip()
            if "GB" in mem_str:
                vram = float(mem_str.replace("GB", "").strip())
        return GPUInfo(name=name, vram_gb=vram, type="apple-silicon")
    except Exception:
        return GPUInfo(name="Apple Silicon (unknown)", vram_gb=0.0, type="apple-silicon")


def _detect_cuda() -> Optional[GPUInfo]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            # torch renamed ``total_mem`` -> ``total_memory`` in newer releases;
            # support both so detection works across torch versions.
            total = getattr(props, "total_memory", None)
            if total is None:
                total = getattr(props, "total_mem", 0)
            vram = total / (1024**3)
            return GPUInfo(name=name, vram_gb=round(vram, 1), type="cuda")
    except ImportError:
        pass
    # Fallback: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            name = parts[0].strip()
            vram = float(parts[1].strip()) / 1024.0  # MiB -> GiB
            return GPUInfo(name=name, vram_gb=round(vram, 1), type="cuda")
    except (FileNotFoundError, Exception):
        pass
    return None


def _check_mlx() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import mlx  # noqa: F401
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def _check_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def detect_hardware() -> HardwareInfo:
    """Detect all hardware capabilities."""
    import multiprocessing

    cpu_name = platform.processor() or "Unknown CPU"
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                cpu_name = result.stdout.strip()
        except Exception:
            pass

    # RAM
    ram_gb = 0.0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        if sys.platform == "darwin":
            try:
                result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
                ram_gb = round(int(result.stdout.strip()) / (1024**3), 1)
            except Exception:
                pass

    # GPU
    gpu = _detect_cuda()
    if gpu is None:
        gpu = _detect_apple_silicon()
    if gpu is None:
        gpu = GPUInfo(name="None", vram_gb=0.0, type="none")

    mlx_ok = _check_mlx()
    vllm_ok = _check_vllm()
    torch_ok = _check_torch()

    # Recommended backend
    if gpu.type == "apple-silicon" and mlx_ok:
        backend = "mlx"
    elif gpu.type == "cuda" and vllm_ok:
        backend = "vllm"
    elif gpu.type == "cuda" and torch_ok:
        backend = "vllm"  # will fall back to torch if vllm not installed
    else:
        backend = "cpu"

    labels: list[str] = []
    if gpu.type == "apple-silicon":
        labels.append("Apple Silicon")
    if gpu.type == "cuda":
        labels.append("CUDA GPU")
    if mlx_ok:
        labels.append("MLX")
    if vllm_ok:
        labels.append("vLLM")
    if torch_ok:
        labels.append("PyTorch")
    if not labels:
        labels.append("CPU-only")

    return HardwareInfo(
        cpu_name=cpu_name,
        cpu_cores=multiprocessing.cpu_count(),
        ram_gb=ram_gb,
        os=platform.system() + " " + platform.release(),
        arch=platform.machine(),
        gpu=gpu,
        python_version=sys.version.split()[0],
        mlx_available=mlx_ok,
        vllm_available=vllm_ok,
        torch_available=torch_ok,
        recommended_backend=backend,
        labels=labels,
    )
