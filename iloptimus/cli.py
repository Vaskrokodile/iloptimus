"""Optimus Studio CLI — start the server and open the browser."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import webbrowser


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8.

    On Windows the default console encoding is cp1252, which cannot encode
    the Unicode box-drawing characters used in the startup banner (or any
    non-Latin-1 output from the pipeline). Python 3.7+ supports
    ``reconfigure`` on the text stream wrapper; if it is unavailable we
    silently fall back to the platform default.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def _ensure_disk_env() -> None:
    """Redirect temp/cache dirs to the data drive on Windows.

    The C: drive on this machine is space-constrained. torch.compile's
    inductor cache and the OS temp dir can fill it up and crash compilation.
    Redirect them to the ILOPTIMUS_HOME drive (E:) if that env var is set.
    """
    import os

    home = os.environ.get("ILOPTIMUS_HOME")
    if not home:
        return
    home = os.path.abspath(home)
    # Only redirect if the home dir is on a different drive than the system temp
    sys_temp = os.environ.get("TEMP", os.environ.get("TMP", ""))
    if sys_temp and os.path.exists(sys_temp):
        sys_drive = os.path.splitdrive(sys_temp)[0]
        home_drive = os.path.splitdrive(home)[0]
        if sys_drive != home_drive:
            tmp_dir = os.path.join(home, "..", "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            os.environ["TEMP"] = tmp_dir
            os.environ["TMP"] = tmp_dir
            os.environ["TMPDIR"] = tmp_dir
    # Always set the inductor cache to the data drive
    inductor_cache = os.path.join(home, "..", "torch-inductor-cache")
    os.makedirs(inductor_cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache


def main():
    _ensure_utf8_stdout()
    _ensure_disk_env()
    parser = argparse.ArgumentParser(
        prog="iloptimus",
        description="Optimus Studio — a full local harness for open-source models (IL + PQLoRA).",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the Optimus Studio server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    serve_parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    # version
    subparsers.add_parser("version", help="Print version")

    subparsers.add_parser("hardware", help="Detect and print hardware info")
    subparsers.add_parser("doctor", help="Verify this machine can run local models and training")
    subparsers.add_parser("data-dir", help="Print the folder containing models, environments, and runs")
    desktop_parser = subparsers.add_parser("install-desktop", help="Install the native macOS desktop app")
    desktop_parser.add_argument("--force", action="store_true", help="Replace an existing app")
    subparsers.add_parser("desktop", help="Open the native desktop app")

    args = parser.parse_args()

    if args.command == "version":
        from . import __version__
        print(f"iloptimus {__version__}")
        return

    if args.command == "data-dir":
        from .core.storage import ensure_app_dirs
        print(ensure_app_dirs())
        return

    if args.command == "install-desktop":
        from .desktop import install_macos_app

        try:
            installed = install_macos_app(force=args.force)
        except (RuntimeError, FileExistsError, subprocess.CalledProcessError) as error:
            print(f"Desktop installation failed: {error}", file=sys.stderr)
            raise SystemExit(2) from error
        print(f"Installed Optimus Studio at {installed}")
        return

    if args.command == "desktop":
        from .desktop import launch_macos_app

        try:
            launch_macos_app()
        except (RuntimeError, FileExistsError, subprocess.CalledProcessError) as error:
            print(f"Could not open Optimus Studio: {error}", file=sys.stderr)
            raise SystemExit(2) from error
        return

    if args.command in {"hardware", "doctor"}:
        from .core import detect_hardware
        hw = detect_hardware()
        print(f"CPU: {hw.cpu_name} ({hw.cpu_cores} cores)")
        print(f"RAM: {hw.ram_gb} GB")
        print(f"OS:  {hw.os} ({hw.arch})")
        print(f"GPU: {hw.gpu.name} ({hw.gpu.type})")
        if hw.gpu.type == "apple-silicon":
            print(f"     Unified memory: {hw.gpu.vram_gb} GB")
        elif hw.gpu.type == "cuda":
            print(f"     VRAM: {hw.gpu.vram_gb} GB")
        print(f"Python: {hw.python_version}")
        print(f"MLX:    {hw.mlx_available}")
        print(f"vLLM:   {hw.vllm_available}")
        print(f"PyTorch: {hw.torch_available}")
        print(f"Recommended backend: {hw.recommended_backend}")
        print(f"Available memory for models: {hw.total_memory_gb:.1f} GB")
        if args.command == "doctor":
            print()
            if hw.recommended_backend == "mlx" and hw.mlx_available:
                print("Ready: local download, chat, IL, and GRPO training are available through MLX.")
            elif hw.recommended_backend == "vllm" and (hw.vllm_available or hw.torch_available):
                print(
                    "Ready: local download, chat, IL, and GRPO training are available through "
                    "vLLM + HuggingFace Transformers + PEFT."
                )
            else:
                print(
                    "Not ready for training: Optimus Studio needs Apple Silicon (MLX) or an NVIDIA CUDA GPU "
                    "(vLLM / PyTorch)."
                )
                raise SystemExit(2)
        return

    if args.command == "serve" or args.command is None:
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 7860)
        no_browser = getattr(args, "no_browser", False)

        url = f"http://{host}:{port}"

        print()
        print("  ╔══════════════════════════════════════════╗")
        from . import __version__
        print(f"  ║          OPTIMUS STUDIO  v{__version__:<15}║")
        print("  ║   Local harness for open-source models   ║")
        print("  ╚══════════════════════════════════════════╝")
        print()
        from .core.storage import ensure_app_dirs
        print(f"  → Local app: {url}")
        print(f"  → Data folder: {ensure_app_dirs()}")
        print("  → Press Ctrl+C to stop")
        print()

        if not no_browser:
            # Open browser after a short delay
            def _open_browser():
                time.sleep(1.5)
                webbrowser.open(url)
            threading.Thread(target=_open_browser, daemon=True).start()

        try:
            import uvicorn
            uvicorn.run(
                "iloptimus.server:app",
                host=host,
                port=port,
                log_level="info",
                reload=False,
            )
        except ImportError:
            print("Error: uvicorn not installed. Run: pip install uvicorn")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
