"""IL Optimus CLI — start the server and open the browser."""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser


def main():
    parser = argparse.ArgumentParser(
        prog="iloptimus",
        description="IL Optimus — run Intuition Learning pipelines locally with a web frontend.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the IL Optimus server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    serve_parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    # version
    subparsers.add_parser("version", help="Print version")

    subparsers.add_parser("hardware", help="Detect and print hardware info")
    subparsers.add_parser("doctor", help="Verify this machine can run local models and training")
    subparsers.add_parser("data-dir", help="Print the folder containing models, environments, and runs")

    args = parser.parse_args()

    if args.command == "version":
        from . import __version__
        print(f"iloptimus {__version__}")
        return

    if args.command == "data-dir":
        from .core.storage import ensure_app_dirs
        print(ensure_app_dirs())
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
            else:
                print("Not ready for training: IL Optimus 0.2 currently requires Apple Silicon with MLX.")
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
        print(f"  ║          IL OPTIMUS  v{__version__:<18}║")
        print("  ║   Intuition Learning Pipeline Studio     ║")
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
