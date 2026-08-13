"""Build and launch the small native macOS shell for the local IL Optimus service."""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


def default_app_path() -> Path:
    return Path.home() / "Applications" / "IL Optimus.app"


def install_macos_app(destination: Path | None = None, *, force: bool = False) -> Path:
    if platform.system() != "Darwin":
        raise RuntimeError("The native desktop shell currently supports macOS 13 or newer")
    if not shutil.which("xcrun"):
        raise RuntimeError("Apple Command Line Tools are required (run: xcode-select --install)")

    target = (destination or default_app_path()).expanduser().resolve()
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to replace it")
    source = Path(__file__).parent / "resources" / "desktop" / "macos"
    if not (source / "ILOptimusApp.swift").exists():
        raise RuntimeError("Desktop launcher resources are missing from this installation")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="iloptimus-desktop-") as temporary:
        staged = Path(temporary) / target.name
        contents = staged / "Contents"
        executable_dir = contents / "MacOS"
        executable_dir.mkdir(parents=True)
        shutil.copy2(source / "Info.plist", contents / "Info.plist")
        subprocess.run(
            [
                "xcrun", "swiftc", str(source / "ILOptimusApp.swift"),
                "-o", str(executable_dir / "ILOptimus"),
                "-framework", "AppKit", "-framework", "WebKit",
            ],
            check=True,
        )
        subprocess.run(["codesign", "--force", "--sign", "-", str(staged)], check=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(staged, target)
    return target


def launch_macos_app(path: Path | None = None) -> None:
    target = (path or default_app_path()).expanduser().resolve()
    if not target.exists():
        target = install_macos_app(target)
    subprocess.run(["open", str(target)], check=True)
