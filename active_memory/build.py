#!/usr/bin/env python3
"""Build script for creating a compiled binary of active-memory CLI.

Uses Nuitka to compile Python to native C code, producing a single
executable that doesn't expose source code.

Requirements:
    pip install nuitka ordered-set zstandard

Usage:
    python build.py                    # Build for current platform
    python build.py --onefile          # Single-file executable
    python build.py --target macos     # Cross-compile (limited)

Output:
    dist/active-memory                 # The compiled binary
    dist/active-memory.exe             # On Windows
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Verify build dependencies are available."""
    errors = []

    # Check Nuitka
    try:
        import nuitka
        print(f"  ✓ Nuitka {nuitka.__version__}")
    except ImportError:
        errors.append("Nuitka not installed: pip install nuitka")

    # Check C compiler
    if platform.system() == "Windows":
        # Nuitka will use MSVC or MinGW
        pass
    else:
        cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
        if cc:
            print(f"  ✓ C compiler: {cc}")
        else:
            errors.append("No C compiler found. Install gcc or clang.")

    # Check ordered-set (Nuitka performance dependency)
    try:
        import ordered_set
        print(f"  ✓ ordered-set")
    except ImportError:
        print("  ⚠ ordered-set not installed (optional but improves build speed)")

    if errors:
        print("\n  Missing dependencies:")
        for e in errors:
            print(f"    ✗ {e}")
        sys.exit(1)


def build(onefile: bool = True, standalone: bool = True):
    """Run Nuitka compilation."""
    print("\n═══════════════════════════════════════════")
    print("  Building active-memory binary")
    print("═══════════════════════════════════════════\n")

    check_dependencies()

    # Clean previous builds
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()

    # Build command
    cmd = [
        sys.executable, "-m", "nuitka",

        # -- Compilation mode --
        "--standalone" if standalone else "",
        "--onefile" if onefile else "",

        # -- Output --
        f"--output-dir=dist",
        "--output-filename=active-memory",

        # -- Include our package --
        "--include-package=active_memory",

        # -- Include data files --
        "--include-package-data=active_memory",

        # -- Optimisation --
        "--lto=yes",                          # Link-time optimisation
        "--python-flag=no_docstrings",        # Strip docstrings
        "--python-flag=no_asserts",           # Strip asserts

        # -- Remove debug info --
        "--python-flag=-O",                   # Optimised bytecode

        # -- Follow imports for our deps --
        "--follow-import-to=active_memory",
        "--follow-import-to=numpy",

        # -- Don't follow into stuff we don't need compiled --
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=setuptools",
        "--nofollow-import-to=pip",

        # -- Plugin for numpy --
        "--enable-plugin=numpy",

        # -- Metadata --
        "--product-name=active-memory",
        "--product-version=0.1.0",
        "--file-description=Active memory management for LLM context windows",

        # -- Entry point --
        "active_memory/cli.py",
    ]

    # Remove empty strings
    cmd = [c for c in cmd if c]

    print(f"  Command: {' '.join(cmd[:6])}...\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print("\n  ✗ Build failed")
        sys.exit(1)

    # Find the output
    binary_name = "active-memory.exe" if platform.system() == "Windows" else "active-memory"
    output_candidates = list(dist_dir.rglob(binary_name))

    if output_candidates:
        output = output_candidates[0]
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"\n  ✓ Build successful!")
        print(f"  ✓ Binary: {output}")
        print(f"  ✓ Size: {size_mb:.1f} MB")

        # Move to dist root if nested
        final = dist_dir / binary_name
        if output != final:
            shutil.move(str(output), str(final))
            print(f"  ✓ Moved to: {final}")
    else:
        print("\n  ⚠ Build completed but binary not found in dist/")


def build_for_distribution():
    """Build binaries for multiple platforms (CI/CD helper)."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    print(f"  Platform: {system}-{machine}")

    build(onefile=True)

    # Rename with platform suffix
    binary_name = "active-memory.exe" if system == "windows" else "active-memory"
    src = Path("dist") / binary_name
    if src.exists():
        suffix = ".exe" if system == "windows" else ""
        dst = Path("dist") / f"active-memory-{system}-{machine}{suffix}"
        shutil.move(str(src), str(dst))
        print(f"  ✓ Renamed to: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build active-memory binary")
    parser.add_argument("--onefile", action="store_true", default=True,
                        help="Produce a single executable (default)")
    parser.add_argument("--no-onefile", action="store_true",
                        help="Produce a directory distribution instead")
    parser.add_argument("--dist", action="store_true",
                        help="Build for distribution (adds platform suffix)")

    args = parser.parse_args()

    if args.dist:
        build_for_distribution()
    else:
        build(onefile=not args.no_onefile)
