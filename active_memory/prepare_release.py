#!/usr/bin/env python3
"""Prepare a minimal binary distribution directory.

This script creates a clean release directory containing:
  - README.md (public version)
  - LICENSE
  - build.yml
  - dist/ binaries (if present)
  - install.sh

Usage:
    python prepare_release.py
    python prepare_release.py --repo owner/active-memory
    python prepare_release.py --publish --repo owner/active-memory
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def prepare(output_dir: str = "dist-repo", repo_slug: str = "owner/active-memory"):
    """Create a clean public distribution directory."""
    out = Path(output_dir)

    # Clean
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    repo_root = Path(__file__).resolve().parents[1]

    # Copy public-facing files
    readme_src = repo_root / "README_PUBLIC.md"
    if not readme_src.exists():
        readme_src = repo_root / "README.md"
    shutil.copy(readme_src, out / "README.md")
    shutil.copy(repo_root / "LICENSE", out / "LICENSE")

    # Copy install script
    install_sh = out / "install.sh"
    install_sh.write_text(INSTALL_SCRIPT.replace("__REPO_SLUG__", repo_slug))
    install_sh.chmod(0o755)

    # Copy GitHub Actions workflow
    workflows = out / ".github" / "workflows"
    workflows.mkdir(parents=True)
    workflow_src = repo_root / "build.yml"
    if not workflow_src.exists():
        workflow_src = repo_root / ".github" / "workflows" / "build.yml"
    if workflow_src.exists():
        shutil.copy(workflow_src, workflows / "build.yml")

    # Copy binaries if they exist
    dist = repo_root / "dist"
    if dist.exists():
        out_dist = out / "dist"
        out_dist.mkdir()
        for f in dist.iterdir():
            if f.is_file() and not f.name.startswith("."):
                shutil.copy(f, out_dist / f.name)
                print(f"  ✓ Copied binary: {f.name}")

    # Create .gitignore
    (out / ".gitignore").write_text(
        "*.py\n"
        "*.pyc\n"
        "__pycache__/\n"
        "*.egg-info/\n"
        ".env\n"
        "*.pkl\n"
    )

    print(f"\n  ✓ Public repo prepared at: {out}/")
    print(f"  ✓ Files:")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"      {f.relative_to(out)}")

    return out


INSTALL_SCRIPT = """#!/bin/bash
# Install active-memory binary
set -e

VERSION="${1:-latest}"
REPO="__REPO_SLUG__"

# Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
    linux)  PLATFORM="linux" ;;
    darwin) PLATFORM="macos" ;;
    *)      echo "Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64|amd64)  ARCH="x86_64" ;;
    arm64|aarch64) ARCH="arm64" ;;
    *)             echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

BINARY="active-memory-${PLATFORM}-${ARCH}"

echo "Downloading active-memory for ${PLATFORM}-${ARCH}..."

if [ "$VERSION" = "latest" ]; then
    URL="https://github.com/${REPO}/releases/latest/download/${BINARY}"
else
    URL="https://github.com/${REPO}/releases/download/${VERSION}/${BINARY}"
fi

curl -fsSL "$URL" -o /tmp/active-memory
chmod +x /tmp/active-memory

# Install
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
if [ -w "$INSTALL_DIR" ]; then
    mv /tmp/active-memory "$INSTALL_DIR/active-memory"
else
    sudo mv /tmp/active-memory "$INSTALL_DIR/active-memory"
fi

echo "✓ Installed active-memory to $INSTALL_DIR/active-memory"
echo "  Run: active-memory --help"
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare binary distribution directory"
    )
    parser.add_argument(
        "--output", "-o", default="dist-repo",
        help="Output directory (default: dist-repo)"
    )
    parser.add_argument(
        "--publish", action="store_true",
        help="Also init git repo in the output directory"
    )
    parser.add_argument(
        "--repo",
        default="owner/active-memory",
        help="GitHub repo slug for install script links (default: owner/active-memory)",
    )

    args = parser.parse_args()
    out = prepare(args.output, args.repo)

    if args.publish:
        subprocess.run(["git", "init"], cwd=out)
        subprocess.run(["git", "add", "."], cwd=out)
        subprocess.run(
            ["git", "commit", "-m", "release: binary distribution"],
            cwd=out,
        )
        print("\n  Git repo initialized. Push with:")
        print(f"    cd {out}")
        print(f"    git remote add origin git@github.com:{args.repo}.git")
        print(f"    git push -u origin main")
