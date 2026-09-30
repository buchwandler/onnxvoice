#!/usr/bin/env python
"""Build the onnxvoice documentation with Sphinx."""

import shutil
import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
BUILD_DIR = DOCS_DIR / "_build"
TARGETS = {
    "html",
    "dirhtml",
    "latex",
    "text",
    "man",
    "changes",
    "linkcheck",
    "doctest",
}


def main() -> int:
    """Build the requested Sphinx target."""
    target = sys.argv[1] if len(sys.argv) > 1 else "html"

    if target == "help":
        print(
            "Usage: python make.py [clean|html|dirhtml|latex|text|man|changes|"
            "linkcheck|doctest|all]"
        )
        return 0

    if target == "clean":
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        print(f"Cleaned {BUILD_DIR}")
        return 0

    if target != "all" and target not in TARGETS:
        print(f"Unknown target: {target}", file=sys.stderr)
        print("Use 'help' target for help", file=sys.stderr)
        return 1

    targets = ("html", "dirhtml", "latex") if target == "all" else (target,)
    for builder in targets:
        output_dir = BUILD_DIR / builder
        command = [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            builder,
            str(DOCS_DIR),
            str(output_dir),
        ]
        print(f"Building {builder} documentation...")
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode

    print(f"Documentation build finished in {BUILD_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
