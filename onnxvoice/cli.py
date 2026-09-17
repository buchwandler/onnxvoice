from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manager import OnnxVoice
from .validation import verify_installation


def _manager(args: argparse.Namespace) -> OnnxVoice:
    return OnnxVoice(cache_dir=args.cache_dir, offline=getattr(args, "offline", False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onnxvoice")
    parser.add_argument("--cache-dir", type=Path, help="Override the shared onnxvoice cache")
    parser.add_argument("--offline", action="store_true", help="Disable network access")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List catalog entries")
    list_parser.add_argument("--system", choices=("piper", "kokoro"))
    list_parser.add_argument("--installed", action="store_true")
    list_parser.add_argument("--language")
    list_parser.add_argument("--quality")
    list_parser.add_argument("--refresh", action="store_true")

    install_parser = sub.add_parser("install", help="Install a model/voice from a catalog")
    install_parser.add_argument("ref", help="e.g. piper:en_US-lessac-medium or kokoro:v1.0")
    install_parser.add_argument("--quality")
    install_parser.add_argument("--refresh", action="store_true")
    install_parser.add_argument("--force", action="store_true")

    path_parser = sub.add_parser("path", help="Print an installed model/voice path")
    path_parser.add_argument("ref")

    show_parser = sub.add_parser("show", help="Show an installed manifest")
    show_parser.add_argument("ref")

    remove_parser = sub.add_parser("remove", help="Remove an installation")
    remove_parser.add_argument("ref")

    verify_parser = sub.add_parser("verify", help="Verify checksums for an installation")
    verify_parser.add_argument("ref")

    cache_parser = sub.add_parser("cache", help="Show cache information")
    cache_sub = cache_parser.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("info")
    cache_sub.add_parser("gc")

    import_parser = sub.add_parser(
        "import", help="Import external runtime files into the shared store"
    )
    import_parser.add_argument("--system", choices=("piper", "kokoro"), required=True)
    import_parser.add_argument("--id", required=True, dest="item_id")
    import_parser.add_argument("--model", type=Path, required=True)
    import_parser.add_argument("--config", type=Path)
    import_parser.add_argument("--voices", type=Path)
    import_parser.add_argument("--sample-rate", type=int)
    import_parser.add_argument("--force", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manager = _manager(args)

    try:
        if args.command == "list":
            items = manager.list(
                args.system,
                installed=args.installed,
                language=args.language,
                quality=args.quality,
                refresh=args.refresh,
            )
            for item in items:
                if hasattr(item, "ref"):
                    suffix = ""
                    if getattr(item, "metadata", {}).get("quality"):
                        suffix = f" [{item.metadata['quality']}]"
                    print(f"{item.ref}{suffix}")
            return 0

        if args.command == "install":
            installed = manager.install(
                args.ref,
                quality=args.quality,
                refresh=args.refresh,
                force=args.force,
            )
            print(installed.path)
            return 0

        if args.command == "path":
            print(manager.where(args.ref))
            return 0

        if args.command == "show":
            system, item_id = args.ref.split(":", 1)
            installation = manager.store.get(system, item_id)
            print((installation.path / "manifest.json").read_text(encoding="utf-8"), end="")
            return 0

        if args.command == "remove":
            manager.remove(args.ref)
            return 0

        if args.command == "verify":
            system, item_id = args.ref.split(":", 1)
            report = verify_installation(manager.store, manager.store.get(system, item_id))
            print(json.dumps({"ok": report.ok, "checks": report.checks}))
            return 0

        if args.command == "cache":
            if args.cache_command == "info":
                print(manager.store.root)
                print(f"installed={len(manager.installed())}")
                return 0
            if args.cache_command == "gc":
                print(f"removed_blobs={manager.store.gc()}")
                return 0

        if args.command == "import":
            installed = manager.import_model(
                system=args.system,
                item_id=args.item_id,
                model=args.model,
                config=args.config,
                voices=args.voices,
                sample_rate=args.sample_rate,
                force=args.force,
            )
            print(installed.path)
            return 0
    except Exception as exc:
        print(f"onnxvoice: {exc}", file=sys.stderr)
        return 2

    parser.error("unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
