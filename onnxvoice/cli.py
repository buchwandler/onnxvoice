from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .catalog_tools import pocket as pocket_catalog_tools
from .catalog_tools.piper import (
    DEFAULT_REPOSITORY,
    DEFAULT_REVISION,
    fetch_and_build_catalog,
    load_catalog,
    verify_catalog,
)
from .manager import OnnxVoice
from .systems import registered_systems
from .validation import verify_installation


def _manager(args: argparse.Namespace) -> OnnxVoice:
    return OnnxVoice(cache_dir=args.cache_dir, offline=getattr(args, "offline", False))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onnxvoice")
    parser.add_argument("--cache-dir", type=Path, help="Override the shared onnxvoice cache")
    parser.add_argument("--offline", action="store_true", help="Disable network access")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List catalog entries")
    list_parser.add_argument("--system", choices=registered_systems())
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
    import_parser.add_argument("--system", choices=registered_systems(), required=True)
    import_parser.add_argument("--id", required=True, dest="item_id")
    import_parser.add_argument("--model", type=Path, required=True)
    import_parser.add_argument("--config", type=Path)
    import_parser.add_argument("--voices", type=Path)
    import_parser.add_argument("--sample-rate", type=int)
    import_parser.add_argument("--force", action="store_true")

    catalog_parser = sub.add_parser("catalog", help="Build and verify external catalogs")
    catalog_sub = catalog_parser.add_subparsers(dest="catalog_system", required=True)

    # Piper catalog subparser
    piper_parser = catalog_sub.add_parser("piper", help="Manage the Piper catalog")
    piper_sub = piper_parser.add_subparsers(dest="catalog_action", required=True)

    piper_build_parser = piper_sub.add_parser("build", help="Build a pinned Piper catalog")
    piper_build_parser.add_argument("--output", type=Path, required=True)
    piper_build_parser.add_argument("--source-output", type=Path)
    piper_build_parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    piper_build_parser.add_argument("--revision", default=DEFAULT_REVISION)

    piper_verify_parser = piper_sub.add_parser("verify", help="Verify a Piper catalog")
    piper_verify_parser.add_argument("--catalog", type=Path, required=True)
    piper_verify_parser.add_argument("--source", type=Path)

    # Pocket catalog subparser
    pocket_parser = catalog_sub.add_parser("pocket", help="Manage the Pocket catalog")
    pocket_sub = pocket_parser.add_subparsers(dest="catalog_action", required=True)

    pocket_build_parser = pocket_sub.add_parser("build", help="Build a pinned Pocket catalog")
    pocket_build_parser.add_argument("--output", type=Path, required=True)
    pocket_build_parser.add_argument("--source-output", type=Path)
    pocket_build_parser.add_argument(
        "--repository", default=pocket_catalog_tools.DEFAULT_REPOSITORY
    )
    pocket_build_parser.add_argument("--revision", default=pocket_catalog_tools.DEFAULT_REVISION)

    pocket_verify_parser = pocket_sub.add_parser("verify", help="Verify a Pocket catalog")
    pocket_verify_parser.add_argument("--catalog", type=Path, required=True)
    pocket_verify_parser.add_argument("--source", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "catalog":
            if args.catalog_system == "piper":
                if args.catalog_action == "build":
                    catalog = fetch_and_build_catalog(
                        repository=args.repository,
                        revision=args.revision,
                    )
                    verify_catalog(catalog)
                    _write_json(args.output, catalog)
                    if args.source_output is not None:
                        _write_json(args.source_output, {"schema": 1, **catalog["source"]})
                    print(f"Wrote {len(catalog['voices'])} voices to {args.output}")
                    return 0
                if args.catalog_action == "verify":
                    catalog = load_catalog(args.catalog)
                    if args.source is not None:
                        source = json.loads(args.source.read_text(encoding="utf-8"))
                        expected = {"schema": 1, **catalog["source"]}
                        if source != expected:
                            raise ValueError("Source metadata does not match catalog provenance")
                    print(f"Verified {len(catalog['voices'])} Piper voices")
                    return 0
            elif args.catalog_system == "pocket":
                if args.catalog_action == "build":
                    catalog = pocket_catalog_tools.build_catalog(
                        repository=args.repository,
                        revision=args.revision,
                    )
                    pocket_catalog_tools.verify_catalog(catalog)
                    _write_json(args.output, catalog)
                    if args.source_output is not None:
                        _write_json(args.source_output, {"schema": 1, **catalog["source"]})
                    print(f"Wrote {len(catalog['bundles'])} bundles to {args.output}")
                    return 0
                if args.catalog_action == "verify":
                    catalog = pocket_catalog_tools.load_catalog(args.catalog)
                    if args.source is not None:
                        source = json.loads(args.source.read_text(encoding="utf-8"))
                        expected = {"schema": 1, **catalog["source"]}
                        if source != expected:
                            raise ValueError("Source metadata does not match catalog provenance")
                    print(f"Verified {len(catalog['bundles'])} Pocket bundles")
                    return 0
            else:
                raise ValueError(f"Unsupported catalog system: {args.catalog_system}")
        manager = _manager(args)
        if args.command == "list":
            items = manager.list(
                args.system,
                installed=args.installed,
                language=args.language,
                quality=args.quality,
                refresh=args.refresh,
            )
            for item in items:
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
