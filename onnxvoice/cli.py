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


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add common filter arguments to a parser."""
    parser.add_argument("--system", choices=registered_systems(), help="Filter by system")
    parser.add_argument("--kind", choices=["voice", "model", "bundle"], help="Filter by kind")
    parser.add_argument(
        "--lang", "--language", dest="language", help="Filter by language (e.g. en-US, en)"
    )
    parser.add_argument(
        "--gender", choices=["male", "female", "neutral", "unknown"], help="Filter by gender"
    )
    parser.add_argument("--quality", help="Filter by quality")
    parser.add_argument("--distribution", help="Filter by distribution")
    parser.add_argument(
        "--status", choices=["installed", "available", "local"], help="Filter by status"
    )
    parser.add_argument(
        "--format", choices=["table", "plain", "json", "tsv"], default="table", help="Output format"
    )


def _add_refresh_args(parser: argparse.ArgumentParser) -> None:
    """Add refresh/offline arguments."""
    parser.add_argument("--refresh", action="store_true", help="Refresh catalogs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onnxvoice")
    parser.add_argument("--cache-dir", type=Path, help="Override the shared onnxvoice cache")
    parser.add_argument("--offline", action="store_true", help="Disable network access")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- list ---
    list_parser = sub.add_parser(
        "list", help="List catalog and installed entries (merged inventory)"
    )
    _add_filter_args(list_parser)
    _add_refresh_args(list_parser)
    list_parser.add_argument(
        "--installed", action="store_true", help="Show only installed entries (legacy alias)"
    )

    # --- voices ---
    voices_parser = sub.add_parser("voices", help="List and resolve stable short voice selectors")
    voices_sub = voices_parser.add_subparsers(dest="voices_command", required=True)

    voices_list_parser = voices_sub.add_parser(
        "list", help="List catalog voices with stable selectors"
    )
    voices_list_parser.add_argument("--system", choices=["kokoro", "piper"])
    voices_list_parser.add_argument("--lang", "--language", dest="language")
    voices_list_parser.add_argument("--refresh", action="store_true")
    voices_list_parser.add_argument("--include-retired", action="store_true")
    voices_list_parser.add_argument(
        "--no-unassigned",
        dest="include_unassigned",
        action="store_false",
        help="Hide current catalog voices without a registry assignment",
    )
    voices_list_parser.set_defaults(include_unassigned=True)
    voices_list_parser.add_argument(
        "--format", choices=["table", "plain", "json", "tsv"], default="table"
    )

    voices_show_parser = voices_sub.add_parser(
        "show", help="Show the full identity behind a short selector"
    )
    voices_show_parser.add_argument("selector")
    voices_show_parser.add_argument("--refresh", action="store_true")
    voices_show_parser.add_argument("--include-retired", action="store_true")
    voices_show_parser.add_argument(
        "--format", choices=["table", "plain", "json", "tsv"], default="table"
    )
    # --- installed ---
    installed_parser = sub.add_parser(
        "installed", help="List installed entries (local-only, no network)"
    )
    _add_filter_args(installed_parser)

    # --- install ---
    install_parser = sub.add_parser("install", help="Install a model/voice from a catalog")
    install_parser.add_argument("ref", help="e.g. piper:en_US-lessac-medium or kokoro:v1.0")
    install_parser.add_argument("--quality")
    install_parser.add_argument("--distribution")
    install_parser.add_argument("--refresh", action="store_true")
    install_parser.add_argument("--force", action="store_true")

    # --- updates ---
    updates_parser = sub.add_parser("updates", help="Check for available updates")
    _add_filter_args(updates_parser)
    updates_parser.add_argument("--cached", action="store_true", help="Use cached catalogs only")

    # --- update ---
    update_parser = sub.add_parser("update", help="Update an installed asset")
    update_parser.add_argument("ref", nargs="?", help="Reference to update")
    update_parser.add_argument("--quality")
    update_parser.add_argument("--distribution")
    update_parser.add_argument(
        "--all", action="store_true", dest="update_all", help="Update all outdated assets"
    )
    update_parser.add_argument("--force", action="store_true")

    # --- info ---
    info_parser = sub.add_parser("info", help="Show detailed info about an installed asset")
    info_parser.add_argument("ref")
    info_parser.add_argument("--quality")
    info_parser.add_argument("--distribution")
    info_parser.add_argument(
        "--check-updates", action="store_true", help="Check for updates (requires network)"
    )

    # --- diagnostics ---
    doctor_parser = sub.add_parser("doctor", help="Show non-secret download diagnostics")
    doctor_parser.add_argument("--system", choices=["pocket"], default="pocket")
    doctor_parser.add_argument("--format", choices=["plain", "json"], default="plain")

    # --- path ---
    path_parser = sub.add_parser("path", help="Print an installed model/voice path")
    path_parser.add_argument("ref")
    path_parser.add_argument("--quality")
    path_parser.add_argument("--distribution")

    # --- show ---
    show_parser = sub.add_parser("show", help="Show an installed manifest")
    show_parser.add_argument("ref")
    show_parser.add_argument("--quality")
    show_parser.add_argument("--distribution")

    # --- remove ---
    remove_parser = sub.add_parser("remove", help="Remove an installation")
    remove_parser.add_argument("ref", nargs="+", help="One or more references to remove")
    remove_parser.add_argument("--quality")
    remove_parser.add_argument("--distribution")
    remove_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be removed without removing"
    )
    remove_parser.add_argument(
        "--gc", action="store_true", help="Run garbage collection after removal"
    )
    remove_parser.add_argument(
        "--all-variants", action="store_true", help="Remove all installed variants of the ref"
    )
    remove_parser.add_argument(
        "--yes", action="store_true", help="Skip confirmation for multi-variant removal"
    )

    # --- verify ---
    verify_parser = sub.add_parser("verify", help="Verify checksums for an installation")
    verify_parser.add_argument("ref")
    verify_parser.add_argument("--quality")
    verify_parser.add_argument("--distribution")

    # --- cache ---
    cache_parser = sub.add_parser("cache", help="Show cache information")
    cache_sub = cache_parser.add_subparsers(dest="cache_command", required=True)

    cache_info_parser = cache_sub.add_parser("info", help="Show cache usage")
    cache_info_parser.add_argument("--format", choices=["table", "json"], default="table")

    cache_gc_parser = cache_sub.add_parser("gc", help="Remove orphaned blobs")
    cache_gc_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be removed"
    )

    # --- import ---
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

    # --- catalog ---
    catalog_parser = sub.add_parser("catalog", help="Build and verify external catalogs")
    catalog_sub = catalog_parser.add_subparsers(dest="catalog_system", required=True)

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


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _voice_payload(record) -> dict[str, Any]:
    identity = record.identity
    system = record.system
    asset_id = record.asset_id
    voice_id = record.voice_id
    backing_ref = (
        identity.backing_ref
        if identity is not None
        else (f"{system}:{asset_id}" if system and asset_id else None)
    )
    return {
        "selector": record.selector,
        "language_key": identity.language if identity is not None else None,
        "engine_code": identity.engine_code if identity is not None else None,
        "slot": identity.slot if identity is not None else None,
        "system": system,
        "asset_id": asset_id,
        "voice_id": voice_id,
        "backing_ref": backing_ref,
        "state": record.state,
        "available": record.available,
        "languages": list(record.languages),
        "gender": record.gender,
    }


def _render_voice_records(records, output_format: str) -> None:
    columns = ["SELECTOR", "SYSTEM", "ASSET", "VOICE", "LANG", "STATUS"]
    rows = []
    for record in records:
        status = "available" if record.available else record.state
        rows.append(
            [
                record.selector or "-",
                record.system or "-",
                record.asset_id or "-",
                record.voice_id or "-",
                ",".join(record.languages) or "-",
                status,
            ]
        )
    if output_format == "json":
        print(
            json.dumps(
                {"items": [_voice_payload(record) for record in records]}, indent=2, sort_keys=True
            )
        )
        return
    from ._table import render_inventory

    render_inventory(columns, rows, format=output_format)


def _cmd_voices_list(args: argparse.Namespace) -> int:
    manager = _manager(args)
    records = manager.list_voices(
        system=args.system,
        language=args.language,
        refresh=args.refresh,
        include_retired=args.include_retired,
        include_unassigned=args.include_unassigned,
    )
    _render_voice_records(records, args.format)
    return 0


def _cmd_voices_show(args: argparse.Namespace) -> int:
    manager = _manager(args)
    identity = manager.resolve_voice_selector(args.selector, include_retired=args.include_retired)
    record = None
    try:
        record = next(
            (
                item
                for item in manager.list_voices(
                    system=identity.system,
                    language=identity.language,
                    refresh=args.refresh,
                    include_retired=args.include_retired,
                    include_unassigned=True,
                )
                if item.identity is not None
                and item.identity.canonical_key == identity.canonical_key
            ),
            None,
        )
    except Exception:
        # Showing a known identity remains useful when the catalog is offline.
        record = None
    if record is None:
        from .types import VoiceRecord

        record = VoiceRecord(
            identity=identity,
            available=False,
            catalog_item=None,
            languages=(identity.language,),
            gender="unknown",
        )
    if args.format == "json":
        print(json.dumps(_voice_payload(record), indent=2, sort_keys=True))
    elif args.format in {"plain", "tsv"}:
        _render_voice_records([record], args.format)
    else:
        payload = _voice_payload(record)
        for key in (
            "selector",
            "system",
            "asset_id",
            "voice_id",
            "backing_ref",
            "language_key",
            "engine_code",
            "slot",
            "state",
            "available",
        ):
            print(f"{key.replace('_', ' ').title():16} {payload[key]}")
    return 0


def _cmd_voices(args: argparse.Namespace) -> int:
    if args.voices_command == "list":
        return _cmd_voices_list(args)
    if args.voices_command == "show":
        return _cmd_voices_show(args)
    raise ValueError(f"unknown voices command: {args.voices_command}")


def _cmd_list(args: argparse.Namespace) -> int:
    from ._table import format_bytes, render_inventory

    manager = _manager(args)

    # Legacy --installed flag
    if args.installed:
        return _cmd_installed(args)

    records = manager.inventory(
        system=args.system,
        language=args.language,
        gender=args.gender,
        kind=args.kind,
        quality=args.quality,
        distribution=args.distribution,
        status=getattr(args, "status", None),
        refresh=args.refresh,
    )

    columns = [
        "STATUS",
        "SYSTEM",
        "KIND",
        "ID",
        "LANG",
        "GENDER",
        "QUALITY",
        "DIST",
        "SIZE",
        "VERSION",
        "UPDATE",
    ]
    rows = []
    for r in records:
        rows.append(
            [
                r.status,
                r.system,
                r.kind,
                r.id,
                r.language_codes[0] if r.language_codes else "-",
                r.gender,
                r.quality or "-",
                r.distribution or "-",
                format_bytes(r.size_bytes),
                r.version or "-",
                r.update_status if r.update_status != "not_checked" else "-",
            ]
        )

    render_inventory(columns, rows, format=args.format)
    return 0


def _cmd_installed(args: argparse.Namespace) -> int:
    from ._table import format_bytes, render_inventory

    manager = _manager(args)
    records = manager.inventory(
        system=args.system,
        language=args.language,
        gender=args.gender,
        kind=args.kind,
        quality=args.quality,
        distribution=args.distribution,
        installed_only=True,
    )

    columns = [
        "SYSTEM",
        "KIND",
        "ID",
        "LANG",
        "GENDER",
        "QUALITY",
        "DIST",
        "SIZE",
        "VERSION",
        "PATH",
    ]
    rows = []
    for r in records:
        path = str(r.installation.path) if r.installation else "-"
        rows.append(
            [
                r.system,
                r.kind,
                r.id,
                r.language_codes[0] if r.language_codes else "-",
                r.gender,
                r.quality or "-",
                r.distribution or "-",
                format_bytes(r.size_bytes),
                r.version or "-",
                path,
            ]
        )

    render_inventory(columns, rows, format=args.format)
    return 0


def _cmd_updates(args: argparse.Namespace) -> int:
    from ._table import format_bytes, render_inventory

    manager = _manager(args)

    # --cached means use offline mode for catalog
    refresh = not getattr(args, "cached", False)

    records = manager.check_updates(
        system=args.system,
        language=args.language,
        kind=args.kind,
        refresh=refresh and not args.offline,
    )

    # Filter to only show installed items (updates are about installed items)
    records = [r for r in records if r.installed]

    columns = [
        "STATUS",
        "SYSTEM",
        "KIND",
        "ID",
        "LANG",
        "QUALITY",
        "DIST",
        "SIZE",
        "VERSION",
        "UPDATE",
    ]
    rows = []
    for r in records:
        rows.append(
            [
                r.status,
                r.system,
                r.kind,
                r.id,
                r.language_codes[0] if r.language_codes else "-",
                r.quality or "-",
                r.distribution or "-",
                format_bytes(r.size_bytes),
                r.version or "-",
                r.update_status,
            ]
        )

    render_inventory(columns, rows, format=args.format)
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    manager = _manager(args)

    if args.update_all:
        # Update all items with update_available
        records = manager.check_updates(refresh=not args.offline)
        to_update = [r for r in records if r.update_status == "update_available" and r.installed]

        if not to_update:
            print("Everything is up to date.")
            return 0

        updated = 0
        for r in to_update:
            try:
                manager.update(
                    r.ref,
                    quality=r.quality,
                    distribution=r.distribution,
                    force=args.force,
                    refresh=False,
                )
                updated += 1
                print(f"Updated {r.ref}")
            except Exception as exc:
                print(f"Failed to update {r.ref}: {exc}", file=sys.stderr)

        print(f"\nUpdated {updated}/{len(to_update)} items.")
        return 0

    if not args.ref:
        print("onnxvoice: either REF or --all is required", file=sys.stderr)
        return 2

    installed = manager.update(
        args.ref,
        quality=args.quality,
        distribution=args.distribution,
        force=args.force,
    )
    print(f"Updated {installed.ref} -> {installed.path}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .huggingface import huggingface_diagnostics

    report = {
        "system": args.system,
        **huggingface_diagnostics(offline=args.offline),
    }
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("Pocket download diagnostics")
    print(f"Hugging Face integration : {report['huggingface_hub']}")
    print(f"Credentials             : {report['credentials']}")
    print(f"Offline mode            : {'yes' if report['offline'] else 'no'}")
    print(f"Implicit token disabled : {'yes' if report['implicit_token_disabled'] else 'no'}")
    print(f"Gated repository access : {report['gated_access']}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from ._table import format_bytes

    manager = _manager(args)

    check_updates = getattr(args, "check_updates", False)
    quality = getattr(args, "quality", None)
    distribution = getattr(args, "distribution", None)

    try:
        installation = manager.resolve(args.ref, quality=quality, distribution=distribution)
    except Exception as exc:
        print(f"onnxvoice: {exc}", file=sys.stderr)
        return 2

    from .inventory import (
        effective_distribution,
        effective_quality,
        gender_from_metadata,
        language_codes_from_metadata,
        logical_size_bytes,
        version_label,
    )

    meta = installation.metadata
    langs = language_codes_from_metadata(meta)
    gender = gender_from_metadata(meta)
    qual = effective_quality(meta)
    dist = effective_distribution(meta)
    version = version_label(installation)
    size = logical_size_bytes(installation)

    update_status = "not checked"
    if check_updates:
        from .inventory import compare_installation_to_catalog

        try:
            item = manager.catalog.resolve(args.ref, quality=quality, distribution=distribution)
            comparison = compare_installation_to_catalog(installation, item)
            update_status = comparison.status
        except Exception:
            update_status = "check failed"

    lines = [
        f"Reference       {installation.ref}",
        f"Status          {'installed' if not meta.get('external') else 'local'}",
        f"Kind            {installation.kind}",
        f"Language        {', '.join(langs) if langs else '-'}",
        f"Gender          {gender}",
        f"Quality         {qual or '-'}",
        f"Distribution    {dist or '-'}",
        f"Installed size  {format_bytes(size)}",
        f"Version         {version or '-'}",
        f"Update          {update_status}",
        f"Path            {installation.path}",
        f"Artifacts       {len(installation.artifacts)}",
    ]
    print("\n".join(lines))
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    from ._table import format_bytes

    manager = _manager(args)
    refs = args.ref

    if args.dry_run:
        # Show what would be removed
        for ref in refs:
            try:
                installations = manager.find_installed(ref)
            except Exception as exc:
                print(f"onnxvoice: {exc}", file=sys.stderr)
                return 2

            if not installations:
                print(f"Not installed: {ref}", file=sys.stderr)
                return 2

            if args.all_variants:
                for inst in installations:
                    size = sum(a.size for a in inst.artifacts)
                    print(
                        f"Would remove {inst.ref} [{inst.metadata.get('selected_quality', '-')}] [{inst.metadata.get('selected_distribution', '-')}] ({format_bytes(size)})"
                    )
            else:
                quality = getattr(args, "quality", None)
                distribution = getattr(args, "distribution", None)
                if len(installations) > 1 and quality is None and distribution is None:
                    print(f"onnxvoice: multiple installations match {ref}:", file=sys.stderr)
                    for inst in installations:
                        q = inst.metadata.get("selected_quality", "-")
                        d = inst.metadata.get("selected_distribution", "-")
                        size = sum(a.size for a in inst.artifacts)
                        print(
                            f"  quality={q} distribution={d} ({format_bytes(size)})",
                            file=sys.stderr,
                        )
                    print(
                        "Specify --quality/--distribution or use --all-variants.", file=sys.stderr
                    )
                    return 2
                inst = installations[0]
                size = sum(a.size for a in inst.artifacts)
                print(f"Would remove {inst.ref} ({format_bytes(size)})")
        return 0

    # Actual removal
    total_logical_bytes = 0
    removed_count = 0

    for ref in refs:
        try:
            installations = manager.find_installed(ref)
        except Exception as exc:
            print(f"onnxvoice: {exc}", file=sys.stderr)
            return 2

        if not installations:
            print(f"Not installed: {ref}", file=sys.stderr)
            return 2

        if args.all_variants:
            if not args.yes and len(installations) > 1:
                print(
                    f"onnxvoice: {ref} has {len(installations)} variants. Use --yes to confirm.",
                    file=sys.stderr,
                )
                return 2
            for inst in installations:
                size = sum(a.size for a in inst.artifacts)
                manager.remove(
                    ref,
                    quality=inst.metadata.get("selected_quality"),
                    distribution=inst.metadata.get("selected_distribution"),
                )
                total_logical_bytes += size
                removed_count += 1
                print(f"Removed {inst.ref}")
        else:
            quality = getattr(args, "quality", None)
            distribution = getattr(args, "distribution", None)
            if len(installations) > 1 and quality is None and distribution is None:
                print(f"onnxvoice: multiple installations match {ref}:", file=sys.stderr)
                for inst in installations:
                    q = inst.metadata.get("selected_quality", "-")
                    d = inst.metadata.get("selected_distribution", "-")
                    size = sum(a.size for a in inst.artifacts)
                    print(f"  quality={q} distribution={d} ({format_bytes(size)})", file=sys.stderr)
                print("Specify --quality/--distribution or use --all-variants.", file=sys.stderr)
                return 2

            inst = installations[0]
            size = sum(a.size for a in inst.artifacts)
            manager.remove(ref, quality=quality, distribution=distribution)
            total_logical_bytes += size
            removed_count += 1
            print(f"Removed {ref} ({format_bytes(size)})")

    gc_bytes = 0
    if args.gc:
        gc_report = manager.store.gc()
        gc_bytes = gc_report.removed_bytes
        if gc_report.removed_blobs > 0:
            print(f"GC: removed {gc_report.removed_blobs} orphan blobs ({format_bytes(gc_bytes)})")

    summary_parts = [
        f"Removed {removed_count} installation(s) ({format_bytes(total_logical_bytes)})"
    ]
    if args.gc:
        summary_parts.append(f"reclaimed {format_bytes(gc_bytes)} from GC")
    print(", ".join(summary_parts))

    return 0


def _cmd_cache_info(args: argparse.Namespace) -> int:
    from ._table import format_bytes

    manager = _manager(args)
    usage = manager.store.usage()

    if getattr(args, "format", "table") == "json":
        import json as json_mod

        data = {
            "root": str(usage.root),
            "installation_count": usage.installation_count,
            "install_logical_bytes": usage.install_logical_bytes,
            "blob_apparent_bytes": usage.blob_apparent_bytes,
            "catalog_bytes": usage.catalog_bytes,
            "unique_file_bytes": usage.unique_file_bytes,
            "orphan_blob_count": usage.orphan_blob_count,
            "orphan_blob_bytes": usage.orphan_blob_bytes,
        }
        print(json_mod.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"Cache                 {usage.root}")
        print(f"Installations         {usage.installation_count}")
        print(f"Installed payload     {format_bytes(usage.install_logical_bytes)}")
        print(f"Blob store            {format_bytes(usage.blob_apparent_bytes)}")
        print(f"Catalogs              {format_bytes(usage.catalog_bytes)}")
        print(f"Unique cache files    {format_bytes(usage.unique_file_bytes)}")
        print(f"Orphan blobs          {usage.orphan_blob_count}")
        print(f"Reclaimable blobs     {format_bytes(usage.orphan_blob_bytes)}")
    return 0


def _cmd_cache_gc(args: argparse.Namespace) -> int:
    from ._table import format_bytes

    manager = _manager(args)

    if args.dry_run:
        usage = manager.store.usage()
        if usage.orphan_blob_count == 0:
            print("No orphan blobs to remove.")
        else:
            print(
                f"Would remove {usage.orphan_blob_count} blobs ({format_bytes(usage.orphan_blob_bytes)})"
            )
        return 0

    report = manager.store.gc()
    if report.removed_blobs == 0:
        print("No orphan blobs to remove.")
    else:
        print(f"Removed {report.removed_blobs} blobs ({format_bytes(report.removed_bytes)})")
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        # Catalog commands don't need a manager
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

        # Command dispatch
        if args.command == "doctor":
            return _cmd_doctor(args)

        if args.command == "voices":
            return _cmd_voices(args)

        handlers = {
            "list": _cmd_list,
            "installed": _cmd_installed,
            "updates": _cmd_updates,
            "update": _cmd_update,
            "info": _cmd_info,
            "remove": _cmd_remove,
        }

        if args.command in handlers:
            return handlers[args.command](args)

        # Legacy commands handled directly
        manager = _manager(args)

        if args.command == "install":
            installed = manager.install(
                args.ref,
                quality=args.quality,
                distribution=getattr(args, "distribution", None),
                refresh=args.refresh,
                force=args.force,
            )
            print(installed.path)
            return 0

        if args.command == "path":
            print(
                manager.where(
                    args.ref,
                    quality=getattr(args, "quality", None),
                    distribution=getattr(args, "distribution", None),
                )
            )
            return 0

        if args.command == "show":
            installation = manager.show(
                args.ref,
                quality=getattr(args, "quality", None),
                distribution=getattr(args, "distribution", None),
            )
            print((installation.path / "manifest.json").read_text(encoding="utf-8"), end="")
            return 0

        if args.command == "verify":
            manager.verify(
                args.ref,
                quality=getattr(args, "quality", None),
                distribution=getattr(args, "distribution", None),
            )
            installation = manager.show(
                args.ref,
                quality=getattr(args, "quality", None),
                distribution=getattr(args, "distribution", None),
            )
            report = verify_installation(manager.store, installation)
            print(json.dumps({"ok": report.ok, "checks": report.checks}))
            return 0

        if args.command == "cache":
            if args.cache_command == "info":
                return _cmd_cache_info(args)
            if args.cache_command == "gc":
                return _cmd_cache_gc(args)

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
