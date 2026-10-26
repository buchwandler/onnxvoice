from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ._install_progress import InstallProgress
from .catalog_tools import chatterbox as chatterbox_catalog_tools
from .catalog_tools import kitten as kitten_catalog_tools
from .catalog_tools import kitten2 as kitten2_catalog_tools
from .catalog_tools import pocket as pocket_catalog_tools
from .catalog_tools import qwen3 as qwen3_catalog_tools
from .catalog_tools import supertonic as supertonic_catalog_tools
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
    voices_parser = sub.add_parser("voices", help="List and inspect catalog voices")
    voices_sub = voices_parser.add_subparsers(dest="voices_command", required=True)

    voices_list_parser = voices_sub.add_parser(
        "list", help="List voices exposed by normalized catalogs"
    )
    voices_list_parser.add_argument(
        "--engine", choices=registered_systems(), help="Filter by engine"
    )
    voices_list_parser.add_argument(
        "--target",
        dest="target",
        metavar="TARGET_ID",
        help="Filter by exact catalog target ID (e.g. v1.0)",
    )
    voices_list_parser.add_argument("--lang", "--language", dest="language")
    voices_list_parser.add_argument("--refresh", action="store_true")
    voices_list_parser.add_argument(
        "--format", choices=["table", "plain", "json", "tsv"], default="table"
    )

    voices_show_parser = voices_sub.add_parser("show", help="Show one catalog voice")
    voices_show_parser.add_argument("ref")
    voices_show_parser.add_argument("--refresh", action="store_true")
    voices_show_parser.add_argument(
        "--format", choices=["table", "plain", "json", "tsv"], default="table"
    )
    # --- engine option and capability discovery ---
    engines_parser = sub.add_parser(
        "engines", help="Discover engine targets, options, and capabilities"
    )
    engines_sub = engines_parser.add_subparsers(dest="engines_command", required=True)

    engines_list_parser = engines_sub.add_parser(
        "list", help="List target-level engine capability variants"
    )
    engines_list_parser.add_argument("--engine", help="Filter by engine name")
    engines_list_parser.add_argument(
        "--capability", choices=["voice-cloning"], help="Filter by capability"
    )
    engines_list_parser.add_argument("--refresh", action="store_true", help="Refresh catalogs")
    engines_list_parser.add_argument(
        "--format", choices=["table", "plain", "json", "tsv"], default="table"
    )

    engines_options_parser = engines_sub.add_parser(
        "options", help="Show typed inputs, controls, capabilities, and limits"
    )
    engines_options_parser.add_argument("--engine", required=True, help="Engine name")
    engines_options_parser.add_argument("--target", help="Catalog target ID or alias")
    engines_options_parser.add_argument("--quality", help="Select a quality/profile")
    engines_options_parser.add_argument("--distribution", help="Select a distribution variant")
    engines_options_parser.add_argument("--refresh", action="store_true", help="Refresh catalogs")
    engines_options_parser.add_argument(
        "--format", choices=["table", "plain", "json", "tsv"], default="table"
    )

    # --- installed ---
    installed_parser = sub.add_parser(
        "installed", help="List installed entries (local-only, no network)"
    )
    _add_filter_args(installed_parser)
    installed_parser.add_argument(
        "--sort", choices=["size", "id", "system"], help="Sort installed entries"
    )
    installed_parser.add_argument("--desc", action="store_true", help="Sort in descending order")

    # --- install ---
    install_parser = sub.add_parser("install", help="Install a model/voice from a catalog")
    install_parser.add_argument("ref", help="e.g. piper:en_US-lessac-medium or kokoro:v1.0")
    install_parser.add_argument("--quality")
    install_parser.add_argument("--distribution")
    install_parser.add_argument("--refresh", action="store_true")
    install_parser.add_argument("--force", action="store_true")

    install_parser.add_argument(
        "--no-progress", action="store_true", help="Disable install progress output"
    )
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
    # --- ONNX Runtime management ---
    runtime_parser = sub.add_parser("runtime", help="Inspect and select ONNX Runtime providers")
    runtime_sub = runtime_parser.add_subparsers(dest="runtime_command")
    runtime_parser.set_defaults(runtime_command="status")

    runtime_status_parser = runtime_sub.add_parser(
        "status", help="Show imported runtime and provider policy"
    )
    runtime_status_parser.add_argument("--format", choices=["plain", "json"], default="plain")

    runtime_list_parser = runtime_sub.add_parser(
        "list", help="List installed distributions and reported providers"
    )
    runtime_list_parser.add_argument("--format", choices=["table", "json"], default="table")

    runtime_use_parser = runtime_sub.add_parser("use", help="Persist the default provider policy")
    runtime_use_parser.add_argument(
        "providers", help="Provider alias or ordered comma-separated provider list"
    )

    runtime_sub.add_parser("reset", help="Remove the saved provider preference")

    # --- diagnostics ---
    doctor_parser = sub.add_parser(
        "doctor", help="Show ONNXVoice environment and runtime diagnostics"
    )
    doctor_parser.add_argument("--system", choices=["pocket"], default=None, help=argparse.SUPPRESS)
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
    remove_parser = sub.add_parser(
        "remove", aliases=["uninstall"], help="Remove/uninstall installed assets"
    )
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

    cache_info_parser.add_argument(
        "--details", action="store_true", help="Show per-category and per-path cache metrics"
    )
    cache_gc_parser = cache_sub.add_parser(
        "gc", help="Collect unreachable blobs and Pocket voice-state records"
    )
    cache_gc_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be removed"
    )
    cache_gc_parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also collect the recognized retired Pocket prompt cache",
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

    supertonic_parser = catalog_sub.add_parser("supertonic", help="Manage the Supertonic-3 catalog")
    supertonic_sub = supertonic_parser.add_subparsers(dest="catalog_action", required=True)

    supertonic_build_parser = supertonic_sub.add_parser(
        "build", help="Build a pinned Supertonic-3 catalog"
    )
    supertonic_build_parser.add_argument("--output", type=Path, required=True)
    supertonic_build_parser.add_argument("--source-output", type=Path)
    supertonic_build_parser.add_argument(
        "--repository", default=supertonic_catalog_tools.DEFAULT_REPOSITORY
    )
    supertonic_build_parser.add_argument(
        "--revision", default=supertonic_catalog_tools.DEFAULT_REVISION
    )

    supertonic_verify_parser = supertonic_sub.add_parser(
        "verify", help="Verify a Supertonic-3 catalog offline"
    )
    supertonic_verify_parser.add_argument("--catalog", type=Path, required=True)
    supertonic_verify_parser.add_argument("--source", type=Path)
    kitten_parser = catalog_sub.add_parser("kitten", help="Manage the KittenTTS catalog")
    kitten_sub = kitten_parser.add_subparsers(dest="catalog_action", required=True)

    kitten_build_parser = kitten_sub.add_parser("build", help="Build a pinned KittenTTS catalog")
    kitten_build_parser.add_argument("--seed-catalog", type=Path, required=True)
    kitten_build_parser.add_argument("--output", type=Path, required=True)
    kitten_build_parser.add_argument("--source-output", type=Path)
    kitten_build_parser.add_argument("--revision", default=kitten_catalog_tools.DEFAULT_REVISION)

    kitten_verify_parser = kitten_sub.add_parser(
        "verify", help="Verify a KittenTTS catalog offline"
    )
    kitten_verify_parser.add_argument("--catalog", type=Path, required=True)
    kitten_verify_parser.add_argument("--source", type=Path)
    qwen3_parser = catalog_sub.add_parser("qwen3", help="Manage the Qwen3 WaveKat catalog")
    qwen3_sub = qwen3_parser.add_subparsers(dest="catalog_action", required=True)
    qwen3_build_parser = qwen3_sub.add_parser("build", help="Build a pinned Qwen3 catalog")
    qwen3_build_parser.add_argument("--output", type=Path, required=True)
    qwen3_build_parser.add_argument("--source-output", type=Path)
    qwen3_verify_parser = qwen3_sub.add_parser("verify", help="Verify a Qwen3 catalog offline")
    qwen3_verify_parser.add_argument("--catalog", type=Path, required=True)
    qwen3_verify_parser.add_argument("--source", type=Path)

    kitten2_parser = catalog_sub.add_parser(
        "kitten2", help="Manage the native KittenTTS 2 bundle catalog"
    )
    kitten2_sub = kitten2_parser.add_subparsers(dest="catalog_action", required=True)
    kitten2_build_parser = kitten2_sub.add_parser(
        "build", help="Build a pinned KittenTTS 2 catalog"
    )
    kitten2_build_parser.add_argument(
        "--repository", default=kitten2_catalog_tools.DEFAULT_REPOSITORY
    )
    kitten2_build_parser.add_argument("--revision", default=kitten2_catalog_tools.DEFAULT_REVISION)
    kitten2_build_parser.add_argument("--output", type=Path, required=True)
    kitten2_build_parser.add_argument("--source-output", type=Path, required=True)
    kitten2_verify_parser = kitten2_sub.add_parser(
        "verify", help="Verify a pinned KittenTTS 2 catalog offline"
    )
    kitten2_verify_parser.add_argument("--catalog", type=Path, required=True)
    kitten2_verify_parser.add_argument("--source", type=Path, required=True)
    chatterbox_parser = catalog_sub.add_parser(
        "chatterbox", help="Manage cleanroom Chatterbox ONNX bundles"
    )
    chatterbox_sub = chatterbox_parser.add_subparsers(dest="catalog_action", required=True)
    chatterbox_build = chatterbox_sub.add_parser("build", help="Build a pinned Chatterbox catalog")
    chatterbox_build.add_argument("--seed", type=Path, required=True)
    chatterbox_build.add_argument("--output", type=Path, required=True)
    chatterbox_build.add_argument("--source-output", type=Path, required=True)
    chatterbox_verify = chatterbox_sub.add_parser(
        "verify", help="Verify a Chatterbox catalog offline"
    )
    chatterbox_verify.add_argument("--catalog", type=Path, required=True)
    chatterbox_verify.add_argument("--source", type=Path, required=True)
    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _voice_payload(record) -> dict[str, Any]:
    payload = {
        "ref": record.ref,
        "system": record.system,
        "asset_id": record.asset_id,
        "voice_id": record.voice_id,
        "backing_ref": record.backing_ref,
        "languages": list(record.languages),
        "gender": record.gender,
    }
    for key, value in {
        "language": record.language,
        "locale": record.locale,
        "language_label": record.language_label,
    }.items():
        if value:
            payload[key] = value
    return payload


def _render_voice_records(records, output_format: str) -> None:
    columns = ["REF", "SYSTEM", "ASSET", "VOICE", "LOCALE", "GENDER"]
    rows = [
        [
            record.ref,
            record.system,
            record.asset_id,
            record.voice_id,
            record.locale or "-",
            record.gender,
        ]
        for record in records
    ]
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
        system=args.engine,
        asset_id=args.target,
        language=args.language,
        refresh=args.refresh,
    )
    _render_voice_records(records, args.format)
    return 0


def _cmd_voices_show(args: argparse.Namespace) -> int:
    manager = _manager(args)
    record = manager.resolve_voice(args.ref, refresh=args.refresh)
    if args.format == "json":
        print(json.dumps(_voice_payload(record), indent=2, sort_keys=True))
    else:
        _render_voice_records([record], args.format)
    return 0


def _cmd_voices(args: argparse.Namespace) -> int:
    if args.voices_command == "list":
        return _cmd_voices_list(args)
    if args.voices_command == "show":
        return _cmd_voices_show(args)
    raise ValueError(f"unknown voices command: {args.voices_command}")


def _engine_display_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _engine_detail(parts: list[Any]) -> str:
    return "; ".join(_engine_display_value(part) for part in parts if part not in (None, ""))


def _append_engine_summary(rows: list[list[str]], data: dict[str, Any], prefix: str = "") -> None:
    label = f"{prefix} / " if prefix else ""
    for name, value in (
        ("engine", data.get("engine")),
        ("target", data.get("target")),
        ("reference", data.get("ref")),
        ("quality", data.get("quality")),
        ("distribution", data.get("distribution")),
        ("layout", data.get("layout")),
        ("sample_rate", data.get("sample_rate")),
        ("contract_status", data.get("contract_status")),
    ):
        status = data.get("field_status", {}).get(name, "")
        rows.append(
            [
                f"{label}metadata",
                name,
                "",
                _engine_display_value(value),
                _engine_detail([f"status={status}" if status else None]),
            ]
        )
    discovery = data.get("target_discovery")
    if isinstance(discovery, dict):
        rows.append(
            [
                f"{label}metadata",
                "target_discovery",
                "",
                discovery.get("status", "unknown"),
                _engine_display_value(discovery.get("reason")),
            ]
        )


def _append_engine_contract_rows(
    rows: list[list[str]], data: dict[str, Any], prefix: str = ""
) -> None:
    label = f"{prefix} / " if prefix else ""
    for name, capability in sorted(data.get("capabilities", {}).items()):
        supported = capability.get("supported")
        value = (
            "supported"
            if supported is True
            else "not supported"
            if supported is False
            else capability.get("status", "unknown")
        )
        detail = _engine_detail(
            [
                f"mode={capability.get('mode')}" if capability.get("mode") else None,
                f"requires={_engine_display_value(capability.get('requires', []))}"
                if capability.get("requires")
                else None,
                f"requires_reference_transcript={capability.get('requires_reference_transcript')}"
                if capability.get("requires_reference_transcript") is not None
                else None,
                f"source={capability.get('source')}" if capability.get("source") else None,
                capability.get("reason"),
                *capability.get("constraints", []),
            ]
        )
        rows.append([f"{label}capability", name, "boolean", value, detail])

    for group, key in (("input", "inputs"), ("parameter", "parameters")):
        for option in sorted(data.get(key, []), key=lambda item: item.get("name", "")):
            if option.get("required"):
                value = "required"
            else:
                value = _engine_display_value(option.get("default"))
                if option.get("default_status") not in {None, "known", "not_applicable"}:
                    value = f"{value} ({option.get('default_status')})"
            details: list[Any] = [option.get("description")]
            if option.get("minimum") is not None or option.get("maximum") is not None:
                low = (
                    "-∞"
                    if option.get("minimum") is None
                    else _engine_display_value(option.get("minimum"))
                )
                high = (
                    "+∞"
                    if option.get("maximum") is None
                    else _engine_display_value(option.get("maximum"))
                )
                left = "[" if option.get("minimum_inclusive") is not False else "("
                right = "]" if option.get("maximum_inclusive") is not False else ")"
                details.append(f"range={left}{low}, {high}{right}")
            if option.get("choices"):
                details.append(f"choices={_engine_display_value(option['choices'])}")
            if option.get("supported") is False:
                details.append("supported=false")
            elif option.get("supported") is None:
                details.append("supported=unknown")
            for key_name in ("status", "availability", "constraint_status", "source"):
                if option.get(key_name):
                    details.append(f"{key_name}={option[key_name]}")
            details.extend(option.get("notes", []))
            rows.append(
                [
                    f"{label}{group}",
                    option.get("name", ""),
                    option.get("type", ""),
                    value,
                    _engine_detail(details),
                ]
            )

    for limit in sorted(data.get("limits", []), key=lambda item: item.get("name", "")):
        value = _engine_display_value(limit.get("value"))
        if limit.get("value") is None:
            value = limit.get("status", "unknown")
        detail = _engine_detail(
            [
                f"unit={limit.get('unit')}" if limit.get("unit") else None,
                f"source={limit.get('source')}" if limit.get("source") else None,
                f"constraint_status={limit.get('constraint_status')}"
                if limit.get("constraint_status")
                else None,
                limit.get("description"),
                *limit.get("notes", []),
            ]
        )
        rows.append([f"{label}limit", limit.get("name", ""), limit.get("unit", ""), value, detail])

    for selection in sorted(
        data.get("selection_metadata", []), key=lambda item: item.get("name", "")
    ):
        detail = _engine_detail(
            [
                f"status={selection.get('status')}" if selection.get("status") else None,
                f"source={selection.get('source')}" if selection.get("source") else None,
                selection.get("description"),
            ]
        )
        rows.append(
            [
                f"{label}selection",
                selection.get("name", ""),
                "metadata",
                _engine_display_value(selection.get("value")),
                detail,
            ]
        )

    for target in sorted(data.get("known_targets", []), key=lambda item: item.get("target", "")):
        details = [
            f"ref={target.get('ref')}",
            f"quality={_engine_display_value(target.get('quality'))}",
            f"qualities={_engine_display_value(target.get('qualities', []))}",
            f"distributions={_engine_display_value(target.get('distributions', []))}",
            f"layout={_engine_display_value(target.get('layout'))}",
            f"sample_rate={_engine_display_value(target.get('sample_rate'))}",
        ]
        rows.append(
            [
                f"{label}target",
                target.get("target", ""),
                "catalog",
                "available",
                _engine_detail(details),
            ]
        )


def _render_engine_options(data: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
        return
    rows: list[list[str]] = []
    _append_engine_summary(rows, data)
    variants = data.get("variants")
    if isinstance(variants, list):
        for variant in variants:
            label = (
                variant.get("distribution")
                or variant.get("quality")
                or variant.get("layout")
                or variant.get("target")
                or "variant"
            )
            _append_engine_summary(rows, variant, prefix=str(label))
            _append_engine_contract_rows(rows, variant, prefix=str(label))
    else:
        _append_engine_contract_rows(rows, data)
    from ._table import render_inventory

    render_inventory(["SECTION", "NAME", "TYPE", "VALUE", "DETAIL"], rows, format=output_format)


def _cmd_engines(args: argparse.Namespace) -> int:
    from .engine_options import list_engine_targets, resolve_engine_options

    manager = _manager(args)
    if args.engines_command == "list":
        data = list_engine_targets(
            manager,
            engine=args.engine,
            capability=args.capability,
            refresh=args.refresh,
        )
        if args.format == "json":
            print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            columns = [
                "ENGINE",
                "TARGET",
                "REF",
                "QUALITY",
                "DISTRIBUTION",
                "LAYOUT",
                "SAMPLE_RATE",
                "LICENSE",
                "COMMERCIAL_USE",
                "VOICE_CLONING",
                "MODE",
                "DISCOVERY",
            ]
            rows = [
                [
                    _engine_display_value(row.get("engine")),
                    _engine_display_value(row.get("target")),
                    _engine_display_value(row.get("ref")),
                    _engine_display_value(row.get("quality")),
                    _engine_display_value(row.get("distribution")),
                    _engine_display_value(row.get("layout")),
                    _engine_display_value(row.get("sample_rate")),
                    _engine_display_value(row.get("license_spdx")),
                    _engine_display_value(row.get("commercial_use")),
                    _engine_display_value(row.get("voice_cloning")),
                    _engine_display_value(row.get("cloning_mode")),
                    _engine_display_value(row.get("target_discovery")),
                ]
                for row in data["rows"]
            ]
            from ._table import render_inventory

            render_inventory(columns, rows, format=args.format)
        return 0
    if args.engines_command == "options":
        data = resolve_engine_options(
            manager,
            args.engine,
            target=args.target,
            quality=args.quality,
            distribution=args.distribution,
            refresh=args.refresh,
        )
        _render_engine_options(data, args.format)
        return 0
    raise ValueError(f"unknown engines command: {args.engines_command}")


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

    sort_by = getattr(args, "sort", None)
    if sort_by:

        def sort_key(record):
            common = (
                record.system.casefold(),
                record.id.casefold(),
                record.quality or "",
                record.distribution or "",
            )
            if sort_by == "size":
                return (record.size_bytes, *common)
            if sort_by == "id":
                return (record.id.casefold(), record.system.casefold(), record.quality or "")
            return (record.system.casefold(), record.id.casefold(), record.quality or "")

        records = sorted(records, key=sort_key, reverse=getattr(args, "desc", False))

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


_PROVIDER_INSTALL_HINTS = {
    "CPUExecutionProvider": 'python -m pip install "onnxvoice[cpu]"',
    "CUDAExecutionProvider": 'python -m pip install "onnxvoice[gpu]"',
    "TensorrtExecutionProvider": 'python -m pip install "onnxvoice[gpu]"',
    "DmlExecutionProvider": 'python -m pip install "onnxvoice[directml]"',
    "OpenVINOExecutionProvider": 'python -m pip install "onnxvoice[openvino]"',
}
_PROVIDER_LIST_ALIASES = (
    ("cpu", "CPUExecutionProvider"),
    ("cuda", "CUDAExecutionProvider"),
    ("tensorrt", "TensorrtExecutionProvider"),
    ("directml", "DmlExecutionProvider"),
    ("openvino", "OpenVINOExecutionProvider"),
    ("rocm", "ROCMExecutionProvider"),
    ("coreml", "CoreMLExecutionProvider"),
    ("nnapi", "NnapiExecutionProvider"),
    ("xnnpack", "XnnpackExecutionProvider"),
)


def _jsonable(value: Any) -> Any:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _provider_source_label(source: str) -> str:
    return {
        "default": "built-in default",
        "user_config": "user config",
    }.get(source, source)


def _distribution_label(status: Any) -> str:
    if status.imported_distribution:
        item = next(
            (dist for dist in status.distributions if dist.name == status.imported_distribution),
            None,
        )
        return (
            f"{item.name} {item.version}" if item and item.version else status.imported_distribution
        )
    if status.distribution_state == "ambiguous":
        return "ambiguous (multiple distributions claim the import)"
    if status.distribution_state == "vendor":
        return "vendor/custom build"
    if status.distributions:
        return ", ".join(
            f"{item.name} {item.version}" if item.version else item.name
            for item in status.distributions
        )
    return "none detected"


def _runtime_status_plain(status: Any) -> None:
    print("ONNX Runtime status")
    print()
    print(f"Importable              : {'yes' if status.importable else 'no'}")
    print(f"Distribution             : {_distribution_label(status)}")
    print(f"Distribution state       : {status.distribution_state}")
    if not status.importable and status.distributions:
        print(
            "Installed distributions : "
            + ", ".join(
                f"{item.name} {item.version}" if item.version else item.name
                for item in status.distributions
            )
        )
    if status.module_version is not None:
        print(f"Module version           : {status.module_version}")
    if status.module_path is not None:
        print(f"Module path              : {status.module_path}")
    if status.build_device is not None:
        print(f"Build device             : {status.build_device}")
    if status.build_info is not None:
        print(f"Build information        : {status.build_info}")
    if status.available_providers:
        print("\nProviders reported by ORT:")
        for provider in status.available_providers:
            print(f"  {provider}")
    else:
        print("Providers reported by ORT: none")
    configured = ", ".join(status.configured_providers) or "invalid"
    resolved = ", ".join(status.resolved_providers) or "unavailable"
    print(f"\nConfigured providers    : {configured}")
    print(f"Configuration source    : {_provider_source_label(status.provider_source)}")
    print(f"Resolved providers      : {resolved}")
    if status.import_error:
        print(f"Import error             : {status.import_error}")
    if status.resolution_error:
        print(f"Resolution error         : {status.resolution_error}")
    if not status.importable:
        print("\nInstall a compatible runtime, for example:")
        print('  python -m pip install "onnxvoice[cpu]"')


def _cmd_runtime_status(args: argparse.Namespace) -> int:
    from . import runtime

    status = runtime.runtime_status()
    if getattr(args, "format", "plain") == "json":
        print(json.dumps(_jsonable(status), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _runtime_status_plain(status)
    return 0


def _cmd_runtime_list(args: argparse.Namespace) -> int:
    from . import runtime
    from ._table import render_table

    status = runtime.runtime_status()
    if args.format == "json":
        print(json.dumps(_jsonable(status), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print("Installed ONNX Runtime distributions")
    distribution_rows = []
    for distribution in status.distributions:
        if distribution.owns_import is None:
            owner = "unknown"
        elif status.distribution_state == "ambiguous" and distribution.owns_import:
            owner = "ambiguous"
        else:
            owner = "yes" if distribution.owns_import else "no"
        distribution_rows.append([distribution.name, distribution.version or "-", owner])
    if not distribution_rows:
        distribution_rows.append(["none detected", "-", "no"])
    render_table(["NAME", "VERSION", "IMPORT OWNER"], distribution_rows)

    print("\nProviders reported by imported ORT")
    aliases = {provider: alias for alias, provider in _PROVIDER_LIST_ALIASES}
    provider_rows = []
    for provider in status.available_providers:
        provider_rows.append(
            [
                aliases.get(provider, provider),
                provider,
                "yes" if provider in status.resolved_providers else "no",
            ]
        )
    if not provider_rows:
        provider_rows.append(["none reported", "-", "no"])
    render_table(["ALIAS", "PROVIDER", "SELECTED"], provider_rows)
    if status.distribution_state == "ambiguous":
        print("\nWarning: multiple distributions claim the same onnxruntime import.")
        print("The imported files may have been overwritten by package installation order.")
        print("Use one ONNX Runtime distribution per environment.")
    return 0


def _provider_install_hint(provider: str | None) -> str:
    from . import runtime

    if provider is not None:
        canonical = runtime.normalize_provider_name(provider)
        hint = _PROVIDER_INSTALL_HINTS.get(canonical)
        if hint:
            return hint
        return f"Install a compatible ONNX Runtime build that provides {canonical}."
    return 'Install a compatible runtime, for example: python -m pip install "onnxvoice[gpu]"'


def _cmd_runtime_use(args: argparse.Namespace) -> int:
    from . import runtime
    from .errors import RuntimeContractError
    from .runtime_config import runtime_config_path

    status = runtime.runtime_status()
    if not status.importable:
        print(
            f"Cannot select {args.providers}: ONNX Runtime is not importable in this environment.",
            file=sys.stderr,
        )
        for distribution in status.distributions:
            version = f" {distribution.version}" if distribution.version else ""
            print(f"  Detected distribution: {distribution.name}{version}", file=sys.stderr)
        if status.import_error:
            print(f"  Import error: {status.import_error}", file=sys.stderr)
        print(_provider_install_hint(args.providers), file=sys.stderr)
        return 2

    try:
        preference = runtime.set_default_providers(args.providers)
        resolved = runtime.resolve_provider_preference(
            preference, available=status.available_providers
        )
    except RuntimeContractError as exc:
        try:
            requested = runtime.provider_preference(args.providers, environment={}).requested
            unavailable = [
                runtime.normalize_provider_name(name)
                for name in requested
                if name.lower() != "auto"
                and runtime.normalize_provider_name(name) not in status.available_providers
            ]
        except RuntimeContractError:
            unavailable = []
        if unavailable:
            provider = unavailable[0]
            print(
                f"Cannot select {args.providers}: {provider} is not available in the imported "
                "ONNX Runtime build.",
                file=sys.stderr,
            )
            if status.distributions:
                print("\nDetected distribution:", file=sys.stderr)
                for distribution in status.distributions:
                    version = f" {distribution.version}" if distribution.version else ""
                    print(f"  {distribution.name}{version}", file=sys.stderr)
            print(
                "\nInstall a compatible runtime in this Python environment, then retry:",
                file=sys.stderr,
            )
            print(f"  {_provider_install_hint(provider)}", file=sys.stderr)
        else:
            print(f"Cannot select {args.providers}: {exc}", file=sys.stderr)
        return 2

    print(f"Default ONNX Runtime provider set to: {','.join(preference.requested)}")
    print(f"Resolved now as: {', '.join(resolved)}")
    print(f"Configuration: {runtime_config_path()}")
    return 0


def _cmd_runtime_reset(args: argparse.Namespace) -> int:
    from . import runtime

    runtime.clear_default_providers()
    status = runtime.runtime_status()
    print("Removed saved ONNX Runtime provider preference.")
    print(f"Effective provider source: {_provider_source_label(status.provider_source)}")
    print(
        "Effective provider: "
        + (", ".join(status.resolved_providers) if status.resolved_providers else "unavailable")
    )
    if status.resolution_error:
        print(f"Resolution error: {status.resolution_error}")
    return 0


def _cmd_runtime(args: argparse.Namespace) -> int:
    action = getattr(args, "runtime_command", "status")
    if action == "status":
        return _cmd_runtime_status(args)
    if action == "list":
        return _cmd_runtime_list(args)
    if action == "use":
        return _cmd_runtime_use(args)
    if action == "reset":
        return _cmd_runtime_reset(args)
    raise ValueError(f"unknown runtime command: {action}")


def _cmd_doctor(args: argparse.Namespace) -> int:
    import platform

    from . import __version__, runtime
    from .huggingface import huggingface_diagnostics

    if args.system == "pocket":
        print(
            "warning: `onnxvoice doctor --system pocket` is deprecated; use `onnxvoice doctor`",
            file=sys.stderr,
        )

    status = runtime.runtime_status()
    report = {
        "schema": 1,
        "onnxvoice": {"version": __version__},
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "onnxruntime": _jsonable(status),
        "huggingface": huggingface_diagnostics(offline=args.offline),
    }
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print("ONNXVoice doctor\n")
    print("Python")
    print(f"  executable              : {sys.executable}")
    print(f"  version                 : {platform.python_version()}")
    print("\nONNX Runtime")
    print(f"  importable              : {'yes' if status.importable else 'no'}")
    print(f"  distribution            : {_distribution_label(status)}")
    print(f"  distribution state      : {status.distribution_state}")
    if status.module_version is not None:
        print(f"  module version          : {status.module_version}")
    if status.module_path is not None:
        print(f"  module path             : {status.module_path}")
    if status.build_device is not None:
        print(f"  build device            : {status.build_device}")
    print("  available providers     : " + (", ".join(status.available_providers) or "none"))
    print("  configured providers    : " + (", ".join(status.configured_providers) or "invalid"))
    print(f"  configuration source    : {_provider_source_label(status.provider_source)}")
    print("  resolved providers      : " + (", ".join(status.resolved_providers) or "unavailable"))
    print("  inference readiness     : provider reported by ORT; session not probed")
    if status.import_error:
        print(f"  import error            : {status.import_error}")
    if status.resolution_error:
        print(f"  resolution error        : {status.resolution_error}")

    hf = report["huggingface"]
    print("\nHugging Face")
    print(f"  integration             : {hf['huggingface_hub']}")
    print(f"  credentials             : {hf['credentials']}")
    print(f"  offline mode            : {'yes' if hf['offline'] else 'no'}")
    print(f"  implicit token disabled : {'yes' if hf['implicit_token_disabled'] else 'no'}")
    print(f"  gated repository access : {hf['gated_access']}")
    healthy = status.importable and status.resolution_error is None
    print("\nResult")
    print(f"  runtime configuration   : {'OK' if healthy else 'needs attention'}")
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
    quality = getattr(args, "quality", None)
    distribution = getattr(args, "distribution", None)
    all_variants = getattr(args, "all_variants", False)
    run_gc = getattr(args, "gc", False)

    try:
        plan = manager.plan_remove(
            refs,
            quality=quality,
            distribution=distribution,
            all_variants=all_variants,
            gc=run_gc,
        )
    except Exception as exc:
        print(f"onnxvoice: {exc}", file=sys.stderr)
        return 2

    if all_variants and not getattr(args, "yes", False):
        for ref in refs:
            if len(manager.find_installed(ref)) > 1:
                print(
                    f"onnxvoice: {ref} has multiple variants. Use --yes to confirm.",
                    file=sys.stderr,
                )
                return 2

    selected = []
    for usage in plan.installations:
        matches = [
            installation
            for installation in manager.find_installed(usage.ref)
            if (installation.storage_id or installation.id) == usage.storage_id
        ]
        if not matches:
            print(f"Not installed: {usage.ref} [{usage.storage_id}]", file=sys.stderr)
            return 2
        selected.append(matches[0])

    if args.dry_run:
        for installation, usage in zip(selected, plan.installations, strict=True):
            print(f"Would remove: {installation.ref} [{usage.storage_id}]")
            print(f"Installation logical size: {format_bytes(usage.logical_bytes)}")
            print(
                f"Immediate unique bytes freed: {format_bytes(usage.immediate_reclaimable_bytes)}"
            )
        if run_gc:
            print(
                f"Newly orphaned blobs: {plan.newly_orphaned_blob_count} "
                f"({format_bytes(plan.newly_orphaned_blob_bytes)})"
            )
            if plan.newly_orphaned_auxiliary_count:
                print(
                    f"Newly orphaned Pocket states: {plan.newly_orphaned_auxiliary_count} "
                    f"({format_bytes(plan.newly_orphaned_auxiliary_bytes)})"
                )
            print(f"GC reclaimable after removal: {format_bytes(plan.gc_reclaimable_bytes)}")
            print(
                "Total estimated reclaimable with --gc: "
                f"{format_bytes(plan.total_reclaimable_bytes)}"
            )
            print(
                f"Allocated bytes estimated: {format_bytes(plan.total_reclaimable_allocated_bytes)}"
            )
        return 0

    immediate_bytes = 0
    immediate_allocated_bytes = 0
    logical_bytes = 0
    for installation in selected:
        result = manager.store.remove(
            installation.system, installation.storage_id or installation.id
        )
        immediate_bytes += result.immediate_reclaimable_bytes
        immediate_allocated_bytes += result.immediate_reclaimable_allocated_bytes
        logical_bytes += result.logical_bytes

    gc_apparent_bytes = 0
    gc_allocated_bytes = 0
    if run_gc:
        gc_report = manager.store.gc()
        gc_apparent_bytes = gc_report.removed_apparent_bytes
        gc_allocated_bytes = gc_report.removed_allocated_bytes
        if gc_report.removed_blobs:
            print(
                f"GC: removed {gc_report.removed_blobs} orphan blobs "
                f"({format_bytes(gc_report.removed_bytes)})"
            )
        if gc_report.removed_auxiliary_count:
            print(
                f"GC: removed {gc_report.removed_auxiliary_count} Pocket voice states "
                f"({format_bytes(gc_report.removed_auxiliary_bytes)})"
            )

    total_reclaimed = immediate_bytes + gc_apparent_bytes
    print(f"Removed {len(selected)} installation(s)")
    print(f"Installation logical size: {format_bytes(logical_bytes)}")
    print(f"Immediate bytes reclaimed: {format_bytes(immediate_bytes)}")
    if run_gc:
        print(f"GC bytes reclaimed: {format_bytes(gc_apparent_bytes)}")
        print(f"Total bytes reclaimed: {format_bytes(total_reclaimed)}")
        print(
            "Allocated bytes reclaimed: "
            f"{format_bytes(immediate_allocated_bytes + gc_allocated_bytes)}"
        )
    return 0


def _cmd_cache_info(args: argparse.Namespace) -> int:
    from ._table import format_bytes

    manager = _manager(args)
    usage = manager.store.usage()
    top_level = [
        {
            "path": str(item.path),
            "category": item.category,
            "file_count": item.file_count,
            "apparent_bytes": item.apparent_bytes,
            "unique_apparent_bytes": item.unique_apparent_bytes,
            "allocated_bytes": item.allocated_bytes,
        }
        for item in usage.top_level_usage
    ]
    unmanaged_paths = [str(path) for path in usage.unmanaged_paths]
    legacy_paths = [str(path) for path in usage.legacy_paths]

    if getattr(args, "format", "table") == "json":
        import json as json_mod

        data = {
            "root": str(usage.root),
            # Retain the established flat JSON contract for existing consumers.
            "installation_count": usage.installation_count,
            "install_logical_bytes": usage.install_logical_bytes,
            "blob_apparent_bytes": usage.blob_apparent_bytes,
            "catalog_bytes": usage.catalog_bytes,
            "unique_file_bytes": usage.unique_file_bytes,
            "orphan_blob_count": usage.orphan_blob_count,
            "orphan_blob_bytes": usage.orphan_blob_bytes,
            "auxiliary_bytes": usage.auxiliary_bytes,
            "pocket_voice_state_count": usage.pocket_voice_state_count,
            "pocket_voice_state_bytes": usage.pocket_voice_state_bytes,
            "orphan_auxiliary_count": usage.orphan_auxiliary_count,
            "orphan_auxiliary_bytes": usage.orphan_auxiliary_bytes,
            "filesystem_total_bytes": usage.filesystem_total_bytes,
            "filesystem_used_bytes": usage.filesystem_used_bytes,
            "filesystem_free_bytes": usage.filesystem_free_bytes,
            "cache_apparent_bytes": usage.cache_apparent_bytes,
            "cache_unique_apparent_bytes": usage.cache_unique_apparent_bytes,
            "cache_allocated_bytes": usage.cache_allocated_bytes,
            "install_apparent_bytes": usage.install_apparent_bytes,
            "metadata_bytes": usage.metadata_bytes,
            "lock_bytes": usage.lock_bytes,
            "other_cache_bytes": usage.other_cache_bytes,
            "legacy_cache_bytes": usage.legacy_cache_bytes,
            "hardlinked_install_artifact_count": usage.hardlinked_install_artifact_count,
            "hardlinked_install_bytes": usage.hardlinked_install_bytes,
            "copied_install_artifact_count": usage.copied_install_artifact_count,
            "copied_install_bytes": usage.copied_install_bytes,
            "missing_blob_artifact_count": usage.missing_blob_artifact_count,
            "referenced_blob_count": usage.referenced_blob_count,
            "referenced_blob_bytes": usage.referenced_blob_bytes,
            "referenced_blob_allocated_bytes": usage.referenced_blob_allocated_bytes,
            "orphan_blob_allocated_bytes": usage.orphan_blob_allocated_bytes,
            "reclaimable_now_bytes": usage.reclaimable_now_bytes,
            "reclaimable_now_allocated_bytes": usage.reclaimable_now_allocated_bytes,
            "filesystem": {
                "total_bytes": usage.filesystem_total_bytes,
                "used_bytes": usage.filesystem_used_bytes,
                "free_bytes": usage.filesystem_free_bytes,
            },
            "cache": {
                "apparent_bytes": usage.cache_apparent_bytes,
                "unique_apparent_bytes": usage.cache_unique_apparent_bytes,
                "allocated_bytes": usage.cache_allocated_bytes,
                "reclaimable_now_bytes": usage.reclaimable_now_bytes,
                "reclaimable_now_allocated_bytes": usage.reclaimable_now_allocated_bytes,
            },
            "installations": {
                "count": usage.installation_count,
                "logical_bytes": usage.install_logical_bytes,
                "apparent_bytes": usage.install_apparent_bytes,
                "hardlinked_artifact_count": usage.hardlinked_install_artifact_count,
                "hardlinked_bytes": usage.hardlinked_install_bytes,
                "copied_artifact_count": usage.copied_install_artifact_count,
                "copied_bytes": usage.copied_install_bytes,
                "missing_blob_artifact_count": usage.missing_blob_artifact_count,
            },
            "blobs": {
                "count": usage.referenced_blob_count + usage.orphan_blob_count,
                "apparent_bytes": usage.blob_apparent_bytes,
                "referenced_count": usage.referenced_blob_count,
                "referenced_bytes": usage.referenced_blob_bytes,
                "referenced_allocated_bytes": usage.referenced_blob_allocated_bytes,
                "orphan_count": usage.orphan_blob_count,
                "orphan_bytes": usage.orphan_blob_bytes,
                "orphan_allocated_bytes": usage.orphan_blob_allocated_bytes,
            },
            "catalogs": {"bytes": usage.catalog_bytes},
            "auxiliary": {
                "bytes": usage.auxiliary_bytes,
                "pocket_voice_state_count": usage.pocket_voice_state_count,
                "pocket_voice_state_bytes": usage.pocket_voice_state_bytes,
                "orphan_count": usage.orphan_auxiliary_count,
                "orphan_bytes": usage.orphan_auxiliary_bytes,
            },
            "legacy_or_unmanaged": {
                "bytes": usage.legacy_cache_bytes + usage.other_cache_bytes,
                "legacy_bytes": usage.legacy_cache_bytes,
                "unmanaged_bytes": usage.other_cache_bytes,
                "paths": legacy_paths + unmanaged_paths,
            },
            "top_level_usage": top_level,
            "scan_errors": list(usage.scan_errors),
        }
        print(json_mod.dumps(data, indent=2, sort_keys=True))
        return 0

    print(f"Cache root             {usage.root}")
    print(f"Filesystem free        {format_bytes(usage.filesystem_free_bytes)}")
    print(f"Filesystem total       {format_bytes(usage.filesystem_total_bytes)}")
    print()
    print(f"Cache disk usage       {format_bytes(usage.cache_allocated_bytes)}")
    print(f"Cache apparent size    {format_bytes(usage.cache_apparent_bytes)}")
    print(f"Reclaimable now        {format_bytes(usage.reclaimable_now_bytes)}")
    print()
    print(f"Installations          {usage.installation_count}")
    print(f"Install logical        {format_bytes(usage.install_logical_bytes)}")
    print(f"Install apparent       {format_bytes(usage.install_apparent_bytes)}")
    print(f"  hard-linked          {format_bytes(usage.hardlinked_install_bytes)}")
    print(f"  copied               {format_bytes(usage.copied_install_bytes)}")
    print(f"Blobs                  {format_bytes(usage.blob_apparent_bytes)}")
    print(f"  referenced           {format_bytes(usage.referenced_blob_bytes)}")
    print(f"  orphan/reclaimable   {format_bytes(usage.orphan_blob_bytes)}")
    print(f"Catalogs               {format_bytes(usage.catalog_bytes)}")
    print(f"Auxiliary cache        {format_bytes(usage.auxiliary_bytes)}")
    print(
        f"Pocket voice states    {usage.pocket_voice_state_count} "
        f"({format_bytes(usage.pocket_voice_state_bytes)})"
    )
    print(
        f"Reclaimable Pocket     {usage.orphan_auxiliary_count} "
        f"({format_bytes(usage.orphan_auxiliary_bytes)})"
    )
    print(
        f"Legacy/unmanaged       {format_bytes(usage.legacy_cache_bytes + usage.other_cache_bytes)}"
    )

    if getattr(args, "details", False):
        print()
        print(f"Filesystem used        {format_bytes(usage.filesystem_used_bytes)}")
        print(f"Cache unique apparent  {format_bytes(usage.cache_unique_apparent_bytes)}")
        print(f"Metadata               {format_bytes(usage.metadata_bytes)}")
        print(f"Locks                  {format_bytes(usage.lock_bytes)}")
        print(f"Other cache            {format_bytes(usage.other_cache_bytes)}")
        print(f"Legacy cache           {format_bytes(usage.legacy_cache_bytes)}")
        print(
            f"Hard-linked artifacts  {usage.hardlinked_install_artifact_count} "
            f"({format_bytes(usage.hardlinked_install_bytes)})"
        )
        print(
            f"Copied artifacts       {usage.copied_install_artifact_count} "
            f"({format_bytes(usage.copied_install_bytes)})"
        )
        print(f"Missing artifact blobs {usage.missing_blob_artifact_count}")
        print(f"Referenced blobs       {usage.referenced_blob_count}")
        print(f"Orphan blobs           {usage.orphan_blob_count}")
        print(f"Pocket state orphans   {usage.orphan_auxiliary_count}")
        print("Top-level cache paths:")
        for item in usage.top_level_usage:
            print(
                f"  {item.path.name:<24} {item.category:<14} "
                f"{format_bytes(item.apparent_bytes)} apparent, "
                f"{format_bytes(item.allocated_bytes)} allocated"
            )
        if usage.legacy_paths:
            print("Recognized legacy paths:")
            for path in usage.legacy_paths:
                print(f"  {path}")
        if usage.unmanaged_paths:
            print("Unmanaged paths (report-only):")
            for path in usage.unmanaged_paths:
                print(f"  {path}")
        if usage.scan_errors:
            print("Scan diagnostics:")
            for error in usage.scan_errors:
                print(f"  {error}")
    return 0


def _cmd_cache_gc(args: argparse.Namespace) -> int:
    from ._table import format_bytes

    manager = _manager(args)
    include_legacy = getattr(args, "include_legacy", False)
    plan = manager.store.plan_gc(include_legacy=include_legacy)
    blob_candidates = [
        candidate for candidate in plan.candidates if candidate.category == "orphan_blob"
    ]
    state_candidates = [
        candidate
        for candidate in plan.candidates
        if candidate.category == "orphan_pocket_voice_state"
    ]
    state_records = {
        candidate.path.parent for candidate in state_candidates if candidate.path.suffix == ".json"
    }
    legacy_candidates = [
        candidate for candidate in plan.candidates if candidate.category == "legacy_cache"
    ]

    if args.dry_run:
        if not plan.candidates:
            print("No orphan cache files to remove.")
            return 0
        if blob_candidates:
            print(
                f"Would remove {len(blob_candidates)} blobs "
                f"({format_bytes(sum(item.apparent_bytes for item in blob_candidates))})"
            )
        if state_records:
            print(
                f"Would remove {len(state_records)} Pocket voice states "
                f"({format_bytes(sum(item.apparent_bytes for item in state_candidates))})"
            )
        if legacy_candidates:
            print(
                f"Would remove {len(legacy_candidates)} legacy cache files "
                f"({format_bytes(sum(item.apparent_bytes for item in legacy_candidates))})"
            )
        print(
            f"Total reclaimable: {format_bytes(plan.apparent_bytes)} apparent, "
            f"{format_bytes(plan.allocated_bytes)} allocated"
        )
        return 0

    report = manager.store.gc(plan=plan)
    if report.removed_candidates == 0:
        print("No orphan cache files to remove.")
        return 0
    if report.removed_blobs:
        print(f"Removed {report.removed_blobs} blobs ({format_bytes(report.removed_bytes)})")
    if report.removed_auxiliary_count:
        print(
            f"Removed {report.removed_auxiliary_count} Pocket voice states "
            f"({format_bytes(report.removed_auxiliary_bytes)})"
        )
    if report.removed_legacy_count:
        print(
            f"Removed {report.removed_legacy_count} legacy cache files "
            f"({format_bytes(report.removed_legacy_bytes)})"
        )
    print(
        f"Removed {report.removed_candidates} cache files; "
        f"{format_bytes(report.removed_apparent_bytes)} apparent bytes, "
        f"{format_bytes(report.removed_allocated_bytes)} allocated bytes reclaimed"
    )
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
            elif args.catalog_system == "supertonic":
                if args.catalog_action == "build":
                    catalog = supertonic_catalog_tools.build_catalog(
                        repository=args.repository,
                        revision=args.revision,
                    )
                    supertonic_catalog_tools.verify_catalog(catalog)
                    _write_json(args.output, catalog)
                    if args.source_output is not None:
                        _write_json(
                            args.source_output,
                            {"schema": 1, **catalog["source"]},
                        )
                    print(f"Wrote {len(catalog['bundles'])} bundles to {args.output}")
                    return 0
                if args.catalog_action == "verify":
                    catalog = supertonic_catalog_tools.load_catalog(args.catalog, args.source)
                    print(f"Verified {len(catalog['bundles'])} Supertonic bundles")
                    return 0
            elif args.catalog_system == "kitten":
                if args.catalog_action == "build":
                    seed_catalog = kitten_catalog_tools.load_catalog(args.seed_catalog)
                    catalog = kitten_catalog_tools.build_catalog(
                        seed_catalog,
                        revision=args.revision,
                    )
                    source = kitten_catalog_tools.build_source(catalog)
                    kitten_catalog_tools.verify_catalog(catalog, source)
                    _write_json(args.output, catalog)
                    if args.source_output is not None:
                        _write_json(args.source_output, source)
                    print(f"Wrote {len(catalog['models'])} Kitten models to {args.output}")
                    return 0
                if args.catalog_action == "verify":
                    catalog = kitten_catalog_tools.load_catalog(args.catalog, args.source)
                    print(f"Verified {len(catalog['models'])} Kitten models")
                    return 0
            elif args.catalog_system == "qwen3":
                if args.catalog_action == "build":
                    catalog = qwen3_catalog_tools.build_catalog()
                    source = qwen3_catalog_tools.build_source(catalog)
                    qwen3_catalog_tools.verify_catalog(catalog, source)
                    _write_json(args.output, catalog)
                    if args.source_output is not None:
                        _write_json(args.source_output, source)
                    print(f"Wrote {len(catalog['bundles'])} Qwen3 bundles to {args.output}")
                    return 0
                if args.catalog_action == "verify":
                    catalog = qwen3_catalog_tools.load_catalog(args.catalog, args.source)
                    print(f"Verified {len(catalog['bundles'])} Qwen3 bundles")
                    return 0
            elif args.catalog_system == "kitten2":
                if args.catalog_action == "build":
                    catalog = kitten2_catalog_tools.build_catalog(
                        repository=args.repository,
                        revision=args.revision,
                    )
                    source = kitten2_catalog_tools.build_source(catalog)
                    kitten2_catalog_tools.verify_catalog(catalog, source)
                    _write_json(args.output, catalog)
                    _write_json(args.source_output, source)
                    print(f"Wrote {len(catalog['bundles'])} Kitten2 bundles to {args.output}")
                    return 0
                if args.catalog_action == "verify":
                    catalog = kitten2_catalog_tools.load_catalog(args.catalog, args.source)
                    print(f"Verified {len(catalog['bundles'])} Kitten2 bundles")
                    return 0
            elif args.catalog_system == "chatterbox":
                if args.catalog_action == "build":
                    seed = json.loads(args.seed.read_text(encoding="utf-8"))
                    catalog = chatterbox_catalog_tools.build_catalog(seed)
                    source = chatterbox_catalog_tools.build_source(catalog)
                    chatterbox_catalog_tools.verify_catalog(catalog, source)
                    _write_json(args.output, catalog)
                    _write_json(args.source_output, source)
                    print(f"Wrote {len(catalog['bundles'])} Chatterbox bundles to {args.output}")
                    return 0
                if args.catalog_action == "verify":
                    catalog = chatterbox_catalog_tools.load_catalog(args.catalog, args.source)
                    print(f"Verified {len(catalog['bundles'])} Chatterbox bundles")
                    return 0
            else:
                raise ValueError(f"Unsupported catalog system: {args.catalog_system}")

        # Command dispatch
        if args.command == "runtime":
            return _cmd_runtime(args)
        if args.command == "doctor":
            return _cmd_doctor(args)

        if args.command == "voices":
            return _cmd_voices(args)
        if args.command == "engines":
            return _cmd_engines(args)

        handlers = {
            "list": _cmd_list,
            "installed": _cmd_installed,
            "updates": _cmd_updates,
            "update": _cmd_update,
            "info": _cmd_info,
            "remove": _cmd_remove,
            "uninstall": _cmd_remove,
        }

        if args.command in handlers:
            return handlers[args.command](args)

        # Legacy commands handled directly
        manager = _manager(args)

        if args.command == "install":
            progress = None if args.no_progress else InstallProgress(args.ref)
            try:
                installed = manager.install(
                    args.ref,
                    quality=args.quality,
                    distribution=getattr(args, "distribution", None),
                    refresh=args.refresh,
                    force=args.force,
                    progress=progress,
                )
            finally:
                if progress is not None:
                    progress.close()
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

    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"onnxvoice: {exc}", file=sys.stderr)
        return 2

    parser.error("unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
