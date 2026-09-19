"""Maintenance checks for the append-only voice selector registry."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..catalog import CatalogClient
from ..types import CatalogItem
from ..voice_selectors import (
    VoiceSelectorRegistry,
    catalog_voice_keys,
    get_voice_selector_registry,
    load_voice_selector_registry,
    selector_for_voice,
)


def unassigned_catalog_voices(
    items: Iterable[CatalogItem],
    *,
    registry: VoiceSelectorRegistry | None = None,
) -> tuple[tuple[CatalogItem, str, str], ...]:
    """Return runnable catalog voices that have no stable registry assignment."""
    registry = registry or get_voice_selector_registry()
    return tuple(
        (item, asset_id, voice_id)
        for item, asset_id, voice_id in catalog_voice_keys(items)
        if selector_for_voice(
            system=item.system,
            asset_id=asset_id,
            voice_id=voice_id,
            include_retired=True,
            registry=registry,
        )
        is None
    )


def missing_registry_voices(
    items: Iterable[CatalogItem],
    *,
    registry: VoiceSelectorRegistry | None = None,
) -> tuple[Any, ...]:
    """Return active registry identities absent from the current catalog."""
    registry = registry or get_voice_selector_registry()
    catalog_keys = {
        (item.system, asset_id, voice_id) for item, asset_id, voice_id in catalog_voice_keys(items)
    }
    return tuple(
        identity
        for identity in registry.identities
        if identity.state == "active" and identity.canonical_key not in catalog_keys
    )


def check_registry_and_catalog(
    items: Iterable[CatalogItem] = (),
    *,
    registry: VoiceSelectorRegistry | None = None,
) -> tuple[str, ...]:
    """Return human-readable registry and catalog drift findings."""
    registry = registry or get_voice_selector_registry()
    issues: list[str] = []
    try:
        registry.validate()
    except Exception as exc:
        issues.append(str(exc))
        return tuple(issues)

    item_list = tuple(items)
    if item_list:
        for item, asset_id, voice_id in unassigned_catalog_voices(item_list, registry=registry):
            issues.append(f"unassigned catalog voice: {item.system}:{asset_id}:{voice_id}")
        for identity in missing_registry_voices(item_list, registry=registry):
            issues.append(f"active registry identity missing from catalog: {identity.selector}")
    return tuple(issues)


def _parse_catalog_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("catalog must use SYSTEM=PATH")
    system, path = value.split("=", 1)
    if system not in {"kokoro", "piper"} or not path:
        raise argparse.ArgumentTypeError("catalog must use kokoro=PATH or piper=PATH")
    return system, path


def _load_catalog_items(sources: list[tuple[str, str]]) -> list[CatalogItem]:
    if not sources:
        return []
    with tempfile.TemporaryDirectory(prefix="onnxvoice-selector-check-") as cache_dir:
        client = CatalogClient(
            cache_dir=Path(cache_dir),
            sources=dict(sources),
            offline=True,
        )
        items: list[CatalogItem] = []
        for system, _ in sources:
            items.extend(client.list(system))
        return items


def _voice_payload(
    item: CatalogItem, asset_id: str, voice_id: str, registry: VoiceSelectorRegistry
) -> dict[str, Any]:
    identity = selector_for_voice(
        system=item.system,
        asset_id=asset_id,
        voice_id=voice_id,
        include_retired=True,
        registry=registry,
    )
    return {
        "selector": identity.selector if identity is not None else None,
        "system": item.system,
        "asset_id": asset_id,
        "voice_id": voice_id,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m onnxvoice.catalog_tools.voice_selectors")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--catalog", action="append", type=_parse_catalog_arg, default=[])
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list-unassigned", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.check and not args.list_unassigned:
        args.check = True
    registry = load_voice_selector_registry(args.registry)
    items = _load_catalog_items(args.catalog)
    unassigned = unassigned_catalog_voices(items, registry=registry)
    issues = check_registry_and_catalog(items, registry=registry) if args.check else ()

    if args.as_json:
        output = {
            "ok": not issues,
            "issues": list(issues),
            "unassigned": [
                _voice_payload(item, asset_id, voice_id, registry)
                for item, asset_id, voice_id in unassigned
            ],
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        if args.check:
            if issues:
                for issue in issues:
                    print(f"ERROR: {issue}")
            else:
                print("selector registry: ok")
        if args.list_unassigned:
            for item, asset_id, voice_id in unassigned:
                print(f"{item.system}:{asset_id}:{voice_id}")
            if not unassigned:
                print("no unassigned catalog voices")
    if args.check and issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
