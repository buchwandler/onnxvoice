"""Maintenance checks for the append-only voice selector registry."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from ..catalog import CatalogClient
from ..inventory import language_codes_from_metadata
from ..types import CatalogItem
from ..voice_selectors import (
    VoiceSelectorRegistry,
    catalog_voice_keys,
    format_voice_selector,
    get_voice_selector_registry,
    load_voice_selector_registry,
    parse_voice_selector,
    selector_for_voice,
    voice_selector_systems,
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


def append_unassigned_registry_entries(
    items: Iterable[CatalogItem], registry_data: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Build an append-only registry update for unassigned catalog voices."""
    registry = VoiceSelectorRegistry.from_data(registry_data)
    pending: dict[tuple[str, str, str], tuple[str, str, CatalogItem]] = {}
    for item, asset_id, voice_id in unassigned_catalog_voices(items, registry=registry):
        if item.system == "pocket":
            states = item.metadata.get("voice_states")
            if not isinstance(states, (list, tuple)) or not any(
                isinstance(state, Mapping) and state.get("name") == voice_id for state in states
            ):
                raise ValueError(
                    f"Pocket voice {asset_id}:{voice_id} has no explicit voice_states record"
                )
        languages = language_codes_from_metadata(item.metadata)
        if len(languages) != 1:
            raise ValueError(
                f"{item.system}:{asset_id}:{voice_id} needs exactly one language in catalog metadata"
            )
        try:
            engine_code = registry.engine_codes[item.system]
        except KeyError as exc:
            raise ValueError(f"selector registry has no engine code for {item.system!r}") from exc
        language = parse_voice_selector(
            format_voice_selector(languages[0], engine_code, 1)
        ).language
        key = (item.system, asset_id, voice_id)
        candidate = (language, engine_code, item)
        previous = pending.get(key)
        if previous is not None and previous[:2] != candidate[:2]:
            raise ValueError(f"ambiguous language metadata for {item.system}:{asset_id}:{voice_id}")
        pending[key] = candidate

    next_slots = {
        (identity.language, identity.engine_code): identity.slot for identity in registry.identities
    }
    for identity in registry.identities:
        slot_key = (identity.language, identity.engine_code)
        next_slots[slot_key] = max(next_slots[slot_key], identity.slot)

    ordered = sorted(
        (
            (language, engine_code, key[1], key[2], key[0])
            for key, (language, engine_code, _item) in pending.items()
        ),
    )
    additions: list[dict[str, Any]] = []
    for language, engine_code, asset_id, voice_id, system in ordered:
        slot_key = (language, engine_code)
        slot = next_slots.get(slot_key, 0) + 1
        next_slots[slot_key] = slot
        additions.append(
            {
                "language": language,
                "engine_code": engine_code,
                "slot": slot,
                "system": system,
                "asset_id": asset_id,
                "voice_id": voice_id,
                "state": "active",
            }
        )

    updated = {**registry_data, "entries": [*registry_data["entries"], *additions]}
    VoiceSelectorRegistry.from_data(updated)
    return updated, tuple(additions)


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
    supported = voice_selector_systems()
    if system not in supported or not path:
        raise argparse.ArgumentTypeError(
            f"catalog must use SYSTEM=PATH where SYSTEM is one of: {', '.join(supported)}"
        )
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


def _read_registry_data(path: Path | None) -> dict[str, Any]:
    try:
        if path is None:
            raw = (
                resources.files("onnxvoice")
                .joinpath("data/voice_selectors.json")
                .read_text(encoding="utf-8")
            )
        else:
            raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read selector registry") from exc
    if not isinstance(data, dict):
        raise ValueError("selector registry must be a JSON object")
    return data


def _write_registry(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
    parser.add_argument("--append-unassigned", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write append-unassigned results to the explicit --registry file (preview is default)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.apply and not args.append_unassigned:
        parser.error("--apply requires --append-unassigned")
    if args.apply and args.registry is None:
        parser.error("--apply requires an explicit --registry path")
    if not args.check and not args.list_unassigned and not args.append_unassigned:
        args.check = True
    if args.append_unassigned:
        registry_data = _read_registry_data(args.registry)
        registry = VoiceSelectorRegistry.from_data(registry_data)
    else:
        registry_data = None
        registry = load_voice_selector_registry(args.registry)
    items = _load_catalog_items(args.catalog)
    if args.append_unassigned:
        assert registry_data is not None
        updated, additions = append_unassigned_registry_entries(items, registry_data)
        if args.apply and additions:
            assert args.registry is not None
            _write_registry(args.registry, updated)
        if args.as_json:
            print(
                json.dumps(
                    {
                        "added": list(additions),
                        "applied": bool(args.apply and additions),
                        "registry": str(args.registry) if args.apply else None,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif not additions:
            print("no unassigned catalog voices")
        else:
            for entry in additions:
                selector = format_voice_selector(
                    entry["language"], entry["engine_code"], entry["slot"]
                )
                identity = f"{entry['system']}:{entry['asset_id']}:{entry['voice_id']}"
                print(f"{selector} {identity}")
            if args.apply:
                print(f"Updated selector registry: {args.registry}")
            else:
                print("Preview only; rerun with --apply and an explicit --registry to write.")
        return 0
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
