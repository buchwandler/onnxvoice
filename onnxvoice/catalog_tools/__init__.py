"""Catalog generation and verification helpers."""

from . import pocket as pocket_tools
from . import supertonic as supertonic_tools
from .piper import (
    DEFAULT_REPOSITORY,
    DEFAULT_REVISION,
    CatalogError,
    build_catalog,
    fetch_and_build_catalog,
    fetch_upstream_catalog,
    get_voice,
    list_voices,
    load_catalog,
    resolve_revision,
    verify_catalog,
)
from .supertonic import CatalogError as SupertonicCatalogError
from .supertonic import build_catalog as build_supertonic_catalog
from .supertonic import load_catalog as load_supertonic_catalog
from .supertonic import verify_catalog as verify_supertonic_catalog

__all__ = [
    "CatalogError",
    "DEFAULT_REPOSITORY",
    "DEFAULT_REVISION",
    "build_catalog",
    "fetch_and_build_catalog",
    "fetch_upstream_catalog",
    "get_voice",
    "list_voices",
    "load_catalog",
    "resolve_revision",
    "verify_catalog",
    "pocket_tools",
    "SupertonicCatalogError",
    "build_supertonic_catalog",
    "load_supertonic_catalog",
    "supertonic_tools",
    "verify_supertonic_catalog",
]
