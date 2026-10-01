"""Catalog generation and verification helpers."""

from . import kitten as kitten_tools
from . import pocket as pocket_tools
from . import supertonic as supertonic_tools
from .kitten import CatalogError as KittenCatalogError
from .kitten import build_catalog as build_kitten_catalog
from .kitten import build_source as build_kitten_source
from .kitten import load_catalog as load_kitten_catalog
from .kitten import verify_catalog as verify_kitten_catalog
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
    "KittenCatalogError",
    "build_kitten_catalog",
    "build_kitten_source",
    "load_kitten_catalog",
    "verify_kitten_catalog",
    "kitten_tools",
    "verify_supertonic_catalog",
]
