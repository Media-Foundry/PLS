"""Compatibility loader for the migrated external BIO/protein V4 implementation."""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path


SOURCE_MODULES = (
    "hbond_network_features", "physicochemical_features", "extract_physchem_from_pdb",
    "surface_features", "structural_geometry_features", "enhanced_features_v2",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_v4(source_root: Path):
    """Load V4 despite its stale ``preparation.protein`` package imports.

    This does not hide provenance: callers receive hashes for every external
    source file and must persist them with generated features.
    """
    source_root = source_root.resolve()
    required = [source_root / f"{name}.py" for name in SOURCE_MODULES + ("protein_features_v4",)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(f"missing external V4 source files: {missing}")
    root_string = str(source_root)
    if root_string not in sys.path: sys.path.insert(0, root_string)
    try:
        preparation = importlib.import_module("preparation")
    except ModuleNotFoundError as error:
        if error.name != "preparation":
            raise
        preparation = types.ModuleType("preparation")
        preparation.__path__ = []
        sys.modules["preparation"] = preparation
    package = sys.modules.get("preparation.protein")
    if package is None:
        package = types.ModuleType("preparation.protein")
        package.__path__ = [root_string]
        sys.modules["preparation.protein"] = package
        setattr(preparation, "protein", package)
    for name in SOURCE_MODULES:
        module = importlib.import_module(name)
        sys.modules[f"preparation.protein.{name}"] = module
        setattr(package, name, module)
    module = importlib.import_module("protein_features_v4")
    hashes = {path.name: file_sha256(path) for path in required}
    return module, hashes
