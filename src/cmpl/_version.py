# src/cmpl/_version.py
from __future__ import annotations
from importlib.metadata import PackageNotFoundError, version as _dist_version

# Name of the installed distribution (usually your [project].name)
_DIST_NAME = "cmpl"

def _from_pyproject() -> str:
    # Fallback for editable/dev checkouts where the dist isn’t installed yet
    import sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[2]  # .../src/cmpl -> project root?
    # Walk up until we find pyproject.toml
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    pp = root / "pyproject.toml"
    if not pp.exists():
        return "0+unknown"
    try:
        if sys.version_info >= (3, 11):
            import tomllib  # stdlib on 3.11+
        else:
            import tomli as tomllib  # add 'tomli' as a dev dep if you need this fallback
        with pp.open("rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("version", "0+unknown")
    except Exception:
        return "0+unknown"

try:
    __version__ = _dist_version(_DIST_NAME)
except PackageNotFoundError:
    __version__ = _from_pyproject()

