# File created by: Eisa Hedayati
# Date: 5/21/2024
# Description: This file is developed at CMRR

from importlib import import_module


__all__ = [
    "cg",
]


def __getattr__(name: str):
    if name == "cg":
        module = import_module(".cg", __name__)
        globals()[name] = module
        return module

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    return sorted(set(globals()) | set(__all__))