from importlib import import_module


__all__ = [
    "grappa",
    "sense",
]


_MODULES = {
    "grappa": ".grappa",
    "sense": ".sense",
}


def __getattr__(name: str):
    if name in _MODULES:
        module = import_module(_MODULES[name], __name__)
        globals()[name] = module
        return module

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    return sorted(set(globals()) | set(__all__))