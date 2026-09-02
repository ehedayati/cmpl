from importlib import import_module


__all__ = [
    "grappa_1D",
    "grappa_2D",
    "grappa_1d_recon",
    "grappa_2d_recon",
]


_MODULES = {
    "grappa_1D": ".grappa_1D",
    "grappa_2D": ".grappa_2D",
}


_OBJECTS = {
    "grappa_1d_recon": (
        ".grappa_1D",
        "grappa_1d_recon",
    ),
    "grappa_2d_recon": (
        ".grappa_2D",
        "grappa_2d_recon",
    ),
}


def __getattr__(name: str):
    if name in _MODULES:
        module = import_module(_MODULES[name], __name__)
        globals()[name] = module
        return module

    if name in _OBJECTS:
        module_name, object_name = _OBJECTS[name]
        module = import_module(module_name, __name__)
        obj = getattr(module, object_name)

        globals()[name] = obj
        return obj

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    return sorted(set(globals()) | set(__all__))