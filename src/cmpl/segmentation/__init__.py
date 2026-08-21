from importlib import import_module


__all__ = [
    "AutoSegmentation",
    "extract_extrusion",
]


_OBJECTS = {
    "AutoSegmentation": (
        ".MRISegmentationTool",
        "AutoSegmentation",
    ),
    "extract_extrusion": (
        ".tools",
        "extract_extrusion",
    ),
}


def __getattr__(name: str):
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