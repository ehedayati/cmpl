from importlib import import_module


__all__ = [
    "visualization",
    "side_by_side_view",
    "visualize_segmentation_slice",
    "plot_3D_mri",
]


_OBJECTS = {
    "side_by_side_view": (
        ".visualization",
        "side_by_side_view",
    ),
    "visualize_segmentation_slice": (
        ".visualization",
        "visualize_segmentation_slice",
    ),
    "plot_3D_mri": (
        ".visualization",
        "plot_3D_mri",
    ),
}


def __getattr__(name: str):
    if name == "visualization":
        module = import_module(".visualization", __name__)
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