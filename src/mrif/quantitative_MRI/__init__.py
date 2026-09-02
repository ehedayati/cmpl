from importlib import import_module


__all__ = [
    "mapping",
    "t2_star_two_parametric_2D",
    "t2_star_two_parametric_3D",
    "t2_star_three_parametric_2D",
    "t2_star_three_parametric_3D",
    "calculate_rmse_percentage_s0",
    "reconstruct_images",
]


_OBJECTS = {
    "t2_star_two_parametric_2D": (
        ".mapping",
        "t2_star_two_parametric_2D",
    ),
    "t2_star_two_parametric_3D": (
        ".mapping",
        "t2_star_two_parametric_3D",
    ),
    "t2_star_three_parametric_2D": (
        ".mapping",
        "t2_star_three_parametric_2D",
    ),
    "t2_star_three_parametric_3D": (
        ".mapping",
        "t2_star_three_parametric_3D",
    ),
    "calculate_rmse_percentage_s0": (
        ".mapping",
        "calculate_rmse_percentage_s0",
    ),
    "reconstruct_images": (
        ".mapping",
        "reconstruct_images",
    ),
}


def __getattr__(name: str):
    if name == "mapping":
        module = import_module(".mapping", __name__)
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