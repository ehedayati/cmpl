from importlib import import_module


__all__ = [
    "enhanced_dicom",
    "metadata",
    "geometry",
]


_MODULES = {
    "enhanced_dicom": ".enhanced_dicom",
    "metadata": ".metadata",
    "geometry": ".geometry",
}


_OBJECTS = {
    # metadata.py
    "get_metadata_string": (
        ".metadata",
        "get_metadata_string",
    ),
    "get_metadata_float": (
        ".metadata",
        "get_metadata_float",
    ),
    "get_metadata_vector": (
        ".metadata",
        "get_metadata_vector",
    ),
    "extract_acquisition_metadata": (
        ".metadata",
        "extract_acquisition_metadata",
    ),

    # geometry.py
    "compute_slice_normal": (
        ".geometry",
        "compute_slice_normal",
    ),
    "compute_slice_position": (
        ".geometry",
        "compute_slice_position",
    ),
    "get_slice_position": (
        ".geometry",
        "get_slice_position",
    ),
    "extract_slice_geometry": (
        ".geometry",
        "extract_slice_geometry",
    ),
}


def __getattr__(name: str):
    # Lazy-load known submodules.
    if name in _MODULES:
        module = import_module(
            _MODULES[name],
            __name__,
        )
        globals()[name] = module
        return module

    # Lazy-load explicitly routed helper functions.
    if name in _OBJECTS:
        module_name, object_name = _OBJECTS[name]

        module = import_module(
            module_name,
            __name__,
        )

        obj = getattr(
            module,
            object_name,
        )

        globals()[name] = obj
        return obj

    # Preserve the existing mrif.dicom API:
    # unknown names are looked up in enhanced_dicom.py.
    module = import_module(
        ".enhanced_dicom",
        __name__,
    )

    globals()["enhanced_dicom"] = module

    try:
        obj = getattr(module, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    globals()[name] = obj
    return obj


def __dir__():
    return sorted(
        set(globals())
        | set(__all__)
        | set(_OBJECTS)
    )