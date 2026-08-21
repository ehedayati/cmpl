from importlib import import_module


__all__ = [
    "enhanced_dicom",
]


def _load_enhanced_dicom():
    module = import_module(".enhanced_dicom", __name__)
    globals()["enhanced_dicom"] = module
    return module


def __getattr__(name: str):
    if name == "enhanced_dicom":
        return _load_enhanced_dicom()

    module = _load_enhanced_dicom()

    try:
        obj = getattr(module, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    globals()[name] = obj
    return obj


def __dir__():
    return sorted(set(globals()) | set(__all__))