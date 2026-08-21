# File created by: Eisa Hedayati
# Date: 12/29/2023
# Description: This file is developed at CMRR

from importlib import import_module

from ._version import __version__

__author__ = "Eisa Hedayati"

__all__ = [
    "__version__",
    "utilities",
    "utils",
    "io",
    "visualization",
    "vis",
    "segmentation",
    "seg",
    "quantitative_MRI",
    "qmr",
    "reconstruction",
    "recon",
    "dicom",
]


_MODULE_ALIASES = {
    "utilities": ".utilities",
    "utils": ".utilities",
    "io": ".utilities.io",
    "visualization": ".visualization",
    "vis": ".visualization",
    "segmentation": ".segmentation",
    "seg": ".segmentation",
    "quantitative_MRI": ".quantitative_MRI",
    "qmr": ".quantitative_MRI",
    "reconstruction": ".reconstruction",
    "recon": ".reconstruction",
    "dicom": ".dicom",
}


def __getattr__(name: str):
    if name in _MODULE_ALIASES:
        module = import_module(_MODULE_ALIASES[name], __name__)
        globals()[name] = module
        return module

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    return sorted(set(globals()) | set(__all__))