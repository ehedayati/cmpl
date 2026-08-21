from importlib import import_module


__all__ = [
    # Modules
    "io",
    "utils",
    "df_build",

    # io.py
    "nifti_read",
    "load_dicom_scan_from_dir",
    "update_nifti_data",
    "dicom_to_SimpleITK",
    "itk_to_nifti",
    "itk_mask_correction",

    # utils.py
    "h5_to_nifti",
    "prepare_zipped_dicom",
    "dicom_to_h5",
    "kspace_to_image_space",
    "apply_hamming_filter_4d_numpy",
    "resize_complex_matrix_fft",
    "zero_pad",
    "resize_matrix",

    # df_build.py
    "build_medical_data_frame",
]


_MODULES = {
    "io": ".io",
    "utils": ".utils",
    "df_build": ".df_build",
}


_OBJECTS = {
    # io.py
    "nifti_read": (".io", "nifti_read"),
    "load_dicom_scan_from_dir": (
        ".io",
        "load_dicom_scan_from_dir",
    ),
    "update_nifti_data": (
        ".io",
        "update_nifti_data",
    ),
    "dicom_to_SimpleITK": (
        ".io",
        "dicom_to_SimpleITK",
    ),
    "itk_to_nifti": (
        ".io",
        "itk_to_nifti",
    ),
    "itk_mask_correction": (
        ".io",
        "itk_mask_correction",
    ),

    # utils.py
    "h5_to_nifti": (
        ".utils",
        "h5_to_nifti",
    ),
    "prepare_zipped_dicom": (
        ".utils",
        "prepare_zipped_dicom",
    ),
    "dicom_to_h5": (
        ".utils",
        "dicom_to_h5",
    ),
    "kspace_to_image_space": (
        ".utils",
        "kspace_to_image_space",
    ),
    "apply_hamming_filter_4d_numpy": (
        ".utils",
        "apply_hamming_filter_4d_numpy",
    ),
    "resize_complex_matrix_fft": (
        ".utils",
        "resize_complex_matrix_fft",
    ),
    "zero_pad": (
        ".utils",
        "zero_pad",
    ),
    "resize_matrix": (
        ".utils",
        "resize_matrix",
    ),

    # df_build.py
    "build_medical_data_frame": (
        ".df_build",
        "build_medical_data_frame",
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