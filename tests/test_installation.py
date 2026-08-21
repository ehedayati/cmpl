# File created by: Eisa Hedayati
# Date: 8/21/2026
# Description: This file is developed at CMRR
import subprocess
import sys
from textwrap import dedent


def run_isolated(code: str):
    """
    Run code in a fresh Python interpreter.

    This prevents imports from one test contaminating another test
    through sys.modules.
    """
    return subprocess.run(
        [sys.executable, "-c", dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def assert_success(result):
    if result.returncode != 0:
        raise AssertionError(
            "\nSTDOUT:\n"
            f"{result.stdout}\n"
            "\nSTDERR:\n"
            f"{result.stderr}"
        )


def test_cmpl_import_is_lightweight():
    result = run_isolated(
        """
        import sys
        import cmpl

        assert cmpl.__version__ == "0.2.0"

        optional_modules = [
            "torch",
            "monai",
            "matplotlib",
            "pandas",
            "nibabel",
            "pydicom",
            "SimpleITK",
            "h5py",
            "dicom2nifti",
            "ipywidgets",
            "IPython",
        ]

        loaded = [
            name
            for name in optional_modules
            if name in sys.modules
        ]

        assert not loaded, (
            "Optional packages were loaded by `import cmpl`: "
            f"{loaded}"
        )
        """
    )

    assert_success(result)


def test_reconstruction_namespace_is_lazy():
    result = run_isolated(
        """
        import sys
        import cmpl

        _ = cmpl.recon

        assert "torch" not in sys.modules
        """
    )

    assert_success(result)


def test_grappa_namespace_is_lazy():
    result = run_isolated(
        """
        import sys
        import cmpl

        _ = cmpl.recon.grappa

        assert "torch" not in sys.modules
        """
    )

    assert_success(result)


def test_grappa_function_import():
    result = run_isolated(
        """
        from cmpl.reconstruction.grappa import grappa_1d_recon

        assert callable(grappa_1d_recon)
        """
    )

    assert_success(result)


def test_sense_namespace_is_lazy():
    result = run_isolated(
        """
        import sys
        import cmpl

        _ = cmpl.recon.sense

        assert "torch" not in sys.modules
        """
    )

    assert_success(result)


def test_sense_cg_import():
    result = run_isolated(
        """
        from cmpl.reconstruction.sense.cg import CG_sense_2D

        assert callable(CG_sense_2D)
        """
    )

    assert_success(result)


def test_qmr_namespace_is_lazy():
    result = run_isolated(
        """
        import sys
        import cmpl

        _ = cmpl.qmr

        assert "torch" not in sys.modules
        assert "matplotlib" not in sys.modules
        """
    )

    assert_success(result)


def test_qmr_function_import():
    result = run_isolated(
        """
        from cmpl.quantitative_MRI import reconstruct_images

        assert callable(reconstruct_images)
        """
    )

    assert_success(result)


def test_visualization_works_without_io_dependencies():
    result = run_isolated(
        """
        import sys
        import importlib.abc


        FORBIDDEN = {
            "h5py",
            "nibabel",
            "pydicom",
            "SimpleITK",
            "dicom2nifti",
        }


        class ForbiddenImportFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".", 1)[0]

                if root in FORBIDDEN:
                    raise ImportError(
                        f"Forbidden dependency imported: {fullname}"
                    )

                return None


        sys.meta_path.insert(0, ForbiddenImportFinder())

        from cmpl.visualization import side_by_side_view

        assert callable(side_by_side_view)
        """
    )

    assert_success(result)


def test_io_does_not_load_unrelated_dependencies():
    result = run_isolated(
        """
        import sys

        import cmpl.utilities.io

        forbidden = [
            "torch",
            "matplotlib",
            "pandas",
            "monai",
            "dicom2nifti",
        ]

        loaded = [
            name
            for name in forbidden
            if name in sys.modules
        ]

        assert not loaded, (
            "I/O loaded unrelated dependencies: "
            f"{loaded}"
        )
        """
    )

    assert_success(result)

def test_data_does_not_load_unrelated_dependencies():
    result = run_isolated(
        """
        import sys

        from cmpl.utilities.df_build import build_medical_data_frame

        assert callable(build_medical_data_frame)

        forbidden = [
            "torch",
            "matplotlib",
            "h5py",
            "nibabel",
            "pydicom",
            "SimpleITK",
            "monai",
        ]

        loaded = [
            name
            for name in forbidden
            if name in sys.modules
        ]

        assert not loaded, (
            "Data utilities loaded unrelated dependencies: "
            f"{loaded}"
        )
        """
    )

    assert_success(result)