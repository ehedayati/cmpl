import os
from setuptools import setup, find_packages

# Read the version from setup.py's version argument
version = "0.6.0"  # Keep this synchronized with the setup() call

def write_version_py(version):
    version_path = os.path.join(os.path.dirname(__file__), "cmpl", "_version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")

write_version_py(version)

setup(
    name="cmpl",
    version=version,
    description="CMRR MRI Processing Libraries",
    author="Eisa Hedayati",
    author_email="heday015@umn.edu",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.9.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "h5py>=3.0.0",
        "pydicom>=2.0.0",
        "nibabel>=5.0.0",
        "pytest>=8.0.0",
        "hypothesis>=6.0.0",
    ],
    python_requires=">=3.10",
    setup_requires=["setuptools>=64.0.0", "wheel"],
)
