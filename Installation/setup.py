import setuptools
import os.path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
        return fp.read()


def get_metadata(rel_path):
    namespace = {}
    os.environ["NNAUDIO_DISABLE_CITATION_REMINDER"] = "1"
    try:
        exec(read(rel_path), namespace)
    finally:
        os.environ.pop("NNAUDIO_DISABLE_CITATION_REMINDER", None)
    return namespace


PACKAGE_METADATA = get_metadata("nnAudio2/__init__.py")


setuptools.setup(
    name="nnaudio2",
    version=PACKAGE_METADATA["__version__"],
    author="AMAAI Lab",
    author_email="dorien.herremans@gmail.com",
    description="A fast GPU audio processing toolbox with 1D convolutional neural network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AMAAI-Lab/nnAudio2",
    project_urls={
        "Documentation": "https://amaai-lab.github.io/nnAudio2/",
        "Issues": "https://github.com/AMAAI-Lab/nnAudio2/issues",
        "Source": "https://github.com/AMAAI-Lab/nnAudio2",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Compatible",
    ],
    python_requires=">=3.8",
    install_requires=[
        "scipy>=1.2.0",
        "numpy>=1.14.5,<2",
        "torch>=1.6.0",
    ],
    extras_require={"tests": ["pytest", "librosa"]},
)
