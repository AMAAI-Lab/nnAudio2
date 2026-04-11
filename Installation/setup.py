import setuptools
import codecs
import os.path
import sys

from setuptools.command.develop import develop

try:
    from setuptools.command.editable_wheel import editable_wheel
except ImportError:
    editable_wheel = None

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    bdist_wheel = None

with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_metadata(rel_path):
    namespace = {}
    original_disable_flag = os.environ.get("NNAUDIO_DISABLE_CITATION_REMINDER")
    os.environ["NNAUDIO_DISABLE_CITATION_REMINDER"] = "1"
    try:
        exec(read(rel_path), namespace)
    finally:
        if original_disable_flag is None:
            os.environ.pop("NNAUDIO_DISABLE_CITATION_REMINDER", None)
        else:
            os.environ["NNAUDIO_DISABLE_CITATION_REMINDER"] = original_disable_flag
    return namespace


PACKAGE_METADATA = get_metadata("nnAudio/__init__.py")
_CITATION_REMINDER = PACKAGE_METADATA["_CITATION_REMINDER"]


def emit_citation_reminder():
    sys.stderr.write(_CITATION_REMINDER + "\n")


class CitationReminderDevelop(develop):
    def run(self):
        super().run()
        emit_citation_reminder()


cmdclass = {
    "develop": CitationReminderDevelop,
}


if editable_wheel is not None:
    class CitationReminderEditableWheel(editable_wheel):
        def run(self):
            super().run()
            emit_citation_reminder()


    cmdclass["editable_wheel"] = CitationReminderEditableWheel


if bdist_wheel is not None:
    class CitationReminderBdistWheel(bdist_wheel):
        def run(self):
            super().run()
            emit_citation_reminder()


    cmdclass["bdist_wheel"] = CitationReminderBdistWheel


setuptools.setup(
    name="nnaudio",  # Replace with your own username
    version=PACKAGE_METADATA["__version__"],
    author="KinWaiCheuk",
    author_email="u3500684@connect.hku.hk",
    description="A fast GPU audio processing toolbox with 1D convolutional neural network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AMAAI-Lab/nnAudio",
    project_urls={
        "Documentation": "https://kinwaicheuk.github.io/nnAudio/index.html",
        "Issues": "https://github.com/AMAAI-Lab/nnAudio/issues",
        "Source": "https://github.com/AMAAI-Lab/nnAudio",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "scipy>=1.2.0",
        "numpy>=1.14.5,<2",
        "torch>=1.6.0",
    ],
    extras_require={"tests": ["pytest", "librosa"]},
    cmdclass=cmdclass,
)
