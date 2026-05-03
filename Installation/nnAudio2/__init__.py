import os

__version__ = "2.0.1"

__citation__ = (
    'Roy et al. (2026) nnAudio 2, arXiv. '
    'Cheuk et al. (2020) nnAudio, IEEE Access, doi:10.1109/ACCESS.2020.3019084.'
)


def cite() -> str:
    return __citation__


def show_citation() -> None:
    print(__citation__)


if os.environ.get("NNAUDIO_DISABLE_CITATION_REMINDER", "").lower() not in {"1", "true", "yes"}:
    print(f"[nnAudio2] Please cite Roy et al. (2026) and Cheuk et al. (2020) if you use this library. "
          f"Set NNAUDIO_DISABLE_CITATION_REMINDER=1 to suppress.")
