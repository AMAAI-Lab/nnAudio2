import os
import warnings

__version__ = "2.0.0"

__citation__ = (
    'Anonymous Authors, "nnAudio: '
    'An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D '
    'Convolutional Neural Networks," Under Review.'
)

_CITATION_REMINDER = f"""
============================================================
nnAudio Citation Reminder

If you like nnAudio, please cite our paper (currently under review).

============================================================
""".strip()


class CitationReminderWarning(UserWarning):
    """Shown once per process when nnAudio is imported."""


def cite() -> str:
    return __citation__


def show_citation() -> None:
    print(__citation__)


def _citation_reminder_enabled() -> bool:
    return os.environ.get("NNAUDIO_DISABLE_CITATION_REMINDER", "").lower() not in {
        "1",
        "true",
        "yes",
    }


if _citation_reminder_enabled():
    warnings.warn(_CITATION_REMINDER, CitationReminderWarning, stacklevel=2)
