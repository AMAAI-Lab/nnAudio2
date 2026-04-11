import os
import warnings

__version__ = "0.3.4"

__citation__ = (
    'K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, "nnAudio: '
    'An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D '
    'Convolutional Neural Networks," IEEE Access, vol. 8, '
    'pp. 161981-162003, 2020, doi: 10.1109/ACCESS.2020.3019084.'
)

_CITATION_REMINDER = f"""
============================================================
nnAudio Citation Reminder

If you like nnAudio, please cite:

K. W. Cheuk, H. Anderson, K. Agres and D. Herremans,
"nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion
Toolbox Using 1D Convolutional Neural Networks,"
IEEE Access, vol. 8, pp. 161981-162003, 2020,
doi: 10.1109/ACCESS.2020.3019084.

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
