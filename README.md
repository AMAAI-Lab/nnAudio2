# nnAudio2

**nnAudio2** is an audio feature extraction toolbox for deep learning, built on PyTorch. Spectrograms and other audio transforms are implemented as `nn.Module` layers — they run on-device (CUDA, MPS, or CPU), are fully differentiable, and can be embedded directly inside a neural network. Filter banks (Mel, CQT, STFT kernels) can optionally be made **trainable**.

nnAudio2 is developed and maintained by the [AMAAI Lab](https://amaai-lab.github.io/) at SUTD. It is a modernised successor to [nnAudio](https://github.com/AMAAI-Lab/nnAudio), which is no longer actively maintained. The original nnAudio codebase has been fully overhauled to work with modern PyTorch and the current scientific Python ecosystem.

---

## Installation

```bash
pip install nnaudio2
```

or directly from the repository:

```bash
pip install git+https://github.com/AMAAI-Lab/nnAudio2.git#subdirectory=Installation
```

## Documentation

[https://amaai-lab.github.io/nnAudio2/](https://amaai-lab.github.io/nnAudio2/)

---

## Supported transforms

| Transform | Trainable | Differentiable | Invertible |
|-----------|:---------:|:--------------:|:----------:|
| STFT | ✅ | ✅ | ✅ (uniform bin only) |
| Mel Spectrogram | ✅ | ✅ | — |
| MFCC | ✅ | ✅ | — |
| CQT | ✅ | ✅ | ✅ (CQT1992v2 only, see note) |
| VQT | ✅ | ✅ | — |
| Gammatone | ✅ | ✅ | — |
| CFP | ✅ | ✅ | — |
| Griffin-Lim | — | ✅ | — |

All transforms run on **CUDA**, **MPS (Apple Silicon)**, and **CPU**.

> **Note on inverse STFT:** reliable reconstruction is only guaranteed for the uniform-bin setting (`freq_scale='no'`). Non-uniform variants (`linear`, `log`, `log2`) are analysis-only; attempting inversion raises an explicit error.

> **Note on inverse CQT:** `iCQT` uses iterative Landweber inversion and achieves > 30 dB SNR for signals whose frequency content is within the Nyquist-sampled range of the chosen `hop_length`. Specifically, reconstruction is reliable up to roughly `f < sr / (2 * hop_length / Q)` where `Q ≈ bins_per_octave / (2^(1/bins_per_octave) − 1)`. At `hop_length=512` with default settings, this corresponds to frequencies below ~880 Hz. Wideband signals with a large `hop_length` will have reduced SNR because high-frequency bins are undersampled in time.

---

## What's new in nnAudio2

nnAudio2 modernises the original library for current PyTorch and scientific Python environments. Key improvements:

- **TorchScript compatibility** — resolved compilation failures in STFT and iSTFT by removing dynamic state mutation and module construction from scripted code paths.
- **Correct iSTFT semantics** — inversion is restricted to `freq_scale='no'`; unsupported configurations now raise an explicit `RuntimeError` instead of returning silently degraded output.
- **CFP restored** — compatibility with modern SciPy is fixed.
- **VQT correctness** — VQT now correctly reduces to CQT when `gamma = 0`.
- **Modern dependencies** — tested against current PyTorch, NumPy 2.x, and SciPy releases.
- **Expanded test suite** — regression tests cover new STFT/iSTFT behaviours; the full suite passes in a modern Python environment.

---

## Quick start

```python
import torch
from nnAudio2.features.mel import MelSpectrogram

# Drop the transform in as a model layer
mel = MelSpectrogram(sr=22050, n_fft=1024, hop_length=512, n_mels=128)
mel = mel.to('cuda')   # or 'mps' on Apple Silicon

audio = torch.randn(4, 22050).to('cuda')   # batch of 4 × 1-second clips
spec  = mel(audio)                          # [4, 128, T] — on GPU
```

---

## Changelog & migration

Full changelog: [CHANGELOG.md](CHANGELOG.md)

**Migrating from nnAudio?** See [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) for a concise breakdown of every change, the reasoning behind each fix, and how to verify your environment.

**v2.0.0** (April 2026) — full overhaul of nnAudio. See the nnAudio2 paper for details.

---

## Citation

If you use nnAudio2, please cite **both** papers.

### nnAudio2 (this repository)

> Abhinaba Roy, Junyi Liang, Dorien Herremans. (2026). *nnAudio 2: Overcoming Dynamic Compilation Barriers and Transform Inconsistencies.* arXiv (forthcoming).

```bibtex
@article{roy2026nnaudio2,
  author  = {Roy, Abhinaba and Liang, Junyi and Herremans, Dorien},
  title   = {nnAudio 2: Overcoming Dynamic Compilation Barriers and Transform Inconsistencies},
  journal = {arXiv},
  year    = {2026},
}
```

### Original nnAudio

> K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, "nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks," *IEEE Access*, vol. 8, pp. 161981–162003, 2020. doi: [10.1109/ACCESS.2020.3019084](https://ieeexplore.ieee.org/document/9174990)

```bibtex
@article{cheuk2020nnaudio,
  author  = {Cheuk, Kin Wai and Anderson, Hans and Agres, Kat and Herremans, Dorien},
  journal = {IEEE Access},
  title   = {nnAudio: An on-the-Fly {GPU} Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks},
  year    = {2020},
  volume  = {8},
  pages   = {161981--162003},
  doi     = {10.1109/ACCESS.2020.3019084},
}
```

---

## Contributing

Contributions are welcome. To run the test suite:

```bash
cd Installation
pytest
```

---

## Publishing to PyPI

A GitHub Actions workflow at `.github/workflows/publish-to-pypi.yml` publishes the package when a version tag is pushed.

1. Create a `pypi` environment in the GitHub repository settings and require manual approval.
2. In PyPI, add a Trusted Publisher for `AMAAI-Lab / nnAudio2`, workflow `publish-to-pypi.yml`, environment `pypi`.
3. Bump `__version__` in `Installation/nnAudio2/__init__.py` to match the tag.
4. Push the tag: `git tag v2.0.1 && git push origin v2.0.1`.

---

## Dependencies

- Python ≥ 3.11
- PyTorch ≥ 2.0
- NumPy ≥ 1.14.5
- SciPy ≥ 1.2.0
