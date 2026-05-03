# Changelog

## version 2.0.0 (April 2026) — nnAudio2

Full overhaul of nnAudio for modern PyTorch and the current scientific Python ecosystem.
Package renamed from `nnAudio` to `nnAudio2`; the original nnAudio repository is no longer maintained.

Key changes:

- **TorchScript support** (issue [#132](../../issues/132)): resolved compilation failures in `STFT` and `iSTFT` by removing dynamic state mutation and module construction from scripted code paths, and tightening argument handling for TorchScript-facing helpers.
- **Safe iSTFT semantics** (issue [#136](../../issues/136)): reliable inversion is now restricted to `freq_scale='no'`. Non-uniform variants (`linear`, `log`, `log2`) raise an explicit `RuntimeError` instead of silently returning poor reconstructions.
- **CFP / SciPy compatibility**: updated `cfp.py` to use `scipy.signal.windows.blackmanharris` (modern SciPy location).
- **VQT correctness**: `VQT(gamma=0)` now explicitly delegates to `CQT1992v2`, aligning its output with CQT behavior.
- **Citation reminder**: `import nnAudio2` shows a citation reminder by default; suppress with `NNAUDIO_DISABLE_CITATION_REMINDER=1`.
- **Dependency baseline**: Python ≥ 3.11, PyTorch ≥ 2.0, NumPy 2.x, current SciPy.
- **Inverse CQT (`iCQT`)**: new `nn.Module` that reconstructs a waveform from the `'Complex'` output of `CQT1992v2` using iterative Landweber inversion. The upper frame bound is estimated via power iteration at initialisation; the step size is set to `1.8/B` for guaranteed convergence. The adjoint correctly handles `ReflectionPad1d` boundary folding to match the forward operator exactly. Reconstruction SNR exceeds 30 dB for signals within the Nyquist-sampled frequency range of the chosen `hop_length` (see documentation for the constraint). The module is fully differentiable.
- **Test suite**: 60 tests pass in the modern environment; new regression tests cover TorchScript compilation, iSTFT rejection behavior, and iCQT round-trip SNR, output shape, and gradient flow.

For a full technical summary of what changed and why, see [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md).  
For the detailed development log (per-batch changes, before/after test results), see [MIGRATION_LOG.md](MIGRATION_LOG.md).

---

## version 0.3.0 (19 Nov 2021):
1. Changed module naming. `nnAudio.Spectrogram` will be replaced by `nnAudio.features` in the future releases. Currently, various spectrogram types are accessible via both methods.

## version 0.2.6 (02 Sep 2021): 
1. Add `relu()` to the `nonlinear_func` in `CFP()` to prevent negative values [#105](/../../pull/105).

## version 0.2.5 (06 Aug 2021): 
1. Incorrect inverse STFT calculation is fixed [#100](/../../issues/100).
1. Add more test cases in unit test 
1. Refactor `STFT` and `iSTFT`

## version 0.2.5 (06 Aug 2021): 
1. Incorrect inverse STFT calculation is fixed [#100](/../../issues/100).
1. Add more test cases in unit test 
1. Refactor `STFT` and `iSTFT`
This version can be obtained via `pip install nnAudio==0.2.5`.


## version 0.2.4 (11 June 2021): 
1. CQT2010 bug has been fixed [#85](/../../issues/85).
1. Provide a wider support for scipy versions using `from scipy.fftpack import fft` in [utils.py](https://github.com/AMAAI-Lab/nnAudio/blob/e9b1697963f0fd8e5030b130a30974bc06408baf/Installation/nnAudio/utils.py#L13)
1. Documentation error for STFT has been fixed [#90](/../../issues/90)

This version can be obtained via `pip install nnAudio==0.2.4`.

## version 0.2.3 (10 June 2021): 
Broken package, please ignore this one.

## version 0.2.2 (1 March 2021): 
Added filter scale support to various version of CQT classes as requested in [#54](/../../issues/54). Different normalization methods are also added to the `forward()` method as `normalization_type` under each CQT class. A bug is discovered in CQT2010, the output is problematic [#85](/../../issues/85).

To use this version, do `pip install nnAudio==0.2.2`.

## version 0.2.1 (15 Jan 2021): 
Fixed bugs [#80](/../../issues/80), [#82](/../../issues/82), and fulfilled request [#83](/../../issues/83). nnAudio version can be checked with `nnAudio.__version__` inside python now. Added two more spectrogram types `Gammatonegram()` and `Combined_Frequency_Periodicity()`.

To use this version, do `pip install nnAudio==0.2.1`.

## version 0.2.0 (8 Nov 2020): 
Now it is possible to do `stft_layer.to(device)` to move the spectrogram layers between different devices.
No more `device` argument when creating the spectrogram layers.

To use this version, do `pip install nnAudio==0.2.0`.

## version 0.1.5:
Much better `iSTFT` and `Griffin-Lim`. Now Griffin-Lim is a separated PyTorch class and requires `torch >= 1.6.0` to run. `STFT` has also been refactored and it is less memory consuming now.

To use this version, do `pip install nnAudio==0.1.5`.

## version 0.1.4a0:
Finalized `iSTFT` and `Griffin-Lim`. They are now more accurate and stable.

## version 0.1.2.dev3:
Add `win_length` to `STFT` so that it has the same funcationality as librosa.

## version 0.1.2.dev2: 
Fix bugs where the inverse cannot be done using GPU. And add a separated `iSTFT` layer class

## version 0.1.2.dev1: 
Add Inverse STFT and Griffin-Lim. They are still under development, please use with care.
                    
## version 0.1.1  (1 June 2020): 
Add MFCC
