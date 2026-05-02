# nnAudio2 Migration Summary

This document describes what changed between nnAudio (≤ 0.3.x) and nnAudio2 (2.0.0), why each change was made, and how to verify the results.

For the full per-batch development log, see [MIGRATION_LOG.md](MIGRATION_LOG.md).

---

## Verified environment

| Dependency | Version used |
|------------|-------------|
| Python     | 3.11.10     |
| PyTorch    | 2.10.0      |
| NumPy      | 2.4.3       |
| SciPy      | 1.17.1      |
| librosa    | 0.11.0      |

---

## Changes

### `features/stft.py` — TorchScript support (issue #132)

**Problem:** `torch.jit.script(STFT(...))` and `torch.jit.script(iSTFT(...))` both failed.

Root causes:
1. `STFT.forward()` assigned `self.num_samples` dynamically — TorchScript does not allow setting undeclared attributes.
2. `STFT.forward()` constructed `nn.ConstantPad1d` / `nn.ReflectionPad1d` at runtime — TorchScript does not allow dynamic module construction inside `forward`.
3. `iSTFT` used `refresh_win == None`-style control flow that TorchScript could not type-infer correctly.

Fixes:
- Replaced dynamic attribute assignment with local variables.
- Replaced runtime-constructed padding modules with functional padding calls (`F.pad`).
- Tightened optional-argument handling in `STFT.forward`, `STFT.inverse`, and `iSTFT.forward`.
- Updated `inverse_stft` to use local tensors for window normalisation under scripting, while preserving the eager-mode cache.

Result: `STFT` and `iSTFT` now compile with `torch.jit.script`.

---

### `features/stft.py` — Safe iSTFT semantics (issue #136)

**Problem:** Calling inverse STFT with `freq_scale='linear'` or `freq_scale='log'` silently returned severely degraded audio:

| `freq_scale` | MSE (baseline) | SNR (baseline) |
|---|---|---|
| `'no'`     | ~1.5 × 10⁻¹³ | ~128 dB |
| `'linear'` | ~1.14         | ~−0.5 dB |
| `'log'`    | ~1.67         | ~−2.2 dB |

The reconstruction was not just noisy — it was essentially wrong — but no error was raised.

Fix:
- Added `self.supports_inverse` tracking per frequency scale.
- `STFTBase` now raises `RuntimeError` on any inverse call when `freq_scale != 'no'`.
- Initialization emits a warning when an inverse-capable object is created with a non-uniform scale.

Result:
- `freq_scale='no'`: inverse still works (MSE ~4 × 10⁻¹⁴).
- `freq_scale='linear'` / `'log'` / `'log2'`: raises a clear `RuntimeError` instead of returning bad audio.

---

### `utils.py` — TorchScript helper annotations (issue #132)

**Problem:** After the `stft.py` fixes, TorchScript compilation moved on to fail in the helper functions used by `iSTFT`.

Fixes:
- Added explicit type annotations to `torch_window_sumsquare` and `overlap_add`.
- Changed `fold(..., stride=...)` calls to pass stride as a 2-element tuple rather than a bare integer, matching current TorchScript type expectations.

---

### `features/cfp.py` — SciPy compatibility

**Problem:** `scipy.signal.blackmanharris` was removed from the top-level `scipy.signal` namespace in modern SciPy.

Fix: changed both `Combined_Frequency_Periodicity` and `CFP` to use `scipy.signal.windows.blackmanharris`.

---

### `features/vqt.py` — VQT / CQT alignment

**Problem:** `VQT(gamma=0)` should reduce to CQT, but showed a meaningful numerical mismatch against `CQT1992v2` (max absolute error ~0.089).

Fix: when `gamma == 0`, `VQT.__init__` now creates an internal `CQT1992v2` module and `VQT.forward` delegates to it.

---

### `__init__.py` — Citation reminder

Added import-time citation reminder so users who install the package are prompted to cite the paper.

- `nnAudio2.__citation__` — citation string
- `nnAudio2.cite()` — returns the citation string
- `nnAudio2.show_citation()` — prints it
- `CitationReminderWarning` shown once per process on `import nnAudio2`
- Set `NNAUDIO_DISABLE_CITATION_REMINDER=1` to suppress

---

## Test results

| Stage | Passed | Failed |
|-------|--------|--------|
| Baseline (before any changes) | 44 | 3 |
| After TorchScript fix (#132)  | 46 | 3 |
| After CFP fix                 | 48 | 1 |
| After VQT fix                 | 49 | 0 |
| After iSTFT semantics (#136)  | 54 | 0 |
| After citation reminder       | **57** | **0** |

---

## Running the test suite

```bash
cd Installation
pytest -q
```

To run only STFT/iSTFT tests:

```bash
pytest tests/test_stft.py -q
```

To verify the citation reminder:

```bash
python -c "import nnAudio2"
# suppress with: NNAUDIO_DISABLE_CITATION_REMINDER=1 python -c "import nnAudio2"
```

---

## Known non-blocking warnings

The following warnings appear in the test suite but do not cause failures:

- CFP emits a `divide by zero` warning (pre-existing).
- `torch.stft(return_complex=False)` has a deprecation warning in newer PyTorch.
- `torch.jit.script` itself is deprecated in very recent PyTorch versions.

These are reasonable cleanup targets for a future maintenance pass.
