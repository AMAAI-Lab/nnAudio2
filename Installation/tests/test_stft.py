import pytest
import librosa
import torch
from scipy.signal import chirp, sweep_poly
import sys

sys.path.insert(0, "./")
from nnAudio.Spectrogram import *
from parameters import *
import warnings


gpu_idx = 0  # Choose which GPU to use

# If GPU is avaliable, also test on GPU
if torch.cuda.is_available():
    device_args = ["cpu", f"cuda:{gpu_idx}"]
else:
    warnings.warn("GPU is not avaliable, testing only on CPU")
    device_args = ["cpu"]

# librosa example audio for testing
example_y, example_sr = librosa.load(librosa.example('vibeace', hq=False))


@pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)
@pytest.mark.parametrize("device", [*device_args])
def test_inverse2(n_fft, hop_length, window, device):
    x = torch.tensor(example_y, device=device)
    stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window).to(device)
    istft = iSTFT(n_fft=n_fft, hop_length=hop_length, window=window).to(device)
    X = stft(x.unsqueeze(0), output_format="Complex")
    x_recon = istft(X, length=x.shape[0], onesided=True).squeeze()
    assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-5, atol=1e-3)
    x = torch.randn(4, 16000).to(device)
    X = stft(x, output_format="Complex")
    x_recon = istft(X, length=x.shape[1], onesided=True).squeeze()
    assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)
@pytest.mark.parametrize("device", [*device_args])
def test_inverse(n_fft, hop_length, window, device):
    x = torch.tensor(example_y, device=device)
    stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window, iSTFT=True).to(
        device
    )
    X = stft(x.unsqueeze(0), output_format="Complex")
    x_recon = stft.inverse(X, length=x.shape[0]).squeeze()
    assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-3, atol=1)
    x = torch.randn(4, 16000).to(device)
    X = stft(x, output_format="Complex")
    x_recon = stft.inverse(X, length=x.shape[1]).squeeze()
    assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)
@pytest.mark.parametrize("device", [*device_args])
def test_stft_complex(n_fft, hop_length, window, device):
    x = example_y
    stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Complex")
    X_real, X_imag = X[:, :, :, 0].squeeze(), X[:, :, :, 1].squeeze()
    X_librosa = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window=window)
    real_diff, imag_diff = np.allclose(
        X_real.cpu(), X_librosa.real, rtol=1e-1, atol=1e-1
    ), np.allclose(X_imag.cpu(), X_librosa.imag, rtol=1e-1, atol=1e-1)

    assert real_diff and imag_diff


@pytest.mark.parametrize("n_fft, win_length, hop_length", stft_with_win_parameters)
@pytest.mark.parametrize("device", [*device_args])
def test_stft_complex_winlength(n_fft, win_length, hop_length, device):
    x = example_y
    stft = STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Complex")
    X_real, X_imag = X[:, :, :, 0].squeeze(), X[:, :, :, 1].squeeze()
    X_librosa = librosa.stft(
        x, n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )
    real_diff, imag_diff = np.allclose(
        X_real.cpu(), X_librosa.real, rtol=1e-3, atol=1e-3
    ), np.allclose(X_imag.cpu(), X_librosa.imag, rtol=1e-3, atol=1e-3)
    assert real_diff and imag_diff


@pytest.mark.parametrize("device", [*device_args])
def test_stft_magnitude(device):
    x = example_y
    stft = STFT(n_fft=2048, hop_length=512).to(device)
    X = stft(
        torch.tensor(x, device=device).unsqueeze(0), output_format="Magnitude"
    ).squeeze()
    X_librosa, _ = librosa.core.magphase(librosa.stft(x, n_fft=2048, hop_length=512))
    assert np.allclose(X.cpu(), X_librosa, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("device", [*device_args])
def test_stft_phase(device):
    x = example_y
    stft = STFT(n_fft=2048, hop_length=512).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Phase")
    X_real, X_imag = torch.cos(X).squeeze(), torch.sin(X).squeeze()
    _, X_librosa = librosa.core.magphase(librosa.stft(x, n_fft=2048, hop_length=512))

    real_diff, imag_diff = np.mean(
        np.abs(X_real.cpu().numpy() - X_librosa.real)
    ), np.mean(np.abs(X_imag.cpu().numpy() - X_librosa.imag))

    # I find that np.allclose is too strict for allowing phase to be similar to librosa.
    # Hence for phase we use average element-wise distance as the test metric.
    assert real_diff < 2e-2 and imag_diff < 2e-2


@pytest.mark.parametrize("device", [*device_args])
def test_stft_torchscript(device):
    x = torch.randn(2, 4096, device=device)
    stft = STFT(n_fft=1024, hop_length=256, verbose=False).to(device)
    scripted = torch.jit.script(stft)

    expected = stft(x, output_format="Complex")
    actual = scripted(x, output_format="Complex")

    assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [*device_args])
def test_istft_torchscript(device):
    x = torch.randn(2, 4096, device=device)
    stft = STFT(n_fft=1024, hop_length=256, verbose=False).to(device)
    X = stft(x, output_format="Complex")

    istft = iSTFT(n_fft=1024, hop_length=256, verbose=False).to(device)
    scripted = torch.jit.script(istft)

    expected = istft(X, onesided=True, length=x.shape[-1])
    actual = scripted(X, onesided=True, length=x.shape[-1])

    assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("freq_scale", ["linear", "log", "log2"])
@pytest.mark.parametrize("device", [*device_args])
def test_stft_inverse_nonuniform_freq_scale_raises(freq_scale, device):
    x = torch.randn(2, 4096, device=device)
    with pytest.warns(UserWarning, match="analysis-only"):
        stft = STFT(
            n_fft=1024,
            hop_length=256,
            freq_scale=freq_scale,
            iSTFT=True,
            verbose=False,
        ).to(device)

    X = stft(x, output_format="Complex")

    with pytest.raises(RuntimeError, match="freq_scale='no'"):
        stft.inverse(X, onesided=True, length=x.shape[-1])


@pytest.mark.parametrize("freq_scale", ["linear", "log"])
@pytest.mark.parametrize("device", [*device_args])
def test_istft_nonuniform_freq_scale_raises(freq_scale, device):
    x = torch.randn(2, 4096, device=device)
    stft = STFT(
        n_fft=1024, hop_length=256, freq_scale=freq_scale, verbose=False
    ).to(device)
    X = stft(x, output_format="Complex")

    with pytest.warns(UserWarning, match="analysis-only"):
        istft = iSTFT(
            n_fft=1024, hop_length=256, freq_scale=freq_scale, verbose=False
        ).to(device)

    with pytest.raises(RuntimeError, match="freq_scale='no'"):
        istft(X, onesided=True, length=x.shape[-1])


if torch.cuda.is_available():
    x = torch.randn((4, 44100)).to(
        f"cuda:{gpu_idx}"
    )  # Create a batch of input for the following Data.Parallel test

    @pytest.mark.parametrize("device", [f"cuda:{gpu_idx}"])
    def test_STFT_Parallel(device):
        spec_layer = STFT(
            hop_length=512,
            n_fft=2048,
            window="hann",
            freq_scale="no",
            output_format="Complex",
        ).to(device)
        inverse_spec_layer = iSTFT(
            hop_length=512, n_fft=2048, window="hann", freq_scale="no"
        ).to(device)

        spec_layer_parallel = torch.nn.DataParallel(spec_layer)
        inverse_spec_layer_parallel = torch.nn.DataParallel(inverse_spec_layer)
        spec = spec_layer_parallel(x)
        x_recon = inverse_spec_layer_parallel(spec, onesided=True, length=x.shape[-1])

        assert np.allclose(
            x_recon.detach().cpu(), x.detach().cpu(), rtol=1e-3, atol=1e-3
        )
