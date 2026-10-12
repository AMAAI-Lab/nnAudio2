import pytest
import torch
from scipy.signal import get_window

from nnAudio2.features.stft import STFT, iSTFT


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("window", ["hann", "hamming"])
@pytest.mark.parametrize("win_length", [400, 401, 512])
@pytest.mark.parametrize("scripted", [False, True])
def test_inverse_with_short_window(device, window, win_length, scripted):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    waveform = torch.randn(2, 2048, device=device)
    analysis = STFT(
        n_fft=512, win_length=win_length, hop_length=128, window=window, verbose=False
    ).to(device)
    synthesis = iSTFT(
        n_fft=512, win_length=win_length, hop_length=128, window=window, verbose=False
    ).to(device)
    if scripted:
        synthesis = torch.jit.script(synthesis)
    spectrum = analysis(waveform)
    actual = synthesis(spectrum, onesided=True, length=waveform.shape[-1])
    expected = torch.istft(
        torch.view_as_complex(spectrum.contiguous()),
        n_fft=512,
        hop_length=128,
        win_length=win_length,
        window=torch.as_tensor(get_window(window, win_length), device=device).float(),
        length=waveform.shape[-1],
    )
    torch.testing.assert_close(
        actual, expected, rtol=2e-3, atol=2e-3, check_dtype=False
    )
    torch.testing.assert_close(
        actual, waveform, rtol=2e-3, atol=2e-3, check_dtype=False
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("win_length", [400, 401])
def test_trainable_short_synthesis_window(device, win_length):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    waveform = torch.randn(2, 2048, device=device)
    analysis = STFT(n_fft=512, win_length=win_length, hop_length=128, verbose=False).to(
        device
    )
    synthesis = iSTFT(
        n_fft=512,
        win_length=win_length,
        hop_length=128,
        trainable_window=True,
        verbose=False,
    ).to(device)
    actual = synthesis(analysis(waveform), onesided=True, length=waveform.shape[-1])
    actual.square().mean().backward()
    assert synthesis.window_mask.grad.shape[-2] == 512
    assert torch.isfinite(synthesis.window_mask.grad).all()
