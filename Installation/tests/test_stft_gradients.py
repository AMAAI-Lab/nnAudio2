import pytest
import torch

from nnAudio2.features.stft import STFT


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("scripted", [False, True])
def test_fixed_stft_silent_waveform_gradients(device, scripted):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    layer = STFT(n_fft=256, hop_length=64, output_format="Magnitude", verbose=False)
    layer = layer.to(device)
    if scripted:
        layer = torch.jit.script(layer)
    waveform = torch.zeros(2, 1024, device=device, requires_grad=True)
    magnitude = layer(waveform)
    magnitude.sum().backward()
    assert torch.count_nonzero(magnitude) == 0
    assert torch.isfinite(waveform.grad).all()
    assert torch.count_nonzero(waveform.grad) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_stft_gradient_through_padded_waveforms(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    waveform = torch.zeros(2, 1024, device=device)
    waveform[:, :400] = torch.randn(2, 400, device=device)
    gain = torch.tensor(0.5, device=device, requires_grad=True)
    layer = STFT(n_fft=256, hop_length=64, output_format="Magnitude", verbose=False)
    layer = layer.to(device)
    prediction = layer(gain * waveform)
    with torch.no_grad():
        target = layer(waveform)
    (prediction - target).square().mean().backward()
    assert torch.isfinite(gain.grad)
    torch.testing.assert_close(gain.grad, -target.square().mean())
