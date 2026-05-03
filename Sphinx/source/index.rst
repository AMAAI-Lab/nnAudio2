nnAudio2 |ProjectVersion|
=========================

**nnAudio2** is an audio feature extraction toolbox for deep learning, built on PyTorch.
Spectrograms and other audio transforms are implemented as ``nn.Module`` layers — they run
on-device (CUDA, MPS, or CPU), are fully differentiable, and can be embedded directly
inside a neural network. Filter banks (Mel, CQT, STFT kernels) can optionally be made
**trainable**.

nnAudio2 is developed and maintained by the `AMAAI Lab <https://amaai-lab.github.io/>`_
at SUTD. It is a modernised successor to
`nnAudio <https://github.com/AMAAI-Lab/nnAudio>`_, which is no longer actively
maintained. The original codebase has been fully overhauled to work with modern PyTorch
and the current scientific Python ecosystem.

If you use nnAudio2, please :doc:`cite both papers <citing>`.


Quick Start
-----------

.. code-block:: python

    import torch
    from nnAudio2.features.mel import MelSpectrogram

    mel = MelSpectrogram(sr=22050, n_fft=1024, hop_length=512, n_mels=128)
    mel = mel.to('cuda')          # or 'mps' on Apple Silicon

    audio = torch.randn(4, 22050).to('cuda')   # batch of 4 × 1-second clips
    spec  = mel(audio)                          # [4, 128, T] — on GPU

Because the transform is an ``nn.Module``, it moves with your model and its parameters
participate in backpropagation. Passing ``trainable_mel=True`` or ``trainable_STFT=True``
allows the filter banks themselves to be optimised during training.

For inverse STFT, use the uniform-bin configuration (``freq_scale='no'``). The
non-uniform ``linear``, ``log``, and ``log2`` scales are analysis-only; attempting
inversion raises an explicit error.

The source code is on `GitHub <https://github.com/AMAAI-Lab/nnAudio2>`_.


.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    intro


.. toctree::
    :maxdepth: 1
    :caption: API Documentation

    nnAudio2


.. toctree::
    :maxdepth: 1
    :caption: Examples & Tutorials

    examples


.. toctree::
    :maxdepth: 1
    :caption: GitHub

    github


.. toctree::
    :maxdepth: 1
    :caption: Citation

    citing


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
