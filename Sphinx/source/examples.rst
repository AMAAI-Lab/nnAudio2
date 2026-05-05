Tutorials
=========

Step-by-step tutorials are available in the ``tutorials/`` folder of the
`nnAudio2 repository <https://github.com/AMAAI-Lab/nnAudio2>`_.
Each tutorial is a self-contained Jupyter notebook that can be run locally.

.. list-table::
   :widths: 10 90
   :header-rows: 0

   * - **Part 1**
     - *Computing Mel spectrograms with nnAudio2* — loading audio, initialising
       the ``MelSpectrogram`` layer, and visualising the output.
   * - **Part 2**
     - *Training a linear keyword spotter with trainable basis functions* — embedding
       nnAudio2 inside a ``LightningModule``, enabling ``trainable_mel`` and
       ``trainable_STFT``, and training on Google Speech Commands.
   * - **Part 3**
     - *Evaluation and visualisation* — loading a saved checkpoint, running the test
       set, and plotting the learned Mel filterbank and STFT kernels.
   * - **Part 4**
     - *Using more complex non-linear models* — swapping the linear classifier for a
       BC-ResNet while keeping the nnAudio2 front-end unchanged.
   * - **Part 5**
     - *Fast & differentiable audio features with HuggingFace* — benchmarking librosa,
       torchaudio, and nnAudio2 on MPS/GPU; integrating ``MelSpectrogram`` as the first
       layer of a HuggingFace ``Trainer``-compatible model; enabling ``trainable_mel=True``
       and visualising filterbank adaptation. Demonstrates a **+28 % relative accuracy
       improvement** on Google Speech Commands v0.02 (35-class) over a fixed mel baseline.

To run the tutorials, install the dependencies listed in ``tutorials/requirements.txt``
and open the notebooks in Jupyter:

.. code-block:: bash

    pip install -r tutorials/requirements.txt
    jupyter notebook tutorials/
