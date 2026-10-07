Linear encoding
===============

.. currentmodule:: neural_encoder.linear

.. autosummary::

   FeatureReweighting
   GramPCA
   DistilledMCCA
   CrossViewRidge
   LinearEncoder
   ChunkedLinearEncoder

Feature reliability weighting
-----------------------------

Used both on the input features and, with signal-to-noise weighting, on the
output dimensions of ``LinearEncoder`` and ``ChunkedLinearEncoder``.

.. autoclass:: FeatureReweighting
   :members:

Gram PCA
--------

Exact PCA for data with many more features than samples, such as full-cortex
fMRI. The principal components are computed from the sample-by-sample Gram
matrix, accumulated over chunks of features, so the full measurement matrix
never needs to be in memory at once. It can run on a GPU and can be passed to
``LinearEncoder`` as its PCA stage.

.. autoclass:: GramPCA
   :members:

Distilled MCCA
--------------

.. autoclass:: DistilledMCCA
   :members:

Cross-view ridge denoiser
-------------------------

Learns an affine map from a single measurement to the mean of the other
views of the same sample. It estimates the part of a measurement that is
shared across views and can be applied to individual measurements after
fitting. ``LinearEncoder`` and ``NeuralEncoder`` apply it to the PCA scores by
default.

.. autoclass:: CrossViewRidge
   :members:

Linear encoder
--------------

.. autoclass:: LinearEncoder
   :members:

Chunked linear encoder
----------------------

Fits session standardization, beta preprocessing, and the linear encoder for
data with a very large number of features, such as full-cortex fMRI. The data
are read one chunk of features at a time from an array or memory map, so the
full measurement matrix is never held in memory, and the PCA uses
:class:`GramPCA`. The fitted model matches ``SessionStandardScaler``,
``BetaPreprocessor``, and ``LinearEncoder`` applied in sequence.

.. code-block:: python

   import numpy as np
   from neural_encoder.linear import ChunkedLinearEncoder

   X = np.load("betas.npy", mmap_mode="r")  # (n_measurements, n_vertices), int16
   encoder = ChunkedLinearEncoder(
       pca_kwargs={"n_components": 768, "dtype": "float32", "device": "cuda"},
       random_state=0,
   )
   Z_train = encoder.fit_transform(X, sample_ids=sample_ids[train], rows=train)
   Z_test = encoder.transform(X, rows=test)

.. autoclass:: ChunkedLinearEncoder
   :members:
