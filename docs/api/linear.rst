Linear encoding
===============

.. currentmodule:: neural_encoder.linear

.. autosummary::

   FeatureReweighting
   DistilledMCCA
   CrossViewRidge
   LinearEncoder

Feature reliability weighting
-----------------------------

.. autoclass:: FeatureReweighting
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
fitting. It is not yet part of ``LinearEncoder`` or ``NeuralEncoder``.

.. autoclass:: CrossViewRidge
   :members:

Linear encoder
--------------

.. autoclass:: LinearEncoder
   :members:
