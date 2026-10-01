Preprocessing
=============

.. currentmodule:: neural_encoder.preprocessing

Transformers fitted on training measurements before encoding. They are kept
separate from the encoders so that train/test boundaries and domain-specific
choices stay explicit.

.. autosummary::

   BetaPreprocessor
   SessionStandardScaler
   ConfoundRemover

.. autoclass:: BetaPreprocessor
   :members:

.. autoclass:: SessionStandardScaler
   :members:

.. autoclass:: ConfoundRemover
   :members:
