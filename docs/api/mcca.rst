MCCA implementation
===================

.. currentmodule:: neural_encoder._mcca

The package contains a minimal, adapted implementation of ``mvlearn``'s MCCA
estimator so that the linear encoder does not depend on the unmaintained
external package. Most users should use
:class:`neural_encoder.linear.DistilledMCCA`, which adds the single-projector
distillation used by neural-encoder.

.. autosummary::

   MCCA

.. autoclass:: MCCA
   :members:
