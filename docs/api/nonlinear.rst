Nonlinear refinement
====================

.. currentmodule:: neural_encoder.nonlinear

Estimator and networks
----------------------

.. autosummary::

   NonlinearRefiner
   ResidualNetwork
   ResidualMLP

.. autoclass:: NonlinearRefiner
   :members:

.. autoclass:: ResidualNetwork
   :members: forward
   :show-inheritance:

.. autoclass:: ResidualMLP
   :members: forward
   :show-inheritance:

Losses
------

.. autosummary::

   MultiViewContrastiveLoss
   symmetric_info_nce
   cosine_pull

.. autoclass:: MultiViewContrastiveLoss
   :members:
   :show-inheritance:

.. autofunction:: symmetric_info_nce

.. autofunction:: cosine_pull
