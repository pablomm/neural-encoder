Utilities
=========

.. currentmodule:: neural_encoder.utils

Preprocessing and data organization
-----------------------------------

.. autosummary::

   MeasurementPreprocessor
   ConfoundRemover
   split_views

.. autoclass:: MeasurementPreprocessor
   :members:

.. autoclass:: ConfoundRemover
   :members:

.. autofunction:: split_views

Evaluation
----------

.. autosummary::

   retrieval_metrics
   compute_rsa
   evaluate_pair
   evaluate_views

.. autofunction:: retrieval_metrics

.. autofunction:: compute_rsa

.. autofunction:: evaluate_pair

.. autofunction:: evaluate_views

Training loggers
----------------

.. autosummary::

   TrainingLogger
   RunningAverages
   AveragingLogger
   ConsoleLogger
   WandbLogger

.. autoclass:: TrainingLogger
   :members:

.. autoclass:: RunningAverages
   :members:

.. autoclass:: AveragingLogger
   :members:

.. autoclass:: ConsoleLogger
   :members:

.. autoclass:: WandbLogger
   :members:
