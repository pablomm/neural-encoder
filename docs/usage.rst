Usage
=====

Training from repeated measurements
-----------------------------------

Use ``sample_ids`` when repeated measurements are stored in one matrix. Rows
with the same identifier are treated as views of the same sample. If
``view_ids`` is omitted, view order is inferred from the order in which each
sample's measurements appear.

.. code-block:: python

   from neural_encoder import NeuralEncoder

   encoder = NeuralEncoder(random_state=0)
   encoder.fit(X_train, sample_ids=sample_ids)

   Z_train = encoder.transform(X_train)
   Z_test = encoder.transform(X_test)

When the repetitions are already stored as aligned matrices, use
:meth:`~neural_encoder.NeuralEncoder.fit_views`:

.. code-block:: python

   encoder = NeuralEncoder(random_state=0)
   encoder.fit_views([X1, X2, X3])

   encoded_views = [encoder.transform(X) for X in (X1, X2, X3)]

Each matrix must have shape ``(n_samples, n_features)``, and row ``i`` must
refer to the same sample in every view.

Preprocessing measurements
--------------------------

``MeasurementPreprocessor`` implements the preprocessing used for the neural
measurements in the paper. Its defaults divide measurements by 300, clip
feature values using the 0.004 and 0.996 training quantiles, replace non-finite
values with zero, and center every feature. Learned quantiles and feature
statistics are reused by ``transform``.

.. code-block:: python

   from neural_encoder.utils import MeasurementPreprocessor

   preprocessing = MeasurementPreprocessor()
   X_train = preprocessing.fit_transform(X_train_raw)
   X_test = preprocessing.transform(X_test_raw)

Preprocessing remains independent of ``NeuralEncoder`` so that train/test
boundaries and domain-specific choices stay explicit.

Configuring the encoder
-----------------------

The principal dimensions are constructor arguments. Keyword dictionaries
configure each internal estimator without replacing it:

.. code-block:: python

   encoder = NeuralEncoder(
       n_components_pca=512,
       n_components_mcca=128,
       feature_reweighting_kwargs={"weighting": "sqrt"},
       pca_kwargs={"whiten": False},
       cross_view_ridge_kwargs={"alphas": [1e2, 1e3, 1e4, 1e5]},
       distilled_mcca_kwargs={"mcca_reg": 0.1},
       refiner_kwargs={
           "steps": 2_000,
           "batch_size": 256,
           "optimizer_kwargs": {"lr": 1e-3},
       },
       random_state=0,
   )

The PCA scores are denoised by ``CrossViewRidge`` before MCCA. Pass
``cross_view_ridge=False`` to skip this stage and fit MCCA directly on the PCA
scores, as in the paper architecture:

.. code-block:: python

   encoder = NeuralEncoder(cross_view_ridge=False, random_state=0)

By default, the nonlinear network receives the distilled MCCA embedding. To
use PCA scores as its input while keeping the MCCA output as the residual base:

.. code-block:: python

   encoder = NeuralEncoder(
       refinement_input_stage="pca",
       n_components_pca=512,
       n_components_pca_refinement=1024,
       n_components_mcca=128,
       random_state=0,
   )

If the two PCA dimensions are equal, the fitted PCA is shared by both branches.

Evaluating aligned views
------------------------

``evaluate_views`` computes retrieval and representational similarity analysis
(RSA) for every ordered pair of views and averages the results:

.. code-block:: python

   from neural_encoder.utils import evaluate_views

   metrics, pairs = evaluate_views(encoded_views, return_pairs=True)
   print(metrics)
   print(pairs)

The returned metrics are mean rank, recall at one, matched cosine similarity,
and RSA. Retrieval is directional, so three views produce six pairwise rows.

PyTorch export
--------------

Use :meth:`~neural_encoder.NeuralEncoder.to_pytorch` after fitting to obtain a
single ``torch.nn.Module`` containing the linear and nonlinear stages:

.. code-block:: python

   import torch

   model = encoder.to_pytorch()
   with torch.no_grad():
       Z = model(torch.as_tensor(X_test, dtype=torch.float32))

The returned module uses the fitted residual network itself. Its parameters are
trainable, and changes to that residual network are reflected in the estimator.

Using the stages independently
------------------------------

The stages are also available as standalone estimators when an experiment
requires direct access to intermediate representations:

.. code-block:: python

   from neural_encoder.linear import LinearEncoder
   from neural_encoder.nonlinear import NonlinearRefiner

   linear = LinearEncoder().fit(X_train, sample_ids=sample_ids)
   Z_linear = linear.transform(X_train)
   Z_pca = linear.transform_until(X_train, stage="pca")
   Z_denoised = linear.transform_until(X_train, stage="cross_view_ridge")

   refiner = NonlinearRefiner().fit(
       Z_linear,
       network_input=Z_pca,
       sample_ids=sample_ids,
   )
   Z_refined = refiner.transform(Z_linear, network_input=Z_pca)

See the :doc:`api/index` for all estimator parameters and fitted attributes.
