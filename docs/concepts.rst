Encoder design
==============

The encoder is trained from repeated observations of the same samples. These
may be supplied as aligned view matrices or as one matrix together with sample
identifiers. All observations share the same input feature space.

Linear encoder
--------------

The linear stage learns a single projection that can be applied to any new
measurement:

.. math::

   z = W_{\mathrm{MCCA}}\,R\,P_{\mathrm{PCA}}\,W_{\mathrm{rel}}\,x + b.

``FeatureReweighting`` first scales each feature by its reliability across
views. Reliability is the mean leave-one-view-out Pearson correlation: for
each view, a feature is correlated with the mean of that feature in the other
views. Negative values are clipped before the selected weighting rule is
applied.

PCA then reduces the weighted feature space. ``CrossViewRidge`` (:math:`R`)
denoises the PCA scores: a ridge regression, with its penalty chosen by
leave-one-out cross-validation, predicts from each measurement the mean of the
other views of the same sample. Single measurements are dominated by
view-specific noise, so the fitted map shrinks the directions that do not
repeat across views. ``DistilledMCCA`` then fits MCCA to the aligned, denoised
representations and distils its view-specific projections into a single
least-squares projection. This makes the fitted encoder applicable when
the view identity is unknown or when only one measurement is available. All
stages are affine, so they compose into the single projection above. The
denoising stage can be disabled (``cross_view_ridge=False`` in
``NeuralEncoder``, ``cross_view_ridge=None`` in ``LinearEncoder``) to fit MCCA
directly on the PCA scores, as in the paper.

Nonlinear refinement
--------------------

The nonlinear stage learns a residual correction with a multiview contrastive
objective:

.. math::

   \hat z = z + \alpha\,g(s).

Here, :math:`z` is the distilled MCCA embedding and :math:`g` is a neural
network. By default :math:`s=z`, as in the architecture used in the paper.
The residual branch can instead receive (non-denoised) PCA scores, optionally from a PCA with
a different dimensionality. The trainable scalar :math:`\alpha` controls the
size of the correction.

``NeuralEncoder`` combines both stages. It follows the scikit-learn estimator
interface for fitting and NumPy inference and can export the fitted computation
as one PyTorch module.

Preprocessing
-------------

Preprocessing is intentionally separate from the encoder. The provided
``MeasurementPreprocessor`` supports clipping, scaling, feature or row
centering, feature standardization, and row normalization. When measurements
are acquired in sessions, ``SessionStandardScaler`` standardizes each feature
within each session, using statistics learned from that session's training
measurements. Fit preprocessing on the training measurements and reuse it for
validation or test data.

Confound regression is also separate. ``ConfoundRemover`` learns a
ridge-regularized mapping from a confound design matrix to the measurements and
subtracts the predicted component.

.. seealso::

   :class:`neural_encoder.NeuralEncoder`,
   :class:`neural_encoder.linear.LinearEncoder`,
   :class:`neural_encoder.nonlinear.NonlinearRefiner`, and
   :class:`neural_encoder.utils.MeasurementPreprocessor`.
