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

   z = D\,(W_{\mathrm{MCCA}}\,R\,P_{\mathrm{PCA}}\,W_{\mathrm{rel}}\,x + b).

``FeatureReweighting`` first scales each feature by its reliability across
views. By default, reliability is the mean leave-one-view-out Pearson
correlation: for each view, a feature is correlated with the mean of that
feature in the other views. Negative values are clipped before the selected
weighting rule is applied.

PCA then reduces the weighted feature space. ``CrossViewRidge`` (:math:`R`)
denoises the PCA scores: a ridge regression, with its penalty chosen by
leave-one-out cross-validation, predicts from each measurement the mean of the
other views of the same sample. Single measurements are dominated by
view-specific noise, so the fitted map shrinks the directions that do not
repeat across views. ``DistilledMCCA`` then fits MCCA to the aligned, denoised
representations and distils its view-specific projections into a single
least-squares projection. This makes the fitted encoder applicable when
the view identity is unknown or when only one measurement is available.

Finally, the output dimensions are reweighted (:math:`D`, a diagonal matrix).
MCCA dimensions have similar variance but very different reliability: the
first ones are highly reproducible across views, while the last ones are
mostly noise. Cosine similarities between embeddings would otherwise count
every dimension equally. The default output stage measures the reliability
:math:`r` of each dimension as the mean correlation between pairs of single
views, converts it to the signal-to-noise ratio :math:`r / (1 - r)`, and
multiplies each dimension by the square root of that ratio, rescaled to unit
root mean square. Each dimension then contributes to inner products in
proportion to its signal-to-noise ratio, which is the optimal weighting for
comparing noisy measurements of a shared signal under independent Gaussian
noise. It improves retrieval and makes the result much less sensitive to the
MCCA dimensionality. Pass ``output_reweighting=None`` (``LinearEncoder``) or
``output_reweighting=False`` (``NeuralEncoder``) for the unweighted embedding.

By default the reliabilities are measured on the training embeddings. These
are the measurements MCCA was fitted to, so the estimates are optimistic,
especially for the last dimensions. ``output_reweighting_cv=K`` estimates them
by K-fold cross-validation over samples instead: the stages up to PCA stay
fitted on all samples, the cross-view ridge and MCCA are refitted on K-1
folds, and the reliability of each refitted dimension on the held-out fold is
transferred to the corresponding fitted dimension. Dimensions are matched one
to one by maximal absolute correlation on the held-out samples (Hungarian
assignment, ``output_reweighting_matching="hungarian"``), by position
(``"index"``), or softly by squared correlations (``"soft"``). The
cross-validated estimates are much closer to the reliability on new data, but
with square-root weights the gain in retrieval is small, so the option is off
by default.

All stages are affine, so they compose into the single projection above. The
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
size of the correction. Since :math:`z` is already reliability-weighted by
default, the refined output is left unweighted; ``NonlinearRefiner`` can
optionally reweight its outputs in the same way
(``output_reweighting="default"``), with in-sample or cross-validated
(``output_reweighting_cv``, one extra training per fold) reliabilities.

``NeuralEncoder`` combines both stages. It follows the scikit-learn estimator
interface for fitting and NumPy inference and can export the fitted computation
as one PyTorch module.

Preprocessing
-------------

Preprocessing is intentionally separate from the encoder. The provided
``BetaPreprocessor`` supports clipping, scaling, feature or row
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
   :class:`neural_encoder.preprocessing.BetaPreprocessor`.
