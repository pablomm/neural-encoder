# neural-encoder

**neural-encoder** is a Python package for learning embeddings from repeated measurements and multiview data. It provides a linear encoder and an optional nonlinear refinement stage through a scikit-learn-style API.

The methods are based on [*Platonic Representations in the Human Brain: Unsupervised Recovery of Universal Geometry*](https://arxiv.org/abs/2605.20496). Developed for fMRI, they apply more generally to repeated or multiview measurements with a common feature space.

## Installation

```bash
pip install git+https://github.com/pablomm/neural-encoder.git
```

## Examples

### Linear encoder

`LinearEncoder` combines feature reliability weighting, PCA, and distilled multiset canonical correlation analysis (MCCA). It learns a single projection that emphasizes structure shared across views and can be applied to individual measurements.

`X_train` contains measurements as rows and features as columns. `sample_ids` and `view_ids` identify the sample and repetition or view associated with each row.

```python
from neural_encoder.linear import LinearEncoder

encoder = LinearEncoder(
    pca_kwargs={"n_components": 64},
    distilled_mcca_kwargs={"n_components": 16},
)
Z_train = encoder.fit_transform(
    X_train, sample_ids=sample_ids, view_ids=view_ids,
)
Z_test = encoder.transform(X_test)
```

### Nonlinear refinement

`NonlinearRefiner` learns a residual transformation, `f(z) = z + alpha * g(z)`, using a multiview contrastive objective and a trainable residual coefficient. It accepts embeddings from the linear encoder or any other fixed representation.

```python
from neural_encoder.nonlinear import NonlinearRefiner

refiner = NonlinearRefiner(
    network_kwargs={"hidden_dim": 128},
    steps=2000,
    batch_size=256,
)
Z_train_refined = refiner.fit_transform(
    Z_train, sample_ids=sample_ids, view_ids=view_ids,
)
Z_test_refined = refiner.transform(Z_test)
```

Custom PyTorch networks and losses can be supplied to `NonlinearRefiner`. Both estimators expose their fitted transformations as PyTorch modules through `to_torch()`.

## Citation

If you use this package in your research, please cite:

```bibtex
@misc{marcosmanchon2026platonic,
  title = {Platonic Representations in the Human Brain: Unsupervised Recovery of Universal Geometry},
  author = {Pablo Marcos-Manchón and Rishi Jha and Lluís Fuentemilla},
  year = {2026},
  eprint = {2605.20496},
  archivePrefix = {arXiv},
  primaryClass = {q-bio.NC},
  url = {https://arxiv.org/abs/2605.20496}
}
```
