# neural-encoder

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-8A2BE2)](https://arxiv.org/abs/2605.20496)

**neural-encoder** is a Python package for learning embeddings from repeated measurements and multiview data. It provides a linear encoder and an optional nonlinear refinement stage through a scikit-learn-style API.

The methods are based on [*Platonic Representations in the Human Brain: Unsupervised Recovery of Universal Geometry*](https://arxiv.org/abs/2605.20496). Developed for fMRI, they apply more generally to repeated or multiview measurements with a common feature space.

## Installation

```bash
pip install git+https://github.com/pablomm/neural-encoder.git
```

## Usage

`NeuralEncoder` combines feature reliability weighting, PCA, distilled multiset canonical correlation analysis (MCCA), and nonlinear residual refinement. By default, the residual network receives the MCCA embedding, reproducing the paper architecture. It can instead receive PCA scores while still learning a correction in the MCCA embedding space.

### Repeated measurements

`X_train` contains measurements as rows and features as columns. `sample_ids` identifies which rows are measurements of the same sample. Repetitions are assigned by their order of appearance within each sample.

```python
from neural_encoder import NeuralEncoder

encoder = NeuralEncoder(
    n_components_pca=64,
    n_components_mcca=16,
    refiner_kwargs={
        "network_kwargs": {"hidden_dim": 128},
        "steps": 2000,
        "batch_size": 256,
    },
    random_state=42,
)
Z_train = encoder.fit_transform(
    X_train, sample_ids=sample_ids,
)
Z_test = encoder.transform(X_test)
```

### Aligned views

When the measurements are already separated into aligned views, corresponding rows of `X1`, `X2`, and `X3` represent the same sample.

```python
from neural_encoder import NeuralEncoder

encoder = NeuralEncoder(
    n_components_pca=64,
    n_components_mcca=16,
    refinement_input_stage="pca",
    n_components_pca_refinement=128,
    refiner_kwargs={
        "network_kwargs": {"hidden_dim": 128},
        "steps": 2000,
        "batch_size": 256,
    },
    random_state=42,
)
encoder.fit_views([X1, X2, X3])

Z1, Z2, Z3 = (encoder.transform(X) for X in (X1, X2, X3))
Z_test = encoder.transform(X_test)
```

Preprocessing is applied separately. The complete fitted architecture can be exported to PyTorch:

```python
model = encoder.to_pytorch()
```

To drive the residual branch from PCA scores, set `refinement_input_stage="pca"`. An optional `n_components_pca_refinement` selects a separate PCA dimension; when omitted or equal to `n_components_pca`, both branches share one fitted PCA. Network, loss, and training settings can be configured through `refiner_kwargs`.

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
