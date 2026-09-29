# neural-encoder

**neural-encoder** is a Python package for learning embeddings from repeated measurements and multiview data. It provides a linear encoder and an optional nonlinear refinement stage through a scikit-learn-style API.

The methods are based on [*Platonic Representations in the Human Brain: Unsupervised Recovery of Universal Geometry*](https://arxiv.org/abs/2605.20496). Developed for fMRI, they apply more generally to repeated or multiview measurements with a common feature space.

## Installation

```bash
pip install git+https://github.com/pablomm/neural-encoder.git
```

## Usage

`NeuralEncoder` combines feature reliability weighting, PCA, distilled multiset canonical correlation analysis (MCCA), and nonlinear residual refinement. By default, the residual network receives the MCCA embedding, reproducing the paper architecture. It can instead receive PCA scores while still learning a correction in the MCCA embedding space.

`X_train` contains measurements as rows and features as columns. `sample_ids` and `view_ids` identify the sample and repetition or view associated with each row. Preprocessing is applied separately.

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
    X_train, sample_ids=sample_ids, view_ids=view_ids,
)
Z_test = encoder.transform(X_test)

# Complete fitted architecture for use in PyTorch.
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
