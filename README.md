<p align="center">
  <a href="https://github.com/pablomm/neural-encoder">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/pablomm/neural-encoder/raw/main/docs/assets/neural-encoder-dark.svg">
        <img alt="neural-encoder" src="https://github.com/pablomm/neural-encoder/raw/main/docs/assets/neural-encoder.svg">
    </picture>
  </a>
</p>

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-8A2BE2)](https://arxiv.org/abs/2605.20496)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/neural-encoder.svg)](https://pypi.org/project/neural-encoder/)
[![Run Tests](https://github.com/pablomm/neural-encoder/actions/workflows/test.yml/badge.svg)](https://github.com/pablomm/neural-encoder/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/neural-encoder/badge/?version=latest)](https://neural-encoder.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/pablomm/neural-encoder/blob/main/LICENSE)

**neural-encoder** is a Python package for learning embeddings from repeated measurements and multiview data, designed for fMRI data or, more generally, high-dimensional noisy data. It provides linear and nonlinear encoders.

The methods are based on [*Platonic Representations in the Human Brain: Unsupervised Recovery of Universal Geometry*](https://arxiv.org/abs/2605.20496). Originally developed for single-trial fMRI responses, they apply more generally to repeated or multiview measurements with a common feature space. They project noisy observations into a low-dimensional embedding that captures the signal shared across repetitions or views. The methods require repeated measurements during training, but work with individual measurements at inference time; multiple repetitions are not required for inference.

**neural-encoder** is developed by the [Dynamics of Memory Formation (DMF)](https://www.ub.edu/brainvitge/groups/memory_formation/) group at the University of Barcelona.

## Installation

Via PyPI:

```bash
pip install neural-encoder
```

or directly from GitHub:

```bash
pip install git+https://github.com/pablomm/neural-encoder.git
```

## Encoding

The main class is `NeuralEncoder`, which combines feature reliability weighting, PCA, cross-view ridge denoising, distilled multiset canonical correlation analysis (MCCA), and nonlinear residual refinement.

### Repeated measurements

It can be used with a single `X_train` matrix containing all repetitions and a `sample_ids` array containing the stimulus ID for each row. Rows with the same ID are measurements of the same sample.

```python
from neural_encoder import NeuralEncoder

encoder = NeuralEncoder()
Z_train = encoder.fit_transform(
    X_train, sample_ids=sample_ids,
)
Z_test = encoder.transform(X_test)
```

### Aligned views

Alternatively, pass the repeated measurements as separate, aligned views. Corresponding rows of `X1`, `X2`, and `X3` represent the same sample.

```python
from neural_encoder import NeuralEncoder

encoder = NeuralEncoder()
encoder.fit_views([X1, X2, X3])

Z1, Z2, Z3 = (encoder.transform(X) for X in (X1, X2, X3))
Z_test = encoder.transform(X_test)
```

Preprocessing is applied separately. The complete fitted architecture can be exported to PyTorch:

```python
model = encoder.to_pytorch()
```

See the [documentation](https://neural-encoder.readthedocs.io/) for the full API and usage guide.

## License

**neural-encoder** is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.



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
