.. image:: assets/neural-encoder.svg
   :alt: neural-encoder
   :align: center
   :class: only-light
   :width: 560px

.. image:: assets/neural-encoder-dark.svg
   :alt: neural-encoder
   :align: center
   :class: only-dark
   :width: 560px

neural-encoder
==============

.. raw:: html

   <p class="badges">
     <a href="https://arxiv.org/abs/2605.20496"><img alt="NeurIPS 2026" src="https://img.shields.io/badge/NeurIPS-2026-8A2BE2"></a>
     <a href="https://www.python.org/"><img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white"></a>
     <a href="https://pypi.org/project/neural-encoder/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/neural-encoder.svg"></a>
     <a href="https://github.com/pablomm/neural-encoder/actions/workflows/test.yml"><img alt="Run Tests" src="https://github.com/pablomm/neural-encoder/actions/workflows/test.yml/badge.svg"></a>
     <a href="https://neural-encoder.readthedocs.io/en/latest/"><img alt="Documentation Status" src="https://readthedocs.org/projects/neural-encoder/badge/?version=latest"></a>
     <a href="https://github.com/pablomm/neural-encoder/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
     <a href="https://github.com/pablomm/neural-encoder"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-pablomm%2Fneural--encoder-181717?logo=github"></a>
   </p>

**neural-encoder** learns embeddings from repeated measurements and multiview
data. It was designed for single-trial fMRI responses and other
high-dimensional, noisy measurements in which several observations correspond
to the same underlying sample.

The encoder learns the signal shared across repetitions or views and maps each
measurement to a low-dimensional representation. Repeated measurements are
used during training; after fitting, individual measurements can be encoded
independently.

The method is based on `Platonic Representations in the Human Brain:
Unsupervised Recovery of Universal Geometry
<https://arxiv.org/abs/2605.20496>`_. It combines feature reliability
weighting, PCA, cross-view ridge denoising, distilled multiset canonical
correlation analysis (MCCA), reliability weighting of the embedding
dimensions, and nonlinear residual refinement.

Quick start
-----------

Fit from a matrix containing all repeated measurements. ``sample_ids`` gives
the identity of the underlying sample represented by each row:

.. code-block:: python

   from neural_encoder import NeuralEncoder

   encoder = NeuralEncoder()
   Z_train = encoder.fit_transform(X_train, sample_ids=sample_ids)
   Z_test = encoder.transform(X_test)

You can also fit from separate, aligned views. Corresponding rows of ``X1``,
``X2``, and ``X3`` must describe the same sample:

.. code-block:: python

   encoder = NeuralEncoder()
   encoder.fit_views([X1, X2, X3])

   Z1, Z2, Z3 = (encoder.transform(X) for X in (X1, X2, X3))

See :doc:`usage` for preprocessing, configuration, evaluation, and PyTorch
export.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   concepts
   usage
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   citation
   acknowledgements

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
