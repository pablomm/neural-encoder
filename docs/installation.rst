Installation
============

neural-encoder requires Python 3.12 or later. Install the released package
from PyPI:

.. code-block:: console

   pip install neural-encoder

To install the current development version directly from GitHub:

.. code-block:: console

   pip install git+https://github.com/pablomm/neural-encoder.git

Optional dependencies
---------------------

The core installation includes NumPy, SciPy, scikit-learn, PyTorch, and pandas.
Additional dependency groups are available for specific workflows:

.. code-block:: console

   pip install "neural-encoder[utils]"   # datasets, Pillow, and tqdm
   pip install "neural-encoder[wandb]"   # Weights & Biases logging

For local development, clone the repository and install it in editable mode.
The ``docs`` extra installs Sphinx and the documentation theme:

.. code-block:: console

   git clone https://github.com/pablomm/neural-encoder.git
   cd neural-encoder
   pip install -e ".[docs]"

Build the documentation from the ``docs`` directory:

.. code-block:: console

   cd docs
   make html

The generated site is written to ``docs/_build/html``. On macOS, ``make docs``
also opens it in the default browser.
