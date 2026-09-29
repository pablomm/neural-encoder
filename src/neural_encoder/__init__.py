"""Embeddings from repeated measurements and multiview data."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__version__ = "0.0.1"

__getattr__, __dir__, __all__ = lazy.attach(
    __name__, submod_attrs={"neural_encoder": ["NeuralEncoder", "NeuralEncoderModule"]},
)

if TYPE_CHECKING:
    from .neural_encoder import NeuralEncoder, NeuralEncoderModule
