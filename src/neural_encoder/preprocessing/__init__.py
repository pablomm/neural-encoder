"""Transformers for preparing measurements before encoding."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__, submod_attrs={
        "beta_preprocessor": ["BetaPreprocessor"],
        "session_standard_scaler": ["SessionStandardScaler"],
        "confound_remover": ["ConfoundRemover"],
    },
)

if TYPE_CHECKING:
    from .beta_preprocessor import BetaPreprocessor
    from .session_standard_scaler import SessionStandardScaler
    from .confound_remover import ConfoundRemover

__all__ = ["BetaPreprocessor", "SessionStandardScaler", "ConfoundRemover"]
