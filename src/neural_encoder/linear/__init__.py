"""Linear estimators for learning representations from multiple views."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__, submod_attrs={
        "feature_reweighting": ["FeatureReweighting"],
        "distilled_mcca": ["DistilledMCCA"],
        "gram_pca": ["GramPCA"],
        "cross_view_ridge": ["CrossViewRidge"],
        "linear_encoder": ["LinearEncoder"],
        "chunked_linear_encoder": ["ChunkedLinearEncoder"],
    },
)

if TYPE_CHECKING:
    from .feature_reweighting import FeatureReweighting
    from .distilled_mcca import DistilledMCCA
    from .gram_pca import GramPCA
    from .cross_view_ridge import CrossViewRidge
    from .linear_encoder import LinearEncoder
    from .chunked_linear_encoder import ChunkedLinearEncoder

__all__ = ["FeatureReweighting", "DistilledMCCA", "GramPCA", "CrossViewRidge", "LinearEncoder", "ChunkedLinearEncoder"]
