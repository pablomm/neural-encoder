"""Linear estimators for learning representations from multiple views."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__, submod_attrs={
        "feature_reweighting": ["FeatureReweighting"],
        "distilled_mcca": ["DistilledMCCA"],
        "cross_view_ridge": ["CrossViewRidge"],
        "linear_encoder": ["LinearEncoder"],
    },
)

if TYPE_CHECKING:
    from .feature_reweighting import FeatureReweighting
    from .distilled_mcca import DistilledMCCA
    from .cross_view_ridge import CrossViewRidge
    from .linear_encoder import LinearEncoder

__all__ = ["FeatureReweighting", "DistilledMCCA", "CrossViewRidge", "LinearEncoder"]
