"""PyTorch components and estimators for nonlinear residual refinement."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__, submod_attrs={
        "networks": ["ResidualNetwork", "ResidualMLP"],
        "losses": ["symmetric_info_nce", "cosine_pull", "MultiViewContrastiveLoss"],
        "nonlinear_refiner": ["NonlinearRefiner"],
    },
)

if TYPE_CHECKING:
    from .networks import ResidualNetwork, ResidualMLP
    from .losses import symmetric_info_nce, cosine_pull, MultiViewContrastiveLoss
    from .nonlinear_refiner import NonlinearRefiner
