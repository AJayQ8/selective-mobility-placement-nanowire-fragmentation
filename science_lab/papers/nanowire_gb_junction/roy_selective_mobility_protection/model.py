"""Paper-local spatial-mobility extension of the frozen Roy solver."""

from __future__ import annotations

import numpy as np

from ..roy_2021_reproduction.model import (
    Array,
    PeriodicLattice,
    RoyModelParameters,
    RoyPseudospectralSolver,
)


class SpatialMobilityRoySolver(RoyPseudospectralSolver):
    """Multiply Roy mobility by a fixed factor inside the flux divergence."""

    def __init__(
        self,
        lattice: PeriodicLattice,
        parameters: RoyModelParameters,
        spatial_mobility_factor: Array,
        *,
        fft_workers: int = 1,
    ) -> None:
        super().__init__(lattice, parameters, fft_workers=fft_workers)
        factor = np.asarray(spatial_mobility_factor, dtype=np.float64)
        try:
            np.broadcast_shapes(factor.shape, lattice.shape)
        except ValueError as error:
            raise ValueError(
                "spatial mobility factor is not broadcastable to the lattice"
            ) from error
        if not np.isfinite(factor).all():
            raise ValueError("spatial mobility factor must be finite")
        if float(np.min(factor)) <= 0.0 or float(np.max(factor)) > 1.0:
            raise ValueError(
                "spatial mobility factor must lie strictly inside (0, 1]"
            )
        self.spatial_mobility_factor = np.array(
            factor, dtype=np.float64, copy=True, order="C"
        )
        self.spatial_mobility_factor.setflags(write=False)

    def mobility(self, c: Array) -> Array:
        mobility = super().mobility(c)
        np.multiply(
            mobility,
            self.spatial_mobility_factor,
            out=mobility,
        )
        return mobility

