"""Exact periodic factor-two Fourier interpolation and validation metrics."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
from scipy import fft as scipy_fft
from scipy.signal import resample


Array = np.ndarray


def periodic_fourier_prolong_factor2(
    coarse: Array,
    *,
    workers: int = 1,
) -> Array:
    """Interpolate a real periodic 3-D node field onto coincident half steps.

    ``scipy.signal.resample`` implements the even-length Nyquist split needed
    for real trigonometric interpolation. Applying it axis by axis avoids a
    bespoke frequency-padding convention and limits the largest persistent
    allocation to the current intermediate plus the next one.
    """

    values = np.asarray(coarse)
    if values.ndim != 3:
        raise ValueError("coarse field must be three-dimensional")
    if any(size < 4 or size % 2 for size in values.shape):
        raise ValueError("every coarse dimension must be even and at least four")
    if workers == 0:
        raise ValueError("workers cannot be zero")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("coarse field must be real floating point")
    if not np.isfinite(values).all():
        raise ValueError("coarse field must be finite")

    current = np.asarray(values, dtype=np.float64, order="C")
    with scipy_fft.set_workers(workers):
        for axis, coarse_size in enumerate(values.shape):
            refined = resample(current, 2 * coarse_size, axis=axis)
            if np.iscomplexobj(refined):
                imaginary_linf = float(np.max(np.abs(refined.imag)))
                if imaginary_linf > 5.0e-13:
                    raise RuntimeError(
                        "periodic interpolation produced a non-negligible "
                        f"imaginary part: {imaginary_linf}"
                    )
                refined = refined.real
            if current is not values:
                del current
            current = np.asarray(refined, dtype=np.float64, order="C")
            gc.collect()
    expected = tuple(2 * size for size in values.shape)
    if current.shape != expected or not np.isfinite(current).all():
        raise RuntimeError("invalid factor-two Fourier interpolation output")
    return current


def prolongation_metrics(
    coarse: Array,
    fine: Array,
    *,
    coarse_spacing: float,
    fine_spacing: float,
) -> dict[str, Any]:
    """Measure coincident-node, mean, and physical-mass preservation."""

    coarse_values = np.asarray(coarse, dtype=np.float64)
    fine_values = np.asarray(fine, dtype=np.float64)
    expected = tuple(2 * size for size in coarse_values.shape)
    if fine_values.shape != expected:
        raise ValueError(f"fine shape must be {expected}, got {fine_values.shape}")
    if coarse_spacing <= 0.0 or fine_spacing <= 0.0:
        raise ValueError("spacings must be positive")
    if not np.isclose(
        coarse_spacing,
        2.0 * fine_spacing,
        rtol=0.0,
        atol=32.0 * np.finfo(float).eps,
    ):
        raise ValueError("fine spacing must be exactly half the coarse spacing")
    coincident = fine_values[::2, ::2, ::2]
    difference = coincident - coarse_values
    coarse_mean = float(np.mean(coarse_values, dtype=np.float64))
    fine_mean = float(np.mean(fine_values, dtype=np.float64))
    coarse_mass = float(
        np.sum(coarse_values, dtype=np.float64) * coarse_spacing**3
    )
    fine_mass = float(np.sum(fine_values, dtype=np.float64) * fine_spacing**3)
    return {
        "coarse_shape": list(coarse_values.shape),
        "fine_shape": list(fine_values.shape),
        "coincident_node_max_abs": float(np.max(np.abs(difference))),
        "coincident_node_rmse": float(np.sqrt(np.mean(difference**2))),
        "coarse_mean": coarse_mean,
        "fine_mean": fine_mean,
        "mean_abs_difference": abs(fine_mean - coarse_mean),
        "coarse_physical_mass": coarse_mass,
        "fine_physical_mass": fine_mass,
        "mass_relative_difference": abs(fine_mass - coarse_mass)
        / max(abs(coarse_mass), np.finfo(float).tiny),
        "no_clipping_or_renormalization_applied": True,
    }
