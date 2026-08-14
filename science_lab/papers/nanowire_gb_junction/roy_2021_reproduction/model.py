"""Source-semantic CPU port of Roy et al.'s 90-degree nanowire case.

The implementation follows the released ``NEW DEG90`` CUDA path at commit
``a275dc0``.  It intentionally does not subclass the project's smooth-wire
finite-volume solver.

Real-space fields are float64, while every Fourier transform is quantized to
complex64, matching the released double/``cufftComplex`` split as closely as
SciPy's CPU FFT backend permits.  This is not a bitwise cuFFT reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import fft as scipy_fft


Array = np.ndarray
RELEASED_SOURCE_COMMIT = "a275dc0639c8f1e3dad795234e4a5ea7a7aeb5b3"
UINT32_MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class PeriodicLattice:
    """Periodic integer lattice used by the released Fourier solver."""

    shape: tuple[int, int, int]
    spacing: float = 0.5

    def __post_init__(self) -> None:
        if len(self.shape) != 3:
            raise ValueError("shape must have exactly three entries")
        if any(int(cells) != cells or cells < 4 for cells in self.shape):
            raise ValueError("each lattice dimension must be an integer >= 4")
        if any(cells % 2 for cells in self.shape):
            raise ValueError("the released centered geometry requires even sizes")
        if self.spacing <= 0.0:
            raise ValueError("spacing must be positive")

    @property
    def cell_count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def cell_volume(self) -> float:
        return float(self.spacing**3)

    @property
    def physical_lengths(self) -> tuple[float, float, float]:
        return tuple(float(cells * self.spacing) for cells in self.shape)


@dataclass(frozen=True)
class RoyModelParameters:
    """Released isotropic variable-mobility Cahn--Hilliard parameters."""

    mobility_prefactor: float = 1.0
    barrier: float = 1.0
    kappa: float = 1.0
    stabilizer_alpha: float = 0.5
    timestep: float = 1.0

    def __post_init__(self) -> None:
        if self.mobility_prefactor <= 0.0:
            raise ValueError("mobility_prefactor must be positive")
        if self.barrier <= 0.0 or self.kappa <= 0.0:
            raise ValueError("barrier and kappa must be positive")
        if self.stabilizer_alpha < 0.0:
            raise ValueError("stabilizer_alpha cannot be negative")
        if self.timestep <= 0.0:
            raise ValueError("timestep must be positive")


@dataclass(frozen=True)
class RoyDeg90Definition:
    """Complete released-input definition for one 90-degree case."""

    lattice: PeriodicLattice
    parameters: RoyModelParameters = RoyModelParameters()
    radius_1: int = 12
    radius_2: int = 12
    solid_composition: float = 1.0
    noise_amplitude: float = 1.0e-3
    noise_seed: int = 2292

    def __post_init__(self) -> None:
        nx, ny, nz = self.lattice.shape
        if self.radius_1 <= 0 or self.radius_2 <= 0:
            raise ValueError("radii must be positive")
        if self.solid_composition <= 0.0:
            raise ValueError("solid_composition must be positive")
        if self.noise_amplitude < 0.0:
            raise ValueError("noise_amplitude cannot be negative")
        if self.noise_seed < 0:
            raise ValueError("noise_seed cannot be negative")
        first_center = nx // 2
        second_center = first_center + self.radius_1 + self.radius_2
        if first_center - self.radius_1 < 0:
            raise ValueError("first wire does not fit in the thin direction")
        if second_center + self.radius_2 > nx:
            raise ValueError("second wire does not fit in the thin direction")
        if ny // 2 - self.radius_1 < 0:
            raise ValueError("first wire does not fit in the y direction")
        if nz // 2 - self.radius_2 < 0:
            raise ValueError("second wire does not fit in the z direction")


FROZEN_DEG90 = RoyDeg90Definition(
    lattice=PeriodicLattice((96, 768, 768), 0.5)
)


@dataclass(frozen=True)
class Deg90Masks:
    """Released strict-interior wire masks.

    ``first_wire`` and ``second_wire`` are broadcast views.  ``union`` owns
    the only full three-dimensional Boolean allocation.
    """

    first_wire: Array
    second_wire: Array
    union: Array


@dataclass(frozen=True)
class ReleasedNoiseLayout:
    """Footprint implied by the released ``Nx`` noise stride."""

    source_draw_count: int
    source_row_count: int
    source_row_length: int
    released_stride: int
    unique_target_prefix_length: int
    unique_target_fraction: float
    maximum_write_overlap: int
    standard_geometry_stride: int
    stride_mismatch_present: bool


@dataclass(frozen=True)
class FieldDiagnostics:
    """Cheap scalar checks that do not impose unpublished field bounds."""

    shape: tuple[int, int, int]
    dtype: str
    finite: bool
    minimum: float | None
    maximum: float | None
    mean: float | None
    mass: float | None
    center_gap_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GslTaus2:
    """Pure-Python GSL ``taus2`` stream with the documented six warmups."""

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed cannot be negative")
        seed_value = int(seed) & UINT32_MASK
        if seed_value == 0:
            seed_value = 1

        self.s1 = (69069 * seed_value) & UINT32_MASK
        if self.s1 < 2:
            self.s1 += 2
        self.s2 = (69069 * self.s1) & UINT32_MASK
        if self.s2 < 8:
            self.s2 += 8
        self.s3 = (69069 * self.s2) & UINT32_MASK
        if self.s3 < 16:
            self.s3 += 16
        for _ in range(6):
            self.get_uint32()

    @staticmethod
    def _tausworthe(
        state: int,
        left_a: int,
        right_b: int,
        mask: int,
        left_d: int,
    ) -> int:
        first = ((state & mask) << left_d) & UINT32_MASK
        second = ((((state << left_a) & UINT32_MASK) ^ state) >> right_b)
        return (first ^ second) & UINT32_MASK

    def get_uint32(self) -> int:
        self.s1 = self._tausworthe(
            self.s1, 13, 19, 4_294_967_294, 12
        )
        self.s2 = self._tausworthe(
            self.s2, 2, 25, 4_294_967_288, 4
        )
        self.s3 = self._tausworthe(
            self.s3, 3, 11, 4_294_967_280, 17
        )
        return int(self.s1 ^ self.s2 ^ self.s3)

    def uniform_pos(self) -> float:
        value = self.get_uint32()
        while value == 0:
            value = self.get_uint32()
        return float(value / 4_294_967_296.0)

    def uniform_pos_array(self, count: int) -> Array:
        if count < 0:
            raise ValueError("count cannot be negative")
        return np.fromiter(
            (self.uniform_pos() for _ in range(count)),
            dtype=np.float64,
            count=count,
        )


def source_wave_numbers(cells: int, spacing: float) -> Array:
    """Return the released signed modes, including a positive Nyquist mode."""

    if cells < 2 or cells % 2:
        raise ValueError("cells must be an even integer >= 2")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    indices = np.arange(cells, dtype=np.int64)
    signed = np.where(indices <= cells // 2, indices, indices - cells)
    return 2.0 * np.pi * signed.astype(np.float64) / (cells * spacing)


def strict_deg90_masks(
    lattice: PeriodicLattice,
    radius_1: int,
    radius_2: int,
) -> Deg90Masks:
    """Construct the exact strict-``<R^2`` masks from ``nanowire3D.cu``."""

    definition = RoyDeg90Definition(
        lattice=lattice,
        radius_1=radius_1,
        radius_2=radius_2,
        noise_amplitude=0.0,
    )
    nx, ny, nz = definition.lattice.shape
    i = np.arange(nx, dtype=np.int64)[:, None, None]
    j = np.arange(ny, dtype=np.int64)[None, :, None]
    k = np.arange(nz, dtype=np.int64)[None, None, :]
    first_center_x = nx // 2
    center_y = ny // 2
    second_center_x = first_center_x + radius_1 + radius_2
    center_z = nz // 2

    first_2d = (
        (i - first_center_x) ** 2 + (j - center_y) ** 2
        < radius_1**2
    )
    second_2d = (
        (i - second_center_x) ** 2 + (k - center_z) ** 2
        < radius_2**2
    )
    first = np.broadcast_to(first_2d, lattice.shape)
    second = np.broadcast_to(second_2d, lattice.shape)
    union = np.logical_or(first, second)
    return Deg90Masks(first_wire=first, second_wire=second, union=union)


def initialize_strict_deg90(
    definition: RoyDeg90Definition,
) -> tuple[Array, Deg90Masks]:
    """Return the released binary field before its one-time perturbation."""

    masks = strict_deg90_masks(
        definition.lattice, definition.radius_1, definition.radius_2
    )
    field = np.zeros(definition.lattice.shape, dtype=np.float64)
    field[masks.union] = definition.solid_composition
    return field, masks


def released_noise_layout(lattice: PeriodicLattice) -> ReleasedNoiseLayout:
    """Describe the overlapping target footprint of the released index."""

    nx, ny, nz = lattice.shape
    if nz < nx:
        raise ValueError(
            "the released contiguous-prefix description requires Nz >= Nx"
        )
    rows = nx * ny
    prefix = min(lattice.cell_count, (rows - 1) * nx + nz)
    maximum_overlap = int(np.ceil(nz / nx))
    return ReleasedNoiseLayout(
        source_draw_count=lattice.cell_count,
        source_row_count=rows,
        source_row_length=nz,
        released_stride=nx,
        unique_target_prefix_length=prefix,
        unique_target_fraction=float(prefix / lattice.cell_count),
        maximum_write_overlap=maximum_overlap,
        standard_geometry_stride=nz,
        stride_mismatch_present=bool(nx != nz),
    )


def apply_released_overlapping_noise(
    field: Array,
    amplitude: float,
    seed: int,
    *,
    chunk_size: int = 1_048_576,
) -> ReleasedNoiseLayout:
    """Apply GSL noise using the released overlapping ``Nx`` flat stride."""

    if field.ndim != 3:
        raise ValueError("field must be three-dimensional")
    if not field.flags.c_contiguous:
        raise ValueError("field must be C-contiguous")
    if field.dtype != np.float64:
        raise ValueError("field must use float64 real-space storage")
    if amplitude < 0.0:
        raise ValueError("amplitude cannot be negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    lattice = PeriodicLattice(tuple(int(v) for v in field.shape))
    layout = released_noise_layout(lattice)
    nx, ny, nz = lattice.shape
    rows = nx * ny
    rows_per_chunk = max(1, chunk_size // nz)
    flat = field.ravel()
    generator = GslTaus2(seed)

    for row_start in range(0, rows, rows_per_chunk):
        row_stop = min(rows, row_start + rows_per_chunk)
        row_count = row_stop - row_start
        draws = generator.uniform_pos_array(row_count * nz)
        draws = amplitude * (0.5 - draws)
        draws = draws.reshape(row_count, nz)
        # Preserve the released row-major addition order.  Adjacent segments
        # overlap whenever Nz > Nx, so changing this order would introduce a
        # small but avoidable floating-point difference.
        for local_row in range(row_count):
            source_row = row_start + local_row
            target_start = source_row * nx
            flat[target_start : target_start + nz] += draws[local_row]
    return layout


def digital_gap_diagnostics(
    masks: Deg90Masks,
    lattice: PeriodicLattice,
) -> dict[str, Any]:
    """Measure the strict-mask separation along the shortest center line."""

    nx, ny, nz = lattice.shape
    center_y = ny // 2
    center_z = nz // 2
    first_line = masks.first_wire[:, center_y, center_z]
    second_line = masks.second_wire[:, center_y, center_z]
    first_indices = np.flatnonzero(first_line)
    second_indices = np.flatnonzero(second_line)
    if not first_indices.size or not second_indices.size:
        raise ValueError("center-line masks must each contain solid samples")
    first_end = int(first_indices.max())
    second_start = int(second_indices.min())
    empty_indices = list(range(first_end + 1, second_start))
    return {
        "first_wire_last_solid_index": first_end,
        "second_wire_first_solid_index": second_start,
        "empty_indices_between_wires": empty_indices,
        "empty_node_count": len(empty_indices),
        "nearest_solid_index_separation": second_start - first_end,
        "digital_gap_length": float(len(empty_indices) * lattice.spacing),
        "continuum_center_separation": float(
            (np.mean(second_indices) - np.mean(first_indices))
            * lattice.spacing
        ),
        "center_line_union_connected_at_t0": bool(
            second_start == first_end + 1
        ),
    }


def field_diagnostics(
    field: Array,
    lattice: PeriodicLattice,
    masks: Deg90Masks | None = None,
) -> FieldDiagnostics:
    """Return finite scalar checks without inventing a literature field gate."""

    if field.shape != lattice.shape:
        raise ValueError("field shape does not match lattice")
    finite = bool(np.isfinite(field).all())
    if not finite:
        return FieldDiagnostics(
            shape=tuple(int(v) for v in field.shape),
            dtype=str(field.dtype),
            finite=False,
            minimum=None,
            maximum=None,
            mean=None,
            mass=None,
            center_gap_value=None,
        )
    nx, ny, nz = lattice.shape
    gap_index = nx // 2 + 12
    if masks is not None:
        gap = digital_gap_diagnostics(masks, lattice)
        empty = gap["empty_indices_between_wires"]
        if empty:
            gap_index = int(empty[len(empty) // 2])
    return FieldDiagnostics(
        shape=tuple(int(v) for v in field.shape),
        dtype=str(field.dtype),
        finite=True,
        minimum=float(np.min(field)),
        maximum=float(np.max(field)),
        mean=float(np.mean(field)),
        mass=float(np.sum(field, dtype=np.float64) * lattice.cell_volume),
        center_gap_value=float(field[gap_index, ny // 2, nz // 2]),
    )


class RoyPseudospectralSolver:
    """Periodic mixed-precision translation of the released CUDA update."""

    def __init__(
        self,
        lattice: PeriodicLattice,
        parameters: RoyModelParameters,
        *,
        fft_workers: int = 1,
    ) -> None:
        if fft_workers == 0:
            raise ValueError("fft_workers cannot be zero")
        self.lattice = lattice
        self.parameters = parameters
        self.fft_workers = fft_workers
        kx = source_wave_numbers(lattice.shape[0], lattice.spacing)
        ky = source_wave_numbers(lattice.shape[1], lattice.spacing)
        kz = source_wave_numbers(lattice.shape[2], lattice.spacing)
        self._wave_numbers = (
            kx[:, None, None],
            ky[None, :, None],
            kz[None, None, :],
        )
        self._k2 = (
            self._wave_numbers[0] ** 2
            + self._wave_numbers[1] ** 2
            + self._wave_numbers[2] ** 2
        )
        self._denominator = (
            1.0
            + parameters.stabilizer_alpha
            * parameters.timestep
            * parameters.kappa
            * self._k2**2
        )
        self.last_inverse_imaginary_linf = 0.0

    def mobility(self, c: Array) -> Array:
        return self.parameters.mobility_prefactor * np.sqrt(
            np.abs(c - c * c)
        )

    def _forward_complex64(self, real_field: Array) -> Array:
        real32 = np.asarray(real_field, dtype=np.float32)
        transformed = scipy_fft.fftn(
            real32,
            workers=self.fft_workers,
            overwrite_x=True,
        )
        return np.asarray(transformed, dtype=np.complex64)

    def _composition_and_potential_spectra(
        self, c: Array
    ) -> tuple[Array, Array]:
        if c.shape != self.lattice.shape:
            raise ValueError("field shape does not match lattice")
        if not np.isfinite(c).all():
            raise ValueError("field must be finite")
        p = self.parameters
        c_hat = self._forward_complex64(c)
        bulk = 2.0 * p.barrier * c * (1.0 - c) * (1.0 - 2.0 * c)
        mu_hat = self._forward_complex64(bulk)
        # ``compute_r.cu`` evaluates each real and imaginary component as
        # double arithmetic, then assigns once to ``cufftComplex``.  Work in
        # bounded chunks so the full-grid CPU port preserves that one-round
        # behavior without allocating another full complex128 field.
        flat_c = c_hat.ravel()
        flat_mu = mu_hat.ravel()
        flat_k2 = self._k2.ravel()
        chunk_size = 1_048_576
        for start in range(0, flat_c.size, chunk_size):
            stop = min(flat_c.size, start + chunk_size)
            scaled_k2 = p.kappa * flat_k2[start:stop]
            real = scaled_k2 * flat_c[start:stop].real
            real += flat_mu[start:stop].real
            flat_mu[start:stop].real = real
            imaginary = scaled_k2 * flat_c[start:stop].imag
            imaginary += flat_mu[start:stop].imag
            flat_mu[start:stop].imag = imaginary
        return c_hat, mu_hat

    def chemical_potential_spectrum(self, c: Array) -> Array:
        _, mu_hat = self._composition_and_potential_spectra(c)
        return mu_hat

    def free_energy(self, c: Array) -> float:
        """Evaluate the published isotropic free energy by spectral Parseval."""

        if c.shape != self.lattice.shape:
            raise ValueError("field shape does not match lattice")
        p = self.parameters
        bulk = np.sum(
            p.barrier * c**2 * (1.0 - c) ** 2,
            dtype=np.float64,
        )
        c_hat = self._forward_complex64(c)
        spectral_norm = (
            c_hat.real.astype(np.float64) ** 2
            + c_hat.imag.astype(np.float64) ** 2
        )
        gradient = (
            np.sum(self._k2 * spectral_norm, dtype=np.float64)
            / self.lattice.cell_count
        )
        return float(
            self.lattice.cell_volume
            * (bulk + 0.5 * p.kappa * gradient)
        )

    def propose_step(self, c: Array, timestep: float | None = None) -> Array:
        """Apply Eq. 11 exactly: the stabilizer divides only the increment."""

        p = self.parameters
        dt = p.timestep if timestep is None else float(timestep)
        if dt <= 0.0:
            raise ValueError("timestep must be positive")
        if dt == p.timestep:
            denominator = self._denominator
        else:
            denominator = (
                1.0
                + p.stabilizer_alpha * dt * p.kappa * self._k2**2
            )

        c_hat, mu_hat = self._composition_and_potential_spectra(c)
        mobility = self.mobility(c)
        increment_real = np.zeros(self.lattice.shape, dtype=np.float64)
        increment_imag = np.zeros(self.lattice.shape, dtype=np.float64)

        for wave_number in self._wave_numbers:
            gradient_hat = np.empty_like(mu_hat)
            np.multiply(
                mu_hat,
                wave_number,
                out=gradient_hat,
                casting="unsafe",
            )
            gradient_hat *= np.complex64(1j)
            gradient_complex = scipy_fft.ifftn(
                gradient_hat,
                workers=self.fft_workers,
                overwrite_x=True,
            )
            flux = np.empty(self.lattice.shape, dtype=np.float32)
            np.multiply(
                mobility,
                gradient_complex.real,
                out=flux,
                casting="unsafe",
            )
            flux_hat = scipy_fft.fftn(
                flux,
                workers=self.fft_workers,
                overwrite_x=True,
            )
            flux_hat = np.asarray(flux_hat, dtype=np.complex64)
            increment_real -= wave_number * flux_hat.imag
            increment_imag += wave_number * flux_hat.real

        updated_hat = np.empty_like(c_hat)
        updated_hat.real = c_hat.real + dt * increment_real / denominator
        updated_hat.imag = c_hat.imag + dt * increment_imag / denominator
        updated_complex = scipy_fft.ifftn(
            updated_hat,
            workers=self.fft_workers,
            overwrite_x=True,
        )
        self.last_inverse_imaginary_linf = float(
            np.max(np.abs(updated_complex.imag))
        )
        return np.asarray(updated_complex.real, dtype=np.float64)
