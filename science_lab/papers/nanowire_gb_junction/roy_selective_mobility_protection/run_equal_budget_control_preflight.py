#!/usr/bin/env python3
"""Run the bounded K2/K3 equal-budget numerical preflight."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..roy_2021_reproduction import model as roy_model
from ..roy_2021_reproduction.model import FROZEN_DEG90
from ..roy_2021_reproduction.run_preflight import contact_metrics
from ..roy_2021_reproduction.run_t1000 import four_arm_metrics
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
from ..roy_fixed_gb_bridge.metrics import array_fingerprint
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel import (
    run_t100_conditioned_production as production_helper,
)
from ..roy_gb_junction_sentinel.storage import atomic_json
from . import equal_budget_controls
from . import geometry as selective_geometry
from . import model as selective_model
from .equal_budget_controls import derive_equal_budget_controls
from .model import SpatialMobilityRoySolver
from .run_t100_tubular_position_diagnostic import INWARD_GEOMETRY


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
CONTRACT_PATH = DIRECTORY / "EQUAL_BUDGET_CONTROLS.md"
DEFAULT_OUTPUT = (
    DIRECTORY / "results" / "equal_budget_control_preflight_v1"
)

START_STEP = 100
ACCEPTED_STEPS_PER_CASE = 16
FFT_WORKERS = 12
RUNTIME_SAFETY_FACTOR = 1.3
TARGET_STEP = 2000
MASS_DRIFT_LIMIT = 1.0e-4
ENERGY_REBOUND_LIMIT = 1.0e-6
FIELD_MAGNITUDE_LIMIT = 10.0
RELATIVE_BUDGET_TOLERANCE = 1.0e-10
CASES = ("K2_junction_cap", "K3_uniform")


def _load_source() -> tuple[np.ndarray, dict[str, Any]]:
    verification = source_helper.verify_sources()
    if verification.get("verified") is not True:
        raise RuntimeError("time-100 source verification failed")
    mapped = np.load(
        source_helper.T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    fingerprint = array_fingerprint(mapped)
    checks = {
        "source_verified": verification.get("verified") is True,
        "fingerprint": (
            fingerprint == source_helper.EXPECTED_T100_FINGERPRINT
        ),
        "shape": list(mapped.shape) == list(FROZEN_DEG90.lattice.shape),
        "dtype": mapped.dtype == np.dtype(np.float64),
        "finite": bool(np.isfinite(mapped).all()),
        "read_only": not mapped.flags.writeable,
    }
    if not all(checks.values()):
        raise RuntimeError(f"time-100 source checks failed: {checks}")
    report = {
        "path": str(source_helper.T100_CHECKPOINT_PATH.resolve()),
        "sha256": sha256_path(source_helper.T100_CHECKPOINT_PATH),
        "fingerprint": fingerprint,
        "step": START_STEP,
        "checks": checks,
    }
    return mapped, report


def _contract(
    definition: equal_budget_controls.EqualBudgetControlDefinition,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "equal_budget_control_preflight",
        "start_step": START_STEP,
        "accepted_steps_per_case": ACCEPTED_STEPS_PER_CASE,
        "cases": list(CASES),
        "fft_workers": FFT_WORKERS,
        "budget_definition": "integral phase_interface_weight(c100)*(1-m)*dV",
        "control_definition": definition.to_dict(),
        "source": source,
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "scientific_contract_sha256": sha256_path(CONTRACT_PATH),
            "equal_budget_controls_sha256": sha256_path(
                Path(equal_budget_controls.__file__).resolve()
            ),
            "selective_model_sha256": sha256_path(
                Path(selective_model.__file__).resolve()
            ),
            "selective_geometry_sha256": sha256_path(
                Path(selective_geometry.__file__).resolve()
            ),
            "roy_model_sha256": sha256_path(
                Path(roy_model.__file__).resolve()
            ),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "scope": {
            "discarded_steps": False,
            "accepted_steps_total": (
                ACCEPTED_STEPS_PER_CASE * len(CASES)
            ),
            "production_launched": False,
            "parameter_sweep_launched": False,
            "part_two_started": False,
        },
    }


def _field_record(
    field: np.ndarray,
    solver: SpatialMobilityRoySolver,
    *,
    step: int,
    initial_mass: float,
    elapsed_wall_seconds: float,
) -> dict[str, Any]:
    record = production_helper._diagnostic_record(
        field,
        solver,
        step=step,
        initial_mass=initial_mass,
        elapsed_wall_seconds=elapsed_wall_seconds,
        include_energy=True,
    )
    record["finite"] = bool(record["field"]["finite"])
    record["contact"] = contact_metrics(field, FROZEN_DEG90)
    record["four_arm_topology"] = four_arm_metrics(
        field, FROZEN_DEG90
    )
    return record


def _run_case(
    source: np.ndarray,
    factor: np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    solver = SpatialMobilityRoySolver(
        FROZEN_DEG90.lattice,
        FROZEN_DEG90.parameters,
        factor,
        fft_workers=FFT_WORKERS,
    )
    field = np.array(source, dtype=np.float64, copy=True, order="C")
    initial_mass = float(source_helper.EXPECTED_T100_MASS)
    started = time.perf_counter()
    records = [
        _field_record(
            field,
            solver,
            step=START_STEP,
            initial_mass=initial_mass,
            elapsed_wall_seconds=0.0,
        )
    ]
    proposal_seconds: list[float] = []
    previous_energy = float(records[-1]["free_energy"])
    energy_nonincreasing = True
    for offset in range(1, ACCEPTED_STEPS_PER_CASE + 1):
        proposal_started = time.perf_counter()
        field = solver.propose_step(field)
        proposal_seconds.append(time.perf_counter() - proposal_started)
        record = _field_record(
            field,
            solver,
            step=START_STEP + offset,
            initial_mass=initial_mass,
            elapsed_wall_seconds=time.perf_counter() - started,
        )
        energy = float(record["free_energy"])
        relative_rebound = (
            energy - previous_energy
        ) / max(abs(previous_energy), 1.0)
        if relative_rebound > ENERGY_REBOUND_LIMIT:
            energy_nonincreasing = False
        previous_energy = energy
        records.append(record)

    event = production_helper._event_from_records(records, latched=None)
    final = records[-1]
    topology = final["four_arm_topology"]["thresholds"]
    contact = final["contact"]["thresholds"]
    checks = {
        "finite": bool(final["finite"] and np.isfinite(field).all()),
        "field_magnitude": (
            float(np.max(np.abs(field))) <= FIELD_MAGNITUDE_LIMIT
        ),
        "mass_drift": (
            max(
                abs(float(item["relative_mass_drift"]))
                for item in records
            )
            <= MASS_DRIFT_LIMIT
        ),
        "energy_nonincreasing": energy_nonincreasing,
        "four_arms_intact": all(
            bool(item["all_four_arms_attached"])
            for item in topology.values()
        ),
        "weld_connected": all(
            bool(item["cores_connected_locally"])
            for item in contact.values()
        ),
        "no_robust_event": event.get("detected") is False,
        "accepted_step_count": (
            len(proposal_seconds) == ACCEPTED_STEPS_PER_CASE
        ),
    }
    result = {
        "case": name,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "event_assessment": event,
        "initial_record": records[0],
        "final_record": final,
        "execution": {
            "accepted_steps": len(proposal_seconds),
            "wall_seconds": time.perf_counter() - started,
            "median_proposal_seconds": float(
                statistics.median(proposal_seconds)
            ),
            "maximum_proposal_seconds": float(max(proposal_seconds)),
        },
    }
    del field, solver
    gc.collect()
    return result


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    source, source_report = _load_source()
    definition, k1, k2, k3 = derive_equal_budget_controls(
        source,
        FROZEN_DEG90.lattice,
        inward_geometry=INWARD_GEOMETRY,
        radius=6.0,
        interface_width=float(np.sqrt(8.0)),
    )
    budget_checks = {
        "junction_match": (
            definition.junction_relative_mismatch
            <= RELATIVE_BUDGET_TOLERANCE
        ),
        "uniform_match": (
            definition.uniform_relative_mismatch
            <= RELATIVE_BUDGET_TOLERANCE
        ),
        "finite_factors": bool(
            np.isfinite(k1).all()
            and np.isfinite(k2).all()
            and np.isfinite(k3).all()
        ),
        "bounded_factors": bool(
            0.0 < np.min(k1) <= np.max(k1) <= 1.0
            and 0.0 < np.min(k2) <= np.max(k2) <= 1.0
            and 0.0 < np.min(k3) <= np.max(k3) <= 1.0
        ),
        "junction_factor_is_full_grid": (
            k2.shape == FROZEN_DEG90.lattice.shape
        ),
        "uniform_factor_is_scalar": k3.shape == (1, 1, 1),
        "junction_and_k1_have_same_minimum": bool(
            np.isclose(np.min(k1), np.min(k2), rtol=0.0, atol=1.0e-14)
        ),
    }
    if not all(budget_checks.values()):
        raise RuntimeError(f"control budget checks failed: {budget_checks}")

    contract = _contract(definition, source_report)
    atomic_json(output / "contract.json", contract)
    k2_result = _run_case(source, k2, name=CASES[0])
    k3_result = _run_case(source, k3, name=CASES[1])
    results = [k2_result, k3_result]
    medians = [
        float(item["execution"]["median_proposal_seconds"])
        for item in results
    ]
    conservative_median = max(medians)
    projected_per_case = (
        (TARGET_STEP - START_STEP)
        * conservative_median
        * RUNTIME_SAFETY_FACTOR
    )
    passed = bool(
        all(budget_checks.values())
        and all(item["passed"] for item in results)
    )
    summary = {
        "schema_version": 1,
        "status": "completed",
        "classification": (
            "equal_budget_control_preflight_passed"
            if passed
            else "equal_budget_control_preflight_failed"
        ),
        "passed": passed,
        "contract": contract,
        "budget_checks": budget_checks,
        "factor_summary": {
            "K1": {
                "minimum": float(np.min(k1)),
                "maximum": float(np.max(k1)),
                "support_cell_fraction": float(
                    np.mean(k1 < 1.0 - 1.0e-14)
                ),
            },
            "K2": {
                "minimum": float(np.min(k2)),
                "maximum": float(np.max(k2)),
                "support_cell_fraction": float(
                    np.mean(k2 < 1.0 - 1.0e-14)
                ),
            },
            "K3": {
                "factor": float(k3.item()),
            },
        },
        "cases": results,
        "runtime_projection": {
            "conservative_median_proposal_seconds": conservative_median,
            "safety_factor": RUNTIME_SAFETY_FACTOR,
            "remaining_steps_per_case": TARGET_STEP - START_STEP,
            "projected_seconds_per_case": projected_per_case,
            "projected_hours_per_case": projected_per_case / 3600.0,
            "projected_hours_two_cases_sequential": (
                2.0 * projected_per_case / 3600.0
            ),
            "explicit_notice_required_before_launch": (
                projected_per_case > 3600.0
            ),
        },
        "scope": {
            "production_launched": False,
            "automatic_follow_on_performed": False,
            "part_two_started": False,
        },
    }
    atomic_json(output / "summary.json", summary)
    if not passed:
        raise RuntimeError(
            "equal-budget preflight failed: "
            + json.dumps(
                {
                    item["case"]: {
                        key: value
                        for key, value in item["checks"].items()
                        if not value
                    }
                    for item in results
                },
                sort_keys=True,
            )
        )
    return summary


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    result = run(arguments.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": result["classification"],
                "passed": result["passed"],
                "output": str(arguments.output.resolve()),
                "runtime_projection": result["runtime_projection"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
