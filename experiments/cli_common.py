"""Shared CLI helpers for full-benchmark experiment runners."""

from __future__ import annotations

import argparse

from bcimpute.config import ExperimentConfig, apply_rfeca_only


def add_full_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required flag to actually launch the full benchmark.",
    )
    parser.add_argument(
        "--mechanism",
        choices=["mcar", "mar"],
        default="mcar",
        help="Artificial missingness mechanism (default: mcar).",
    )
    parser.add_argument(
        "--only-rfeca",
        action="store_true",
        help=(
            "Run only RFECA_SVR(k=5/10/20). Use after inductive RFECA fix to avoid "
            "re-running Mean/KNN/MissForest. Note: with seed scheme v2, masks differ "
            "from pre-v2 artifacts — do not merge CSVs with legacy runs unless "
            "--seed-scheme legacy is set."
        ),
    )
    parser.add_argument(
        "--seed-scheme",
        choices=["v2", "legacy"],
        default="v2",
        help=(
            "Missingness seed formula. v2 is collision-free (default). "
            "legacy reproduces masks from frozen 2026-07 full campaigns "
            "(needed only if merging with those artifact CSVs)."
        ),
    )


def configure_full_run(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply CLI overrides (RFECA-only, seed scheme, sequential outer loop)."""
    cfg.n_jobs = 1
    cfg.target_cell_policy = "originally_observed_only"
    cfg.missingness_seed_scheme = args.seed_scheme
    if args.only_rfeca:
        apply_rfeca_only(cfg)
    return cfg


def artifact_prefix(cohort: str, mechanism: str, *, only_rfeca: bool) -> str:
    base = f"{cohort}_full" if mechanism == "mcar" else f"{cohort}_full_{mechanism}"
    if only_rfeca:
        return f"{base}_rfeca"
    return base
