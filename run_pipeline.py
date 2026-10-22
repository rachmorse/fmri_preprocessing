#!/usr/bin/env python3
"""Run the resting-state fMRI preprocessing for the configured cohorts and sessions.

Usage:
    python run_pipeline.py [--dry-run]
"""

import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

from preprocessing.preprocessing_steps import (
    initialize_preprocessing_dirs,
    is_short_run,
    missing_inputs,
    run_preprocessing,
    setup_software,
)

Cohort = Literal["bbhi", "bbhi_senior"]
Session = Literal["ses-01", "ses-02"]

COHORTS: List[Cohort] = ["bbhi", "bbhi_senior"]
SESSIONS: List[Session] = ["ses-01", "ses-02"]
SUBJECTS: List[str] = []  # Leave empty to auto-detect unprocessed subjects

# Steps fail when too many subjects run in parallel so be careful increasing these
N_JOBS = 10
N_JOBS_HIGH_MEM = 3  # fsl_regfilt steps

# Where outputs are written and where the intermediate files are stored
LOCAL_ROOT = Path("/home/rachel/Desktop/preprocessing-updated_reconall")
REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / "logs"

ACPARAMS_FILE = REPO_ROOT / "acparams_hcp.txt"

# Make sure that these version match versions used to process any other subs that will be combined with these
FSL_DIR = Path("/vol/software/fsl_6_0_4")
FREESURFER_HOME = Path("/vol/software/freesurfer-6.0")
SPM_PATH = Path("/home/rachel/spm12")
MATLAB_CMD = "/usr/local/bin/matlab -nodesktop -nosplash"


@dataclass(frozen=True)
class CohortConfig:
    """Paths and naming conventions for one cohort.

    Attributes:
        bids: Shared BIDS folder.
        reconall: Shared recon-all folder.
        reconall_suffix: Suffix after `{subject}_{ses}` in the recon-all folder names.
        work_root: Folder for intermediate files, one subfolder per session.
        output: Local folder for the final outputs, laid out as `{subject}/{ses}`.
        shared_output: Shared folder of finished subjects per session, with a `{subject}` placeholder.
    """

    bids: Path
    reconall: Path
    reconall_suffix: str
    work_root: Path
    output: Path
    shared_output: Dict[Session, str]


COHORT_CONFIG: Dict[Cohort, CohortConfig] = {
    "bbhi": CohortConfig(
        bids=Path("/pool/guttmann/institut/BBHI/MRI/BIDS"),
        reconall=Path("/pool/guttmann/institut/BBHI/MRI/derivatives/reconall_fs6"),
        reconall_suffix="_run-01",
        work_root=LOCAL_ROOT / "bbhi",
        output=LOCAL_ROOT / "bbhi" / "resting_preprocessed",
        shared_output={
            "ses-01": "/pool/guttmann/institut/BBHI/MRI/processed_data/resting_preproc_fs6-recon/{subject}",
            "ses-02": "/pool/guttmann/institut/BBHI/MRI/processed_data/resting_preproc_fs6-recon_tp2/{subject}",
        },
    ),
    "bbhi_senior": CohortConfig(
        bids=Path("/pool/guttmann/institut/UB/Superagers/MRI/BIDS"),
        reconall=Path("/pool/guttmann/institut/UB/Superagers/MRI/derivatives/reconall_fs6"),
        reconall_suffix="",
        work_root=LOCAL_ROOT / "bbhi_senior",
        output=LOCAL_ROOT / "bbhi_senior" / "resting_preprocessed",
        shared_output={
            "ses-01": "/pool/guttmann/institut/UB/Superagers/MRI/resting_preproc_fs6-recon/{subject}/ses-01",
            "ses-02": "/pool/guttmann/institut/UB/Superagers/MRI/resting_preproc_fs6-recon/{subject}/ses-02",
        },
    ),
}


def work_dir(cohort: Cohort, ses: Session) -> Path:
    """Get the folder for the intermediate files of a cohort and session.

    Args:
        cohort: Cohort name.
        ses: Session.

    Returns:
        Path to the intermediate files folder.
    """
    return COHORT_CONFIG[cohort].work_root / ses


def processed_dirs(cohort: Cohort, ses: Session) -> List[str]:
    """Get the templates of a subject's finished output folders, local first and then shared.

    Args:
        cohort: Cohort name.
        ses: Session.

    Returns:
        Folder templates with a `{subject}` placeholder.
    """
    cfg = COHORT_CONFIG[cohort]
    return [f"{cfg.output}/{{subject}}/{ses}", cfg.shared_output[ses]]


def select_subjects(cohort: Cohort, ses: Session) -> List[str]:
    """Get the subjects to process for a cohort and session.

    Uses `SUBJECTS` when set, keeping only those with all inputs and a run of at least 10 min. Otherwise auto-detects
    unprocessed subjects.

    Args:
        cohort: Cohort name.
        ses: Session.

    Returns:
        Sorted subject IDs.
    """
    cfg = COHORT_CONFIG[cohort]

    if not SUBJECTS:
        return sorted(
            initialize_preprocessing_dirs(cfg.reconall, cfg.reconall_suffix, cfg.bids, ses, processed_dirs(cohort, ses))
        )

    subjects = []
    n_short_runs = 0
    for subject in SUBJECTS:
        if not (cfg.bids / subject).is_dir():
            continue
        missing = missing_inputs(subject, ses, cfg.bids, cfg.reconall, cfg.reconall_suffix)
        if missing:
            print(f"Skipping {subject} ({cohort} {ses}), missing inputs:\n  " + "\n  ".join(missing))
            continue
        if is_short_run(subject, ses, cfg.bids):
            n_short_runs += 1
            continue
        subjects.append(subject)

    if n_short_runs:
        print(f"{cohort} {ses}: {n_short_runs} subjects skipped because of a run shorter than 10 min")

    return sorted(subjects)


def format_summary(results: List[dict]) -> str:
    """Build the end-of-pipeline summary.

    Args:
        results: One dict per cohort and session with `cohort`, `ses` and the `steps` from `run_preprocessing`.

    Returns:
        The formatted summary.
    """
    lines = ["", "=" * 60, "PIPELINE SUMMARY", "=" * 60]
    if not results:
        lines.append("No subjects needed processing.")

    for result in results:
        lines.append(f"\n{result['cohort']} {result['ses']}")
        for step in result["steps"]:
            lines.append(f"  {step['step']}: {step['n_successful']} / {step['n_input']} successful")
            if step["failed"]:
                lines.append(f"    Failed: {', '.join(step['failed'])}")

    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    """Select the subjects for every cohort and session, then preprocess them."""
    parser = argparse.ArgumentParser(description="Run resting-state fMRI preprocessing.")
    parser.add_argument("--dry-run", action="store_true", help="List the subjects that would be processed and exit.")
    args = parser.parse_args()

    jobs = []
    for cohort in COHORTS:
        for ses in SESSIONS:
            subjects = select_subjects(cohort, ses)
            print(f"{cohort} {ses}: {len(subjects)} subjects to process {subjects}")
            if subjects:
                jobs.append((cohort, ses, subjects))

    if args.dry_run:
        return

    setup_software(FSL_DIR, FREESURFER_HOME, SPM_PATH, MATLAB_CMD)

    start_time = datetime.datetime.now()
    print(f"Starting pipeline at {start_time}")

    results = []
    for cohort, ses, subjects in jobs:
        cfg = COHORT_CONFIG[cohort]
        print(f"\n{'#' * 60}\n{cohort} {ses}\n{'#' * 60}")
        steps = run_preprocessing(
            subjects,
            ses,
            root_path=str(work_dir(cohort, ses)),
            output_path=str(cfg.output),
            bids_path=str(cfg.bids),
            recon_all_path=str(cfg.reconall),
            recon_suffix=cfg.reconall_suffix,
            acparams_file=ACPARAMS_FILE,
            spm_path=SPM_PATH,
            log_dir=str(LOG_ROOT / f"{cohort}_{ses}"),
            n_jobs=N_JOBS,
            n_jobs_high_mem=N_JOBS_HIGH_MEM,
        )
        results.append({"cohort": cohort, "ses": ses, "steps": steps})

    summary = format_summary(results)
    print(summary)
    LOG_ROOT.mkdir(exist_ok=True)
    (LOG_ROOT / f"pipeline_summary_{start_time:%Y%m%d_%H%M}.txt").write_text(summary + "\n")

    hours = (datetime.datetime.now() - start_time).total_seconds() / 3600
    print(f"Total time: {hours:.2f} hours")


if __name__ == "__main__":
    main()
