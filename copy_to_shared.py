#!/usr/bin/env python3
"""Copy the final preprocessed outputs from the local output folder to the shared folder.

This includes the scrubbed images and JSON sidecars if `scrubbing.py` has been run.
Subjects that already have files in the shared folder are skipped unless --overwrite is given.

NOTE: sudo is required to write to the shared folder.

Usage:
    sudo "$(which python)" copy_to_shared.py [--dry-run] [--overwrite]
"""

import argparse
import os
import shutil

from preprocessing.preprocessing_steps import is_processed
from run_pipeline import COHORT_CONFIG, COHORTS, SESSIONS, SUBJECTS

OUTPUT_DIRS = ["native_T1", "MNI_2mm"]


def plan_copies(local_dir, shared_dir):
    """List a subject's files to copy and the shared files they would replace.

    Args:
        local_dir (str): The subject's local output folder for this session.
        shared_dir (str): The subject's shared output folder for this session.

    Returns:
        tuple: (src, dst) pairs to copy, existing shared files that would be overwritten,
            and shared files with no local equivalent.
    """
    copies, existing, extra_files = [], [], []

    for dirname in OUTPUT_DIRS:
        src_dir = os.path.join(local_dir, dirname)
        dst_dir = os.path.join(shared_dir, dirname)
        local_files = sorted(os.listdir(src_dir))

        for fname in local_files:
            dst = os.path.join(dst_dir, fname)
            copies.append((os.path.join(src_dir, fname), dst))
            if os.path.exists(dst):
                existing.append(dst)

        if os.path.isdir(dst_dir):
            extra_files += [os.path.join(dst_dir, f) for f in sorted(set(os.listdir(dst_dir)) - set(local_files))]

    return copies, existing, extra_files


def copy_files(copies, existing, dry_run):
    """Copy files, stating each one that overwrites an existing shared file.

    Args:
        copies (list): (src, dst) pairs to copy.
        existing (list): Destinations that already exist and will be overwritten.
        dry_run (bool): Whether to only print what would be copied.
    """
    for src, dst in copies:
        if dst in existing:
            action = "[DRY RUN] Would OVERWRITE" if dry_run else "OVERWRITING"
        else:
            action = "[DRY RUN] Would copy" if dry_run else "Copying"
        print(f"{action} {src} -> {dst}")

        if not dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def print_report(n_copied, skipped, stale, dry_run):
    """Print the totals, the skipped subjects, and any shared files that may now be stale.

    Args:
        n_copied (int): Number of files copied.
        skipped (list): Subjects skipped because they already exist in the shared folder.
        stale (dict): Shared files not made by this pipeline, per subject that was overwritten.
        dry_run (bool): Whether this was a dry run.
    """
    print(f"\nTotal files {'to copy' if dry_run else 'copied'}: {n_copied}")

    if skipped:
        print(f"\nSkipped {len(skipped)} subjects that already exist in the shared folder (use --overwrite to replace):")
        for label in skipped:
            print(f"  {label}")

    if stale:
        print("\nWARNING: these files were not made by this pipeline. If reprocessing, check these files are correctly processed:")
        for label, files in stale.items():
            print(f"  {label}:")
            for f in files:
                print(f"    {f}")


def main():
    """Copy every finished local subject in the configured cohorts and sessions to the shared folder."""
    parser = argparse.ArgumentParser(description="Copy final preprocessed outputs to the shared folder.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without copying.")
    parser.add_argument("--overwrite", action="store_true", help="Replace subjects that already exist in the shared folder.")
    args = parser.parse_args()

    n_copied = 0
    skipped = []
    stale = {}

    for cohort in COHORTS:
        cfg = COHORT_CONFIG[cohort]
        if not cfg.output.is_dir():
            continue

        for ses in SESSIONS:
            for subject in SUBJECTS or sorted(os.listdir(cfg.output)):
                local_dir = os.path.join(cfg.output, subject, ses)

                # Only copy subjects whose local output is complete
                if not is_processed(subject, ses, local_dir):
                    continue

                label = f"{cohort} {ses} {subject}"
                copies, existing, extra_files = plan_copies(local_dir, cfg.shared_output[ses].format(subject=subject))

                if existing and not args.overwrite:
                    skipped.append(label)
                    continue

                copy_files(copies, existing, args.dry_run)
                n_copied += len(copies)

                # Files made by other scripts may be based on the replaced outputs
                if existing and extra_files:
                    stale[label] = extra_files

    print_report(n_copied, skipped, stale, args.dry_run)


if __name__ == "__main__":
    main()
