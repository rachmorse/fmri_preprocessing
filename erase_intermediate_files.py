#!/usr/bin/env python3
"""Erase the intermediate preprocessing files of finished subjects to make more space.

Also deletes the local final outputs of subjects that have been copied to the shared folder.

Usage:
    python erase_intermediate_files.py [--dry-run]
"""

import argparse
import os
import shutil

from preprocessing.preprocessing_steps import is_processed
from run_pipeline import COHORTS, SESSIONS, SUBJECTS, processed_dirs, work_dir


def safe_remove(path, dry_run, stats):
    """Delete a file or directory if it exists, or only report it on a dry run.

    Args:
        path (str): The file or directory to delete.
        dry_run (bool): Whether to only print what would be deleted.
        stats (dict): Counts of deleted `dirs` and `files`, updated in place.
    """
    if not os.path.exists(path):
        return

    if os.path.isdir(path):
        if dry_run:
            print(f"[DRY RUN] Would delete directory: {path}")
        else:
            shutil.rmtree(path, ignore_errors=True)
        stats["dirs"] += 1
    else:
        if dry_run:
            print(f"[DRY RUN] Would delete file: {path}")
        else:
            os.remove(path)
        stats["files"] += 1


def remove_unzipped_coregistered(fmri2standard, suj, ses, dry_run, stats):
    """Remove uncompressed coregistered images that have a .nii.gz copy.

    These are left over when a subject is coregistered again; later steps only read the .nii.gz.

    Args:
        fmri2standard (str): The subject's fmri2standard folder.
        suj (str): The subject ID.
        ses (str): The session for the data.
        dry_run (bool): Whether to only print what would be deleted.
        stats (dict): Counts of deleted `dirs` and `files`, updated in place.
    """
    images = {
        "spm_coregister2T1_bold": f"{suj}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii",
        "spm_coregister2T1_sbref": f"{suj}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii",
    }
    for dirname, image in images.items():
        nii_path = os.path.join(fmri2standard, dirname, image)
        if os.path.isfile(f"{nii_path}.gz"):
            safe_remove(nii_path, dry_run, stats)


def erase_session(cohort, ses, dry_run, stats):
    """Erase the intermediate files of every finished subject in one cohort and session.

    Only subjects in `SUBJECTS` are erased when it is set.

    Args:
        cohort (str): The cohort to clean up.
        ses (str): The session for the data.
        dry_run (bool): Whether to only print what would be deleted.
        stats (dict): Counts of deleted `dirs` and `files`, updated in place.
    """
    root = str(work_dir(cohort, ses))
    subjects_root = os.path.join(root, "fmri2standard")

    if not os.path.isdir(subjects_root):
        return

    for suj in os.listdir(subjects_root):
        suj_path = os.path.join(subjects_root, suj)
        if not (os.path.isdir(suj_path) and suj.startswith("sub-")) or (SUBJECTS and suj not in SUBJECTS):
            continue

        # Only proceed if final output is complete, in the local output folder if it exists, otherwise in the shared folder
        local_dir, shared_dir = (d.format(subject=suj) for d in processed_dirs(cohort, ses))
        finished_dir = local_dir if os.path.isdir(local_dir) else shared_dir
        if not is_processed(suj, ses, finished_dir):
            continue

        # Determine if the .tsv file exists
        dvars_file = os.path.join(root, "QC", suj, "dvars_node",
                                  f"{suj}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE_dvars.tsv")

        # fMRItoStandard
        fmri2standard = os.path.join(root, "fmri2standard", suj)
        apply_topup_dir = os.path.join(fmri2standard, "apply_topup")
        safe_remove(apply_topup_dir, dry_run, stats)
        remove_unzipped_coregistered(fmri2standard, suj, ses, dry_run, stats)

        if os.path.isfile(dvars_file):
            # Remove certain directories if the .tsv (QC) file exists
            for dirname in ["binarize_mask", "mask_T1", "vol2vol"]:
                dir_path = os.path.join(fmri2standard, dirname)
                safe_remove(dir_path, dry_run, stats)

            qc_brain_mask = os.path.join(root, "QC", suj, "brain_mask")
            safe_remove(qc_brain_mask, dry_run, stats)

        for dirname in [
            "Corregister_SBref2SEgfm", "eliminate_first_scans", "extract_mask",
            "Mean_SEgfm_AP", "Merge_ap_pa_inputs", "Merge_SEgfm_AP_PA",
            "Topup_SEgfm_estimation", "apply_topup_to_SBref"
        ]:
            dir_path = os.path.join(fmri2standard, dirname)
            safe_remove(dir_path, dry_run, stats)

        # Remove specific files in fmri2standard
        for fname in ["d3.js", "graph.json", "graph1.json", "index.html"]:
            f_path = os.path.join(fmri2standard, fname)
            safe_remove(f_path, dry_run, stats)

        # Nuisance
        nuisance_dir = os.path.join(root, "nuisance_correction", suj)
        for dirname in [
            "AcompCor_mask", "cosine_filter", "filter_regressors_bold",
            "masks_csf_wm", "Merge_txt_inputs", "Merge_wm_csf"
        ]:
            dir_path = os.path.join(nuisance_dir, dirname)
            safe_remove(dir_path, dry_run, stats)

        for fname in ["d3.js", "graph.json", "graph1.json", "index.html"]:
            f_path = os.path.join(nuisance_dir, fname)
            safe_remove(f_path, dry_run, stats)

        # MNI Normalization
        normalization_dir = os.path.join(root, "normalization", suj)
        for fname in [
            f"{suj}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii",
            f"{suj}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii",
            f"{suj}_{ses}_run-01_T1w.nii"
        ]:
            f_path = os.path.join(normalization_dir, fname)
            safe_remove(f_path, dry_run, stats)

        # BIDS copy
        anat_dir_local = os.path.join(root, "BIDS", "anat", f"{suj}_{ses}_run-01_T1w.nii")
        if dry_run:
            print(f"[Check] BIDS/anat local path: {anat_dir_local}")
        safe_remove(anat_dir_local, dry_run, stats)


def is_copied(local_dir, shared_dir):
    """Checks whether every local output file has a matching copy in the shared folder.

    `copy_to_shared.py` keeps the file times, and only sets them once a file is fully copied,
    so a complete copy has the same size and modification time.

    Args:
        local_dir (str): The subject's local output folder for this session.
        shared_dir (str): The subject's shared output folder for this session.

    Returns:
        bool: True if all local files are in the shared folder with the same size and modification time.
    """
    for dirpath, _, filenames in os.walk(local_dir):
        for fname in filenames:
            src = os.path.join(dirpath, fname)
            dst = os.path.join(shared_dir, os.path.relpath(src, local_dir))
            if not os.path.isfile(dst):
                print(f"  Missing in the shared folder: {dst}")
                return False
            src_stat, dst_stat = os.stat(src), os.stat(dst)
            if src_stat.st_size != dst_stat.st_size or int(src_stat.st_mtime) != int(dst_stat.st_mtime):
                print(f"  Different in the shared folder: {dst}")
                return False
    return True


def remove_copied_outputs(cohort, ses, dry_run, stats):
    """Erase the local final outputs of subjects that have been copied to the shared folder.

    Only subjects in `SUBJECTS` are erased when it is set.

    Args:
        cohort (str): The cohort to clean up.
        ses (str): The session for the data.
        dry_run (bool): Whether to only print what would be deleted.
        stats (dict): Counts of deleted `dirs` and `files`, updated in place.
    """
    local_template, shared_template = processed_dirs(cohort, ses)
    local_root = local_template.split("{subject}")[0]

    if not os.path.isdir(local_root):
        return

    for suj in sorted(os.listdir(local_root)):
        if not suj.startswith("sub-") or (SUBJECTS and suj not in SUBJECTS):
            continue

        local_dir, shared_dir = local_template.format(subject=suj), shared_template.format(subject=suj)
        if not os.path.isdir(local_dir) or not is_processed(suj, ses, local_dir):
            continue

        if not is_copied(local_dir, shared_dir):
            print(f"Keeping local outputs of {suj}: not yet copied to the shared folder")
            continue

        safe_remove(local_dir, dry_run, stats)

        # Remove the subject folder too if this was its only session
        subject_dir = os.path.join(local_root, suj)
        if not dry_run and not os.listdir(subject_dir):
            os.rmdir(subject_dir)


def main():
    """Use this script to erase the intermediate files from preprocessing to make more space."""
    parser = argparse.ArgumentParser(description="Erase intermediate files from preprocessing.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without actually deleting.")
    args = parser.parse_args()

    dry_run = args.dry_run
    stats = {"dirs": 0, "files": 0}

    for cohort in COHORTS:
        for ses in SESSIONS:
            erase_session(cohort, ses, dry_run, stats)
            remove_copied_outputs(cohort, ses, dry_run, stats)

    if dry_run:
        print("\n===== DRY RUN SUMMARY =====")
        print(f"Would delete {stats['dirs']} directories")
        print(f"Would delete {stats['files']} files")
    else:
        print("\n===== CLEANUP SUMMARY =====")
        print(f"Total directories deleted: {stats['dirs']}")
        print(f"Total files deleted: {stats['files']}")
    print("===========================\n")


if __name__ == "__main__":
    main()
