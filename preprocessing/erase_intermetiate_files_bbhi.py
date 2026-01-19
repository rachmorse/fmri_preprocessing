#!/usr/bin/env python3

import os
import shutil
import argparse


def main():
    """Use this script to erase the intermediate files from preprocessing to make more space."""
    
    parser = argparse.ArgumentParser(description="Erase intermediate files from preprocessing.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without actually deleting.")
    args = parser.parse_args()
    
    dry_run = args.dry_run

    ses = "ses-02"
    root = "/home/rachel/Desktop/preprocessing-updated_reconall/bbhi"
    if ses == "ses-01":
        shared_root = "/pool/guttmann/institut/BBHI/MRI/processed_data/resting_preproc_fs6-recon"
    else:
        shared_root = "/pool/guttmann/institut/BBHI/MRI/processed_data/resting_preproc_fs6-recon_tp2"

    # stats = {'dirs': 0, 'files': 0}
    stats = [0, 0] # [dirs, files]

    def safe_remove(path):
        if not os.path.exists(path):
            return

        if os.path.isdir(path):
            if dry_run:
                print(f"[DRY RUN] Would delete directory: {path}")
            else:
                shutil.rmtree(path, ignore_errors=True)
            stats[0] += 1
        else:
            if dry_run:
                print(f"[DRY RUN] Would delete file: {path}")
            else:
                os.remove(path)
            stats[1] += 1

    subjects_root = os.path.join(root, "fmri2standard")
    local_finished_root = os.path.join(root, "resting_preprocessed")

    for suj in os.listdir(subjects_root):
        suj_path = os.path.join(subjects_root, suj)
        if not (os.path.isdir(suj_path) and suj.startswith("sub-")):
            continue

        # Check for existence of processed data in either local or shared location
        finished_suj_local = os.path.join(local_finished_root, suj, ses)
        finished_suj_shared = os.path.join(shared_root, suj)

        # Only proceed if final output exists in at least one location
        data_exists_locally = os.path.isdir(finished_suj_local)
        data_exists_shared = os.path.exists(finished_suj_shared)

        if not (data_exists_locally or data_exists_shared):
            continue

        # Determine if the .tsv file exists
        dvars_file = os.path.join(root, "QC", suj, "dvars_node",
                                  f"{suj}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE_dvars.tsv")

        # fMRItoStandard
        fmri2standard = os.path.join(root, "fmri2standard", suj)
        apply_topup_dir = os.path.join(fmri2standard, "apply_topup")
        safe_remove(apply_topup_dir)        

        if os.path.isfile(dvars_file):
            # Remove certain directories if the .tsv (QC) file exists
            for dirname in ["binarize_mask", "mask_T1", "vol2vol"]:
                dir_path = os.path.join(fmri2standard, dirname)
                safe_remove(dir_path)
            
            qc_brain_mask = os.path.join(root, "QC", suj, "brain_mask")
            safe_remove(qc_brain_mask)

        for dirname in [
            "Corregister_SBref2SEgfm", "eliminate_first_scans", "extract_mask",
            "Mean_SEgfm_AP", "Merge_ap_pa_inputs", "Merge_SEgfm_AP_PA",
            "Topup_SEgfm_estimation", "apply_topup_to_SBref"
        ]:
            dir_path = os.path.join(fmri2standard, dirname)
            safe_remove(dir_path)

        # Remove specific files in fmri2standard
        for fname in ["d3.js", "graph.json", "graph1.json", "index.html"]:
            f_path = os.path.join(fmri2standard, fname)
            safe_remove(f_path)

        # Nuisance
        nuisance_dir = os.path.join(root, "nuisance_correction", suj)
        for dirname in [
            "AcompCor_mask", "cosine_filter", "filter_regressors_bold",
            "masks_csf_wm", "Merge_txt_inputs", "Merge_wm_csf"
        ]:
            dir_path = os.path.join(nuisance_dir, dirname)
            safe_remove(dir_path)

        for fname in ["d3.js", "graph.json", "graph1.json", "index.html"]:
            f_path = os.path.join(nuisance_dir, fname)
            safe_remove(f_path)

        # MNI Normalization
        normalization_dir = os.path.join(root, "normalization", suj)
        for fname in [
            f"{suj}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii",
            f"{suj}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii",
            f"{suj}_{ses}_run-01_T1w.nii"
        ]:
            f_path = os.path.join(normalization_dir, fname)
            safe_remove(f_path)

        # BIDS copy 
        anat_dir_local = os.path.join(root, "BIDS", "anat", f"{suj}_{ses}_run-01_T1w.nii")
        if dry_run:
            print(f"[Check] BIDS/anat local path: {anat_dir_local}")
        safe_remove(anat_dir_local)

    if dry_run:
        print("\n===== DRY RUN SUMMARY =====")
        print(f"Would delete {stats[0]} directories")
        print(f"Would delete {stats[1]} files")
    else:
        print("\n===== CLEANUP SUMMARY =====")
        print(f"Total directories deleted: {stats[0]}")
        print(f"Total files deleted: {stats[1]}")
    print("===========================\n")

if __name__ == "__main__":
    main()