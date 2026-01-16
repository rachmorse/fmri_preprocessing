#!/usr/bin/env python3

import os
import shutil


def main():
    """Use this script to erase the intermediate files from preprocessing to make more space."""

    ses = "ses-02"
    root = "/home/rachel/Desktop/preprocessing-updated_reconall"

    total_dirs_deleted = 0
    total_files_deleted = 0

    subjects_root = os.path.join(root, "fmri2standard")
    finished_root = os.path.join(root, "resting_preprocessed")

    for suj in os.listdir(subjects_root):
        suj_path = os.path.join(subjects_root, suj)
        if not (os.path.isdir(suj_path) and suj.startswith("sub-")):
            continue

        finished_suj = os.path.join(finished_root, suj)

        # Only proceed if final output exists
        if not os.path.isdir(finished_suj):
            continue

        # Determine if the .tsv file exists
        dvars_file = os.path.join(root, "QC", suj, "dvars_node",
                                  f"{suj}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE_dvars.tsv")

        # fMRItoStandard
        fmri2standard = os.path.join(root, "fmri2standard", suj)
        apply_topup_dir = os.path.join(fmri2standard, "apply_topup")
        # Remove apply_topup directory if it exists
        if os.path.exists(apply_topup_dir):
            shutil.rmtree(apply_topup_dir, ignore_errors=True)
            total_dirs_deleted += 1         

        if os.path.isfile(dvars_file):
            # Remove certain directories if the .tsv (QC) file exists
            for dirname in ["binarize_mask", "mask_T1", "vol2vol"]:
                dir_path = os.path.join(fmri2standard, dirname)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
                    total_dirs_deleted += 1         
            qc_brain_mask = os.path.join(root, "QC", suj, "brain_mask")
            if os.path.exists(qc_brain_mask):
                shutil.rmtree(qc_brain_mask, ignore_errors=True)
                total_dirs_deleted += 1 

        for dirname in [
            "Corregister_SBref2SEgfm", "eliminate_first_scans", "extract_mask",
            "Mean_SEgfm_AP", "Merge_ap_pa_inputs", "Merge_SEgfm_AP_PA",
            "Topup_SEgfm_estimation", "apply_topup_to_SBref"
        ]:
            dir_path = os.path.join(fmri2standard, dirname)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
                total_dirs_deleted += 1 

        # Remove specific files in fmri2standard
        for fname in ["d3.js", "graph.json", "graph1.json", "index.html"]:
            f_path = os.path.join(fmri2standard, fname)
            if os.path.exists(f_path):
                os.remove(f_path)
                total_files_deleted += 1 

        # Nuisance
        nuisance_dir = os.path.join(root, "nuisance_correction", suj)
        for dirname in [
            "AcompCor_mask", "cosine_filter", "filter_regressors_bold",
            "masks_csf_wm", "Merge_txt_inputs", "Merge_wm_csf"
        ]:
            dir_path = os.path.join(nuisance_dir, dirname)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
                total_dirs_deleted += 1 

        for fname in ["d3.js", "graph.json", "graph1.json", "index.html"]:
            f_path = os.path.join(nuisance_dir, fname)
            if os.path.exists(f_path):
                os.remove(f_path)
                total_files_deleted += 1

        # MNI Normalization
        normalization_dir = os.path.join(root, "normalization", suj)
        for fname in [
            f"{suj}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii",
            f"{suj}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii",
            f"{suj}_{ses}_run-01_T1w.nii"
        ]:
            f_path = os.path.join(normalization_dir, fname)
            if os.path.exists(f_path):
                if os.path.isdir(f_path):
                    shutil.rmtree(f_path, ignore_errors=True)
                    total_dirs_deleted += 1 
                else:
                    os.remove(f_path)
                    total_files_deleted += 1 

        # BIDS copy 
        anat_dir_local = os.path.join(root, "BIDS", "anat", f"{suj}_{ses}_run-01_T1w.nii")

        if os.path.exists(anat_dir_local):
            if os.path.isdir(anat_dir_local):
                shutil.rmtree(anat_dir_local, ignore_errors=True)
                total_dirs_deleted += 1 
            else:
                os.remove(anat_dir_local)
                total_files_deleted += 1 

    print("\n===== CLEANUP SUMMARY =====")
    print(f"Total directories deleted: {total_dirs_deleted}")
    print(f"Total files deleted: {total_files_deleted}")
    print("===========================\n")

if __name__ == "__main__":
    main()