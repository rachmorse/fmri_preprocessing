#!/usr/bin/env python3

import os
import shutil

# Use this script to erase the intermediate files from preprocessing to make more space 

def main():
    ses = "ses-02"
    finished = "/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed"
    root = "/home/rachel/Desktop/Preprocessing/"

    # Loop through each folder in 'finished'
    for folder in os.listdir(finished):
        folder_path = os.path.join(finished, folder)
        if not os.path.isdir(folder_path):
            continue

        suj = os.path.basename(folder_path)

        # Determine if the .tsv file exists
        dvars_file = os.path.join(root, "QC", suj, "dvars_node",
                                  f"{suj}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE_dvars.tsv")

        # fMRItoStandard
        fmri2standard = os.path.join(root, "fmri2standard", suj)
        apply_topup_dir = os.path.join(fmri2standard, "apply_topup")
        # Remove apply_topup directory if it exists
        if os.path.exists(apply_topup_dir):
            shutil.rmtree(apply_topup_dir, ignore_errors=True)

        if os.path.isfile(dvars_file):
            # Remove certain directories if the .tsv (QC) file exists
            for dirname in ["binarize_mask", "mask_T1", "vol2vol"]:
                dir_path = os.path.join(fmri2standard, dirname)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
            qc_brain_mask = os.path.join(root, "QC", suj, "brain_mask")
            if os.path.exists(qc_brain_mask):
                shutil.rmtree(qc_brain_mask, ignore_errors=True)

        for dirname in [
            "Corregister_SBref2SEgfm", "eliminate_first_scans", "extract_mask",
            "Mean_SEgfm_AP", "Merge_ap_pa_inputs", "Merge_SEgfm_AP_PA",
            "Topup_SEgfm_estimation", "apply_topup_to_SBref"
        ]:
            dir_path = os.path.join(fmri2standard, dirname)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)

        # Remove specific files in fmri2standard
        for fname in ["d3.js", "graph.json", "graph1.json", "index.html"]:
            f_path = os.path.join(fmri2standard, fname)
            if os.path.exists(f_path):
                os.remove(f_path)

        # Nuisance
        nuisance_dir = os.path.join(root, "nuisance_correction", suj)
        for dirname in [
            "AcompCor_mask", "cosine_filter", "filter_regressors_bold",
            "masks_csf_wm", "Merge_txt_inputs", "Merge_wm_csf"
        ]:
            dir_path = os.path.join(nuisance_dir, dirname)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)

        for fname in ["d3.js", "graph.json", "graph1.json", "index.html"]:
            f_path = os.path.join(nuisance_dir, fname)
            if os.path.exists(f_path):
                os.remove(f_path)

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
                else:
                    os.remove(f_path)

        # BIDS copy 
        anat_dir_local = os.path.join(root,f"BIDS/anat/{suj}_{ses}_run-01_T1w.nii")

        if os.path.exists(anat_dir_local):
            if os.path.isdir(anat_dir_local):
                shutil.rmtree(anat_dir_local, ignore_errors=True)
            else:
                os.remove(anat_dir_local)

if __name__ == "__main__":
    main()