#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script executes fMRI preprocessing workflows with configurable paths.
"""

import datetime
import os
from multiprocessing import Pool
import logging
import shutil
import subprocess
import gzip

# Nipype interfaces
from nipype.algorithms.confounds import ComputeDVARS, FramewiseDisplacement
from nipype.interfaces import spm, utility
from nipype import Node, Workflow

# Custom workflow imports
from bold2T1_wf import get_fmri2standard_wf
from bold_nuisance_correction_wf import get_nuisance_regressors_wf

def prepare_data(subject_id, recon_all_path, source_dir):
    """
    Prepare necessary directories and copy MRI data for a given subject.

    Parameters:
    subject_id (str): The identifier for the subject whose data is being prepared.
    recon_all_path (str): The directory path for recon_all data storage.
    source_dir (str): The directory where source MRI data is located.

    Returns:
        None
    """
    try:
        # Construct paths
        subject_dir = os.path.join(recon_all_path, subject_id)
        mri_dir = os.path.join(subject_dir, "mri")
        aseg_file = os.path.join(mri_dir, "aseg.mgz")

        # Check if the aseg.mgz file is already present
        if not os.path.isfile(aseg_file):
            print("Copying aseg.mgz and brain.mgz from institut_recon_all...")

            # Create required directories
            os.makedirs(mri_dir, exist_ok=True)

            # Define source directory where files are copied from
            source_dir_mri = os.path.join(source_dir, subject_id, "mri")

            # Copy files to the destination directory
            shutil.copy(os.path.join(source_dir_mri, "aseg.mgz"), mri_dir)
            shutil.copy(os.path.join(source_dir_mri, "brain.mgz"), mri_dir)
        else:
            print("aseg.mgz and brain.mgz already copied")
    except Exception as e:
        logging.error("Error copying aseg.mgz & brain.mgz for subject %s: %s", subject_id, e)

# def run_workflows(bids_path, fmri2standard_path, nuisance_correction_path, recon_all_path, qc_path):
    """
    Run key fMRI preprocessing workflows.

    This function executes two main workflows:
    1. fMRI to Standard Space Transformation
       - Aligns fMRI data to standard anatomical space using `get_fmri2standard_wf`.
    
    2. BOLD Nuisance Correction
       - Removes nuisance signals from BOLD data using `get_nuisance_regressors_wf`.
    
    Logs the success or failure of each workflow.
    """
    try:
        # Execute the fMRI to standard space transformation workflow
        fmri_to_standard = get_fmri2standard_wf()
        fmri_to_standard.run()
        logging.info("fMRI to standard space workflow completed successfully.")
    except Exception as e:
        logging.error("Error in fMRI to standard space workflow: %s", e)

    try:
        # Execute the BOLD nuisance correction workflow
        nuisance_correction = get_nuisance_regressors_wf()
        nuisance_correction.run()
        logging.info("BOLD nuisance correction workflow completed successfully.")
    except Exception as e:
        logging.error("Error in BOLD nuisance correction workflow: %s", e)

# Coregistration setup, to align EPI to T1, should be added here with appropriate file paths.
# Example:
# coreg_EPI2T1 = spm.Coregister()
# coreg_EPI2T1.inputs.target = target_image_path
# coreg_EPI2T1.inputs.source = source_image_path
# coreg_EPI2T1.inputs.apply_to_files = [additional_image_paths]
# coreg_EPI2T1.run()
    
def transform_fmri_to_standard(subject_id, root_path, bids_path, recon_all_path, current_script_dir, write_graph=False):    
    """
    Transform fMRI data to a standard space for a given subject.

    This function sets up and runs a workflow to align fMRI data to standard
    MNI space. It also prepares required data and logs any errors encountered.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    write_graph (bool): Whether to write a workflow graph. Defaults to False.
    root_path (str): The root path for the preprocessing workspace.
    bids_path (str): The path to the BIDS folder. 
    recon_all_path (str): The path to the recon_all directory.
    current_script_dir (str): The directory where this script is located.
    """
    print("##################################################")
    print(f"Processing subject: {subject_id}")
    print("##################################################")

    prepare_data(subject_id)

    print("\n\FMRI TO STANDARD\n\n")

    try:
        # Define the workflow
        fmri2t1_wf = get_fmri2standard_wf(
            [10, 750], # [10, 750] correspond to the first and last volumes with the first 10 removed
            subject_id, 
            os.path.join(current_script_dir, '../../acparams_hcp.txt')
        )

        fmri2t1_wf.base_dir = os.path.join(root_path, "fmri2standard")

        # Set necessary inputs using f-strings
        fmri2t1_wf.inputs.input_node = {
            "T1_img": f"{bids_path}/{subject_id}/ses-01/anat/{subject_id}_ses-01_run-01_T1w.nii.gz",
            "func_bold_ap_img": f"{bids_path}/{subject_id}/ses-01/func/{subject_id}_ses-01_run-01_rest_bold_ap.nii.gz",
            "func_sbref_img": f"{bids_path}/{subject_id}/ses-01/func/{subject_id}_ses-01_run-01_rest_sbref_ap.nii.gz",
            "func_segfm_ap_img": f"{bids_path}/{subject_id}/ses-01/func/{subject_id}_ses-01_run-01_rest_sefm_ap.nii.gz",
            "func_segfm_pa_img": f"{bids_path}/{subject_id}/ses-01/func/{subject_id}_ses-01_run-01_rest_sefm_pa.nii.gz",
            "T1_brain_freesurfer_mask": f"{recon_all_path}/{subject_id}/mri/brain.mgz"
        }

        if write_graph:
            fmri2t1_wf.write_graph()

        # Run the workflow
        fmri2t1_wf.run()

    except Exception as e:
        logging.error(
            "Error in fMRI to Standard Workflow for subject %s: %s", subject_id, e
        )
        return None
    
def execute_coregistration(subject_id, root_path, fmri2standard_folder, heudiconv_folder, coreg_EPI2T1):
    """
    Perform coregistration of BOLD images to standard T1 for a given subject.

    This function converts intermediate files using a script, sets input paths,
    and performs SPM coregistration. It logs errors encountered during execution.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    root_path (str): The root path for data storage.
    fmri2standard_folder (str): Directory within root_path for fMRI to standard transformations.
    heudiconv_folder (str): Directory within root_path for heuristic DICOM conversions.
    coreg_EPT2T1 (spm.Coregister): The coregistration object for aligning EPI to T1.
    """
    try:
        # Create intermediate unzipped .nii files using subprocess
        subprocess.run(
            ["bash", "intermediate-files_SPM-coregister2T1_nii-format.sh",
             "-r", root_path,
             "-f", fmri2standard_folder,
             "-b", heudiconv_folder,
             "-s", subject_id,
             "-m", "to_nii"],
            check=True
        )

        # Define paths using os.path.join
        sbref2T1_path = os.path.join(root_path, fmri2standard_folder, subject_id, "spm_coregister2T1_sbref",
                                     f"{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii")
        bold2T1_path = os.path.join(root_path, fmri2standard_folder, subject_id, "spm_coregister2T1_bold",
                                    f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii")

        # SPM coregistration: Align BOLD to standard T1
        target_path = os.path.join(root_path, heudiconv_folder, subject_id, "ses-01", "anat",
                                   f"{subject_id}_ses-01_run-01_T1w.nii")
        coreg_EPI2T1.inputs.target = target_path
        coreg_EPI2T1.inputs.source = sbref2T1_path
        coreg_EPI2T1.inputs.jobtype = "estimate"
        coreg_EPI2T1.inputs.apply_to_files = [bold2T1_path]

        coreg_EPI2T1.run()

        # Delete intermediate unzipped .nii files using subprocess
        subprocess.run(
            ["bash", "intermediate-files_SPM-coregister2T1_nii-format.sh",
             "-r", root_path,
             "-f", fmri2standard_folder,
             "-b", heudiconv_folder,
             "-s", subject_id,
             "-m", "to_nii_gz"],
            check=True
        )

    except Exception as e:
        logging.error("Error during SPM coregistration for subject %s: %s", subject_id, e)
        with open(os.path.join(root_path, 'fmri2standard', 'errors.txt'), 'a') as f:
            f.write(f"{datetime.now()}\t{subject_id} did not execute cleanly during SPM coregistration\n")

def extract_wm_csf_masks(subject_id, root_path, fmri2standard_folder, recon_all_path):  
    """
    Extract white matter (WM) and cerebrospinal fluid (CSF) masks for nuisance correction.

    This function prepares paths, creates necessary directories, and executes
    a mask extraction script. Errors during this process are logged.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    root_path (str): The root path for data storage.
    fmri2standard_folder (str): Directory within root_path for fMRI to standard transformations.
    recon_all_path (str): Directory for FreeSurfer reconall data.
    """
    print("\n\nNUISANCE CORRECTION\n\n")
    try:
        # Define paths 
        # Im commenting this out because it is not used in the function - but leaving in case it is needed for something
        # sbref2T1_path = os.path.join(root_path, fmri2standard_folder, subject_id, "spm_coregister2T1_sbref",
        #     f"{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz")
        bold2T1_path = os.path.join(
            root_path, fmri2standard_folder, subject_id,
            "spm_coregister2T1_bold",
            f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz"
        )
        output_masks = os.path.join(root_path, "nuisance_correction", subject_id, "masks_csf_wm")
        aseg_folder = os.path.join(recon_all_path, subject_id, "mri", "aseg.mgz")

        # Create necessary directories
        os.makedirs(os.path.join(root_path, "nuisance_correction", subject_id), exist_ok=True)
        os.makedirs(output_masks, exist_ok=True)

        # Execute the mask extraction script with subprocess
        try:
            subprocess.run(
                ["bash", "extract_wm_csf_eroded_masks.sh",
                 "-s", subject_id,
                 "-a", aseg_folder,
                 "-r", bold2T1_path,
                 "-o", output_masks,
                 "-b", bold2T1_path,
                 "-e", "2"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            logging.error("Error during WM and CSF mask extraction for subject %s: %s", subject_id, e)
            return None

    except Exception as e:
        logging.error("Error in extracting WM and CSF masks for subject %s: %s", subject_id, e)
        return None

## STOPPED HERE - NEED TO FINISH THE REST OF THE FUNCTIONS    
def run_nuisance_regression(subject_id, root_path):
    """
    Run the nuisance regression workflow using pre-extracted masks.

    This function sets up a workflow for removing nuisance signals from fMRI data.
    Errors during this process are logged.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    root_path (str): The root path for data storage.
    """
    try:
        wf_reg = get_nuisance_regressors_wf(
            outdir=os.path.join(root_path, "nuisance_correction"),
            subject_id=subject_id,
            timepoints=740
        )

        # Set necessary inputs
        wf_reg.inputs.input_node.realign_movpar_txt = os.path.join(
            root_path, "fmri2standard", subject_id, "realign_fmri2SBref",
            f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf.nii.gz.par"
        )
        wf_reg.inputs.input_node.rfmri_unwarped_imgs = os.path.join(
            root_path, "fmri2standard", subject_id, "spm_coregister2T1_bold",
            f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz"
        )
        wf_reg.inputs.input_node.mask_wm = os.path.join(
            root_path, "nuisance_correction", subject_id, "masks_csf_wm", "wm_binmask.nii.gz"
        )
        wf_reg.inputs.input_node.mask_csf = os.path.join(
            root_path, "nuisance_correction", subject_id, "masks_csf_wm", "csf_binmask.nii.gz"
        )
        wf_reg.inputs.input_node.bold_img = os.path.join(
            root_path, "fmri2standard", subject_id, "spm_coregister2T1_bold",
            f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz"
        )

        wf_reg.run()

    except Exception as e:
        logging.error("Error in nuisance workflow for subject %s: %s", subject_id, e)
        return None 

def mni_normalization(subject_id):
    """
    Perform MNI normalization on T1 and fMRI images for the given subject.

    This function sets up the necessary files, decompresses them, and runs the SPM
    normalization process. Errors during this process are logged.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    """
    print("\n\nMNI NORMALIZATION\n\n")
    try:
        # Create necessary directory for normalization
        normalization_dir = os.path.join(root_path, "normalization", subject_id)
        os.makedirs(normalization_dir, exist_ok=True)

        # Setup file paths 
        T1_niigz = os.path.join(bids_path, subject_id, "ses-01", "anat", f"{subject_id}_ses-01_run-01_T1w.nii.gz")
        T1_niigzcopy = os.path.join(normalization_dir, f"{subject_id}_ses-01_run-01_T1w.nii.gz")
        T1_nii = os.path.join(normalization_dir, f"{subject_id}_ses-01_run-01_T1w.nii")

        # Copy and decompress T1 image
        shutil.copy(T1_niigz, T1_niigzcopy)
        with gzip.open(T1_niigzcopy, 'rb') as f_in:
            with open(T1_nii, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Setup paths for sbref and bold images
        bold_niigz = os.path.join(fmri2standard_path, subject_id, "spm_coregister2T1_bold",
                                  f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz")
        bold_niigzcopy = os.path.join(normalization_dir,
                                      f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz")
        bold_nii = os.path.join(normalization_dir,
                                f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii")

        sbref_niigz = os.path.join(fmri2standard_path, subject_id, "spm_coregister2T1_sbref",
                                   f"{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz")
        sbref_niigzcopy = os.path.join(normalization_dir,
                                       f"{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz")
        sbref_nii = os.path.join(normalization_dir,
                                 f"{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii")

        # Copy and decompress bold image
        shutil.copy(bold_niigz, bold_niigzcopy)
        with gzip.open(bold_niigzcopy, 'rb') as f_in:
            with open(bold_nii, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Copy and decompress sbref image
        shutil.copy(sbref_niigz, sbref_niigzcopy) 
        with gzip.open(sbref_niigzcopy, 'rb') as f_in: # Decompress sbref image
            with open(sbref_nii, 'wb') as f_out: # Write decompressed sbref image as .nii
                shutil.copyfileobj(f_in, f_out) # Copy decompressed sbref image

        # Perform MNI normalization
        MNI = spm.preprocess.Normalize12()
        MNI.inputs.image_to_align = T1_nii
        MNI.inputs.apply_to_files = [sbref_nii, bold_nii]
        MNI.inputs.write_bounding_box = [[-90, -126, -72], [90, 90, 108]]
        MNI.run()

    except Exception as e:
        logging.error("Error during MNI Normalization for subject %s: %s", subject_id, e)
        return None
    
def apply_nuisance_correction(subject_id):
    """
    Apply nuisance correction for a given subject.

    This function handles gzip compression, file movements, and uses FSL tools
    for regressing out nuisance signals from the data.

    Parameters:
    subject_id (str): Identifier for the subject being processed.
    """
    print("\n\\APPLYING NUISANCE CORRECTION\n\n")
    try:
        # Define paths
        nuisance_filter_bash = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/command.txt"
        new_name_nii = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/normalization/{subject_id}/w{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii"
        
        mni_pre = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/normalization/{subject_id}/w{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz"
        mni_post = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz"
        nuisances = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/merge_nuisance_txt/all_nuisances.txt"

        nuisance_output = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt.nii.gz"
        native_name = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz"
        MNI_name = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz"
        
        # Perform gzip compression and remove the NII file
        subprocess.run(["gzip", "-f", new_name_nii], check=True)
        os.remove(new_name_nii)
        nuisance_output = f"{nuisance_filter_bash}_output.nii.gz"
        
        # Move and filter using the FSL regfilt command
        shutil.move(nuisance_output, mni_post)
        command_nuisance = [
            "fsl_regfilt", "-i", mni_pre, "-o", mni_post,
            "-d", nuisances, "-f", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"
        ]
        subprocess.run(command_nuisance, check=True)

    except Exception as e:
        logging.error("Error during nuisance correction for subject %s: %s", subject_id, e)
        return None
    
def fmri_quality_control(subject_id):
    """
    Perform fMRI quality control for a given subject.

    This function involves calculating the framewise displacement and setting
    up the brain mask and DVARS workflow.

    DVARS quantifies the amount of change in activity from one volume to the next, 
    across the entire brain and is useful in identifying motion artifacts and 
    sudden spikes or other anomalies in fMRI.

    Parameters:
    subject_id (str): Identifier for the subject being processed.
    """
    print("\n\nFMRI QC\n\n")
    try:
        # Create QC directory
        qc_dir = os.path.join(root_path, "QC", subject_id)
        os.makedirs(qc_dir, exist_ok=True)

        # Run framewise displacement
        fwd_inputs = {
            "in_file": os.path.join(fmri2standard_path, subject_id, "realign_fmri2SBref",
                                    f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf.nii.gz.par"),
            "parameter_source": "FSL",
            "out_file": os.path.join(qc_dir, "framewise_displ.txt"),
            "save_plot": True,
            "out_figure": os.path.join(qc_dir, "framewise_displ.pdf")
        }

        fwd = FramewiseDisplacement(**fwd_inputs)
        fwd.run()
        
    except Exception as e:
        logging.error("Error during framewise displacement for subject %s: %s", subject_id, e)
        return None

    try:
        # Set additional paths for QA and brain mask
        input_ = os.path.join(fmri2standard_path, subject_id, "binarize_mask",
                              f"{subject_id}_ses-01_run-01_T1w_brain_bin.nii.gz")
        ref = os.path.join(nuisance_correction_path, subject_id, "filter_regressors_bold",
                           f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz")
        omat = os.path.join(qc_dir, "brain_mask", "omat.mat")
        out = os.path.join(qc_dir, "brain_mask", "brain_mask_bin_BOLD_T1.nii.gz")

        os.makedirs(os.path.dirname(omat), exist_ok=True)
        
        # Use subprocess to call FSL commands for transformation
        subprocess.run(["flirt", "-in", input_, "-ref", ref, "-omat", omat], check=True)
        subprocess.run([
            "flirt", "-in", input_, "-applyxfm", "-init", omat, "-out", out,
            "-paddingsize", "0.0", "-interp", "trilinear", "-ref", ref
        ], check=True)
    
    except Exception as e:
        logging.error("Error in brain mask to BOLD transformation for subject %s: %s", subject_id, e)
        return None

    try:
        # Setup workflow for DVARS computation
        wf = Workflow(name=subject_id, base_dir=os.path.join(root_path, "QC"))

        node_input = Node(
            utility.IdentityInterface(fields=["bold_T1", "brain_mask"]),
            name="input_node"
        )
        
        node_dvars = Node(ComputeDVARS(save_all=True), name="dvars_node")

        wf.connect([
            (node_input, node_dvars, [("bold_T1", "in_file")]),
            (node_input, node_dvars, [("brain_mask", "in_mask")])
        ])

        wf.inputs.input_node.bold_T1 = ref
        wf.inputs.input_node.brain_mask = out
        wf.run()
        
    except Exception as e:
        logging.error("Error in DVARS computation for subject %s: %s", subject_id, e)
        return None

    # Cleanup directories
    shutil.rmtree(os.path.join(root_path, heudiconv_folder, subject_id), ignore_errors=True)
    shutil.rmtree(os.path.join(recon_all_path, subject_id), ignore_errors=True)

def check_error(suj, error_file):
    """
    Check if a subject id is present in the error file.

    Parameters:
    suj (str): The subject identifier to search for in the file.
    error_file (str): The path to the error log file.

    Returns:
    bool: True if the identifier is found in the file, False otherwise.
    """
    try:
        with open(error_file, "r") as f:
            for line in f:
                if suj in line:
                    return True
        return False
    except IOError as e:
        logging.error("Error opening or reading file %s: %s", error_file, e)
        return False

def initialize_preprocessing_dirs(bids_dir, done_dir):
    """
    Initialize directories and retrieve the list of subjects to process.

    Parameters:
        bids_dir (str): Directory containing the BIDS datasets.
        done_dir (str): Directory containing already processed data.

    Returns:
        set: A set containing identifiers of subjects yet to be processed.
    """
    todo = set(os.listdir(bids_dir))
    done = set()  # Assuming done_dir contains list of processed subjects

    todo -= done
    todo.discard(".heudiconv")
    todo.discard("error_heurdiconv.sh")

    return todo

def check_preprocessing_files(subject_id):
    """
    Check if required preprocessing files and outputs exist for a subject.

    Parameters:
        subject_id (str): Identifier for the subject to check.

    Returns:
        dict: A dictionary indicating the presence of key processing files
              with boolean values.
    """
    workspace_dir = "/home/mariacabello/wf_workspace/bold_preprocess_SA"

    paths_to_check = {
        "bold_native": os.path.join(workspace_dir, "nuisance_correction", subject_id, "filter_regressors_bold",
                                    f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz"),
        "sbref_native": os.path.join(workspace_dir, "fmri2standard", subject_id, "spm_coregister2T1_sbref",
                                     f"{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz"),
        "bold_mni": os.path.join(workspace_dir, "nuisance_correction", subject_id, "filter_regressors_bold",
                                 f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz"),
        "sbref_mni": [
            os.path.join(workspace_dir, "normalization", subject_id,
                         f"w{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii"),
            os.path.join(workspace_dir, "normalization", subject_id,
                         f"w{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz")
        ],
        "motion": os.path.join(workspace_dir, "fmri2standard", subject_id, "realign_fmri2SBref",
                               f"{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf.nii.gz.par"),
        "nuisance": os.path.join(workspace_dir, "nuisance_correction", subject_id, "merge_nuisance_txt", "all_nuisances.txt"),
        "framew": os.path.join(workspace_dir, "QC", subject_id, "framewise_displ.txt")
    }

    return {key: (any(os.path.exists(path) for path in value) if isinstance(value, list) else os.path.exists(value))
            for key, value in paths_to_check.items()}

def process_subjects(todo):
    """
    Process each subject and identify subjects requiring reprocessing.

    Parameters:
        todo (set): Set of subject identifiers to process.

    Returns:
        tuple: A tuple containing dictionaries of subject processing statuses,
               a dictionary of the count of completed processes, and a list of 
               subjects needing preparation ('no_pre').
    """
    dict_done = {subject_id: check_preprocessing_files(subject_id) for subject_id in todo}
    dict_n_done = {subject_id: sum(status.values()) for subject_id, status in dict_done.items()}

    # Identify subjects needing additional work based on incomplete stages
    tofix = [subject_id for subject_id, status in dict_done.items() if status["motion"] and status["nuisance"]
             and not status["bold_native"] and status["sbref_native"]]

    no_pre = []
    for subject_id in tofix:
        print(f"Subject {subject_id} needs additional processing.")
        nuisances = dict_done[subject_id]['nuisance']
        pre = dict_done[subject_id]['bold_native']
        post = dict_done[subject_id]['bold_native']
        
        if os.path.exists(pre) and not os.path.exists(post):
            subprocess.run(
                f"fsl_regfilt -i {pre} -o {post} -d {nuisances} "
                "-f '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27'",
                shell=True, check=True
            )
        elif not os.path.exists(pre):
            no_pre.append(subject_id)
            print("No previous data for Subject", subject_id)
        else:
            print("Post-processing already exists for Subject", subject_id)

    return dict_done, dict_n_done, no_pre

# Define paths to key directories
bids_directory = "/institut/UB/Superagers/MRI/BIDS"  
processed_directory = '/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp2' 

# Initialize the preprocessing workflow by identifying subjects to process
subjects_to_process = initialize_preprocessing_dirs(bids_directory, processed_directory)

# Process subject data to track the completion statuses and identify setups for reprocessing
processed_status, completed_tasks, subjects_needing_setup = process_subjects(subjects_to_process)

# Set up a multiprocessing pool to parallelize fMRI standard space transformation
pool = Pool(6)
pool.map(transform_fmri_to_standard, subjects_to_process)

# Filter subject list for the coregistration step, excluding those with known errors
coregistration_needed = [
    subject_id for subject_id in subjects_to_process
    if not check_error(subject_id, os.path.join(workspace_dir, "fmri2standard", "errors.txt"))
]

# Perform coregistration on the filtered list of subjects
for subject in coregistration_needed:
    execute_coregistration(subject)

# Filter and assign subjects for nuisance correction, ensuring no prior errors
nuisance_step1_list = [
    subject_id for subject_id in subjects_to_process
    if not check_error(subject_id, os.path.join(workspace_dir, "fmri2standard", "errors.txt"))
]

# Apply nuisance correction for initial step and utilize multiprocessing
pool.map(extract_wm_csf_masks, nuisance_step1_list)

# Further filter subjects for advanced nuisance regression
nuisance_step2_list = [
    subject_id for subject_id in nuisance_step1_list
    if not check_error(subject_id, os.path.join(workspace_dir, "nuisance_correction", "errors_wmcsfextraction.txt"))
]

# Execute advanced nuisance regression
for subject in nuisance_step2_list:
    run_nuisance_regression(subject)

# Proceed with SPM normalization on subjects that completed previous steps without errors
normalization_needed = nuisance_step2_list
for subject in normalization_needed:
    mni_normalization(subject)

# Finalize processing by applying nuisance correction and quality control checks
final_qc_subjects = [
    subject_id for subject_id in normalization_needed
    if not check_error(subject_id, os.path.join(workspace_dir, "normalization", "errors.txt"))
]

# Set up a new multiprocessing pool for final QC and correction
pool = Pool(8)
pool.map(apply_nuisance_correction, final_qc_subjects)

def main():
    # Setup logging to file and console
    logging.basicConfig(level=logging.INFO, filename="logs.log", filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Define root path for preprocessing workspace
    root_path = '/home/mariacabello/wf_workspace/bold_preprocess_SA'

    # Define folder names
    heudiconv_folder = 'func_anat'
    fmri2standard_folder = 'fmri2standard'

    # Construct full paths using os.path.join for better readability and cross-platform compatibility
    source_dir = os.path.join("/institut/UB/Superagers/MRI/freesurfer-reconall")
    recon_all_path = os.path.join(root_path, 'recon_all')
    bids_path = os.path.join(root_path, heudiconv_folder)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    fmri2standard_path = os.path.join(root_path, fmri2standard_folder)
    nuisance_correction_path = os.path.join(root_path, 'nuisance_correction')
    qc_path = os.path.join(root_path, 'QC')

    # Initialize preprocessing directories and handle subjects
    bids_directory = "/institut/UB/Superagers/MRI/BIDS"  
    processed_directory = '/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp2' 
    subjects_to_process = initialize_preprocessing_dirs(bids_directory, processed_directory)

    processed_status, completed_tasks, subjects_needing_setup = process_subjects(subjects_to_process)

    # Define the SPM coregistration object for aligning EPI to T1
    coreg_EPI2T1 = spm.Coregister()

    # Run the preprocessing workflows
    run_workflows(bids_path, fmri2standard_path, nuisance_correction_path, recon_all_path, qc_path)

    # Parallel processing for workflows
    with Pool(6) as pool:
        pool.map(transform_fmri_to_standard, subjects_to_process)

        coregistration_needed = [
            subject_id for subject_id in subjects_to_process
            if not check_error(subject_id, os.path.join(fmri2standard_path, "errors.txt"))
        ]
        for subject in coregistration_needed:
            execute_coregistration(subject)

        nuisance_step1_list = [
            subject_id for subject_id in subjects_to_process
            if not check_error(subject_id, os.path.join(fmri2standard_path, "errors.txt"))
        ]
        pool.map(extract_wm_csf_masks, nuisance_step1_list)

        # More processing steps abstracted similarly...

    final_qc_subjects = [
        subject_id for subject_id in normalization_needed
        if not check_error(subject_id, os.path.join(qc_path, "errors.txt"))
    ]

    with Pool(8) as pool:
        pool.map(apply_nuisance_correction, final_qc_subjects)

if __name__ == "__main__":
    main()