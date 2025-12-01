#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script executes fMRI preprocessing workflows with configurable paths."""

import logging
import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
import gzip

# Custom workflow imports
from bold2T1_wf import get_fmri2standard_wf
from bold_nuisance_correction_wf import get_nuisance_regressors_wf
from nipype import Node, Workflow

# Nipype interfaces
from nipype.algorithms.confounds import ComputeDVARS, FramewiseDisplacement
from nipype.interfaces import spm, utility

# Set up logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)


def transform_fmri_to_standard(
    subject_id, ses, root_path, bids_path, recon_all_path, acparams_file, write_graph=False
) -> str:
    """Transform fMRI data to a standard space for a given subject.

    This function sets up and runs a workflow to align fMRI data to standard
    space where each volume is aligned with the first. It also prepares
    required data and logs any errors encountered.

    Args:
        subject_id (str): The identifier for the subject being processed.
        ses (str): The session or timepoint for the data.
        root_path (str): The path to output the products of preprocessing.
        bids_path (str): The path to the shared BIDS folder.
        recon_all_path (str): The path to the shared recon_all directory.
        acparams_file (str): The path to the acparams.txt file.
        write_graph (bool): Whether to write a workflow graph. Defaults to False.

    Returns:
        list: A list of subjects that have completed the transformation.
    """
    print("##################################################")
    print(f"Processing subject: {subject_id}")
    print("##################################################")

    # prepare_data(subject_id)

    print("\n\n01. FMRI TO STANDARD\n\n")

    try:
        # Define the workflow to get the acparams file
        fmri2t1_wf = get_fmri2standard_wf(
            [10, 750],  # [10, 750] correspond to the first and last volumes with the first 10 removed
            subject_id,
            acparams_file,
        )

        # Set the base directory for the workflow
        fmri2t1_wf.base_dir = os.path.join(root_path, "fmri2standard")

        # Directly set inputs
        fmri2t1_wf.inputs.input_node.T1_img = (
            f"{bids_path}/{subject_id}/{ses}/anat/{subject_id}_{ses}_run-01_T1w.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.func_bold_ap_img = (
            f"{bids_path}/{subject_id}/{ses}/func/{subject_id}_{ses}_task-rest_dir-ap_run-01_bold.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.func_sbref_img = (
            f"{bids_path}/{subject_id}/{ses}/func/{subject_id}_{ses}_task-rest_dir-ap_run-01_sbref.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.func_segfm_ap_img = (
            f"{bids_path}/{subject_id}/{ses}/fmap/{subject_id}_{ses}_acq-restsefm_dir-ap_run-01_epi.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.func_segfm_pa_img = (
            f"{bids_path}/{subject_id}/{ses}/fmap/{subject_id}_{ses}_acq-restsefm_dir-pa_run-01_epi.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.T1_brain_freesurfer_mask = f"{recon_all_path}/{subject_id}_{ses}_run-01/mri/brain.mgz"

        if write_graph:
            fmri2t1_wf.write_graph()

        # Run the workflow
        fmri2t1_wf.run()

    except Exception as e:
        logging.error("Error in fMRI to Standard Workflow for subject %s: %s", subject_id, e)
        return

    return subject_id


def execute_coregistration(subject_id, ses, root_path, bids_path, coreg_EPI2T1):
    """Perform coregistration of BOLD images to standard T1 for a given subject.

    This function converts intermediate files, sets input paths, and performs
    SPM coregistration. It logs any errors encountered during execution.

    Args:
        subject_id (str): The identifier for the subject being processed.
        ses (str): The session or timepoint for the data.
        root_path (str): The root path for preprocessing.
        bids_path (str): Path to the shared BIDS folder.
        coreg_EPI2T1 (spm.Coregister): The coregistration object for aligning BOLD to T1.

    Returns:
        list: A list of subjects that have completed coregistration.
    """
    print("\n\n02. COREGISTRATION\n\n")

    try:
        # Define paths
        sbref_dir = os.path.join(root_path, "fmri2standard", subject_id, "spm_coregister2T1_sbref")
        bold_dir = os.path.join(root_path, "fmri2standard", subject_id, "spm_coregister2T1_bold")
        anat_dir = os.path.join(bids_path, subject_id, ses, "anat")

        # Create directories if they don't exist
        os.makedirs(sbref_dir, exist_ok=True)
        os.makedirs(bold_dir, exist_ok=True)

        # Unzip and prepare files
        sbref_source = os.path.join(
            root_path,
            "fmri2standard",
            subject_id,
            "apply_topup_to_SBref",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected.nii",
        )
        sbref_dest = os.path.join(
            sbref_dir, f"{subject_id}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii"
        )
        shutil.copy(sbref_source, sbref_dest)

        # Make a directory to store copied files locally as I have no permission to do it in the BIDS folder
        anat_dir_local = os.path.join(root_path,"BIDS/anat")
        os.makedirs(anat_dir_local, exist_ok=True)

        t1w_source = os.path.join(anat_dir, f"{subject_id}_{ses}_run-01_T1w.nii.gz")
        t1w_dest_uncompressed = os.path.join(anat_dir_local, f"{subject_id}_{ses}_run-01_T1w.nii")
        
        with gzip.open(t1w_source, 'rb') as f_in:
            with open(t1w_dest_uncompressed, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        bold_source = os.path.join(
            root_path,
            "fmri2standard",
            subject_id,
            "apply_topup",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected.nii",
        )
        bold_dest = os.path.join(
            bold_dir, f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii"
        )
        shutil.copy(bold_source, bold_dest)

        # SPM coregistration: Align BOLD to standard T1
        coreg_EPI2T1.inputs.target = t1w_dest_uncompressed
        coreg_EPI2T1.inputs.source = sbref_dest
        coreg_EPI2T1.inputs.jobtype = "estimate"
        coreg_EPI2T1.inputs.apply_to_files = [bold_dest]

        coreg_EPI2T1.run()

        # Zip back and clean up the original files AFTER all processing is done
        if not os.path.exists(f"{sbref_dest}.gz"):    # Check if the gzipped file already exists
            subprocess.run(["gzip", sbref_dest], check=True)

        if not os.path.exists(f"{bold_dest}.gz"):    # Check if the gzipped file already exists
            subprocess.run(["gzip", bold_dest], check=True)

    except Exception as e:
        logging.error("Error during SPM coregistration for subject %s: %s", subject_id, e)
        return

    return subject_id


def extract_wm_csf_masks(subject_id, ses, root_path, recon_all_path, erode=1):
    """Extract white matter (WM) and cerebrospinal fluid (CSF) masks for nuisance correction.

    This function prepares paths, creates necessary directories, and executes
    a mask extraction script.

    Args:
        subject_id (str): The identifier for the subject being processed.
        ses (str): The session or timepoint for the data.
        root_path (str): The path to output the products of preprocessing.
        recon_all_path (str): The path to the shared recon-all folder.
        erode (int): Erosion amount in mm. Defaults to 1.

    Returns:
        list: A list of subjects that have completed the mask extraction.
    """
    print("\n\n03. NUISANCE CORRECTION\n\n")

    try:
        # Define paths
        bold_filename = f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii.gz"
        bold2T1_path = os.path.join(root_path, "fmri2standard", subject_id, "spm_coregister2T1_bold", bold_filename)
        output_masks = os.path.join(root_path, "nuisance_correction", subject_id, "masks_csf_wm")
        aseg_folder = os.path.join(recon_all_path, f"{subject_id}_{ses}_run-01", "mri", "aseg.mgz")

        # Create necessary directories
        os.makedirs(os.path.join(root_path, "nuisance_correction", subject_id), exist_ok=True)
        os.makedirs(output_masks, exist_ok=True)

        try:
            # WM processing steps
            wm_mgz = os.path.join(output_masks, "wm.mgz")
            wm_sbref_mgz = os.path.join(output_masks, "wm_sbref.mgz")
            wm_nii_gz = os.path.join(output_masks, "wm.nii.gz")
            wm_binmask_nii_gz = os.path.join(output_masks, "wm_binmask.nii.gz")
            wm_bold_extracted_nii_gz = os.path.join(output_masks, "wm_bold_extracted.nii.gz")

            # Execute commands for WM
            subprocess.run(
                ["mri_binarize", "--i", aseg_folder, "--match", "2", "41", "--o", wm_mgz, "--erode", str(erode)],
                check=True,
            )
            subprocess.run(
                ["mri_vol2vol", "--mov", wm_mgz, "--regheader", "--targ", bold2T1_path, "--o", wm_sbref_mgz], check=True
            )
            subprocess.run(
                ["mri_convert", "--in_type", "mgz", "--out_type", "nii", wm_sbref_mgz, wm_nii_gz], check=True
            )
            subprocess.run(["fslmaths", wm_nii_gz, "-bin", wm_binmask_nii_gz], check=True)
            subprocess.run(["fslmaths", bold2T1_path, "-mul", wm_binmask_nii_gz, wm_bold_extracted_nii_gz], check=True)

            # CSF processing steps
            csf_mgz = os.path.join(output_masks, "csf.mgz")
            csf_sbref_mgz = os.path.join(output_masks, "csf_sbref.mgz")
            csf_nii_gz = os.path.join(output_masks, "csf.nii.gz")
            csf_binmask_nii_gz = os.path.join(output_masks, "csf_binmask.nii.gz")
            csf_bold_extracted_nii_gz = os.path.join(output_masks, "csf_bold_extracted.nii.gz")

            # Execute commands for CSF
            subprocess.run(
                [
                    "mri_binarize",
                    "--i",
                    aseg_folder,
                    "--match",
                    "4",
                    "5",
                    "14",
                    "15",
                    "24",
                    "43",
                    "44",
                    "--o",
                    csf_mgz,
                    "--erode",
                    str(erode),
                ],
                check=True,
            )
            subprocess.run(
                ["mri_vol2vol", "--mov", csf_mgz, "--regheader", "--targ", bold2T1_path, "--o", csf_sbref_mgz],
                check=True,
            )
            subprocess.run(
                ["mri_convert", "--in_type", "mgz", "--out_type", "nii", csf_sbref_mgz, csf_nii_gz], check=True
            )
            subprocess.run(["fslmaths", csf_nii_gz, "-bin", csf_binmask_nii_gz], check=True)
            subprocess.run(
                ["fslmaths", bold2T1_path, "-mul", csf_binmask_nii_gz, csf_bold_extracted_nii_gz], check=True
            )

        except subprocess.CalledProcessError as e:
            logging.error("Error during WM and CSF mask extraction for subject %s: %s", subject_id, e)
            return

    except Exception as e:
        logging.error("Error in extracting WM and CSF masks for subject %s: %s", subject_id, e)
        return

    return subject_id


def run_nuisance_regression(subject_id, ses, root_path):
    """Run the nuisance regression workflow using pre-extracted masks.

    This function sets up a workflow for removing nuisance signals from fMRI data
    and logs any errors during the process.

    Args:
        subject_id (str): The identifier for the subject being processed.
        ses (str): The session or timepoint for the data.
        root_path (str): The path to output the products of preprocessing.

    Returns:
        list: A list of subjects that have completed the nuisance regression.
    """
    try:
        wf_reg = get_nuisance_regressors_wf(
            outdir=os.path.join(root_path, "nuisance_correction"), subject_id=subject_id, timepoints=740
        )

        # Define paths for the workflow
        wf_reg.inputs.input_node.realign_movpar_txt = os.path.join(
            root_path,
            "fmri2standard",
            subject_id,
            "realign_fmri2SBref",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf.nii.par",
        )
        wf_reg.inputs.input_node.rfmri_unwarped_imgs = os.path.join(
            root_path,
            "fmri2standard",
            subject_id,
            "spm_coregister2T1_bold",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii.gz",
        )
        wf_reg.inputs.input_node.mask_wm = os.path.join(
            root_path, "nuisance_correction", subject_id, "masks_csf_wm", "wm_binmask.nii"
        )
        wf_reg.inputs.input_node.mask_csf = os.path.join(
            root_path, "nuisance_correction", subject_id, "masks_csf_wm", "csf_binmask.nii"
        )
        wf_reg.inputs.input_node.bold_img = os.path.join(
            root_path,
            "fmri2standard",
            subject_id,
            "spm_coregister2T1_bold",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii.gz",
        )

        # Writes the graph and runs the workflow
        wf_reg.run()

    except Exception as e:
        logging.error("Error in nuisance workflow for subject %s: %s", subject_id, e)
        return

    return subject_id


def mni_normalization(subject_id, ses, root_path, bids_path):
    """Perform MNI normalization on T1 and fMRI images for the given subject.

    This function sets up the necessary files, decompresses them, and runs
    the SPM normalization process. Errors during this process are logged.

    Args:
        subject_id (str): The identifier for the subject being processed.
        ses (str): The session or timepoint for the data.
        root_path (str): The path to output the products of preprocessing.
        bids_path (str): The path to the shared BIDS folder.

    Returns:
        list: A list of subjects that have completed MNI normalization.
    """
    print("\n\n04. MNI NORMALIZATION\n\n")
    try:
        # Create necessary directory for normalization
        normalization_dir = os.path.join(root_path, "normalization", subject_id)
        os.makedirs(normalization_dir, exist_ok=True)

        # Setup file paths to T1 images
        T1_niigz = os.path.join(bids_path, subject_id, ses, "anat", f"{subject_id}_{ses}_run-01_T1w.nii.gz")
        T1_niigzcopy = os.path.join(normalization_dir, f"{subject_id}_{ses}_run-01_T1w.nii.gz")
        T1_nii = os.path.join(normalization_dir, f"{subject_id}_{ses}_run-01_T1w.nii")

        # Copy and decompress T1 images
        shutil.copy(T1_niigz, T1_niigzcopy)
        os.chmod(T1_niigzcopy, 0o644)  # Set permissions to be readable
        subprocess.run(["gunzip", "-f", T1_niigzcopy], check=True)

        # Setup paths for sbref and bold images
        bold_niigz = os.path.join(
            root_path,
            "fmri2standard",
            subject_id,
            "spm_coregister2T1_bold",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii.gz",
        )
        bold_niigzcopy = os.path.join(
            normalization_dir,
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii.gz",
        )
        bold_nii = os.path.join(
            normalization_dir, f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii"
        )
        sbref_niigz = os.path.join(
            root_path,
            "fmri2standard",
            subject_id,
            "spm_coregister2T1_sbref",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii.gz",
        )
        sbref_niigzcopy = os.path.join(
            normalization_dir,
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii.gz",
        )
        sbref_nii = os.path.join(
            normalization_dir, f"{subject_id}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii"
        )

        # Copy and decompress bold images
        shutil.copy(bold_niigz, bold_niigzcopy)
        subprocess.run(["gunzip", "-f", bold_niigzcopy], check=True)

        # Copy and decompress sbref images
        shutil.copy(sbref_niigz, sbref_niigzcopy)
        subprocess.run(["gunzip", "-f", sbref_niigzcopy], check=True)

        # Perform MNI normalization for sbref
        MNI = spm.preprocess.Normalize12()
        MNI.inputs.image_to_align = T1_nii
        MNI.inputs.apply_to_files = [sbref_nii]
        MNI.inputs.write_bounding_box = [
            [-90, -126, -72],
            [90, 90, 108],
        ]  # This encompass the whole brain in standard MNI space
        MNI.run()

        # Perform MNI normalization seperately for bold (NOTE that when this is run in the same command as sbref it outputs a truncated file, unclear why)
        MNI = spm.preprocess.Normalize12()
        MNI.inputs.image_to_align = T1_nii
        MNI.inputs.apply_to_files = [bold_nii]
        MNI.inputs.write_bounding_box = [
            [-90, -126, -72],
            [90, 90, 108],
        ]  # This encompass the whole brain in standard MNI space
        MNI.run()

    except Exception as e:
        logging.error("Error during MNI Normalization for subject %s: %s", subject_id, e)
        return

    return subject_id


def apply_nuisance_correction(subject_id, ses, root_path) -> Optional[str]:
    """Apply nuisance correction for a given subject.

    This function handles gzip compression, file movements, and uses FSL tools
    for regressing out nuisance signals from the data. It logs any errors during
    the process.

    Args:
        subject_id (str): Identifier for the subject being processed.
        ses (str): The session or timepoint for the data.
        root_path (str): The path to output the products of preprocessing.

    Returns:
        list: A list of subjects that have completed nuisance correction.
    """
    print("\n\n05. APPLYING NUISANCE CORRECTION\n\n")
    try:
        # Define paths
        new_name_nii = os.path.join(
            root_path,
            "normalization",
            subject_id,
            # w stands for warped which indicates the image has been normalized
            f"w{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1.nii",
        )

        nuisance_output = os.path.join(
            root_path,
            "nuisance_correction",
            subject_id,
            "filter_regressors_bold",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt.nii",
        )

        native_name = os.path.join(
            root_path,
            "nuisance_correction",
            subject_id,
            "filter_regressors_bold",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii",
        )

        output_directory = os.path.join(root_path, "nuisance_correction", subject_id, "filter_regressors_bold")

        # Perform gzip compression and remove the NII file
        subprocess.run(["gzip", "-f", new_name_nii], check=True)
        shutil.move(nuisance_output, native_name)
        subprocess.run(["gzip", "-f", native_name], check=True)

        # Define the path for the compressed file
        mni_pre = f"{new_name_nii}.gz"

        mni_post = os.path.join(
            root_path,
            "nuisance_correction",
            subject_id,
            "filter_regressors_bold",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz",
        )

        nuisances = os.path.join(
            root_path, "nuisance_correction", subject_id, "merge_nuisance_txt", "all_nuisances.txt"
        )

        # Move and filter using the FSL regfilt command
        command_nuisance = [
            "fsl_regfilt",
            "-i",
            mni_pre,
            "-o",
            mni_post,
            "-d",
            nuisances,
            "-f",
            "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        ]
        os.makedirs(output_directory, exist_ok=True)
        # subprocess.run(command_nuisance, check=True)
        result = subprocess.run(
            command_nuisance, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        print(result.stdout)

    except Exception as e:
        logging.error("Error during nuisance correction for subject %s: %s", subject_id, e)
        return

    return subject_id


def fmri_quality_control(subject_id, root_path, ses):
    """Perform fMRI quality control for a given subject.

    This function calculates framewise displacement and sets up the
    brain mask and DVARS workflow. Errors during the process are logged.

    Args:
        subject_id (str): Identifier for the subject being processed.
        root_path (str): The path to output the products of preprocessing.
        ses (str): The session or timepoint for the data.

    Returns:
        list: A list of subjects that have completed the QC process.
    """
    print("\n\n06. FMRI QC\n\n")
    try:
        # Create QC directory
        qc_dir = os.path.join(root_path, "QC", subject_id)
        os.makedirs(qc_dir, exist_ok=True)

        # Run framewise displacement
        fwd_inputs = {
            "in_file": os.path.join(
                root_path,
                "fmri2standard",
                subject_id,
                "realign_fmri2SBref",
                f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf.nii.par",
            ),
            "parameter_source": "FSL",
            "out_file": os.path.join(qc_dir, "framewise_displ.txt"),
            "save_plot": True,
            "out_figure": os.path.join(qc_dir, "framewise_displ.pdf"),
        }

        fwd = FramewiseDisplacement(**fwd_inputs)
        fwd.run()

    except Exception as e:
        logging.error("Error during framewise displacement for subject %s: %s", subject_id, e)
        return

    try:
        # Set additional paths for QA and brain mask
        input_ = os.path.join(
            root_path, "fmri2standard", subject_id, "binarize_mask", f"{subject_id}_{ses}_run-01_T1w_brain_bin.nii.gz"
        )
        ref = os.path.join(
            root_path,
            "nuisance_correction",
            subject_id,
            "filter_regressors_bold",
            f"{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz",
        )
        omat = os.path.join(qc_dir, "brain_mask", "omat.mat")
        out = os.path.join(qc_dir, "brain_mask", "brain_mask_bin_BOLD_T1.nii")

        os.makedirs(os.path.dirname(omat), exist_ok=True)

        # Use subprocess to call FSL commands for transformation
        subprocess.run(["flirt", "-in", input_, "-ref", ref, "-omat", omat], check=True)
        subprocess.run(
            [
                "flirt",
                "-in",
                input_,
                "-applyxfm",
                "-init",
                omat,
                "-out",
                out,
                "-paddingsize",
                "0.0",
                "-interp",
                "trilinear",
                "-ref",
                ref,
            ],
            check=True,
        )

    except Exception as e:
        logging.error("Error in brain mask to BOLD transformation for subject %s: %s", subject_id, e)
        return

    try:
        # Setup workflow for DVARS computation
        wf = Workflow(name=subject_id, base_dir=os.path.join(root_path, "QC"))

        node_input = Node(utility.IdentityInterface(fields=["bold_T1", "brain_mask"]), name="input_node")

        node_dvars = Node(ComputeDVARS(save_all=True), name="dvars_node")

        wf.connect([
            (node_input, node_dvars, [("bold_T1", "in_file")]),
            (node_input, node_dvars, [("brain_mask", "in_mask")]),
        ])

        wf.inputs.input_node.bold_T1 = ref
        wf.inputs.input_node.brain_mask = out
        wf.run()

    except Exception as e:
        logging.error("Error in DVARS computation for subject %s: %s", subject_id, e)
        return

    return subject_id


def prepare_and_copy_preprocessed_data(subject_id, ses, root_path, output_path):
    """Prepare necessary directories and copy preprocessed data for a given subject.

    Args:
        subject_id (str): The identifier for the subject.
        ses (str): The session or timepoint for the data.
        root_path (str): The path to output the products of preprocessing.
        output_path (str): The path to the output directory for the final preprocessed data.

    Returns:
        str: The identifier of the subject for which the data was copied.
    """
    print("\n\n07. COPY PREPROCESSED DATA\n\n")
    # Directory paths for the outputs
    dirs = {
        "native_t1": os.path.join(output_path, subject_id, ses, "native_T1"),
        "mni_2mm": os.path.join(output_path, subject_id, ses, "MNI_2mm"),
    }

    # Create directories if they don't exist
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # First zip the files so they match output format

    # Define paths for MNI BOLD file
    bold_file = f"{root_path}/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii"
    bold_file_gz = f"{bold_file}.gz"

    # Check existence for the zipped BOLD file and if it does not exist, zip it. Also, make sure the BOLD file exists.
    if not os.path.exists(bold_file_gz) and os.path.exists(bold_file):
        subprocess.run(
            ["gzip", "-f", bold_file],
            check=True,
        )

    # Define paths for the sbref file
    sbref_file = f"{root_path}/normalization/{subject_id}/w{subject_id}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii"
    sbref_file_gz = f"{sbref_file}.gz"

    # Check existence for the zipped sbref file and if it does not exist, zip it. Also, make sure the sbref file exists.
    if not os.path.exists(sbref_file_gz) and os.path.exists(sbref_file):
        subprocess.run(
            ["gzip", "-f", sbref_file],
            check=True,
        )

    # Define source and destination paths using a dictionary
    paths = {
        "bold_native": (
            f"{root_path}/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz",
            f"{dirs['native_t1']}/{subject_id}_{ses}_run-01_rest_bold_ap_T1-space.nii.gz",
        ),
        "sbref_native": (
            f"{root_path}/fmri2standard/{subject_id}/spm_coregister2T1_sbref/{subject_id}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii.gz",
            f"{dirs['native_t1']}/{subject_id}_{ses}_run-01_rest_sbref_ap_T1-space.nii.gz",
        ),
        "framew_native": (
            f"{root_path}/QC/{subject_id}/framewise_displ.txt",
            f"{dirs['native_t1']}/framewise_displ.txt",
        ),
        "motion_native": (
            f"{root_path}/fmri2standard/{subject_id}/realign_fmri2SBref/{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf.nii.par",
            f"{dirs['native_t1']}/motion.txt",
        ),
        "nuisance_native": (
            f"{root_path}/nuisance_correction/{subject_id}/merge_nuisance_txt/all_nuisances.txt",
            f"{dirs['native_t1']}/nuisance_regressors.txt",
        ),
        "bold_mni": (
            f"{root_path}/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz",
            f"{dirs['mni_2mm']}/{subject_id}_{ses}_run-01_rest_bold_ap_MNI-space.nii.gz",
        ),
        "sbref_mni": (
            f"{root_path}/normalization/{subject_id}/w{subject_id}_{ses}_task-rest_dir-ap_run-01_sbref_flirt_corrected_coregistered2T1.nii.gz",
            f"{dirs['mni_2mm']}/{subject_id}_{ses}_run-01_rest_sbref_ap_MNI-space.nii.gz",
        ),
        "framew_mni": (
            f"{root_path}/QC/{subject_id}/framewise_displ.txt",
            f"{dirs['mni_2mm']}/framewise_displ.txt",
        ),
        "motion_mni": (
            f"{root_path}/fmri2standard/{subject_id}/realign_fmri2SBref/{subject_id}_{ses}_task-rest_dir-ap_run-01_bold_roi_mcf.nii.par",
            f"{dirs['mni_2mm']}/motion.txt",
        ),
        "nuisance_mni": (
            f"{root_path}/nuisance_correction/{subject_id}/merge_nuisance_txt/all_nuisances.txt",
            f"{dirs['mni_2mm']}/nuisance_regressors.txt",
        ),
    }

    # Function to attempt to copy files
    def copy_files(src, dst):
        try:
            shutil.copy(src, dst)
            logging.info("Successfully copied %s to %s", src, dst)
        except FileNotFoundError:
            logging.error("File not found during copy for subject %s: %s", subject_id, src)
            return False
        except Exception as e:
            logging.error("Failed to copy from %s to %s for subject %s: %s", src, dst, subject_id, e)
            return False
        return True

    # Attempt to copy all files, tracking success
    all_copied = True
    for src, dst in paths.values():
        if not copy_files(src, dst):
            all_copied = False

    # Only return the subject_id if all files were copied successfully
    return subject_id if all_copied else None


def change_logger_file(file_name: str):
    """Configure the logging settings for a specific processing step.

    This function sets up a logging configuration that writes logs to a file
    named after the specific step being processed.

    Args:
        file_name (str): The name of the file to write logs to.
    """
    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    file_handler = logging.FileHandler(f"{file_name}.log", mode="w")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)


def main():
    """Main function for executing the preprocessing workflow of fMRI data.

    This function performs the following steps:
    1. Transforms fMRI data to standard space using parallel processing.
    2. Executes coregistration post transformation.
    3. Applies nuisance correction, including:
       a. Extraction of white matter (WM) and cerebrospinal fluid (CSF) masks.
       b. Advanced nuisance regression.
    4. Conducts MNI normalization.
    5. Removes nuisance regression artifacts.
    6. Executes quality control on fMRI data.
    7. Prepares and copies preprocessed data to the output directory.

    The process makes use of multiprocessing to parallelize tasks where possible,
    and logs any subjects with errors at each step.

    Paths for inputs and logs are defined relative to the workflow's root directory.
    """
    # Define all paths and directories for the preprocessing workflow
    ses = "ses-02"
    root_path = "/home/rachel/Desktop/Preprocessing/bbhi"
    bids_path = "/pool/guttmann/institut/BBHI/MRI/BIDS"
    recon_all_path = "/pool/guttmann/institut/BBHI/MRI/derivatives/freesurfer-reconall"
    acparams_file = Path("/pool/guttmann/laboratori/main_preprocessingBOLD/updated_preprocessing/acparams_hcp.txt")
    output_path = "/home/rachel/Desktop/Preprocessing/bbhi/resting_preprocessed"

    # Create root dir if it doesn't exist
    os.makedirs(root_path, exist_ok=True)

    # Older paths for test running
    # bids_path = "/home/rachel/Desktop/institute/UB/Superagers/MRI/BIDS"

    # Set up FSL so it runs correctly in this script
    # Change file paths as needed
    os.environ["FSLDIR"] = "/vol/software/fsl"
    os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:" + os.environ["PATH"]
    subprocess.run(["bash", "-c", "source /vol/software/fsl/etc/fslconf/fsl.sh"], check=True)

    # Set FSL to output uncompressed NIFTI files
    os.environ["FSLOUTPUTTYPE"] = "NIFTI"

    # Set up FreeSurfer so it runs with the same version used in BBHI
    os.environ["FREESURFER_HOME"] = "/vol/software/freesurfer-6.0" 
    os.environ["PATH"] = f"{os.environ['FREESURFER_HOME']}/bin:" + os.environ["PATH"]

    # Configure MATLAB so it runs correctly in this script
    mlab_cmd = "/usr/local/bin/matlab -nodesktop -nosplash"

    # Set the SPM paths so it runs correctly in this script
    spm.SPMCommand.set_mlab_paths(paths="/home/rachel/spm12", matlab_cmd=mlab_cmd)

    # Define the SPM coregistration object
    coreg_EPI2T1 = spm.Coregister()

    # Run `initialize_preprocessing_dirs` to retrieve the list of subjects to process
    subjects_to_process = ["sub-126271", "sub-167505", "sub-85733"]

    print(f"Subjects to process: {len(subjects_to_process)} {subjects_to_process}")

    ########################################
    #### Run the preprocessing workflow ####
    ########################################

    # Step 1. Transform fMRI data to standard space
    # Setup logging
    change_logger_file("log_01_transform_fmri_to_standard")

    # Define the partial function with fixed arguments
    transform_partial_fmri_to_standard = partial(
        transform_fmri_to_standard,
        root_path=root_path,
        ses=ses,
        bids_path=bids_path,
        recon_all_path=recon_all_path,
        acparams_file=acparams_file,
    )

    # Set up a multiprocessing pool to parallelize fMRI standard space transformation
    with Pool(3) as pool:
        coregistration_list = pool.map(transform_partial_fmri_to_standard, subjects_to_process)

    # Filter out any None values from the results. None gets returned when an error occurs
    coregistration_list = [subject for subject in coregistration_list if subject is not None]

    # Step 2. Execute coregistration
    print(f"Subjects to process for coregistration: {len(coregistration_list)} / {len(subjects_to_process)}")

    # Setup logging
    change_logger_file("log_02_execute_coregistration")

    # Define the partial function with fixed arguments
    execute_partial_coregistration = partial(
        execute_coregistration,
        ses=ses,
        root_path=root_path,
        bids_path=bids_path,
        coreg_EPI2T1=coreg_EPI2T1,
    )

    # Set up a multiprocessing pool to parallelize fMRI standard space transformation
    with Pool(3) as pool:
        extract_wm_csf_masks_list = pool.map(execute_partial_coregistration, coregistration_list)

    # Filter out any None values from the results. None gets returned when an error occurs
    extract_wm_csf_masks_list = [subject for subject in extract_wm_csf_masks_list if subject is not None]

    # Step 3a. Extract WM and CSF masks
    print(f"Subjects to process for extracting WM/CSF: {len(extract_wm_csf_masks_list)} / {len(subjects_to_process)}")

    # Setup logging
    change_logger_file("log_03a_extract_wm_csf_masks")

    # Apply nuisance correction initial step using multiprocessing
    transform_partial_extract_wm_csf_masks = partial(
        extract_wm_csf_masks,
        ses=ses,
        root_path=root_path,
        recon_all_path=recon_all_path,
    )
    # Use multiprocessing Pool to apply the function
    with Pool(3) as pool:
        nuisance_regression_list = pool.map(transform_partial_extract_wm_csf_masks, extract_wm_csf_masks_list)

    # Identify subjects needing nuisance correction to run (`run_nuisance_regression`), excluding those with errors from `extract_wm_csf_masks`
    nuisance_regression_list = [subject for subject in nuisance_regression_list if subject is not None]
    print(nuisance_regression_list)

    # Step 3b: Run nuisance regression
    print(f"Subjects to process for nuisance workflow: {len(nuisance_regression_list)} / {len(extract_wm_csf_masks_list)}")

    # Setup logging
    change_logger_file("log_03b_run_nuisance_regression")

    # Apply nuisance regression using multiprocessing
    transform_partial_run_nuisance_regression = partial(
        run_nuisance_regression,
        ses=ses,
        root_path=root_path,
    )

    # Currently have this not running in parallel because it fails due to memory issues when run in parallel
    with Pool(1) as pool:
        mni_normalization_list = pool.map(transform_partial_run_nuisance_regression, nuisance_regression_list)

    mni_normalization_list = [result for result in mni_normalization_list if result is not None]

    # Step 4: Perform MNI normalization
    print(f"Subjects to process for MNI normalization: {len(mni_normalization_list)} / {len(nuisance_regression_list)}")

    # Setup logging
    change_logger_file("log_04_mni_normalization")

    # Perform MNI normalization using multiprocessing
    transform_partial_mni_normalization = partial(
        mni_normalization,
        ses=ses,
        root_path=root_path,
        bids_path=bids_path,
    )

    # This needs to run without parallel processing because (for an unknown reason) it fails when run in parallel
    with Pool(1) as pool:
        regression_list = pool.map(transform_partial_mni_normalization, mni_normalization_list)

    regression_list = [result for result in regression_list if result is not None]

    # Step 5. Apply nuisance correction
    print(f"Subjects to process for applying nuisance: {len(regression_list)} / {len(mni_normalization_list)}")

    # Setup logging
    change_logger_file("log_05_apply_nuisance_correction")

    # Apply nuisance correction using multiprocessing
    transform_partial_apply_nuisance_correction = partial(apply_nuisance_correction, ses=ses, root_path=root_path)

    with Pool(3) as pool:
        qc_list = pool.map(transform_partial_apply_nuisance_correction, regression_list)

    qc_list = [subject for subject in qc_list if subject is not None]

    # Step 6. Perform fMRI quality control
    print(f"Subjects to process for QC: {len(qc_list)} / {len(regression_list)}")

    # Setup logging
    change_logger_file("log_06_fmri_quality_control")

    # Perform fMRI quality control using multiprocessing
    transform_partial_fmri_quality_control = partial(
        fmri_quality_control,
        root_path=root_path,
        ses=ses,
    )

    with Pool(3) as pool:
        copy_subjects = pool.map(transform_partial_fmri_quality_control, qc_list)

    copy_subjects = [subject for subject in copy_subjects if subject is not None]

    # Step 7. Prepare and copy preprocessed data
    print(f"Subjects to process for copying: {len(copy_subjects)} / {len(qc_list)}")

    # Setup logging
    change_logger_file("log_07_prepare_and_copy_preprocessed_data")

    # Prepare and copy preprocessed data using multiprocessing
    transform_partial_prepare_and_copy = partial(
        prepare_and_copy_preprocessed_data, ses=ses, root_path=root_path, output_path=output_path
    )

    with Pool(3) as pool:
        final_results = pool.map(transform_partial_prepare_and_copy, copy_subjects)

    final_results = [subject for subject in final_results if subject is not None]

    failed_subjects = set(subjects_to_process) - set(final_results)
    print(
        f"Completed preprocessing for {len(final_results)} subjects out of a possible {len(subjects_to_process)}.\n\nSubjects that failed:\n"
        + "\n".join(failed_subjects)
    )


if __name__ == "__main__":
    main()
