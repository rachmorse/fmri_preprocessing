#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script executes fMRI preprocessing workflows with configurable paths."""

import gzip
import logging
import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path

# NOTE this is supposed to be needed for the QC framewise displacement figure. Check that this is correct
# Custom workflow imports
from bold2T1_wf import get_fmri2standard_wf
from bold_nuisance_correction_wf import get_nuisance_regressors_wf
from nipype import Node, Workflow

# Nipype interfaces
from nipype.algorithms.confounds import ComputeDVARS, FramewiseDisplacement
from nipype.interfaces import spm, utility

# def prepare_data(subject_id, recon_all_path, source_dir):
#     """
#     Prepare necessary directories and copy MRI data for a given subject.

#     Parameters:
#     subject_id (str): The identifier for the subject whose data is being prepared.
#     recon_all_path (str): The directory path for recon_all data storage.
#     subject_reconall (str): The directory where each subject's recon-all is.
#     mri_dir (str): The directory with each subject's recon-all called MRI.
#     aseg_file (str): The directory with each subject's recon-all with the aseg.mgz file.
#     """
#     try:
#         # Construct paths
#         subject_reconall = os.path.join(recon_all_path, subject_id)
#         mri_dir = os.path.join(subject_reconall, "mri")
#         aseg_file = os.path.join(mri_dir, "aseg.mgz")

#         # Check if the aseg.mgz file is already present
#         if not os.path.isfile(aseg_file):
#             print("Copying aseg.mgz and brain.mgz from institut_recon_all...")

#             # Create required directories
#             os.makedirs(mri_dir, exist_ok=True)

#             # Define source directory where files are copied from
#             source_dir_mri = os.path.join(source_dir, subject_id, "mri")

#             # Copy files to the destination directory
#             shutil.copy(os.path.join(source_dir_mri, "aseg.mgz"), mri_dir)
#             shutil.copy(os.path.join(source_dir_mri, "brain.mgz"), mri_dir)
#         else:
#             print("aseg.mgz and brain.mgz already copied")
#     except Exception as e:
#         logging.error("Error copying aseg.mgz & brain.mgz for subject %s: %s", subject_id, e)


def transform_fmri_to_standard(subject_id, root_path, bids_path, recon_all_path, acparams_file, write_graph=False):
    """Transform fMRI data to a standard space for a given subject.

    This function sets up and runs a workflow to align fMRI data to standard
    space where each volume is aligned with the first. It also prepares
    required data and logs any errors encountered.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    root_path (str): The root path for the preprocessing workspace.
    bids_path (str): The path to the shared BIDS folder.
    recon_all_path (str): The path to the recon_all directory.
    acparams_file (str): The path to the acparams.txt file.
    write_graph (bool): Whether to write a workflow graph. Defaults to False.

    Returns:
    list: A list of subjects that have completed the transformation.
    """
    print("##################################################")
    print(f"Processing subject: {subject_id}")
    print("##################################################")

    # prepare_data(subject_id)

    completed_subjects = []

    print("\n\\FMRI TO STANDARD\n\n")

    try:
        # Define the workflow to get the acparams file
        fmri2t1_wf = get_fmri2standard_wf(
            [10, 750],  # [10, 750] correspond to the first and last volumes with the first 10 removed
            subject_id,
            acparams_file,
        )

        # Set the base directory for the workflow
        fmri2t1_wf.base_dir = os.path.join(root_path, "fmri2standard")

        # Directly set inputs like the working code does
        fmri2t1_wf.inputs.input_node.T1_img = (
            f"{bids_path}/{subject_id}/ses-02/anat/{subject_id}_ses-02_run-01_T1w.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.func_bold_ap_img = (
            f"{bids_path}/{subject_id}/ses-02/func/{subject_id}_ses-02_task-rest_dir-ap_run-01_bold.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.func_sbref_img = (
            f"{bids_path}/{subject_id}/ses-02/func/{subject_id}_ses-02_task-rest_dir-ap_run-01_sbref.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.func_segfm_ap_img = (
            f"{bids_path}/{subject_id}/ses-02/fmap/{subject_id}_ses-02_acq-restsefm_dir-ap_run-01_epi.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.func_segfm_pa_img = (
            f"{bids_path}/{subject_id}/ses-02/fmap/{subject_id}_ses-02_acq-restsefm_dir-pa_run-01_epi.nii.gz"
        )
        fmri2t1_wf.inputs.input_node.T1_brain_freesurfer_mask = f"{recon_all_path}/{subject_id}_ses-02/mri/brain.mgz"

        if write_graph:
            fmri2t1_wf.write_graph()

        # Run the workflow
        fmri2t1_wf.run()

        completed_subjects.append(subject_id)

    except Exception as e:
        logging.error("Error in fMRI to Standard Workflow for subject %s: %s", subject_id, e)
        return completed_subjects

    return completed_subjects


def execute_coregistration(subject_id, root_path, fmri2standard_folder, bids_path, coreg_EPI2T1):
    """Perform coregistration of BOLD images to standard T1 for a given subject.

    This function converts intermediate files, sets input paths, and performs
    SPM coregistration. It logs any errors encountered during execution.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    root_path (str): The root path for preprocessing.
    fmri2standard_folder (str): Directory for storing fMRI to standard transformations.
    bids_path (str): Directory for BIDS data.
    coreg_EPI2T1 (spm.Coregister): The coregistration object for aligning BOLD to T1.

    Returns:
    list: A list of subjects that have completed coregistration.
    """
    completed_subjects = []

    try:
        # Create intermediate unzipped .nii files using subprocess
        subprocess.run(
            [
                "bash",
                "intermediate-files_SPM-coregister2T1_nii-format.sh",
                "-r",
                root_path,
                "-f",
                fmri2standard_folder,
                "-b",
                bids_path,
                "-s",
                subject_id,
                "-m",
                "to_nii",
            ],
            check=True,
        )

        # Define paths
        sbref2T1_path = os.path.join(
            root_path,
            fmri2standard_folder,
            subject_id,
            "spm_coregister2T1_sbref",
            f"{subject_id}_ses-02_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii",
        )
        bold2T1_path = os.path.join(
            root_path,
            fmri2standard_folder,
            subject_id,
            "spm_coregister2T1_bold",
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii",
        )

        # SPM coregistration: Align BOLD to standard T1
        target_path = os.path.join(
            root_path, bids_path, subject_id, "ses-02", "anat", f"{subject_id}_ses-02_run-01_T1w.nii"
        )
        coreg_EPI2T1.inputs.target = target_path
        coreg_EPI2T1.inputs.source = sbref2T1_path
        coreg_EPI2T1.inputs.jobtype = "estimate"
        coreg_EPI2T1.inputs.apply_to_files = [bold2T1_path]

        coreg_EPI2T1.run()

        # Delete intermediate unzipped .nii files using subprocess
        subprocess.run(
            [
                "bash",
                "intermediate-files_SPM-coregister2T1_nii-format.sh",
                "-r",
                root_path,
                "-f",
                fmri2standard_folder,
                "-b",
                bids_path,
                "-s",
                subject_id,
                "-m",
                "to_nii_gz",
            ],
            check=True,
        )

        completed_subjects.append(subject_id)

    except Exception as e:
        logging.error("Error during SPM coregistration for subject %s: %s", subject_id, e)
        return completed_subjects

    return completed_subjects


def extract_wm_csf_masks(subject_id, root_path, fmri2standard_folder, recon_all_path):
    """Extract white matter (WM) and cerebrospinal fluid (CSF) masks for nuisance correction.

    This function prepares paths, creates necessary directories, and executes
    a mask extraction script. It logs any errors encountered during the process.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    root_path (str): The root path for data storage.
    fmri2standard_folder (str): Directory for fMRI to standard transformations.
    recon_all_path (str): Directory for FreeSurfer reconall data.

    Returns:
    list: A list of subjects that have completed the mask extraction.
    """
    completed_subjects = []

    print("\n\nNUISANCE CORRECTION\n\n")

    try:
        # Define paths
        # Im commenting this out because it is not used in the function - but leaving in case it is needed for something
        # sbref2T1_path = os.path.join(root_path, fmri2standard_folder, subject_id, "spm_coregister2T1_sbref",
        #     f"{subject_id}_ses-02_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz")
        bold2T1_path = os.path.join(
            root_path,
            fmri2standard_folder,
            subject_id,
            "spm_coregister2T1_bold",
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz",
        )
        output_masks = os.path.join(root_path, "nuisance_correction", subject_id, "masks_csf_wm")
        aseg_folder = os.path.join(recon_all_path, subject_id, "mri", "aseg.mgz")

        # Create necessary directories
        os.makedirs(os.path.join(root_path, "nuisance_correction", subject_id), exist_ok=True)
        os.makedirs(output_masks, exist_ok=True)

        # Execute the mask extraction script with subprocess
        try:
            subprocess.run(
                [
                    "bash",
                    "extract_wm_csf_eroded_masks.sh",
                    "-s",
                    subject_id,
                    "-a",
                    aseg_folder,
                    "-r",
                    bold2T1_path,
                    "-o",
                    output_masks,
                    "-b",
                    bold2T1_path,
                    "-e",
                    "2",
                ],
                check=True,
            )

            completed_subjects.append(subject_id)

        except subprocess.CalledProcessError as e:
            logging.error("Error during WM and CSF mask extraction for subject %s: %s", subject_id, e)

    except Exception as e:
        logging.error("Error in extracting WM and CSF masks for subject %s: %s", subject_id, e)

    return completed_subjects


def run_nuisance_regression(subject_id, root_path):
    """Run the nuisance regression workflow using pre-extracted masks.

    This function sets up a workflow for removing nuisance signals from fMRI data
    and logs any errors during the process.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    root_path (str): The root path for data storage.

    Returns:
    list: A list of subjects that have completed the nuisance regression.
    """
    completed_subjects = []

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
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf.nii.gz.par",
        )
        wf_reg.inputs.input_node.rfmri_unwarped_imgs = os.path.join(
            root_path,
            "fmri2standard",
            subject_id,
            "spm_coregister2T1_bold",
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz",
        )
        wf_reg.inputs.input_node.mask_wm = os.path.join(
            root_path, "nuisance_correction", subject_id, "masks_csf_wm", "wm_binmask.nii.gz"
        )
        wf_reg.inputs.input_node.mask_csf = os.path.join(
            root_path, "nuisance_correction", subject_id, "masks_csf_wm", "csf_binmask.nii.gz"
        )
        wf_reg.inputs.input_node.bold_img = os.path.join(
            root_path,
            "fmri2standard",
            subject_id,
            "spm_coregister2T1_bold",
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz",
        )

        # Writes the graph and runs the workflow
        wf_reg.run()

        completed_subjects.append(subject_id)

    except Exception as e:
        logging.error("Error in nuisance workflow for subject %s: %s", subject_id, e)
        return completed_subjects


def mni_normalization(subject_id, root_path, bids_path, fmri2standard_path):
    """Perform MNI normalization on T1 and fMRI images for the given subject.

    This function sets up the necessary files, decompresses them, and runs
    the SPM normalization process. Errors during this process are logged.

    Parameters:
    subject_id (str): The identifier for the subject being processed.
    root_path (str): The root path for data storage.
    bids_path (str): The path to the BIDS folder.
    fmri2standard_path (str): The path to the fMRI to standard transformations.

    Returns:
    list: A list of subjects that have completed MNI normalization.
    """
    completed_subjects = []

    print("\n\nMNI NORMALIZATION\n\n")
    try:
        # Create necessary directory for normalization
        normalization_dir = os.path.join(root_path, "normalization", subject_id)
        os.makedirs(normalization_dir, exist_ok=True)

        # Setup file paths
        T1_niigz = os.path.join(bids_path, subject_id, "ses-02", "anat", f"{subject_id}_ses-02_run-01_T1w.nii.gz")
        T1_niigzcopy = os.path.join(normalization_dir, f"{subject_id}_ses-02_run-01_T1w.nii.gz")
        T1_nii = os.path.join(normalization_dir, f"{subject_id}_ses-02_run-01_T1w.nii")

        # Copy and decompress T1 image
        shutil.copy(T1_niigz, T1_niigzcopy)
        with gzip.open(T1_niigzcopy, "rb") as f_in, open(T1_nii, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        # NOTE This is the part to improve without having to change the file formats
        # Setup paths for sbref and bold images
        bold_niigz = os.path.join(
            fmri2standard_path,
            subject_id,
            "spm_coregister2T1_bold",
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz",
        )
        bold_niigzcopy = os.path.join(
            normalization_dir, f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz"
        )
        bold_nii = os.path.join(
            normalization_dir, f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii"
        )
        sbref_niigz = os.path.join(
            fmri2standard_path,
            subject_id,
            "spm_coregister2T1_sbref",
            f"{subject_id}_ses-02_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz",
        )
        sbref_niigzcopy = os.path.join(
            normalization_dir, f"{subject_id}_ses-02_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz"
        )
        sbref_nii = os.path.join(
            normalization_dir, f"{subject_id}_ses-02_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii"
        )

        # Copy and decompress bold image
        shutil.copy(bold_niigz, bold_niigzcopy)
        with gzip.open(bold_niigzcopy, "rb") as f_in, open(bold_nii, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Copy and decompress sbref image
        shutil.copy(sbref_niigz, sbref_niigzcopy)
        with (
            gzip.open(sbref_niigzcopy, "rb") as f_in,
            open(sbref_nii, "wb") as f_out,
        ):  # Write decompressed sbref image as .nii
            shutil.copyfileobj(f_in, f_out)  # Copy decompressed sbref image

        # Perform MNI normalization
        #### NOTE that Maria has the following lines twice, so check why
        MNI = spm.preprocess.Normalize12()
        MNI.inputs.image_to_align = T1_nii
        MNI.inputs.apply_to_files = [sbref_nii, bold_nii]
        MNI.inputs.write_bounding_box = [[-90, -126, -72], [90, 90, 108]]
        MNI.run()

        completed_subjects.append(subject_id)

    except Exception as e:
        logging.error("Error during MNI Normalization for subject %s: %s", subject_id, e)
        return completed_subjects


def apply_nuisance_correction(subject_id, root_path):
    """Apply nuisance correction for a given subject.

    This function handles gzip compression, file movements, and uses FSL tools
    for regressing out nuisance signals from the data. It logs any errors during
    the process.

    Parameters:
    subject_id (str): Identifier for the subject being processed.
    root_path (str): The root path for preprocessing.

    Returns:
    list: A list of subjects that have completed nuisance correction.
    """
    completed_subjects = []

    print("\n\\APPLYING NUISANCE CORRECTION\n\n")
    try:
        # Define paths
        #### NOTE this line is not used in the script but exists in Maria's
        # nuisance_filter_bash = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/command.txt"
        new_name_nii = os.path.join(
            root_path,
            "normalization",
            subject_id,
            f"w{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii",
        )

        nuisance_output = os.path.join(
            root_path,
            "nuisance_correction",
            subject_id,
            "filter_regressors_bold",
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt.nii.gz",
        )

        native_name = os.path.join(
            root_path,
            "nuisance_correction",
            subject_id,
            "filter_regressors_bold",
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz",
        )
        #### NOTE this line is not used in the script but exists in Maria's
        # MNI_name = f"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz"
        output_directory = os.path.join(root_path, "nuisance_correction", subject_id, "filter_regressors_bold")

        # Perform gzip compression and remove the NII file
        subprocess.run(["gzip", "-f", new_name_nii], check=True)
        os.remove(new_name_nii)
        shutil.move(nuisance_output, native_name)

        mni_pre = os.path.join(
            root_path,
            "normalization",
            subject_id,
            f"w{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz",
        )

        mni_post = os.path.join(
            root_path,
            "nuisance_correction",
            subject_id,
            "filter_regressors_bold",
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz",
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
        subprocess.run(command_nuisance, check=True)

        completed_subjects.append(subject_id)

    except Exception as e:
        logging.error("Error during nuisance correction for subject %s: %s", subject_id, e)
        return completed_subjects


def fmri_quality_control(
    subject_id, root_path, fmri2standard_path, nuisance_correction_path, bids_path, recon_all_path
):
    """Perform fMRI quality control for a given subject.

    This function calculates framewise displacement and sets up the
    brain mask and DVARS workflow. Errors during the process are logged.

    Parameters:
    subject_id (str): Identifier for the subject being processed.
    root_path (str): The root path for data storage.
    fmri2standard_path (str): The path to the fMRI to standard transformations.
    nuisance_correction_path (str): The path to the nuisance correction data.
    bids_path (str): The directory for BIDS data.
    recon_all_path (str): The path to the recon_all directory.

    Returns:
    list: A list of subjects that have completed the QC process.
    """
    print("\n\nFMRI QC\n\n")
    completed_subjects = []

    try:
        # Create QC directory
        qc_dir = os.path.join(root_path, "QC", subject_id)
        os.makedirs(qc_dir, exist_ok=True)

        # Run framewise displacement
        fwd_inputs = {
            "in_file": os.path.join(
                fmri2standard_path,
                subject_id,
                "realign_fmri2SBref",
                f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf.nii.gz.par",
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
        return completed_subjects

    try:
        # Set additional paths for QA and brain mask
        input_ = os.path.join(
            fmri2standard_path, subject_id, "binarize_mask", f"{subject_id}_ses-02_run-01_T1w_brain_bin.nii.gz"
        )
        ref = os.path.join(
            nuisance_correction_path,
            subject_id,
            "filter_regressors_bold",
            f"{subject_id}_ses-02_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz",
        )
        omat = os.path.join(qc_dir, "brain_mask", "omat.mat")
        out = os.path.join(qc_dir, "brain_mask", "brain_mask_bin_BOLD_T1.nii.gz")

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
        return completed_subjects

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
        return completed_subjects

    # If all steps up to this point are successful, mark the subject as completed
    completed_subjects.append(subject_id)

    # Cleanup directories to free space
    shutil.rmtree(os.path.join(root_path, bids_path, subject_id), ignore_errors=True)
    shutil.rmtree(os.path.join(recon_all_path, subject_id), ignore_errors=True)

    return completed_subjects


def initialize_preprocessing_dirs(bids_dir, processed_directory):
    """Initialize directories and retrieve the list of subjects to process.

    Parameters:
        bids_dir (str): Directory containing the BIDS datasets.
        processed_directory (str): Directory containing processed data.

    Returns:
        set: A set containing identifiers of subjects yet to be processed.
    """
    subjects_to_process = set(os.listdir(bids_dir))
    done = set(os.listdir(processed_directory))

    subjects_to_process -= done  # Subtract processed subjects
    subjects_to_process.discard(".heudiconv")
    subjects_to_process.discard("error_heurdiconv.sh")

    return subjects_to_process


def setup_logging(step_name):
    """Configure the logging settings for a specific processing step.

    This function sets up a logging configuration that writes logs to a file
    named after the specific step being processed.

    Parameters:
    step_name (str): The name of the processing step to log.
    """
    log_file = f"{step_name}_logs.log"
    logging.basicConfig(
        level=logging.INFO,  # Can change this to only log errors
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main():
    """Main function for executing the preprocessing workflow of fMRI data.

    This function performs the following steps:
    1. Transforms fMRI data to standard space using parallel processing.
    2. Executes coregistration for valid subjects post transformation.
    3. Applies nuisance correction, including:
       a. Extraction of white matter (WM) and cerebrospinal fluid (CSF) masks.
       b. Advanced nuisance regression.
    4. Conducts MNI normalization.
    5. Removes nuisance regression artifacts.
    6. Executes quality control on fMRI data.

    The process makes use of multiprocessing to parallelize tasks where possible,
    and logs any subjects with errors at each step.

    Paths for inputs and logs are defined relative to the workflow's root directory.
    """
    # Setup logging to file and console
    logging.basicConfig(
        level=logging.ERROR, filename="logs.log", filemode="w", format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Define all paths and directories for the preprocessing workflow
    root_path = "/home/rachel/Desktop/Preprocessing"
    fmri2standard_folder = "fmri2standard"
    source_dir = os.path.join("/home/rachel/Desktop/institute/UB/Superagers/MRI/freesurfer-reconall")
    mri_path = "/home/rachel/Desktop/institute/UB/Superagers/MRI"
    bids_path = os.path.join(mri_path, "BIDS")
    recon_all_path = os.path.join(mri_path, "freesurfer-reconall")
    acparams_file = Path("/pool/guttmann/laboratori/main_preprocessingBOLD/updated_preprocessing/acparams_hcp.txt")
    fmri2standard_path = os.path.join(root_path, fmri2standard_folder)
    nuisance_correction_path = os.path.join(root_path, "nuisance_correction")
    qc_path = os.path.join(root_path, "QC")
    bids_dir = "/home/rachel/Desktop/institute/UB/Superagers/MRI/BIDS"
    processed_directory = "/home/rachel/Desktop/institute/UB/Superagers/MRI/processed_data/fMRI-preprocessed_tp2"

    # Define the SPM coregistration object
    coreg_EPI2T1 = spm.Coregister()

    # Run `initialize_preprocessing_dirs` to retrieve the list of subjects to process
    subjects_to_process = initialize_preprocessing_dirs(bids_dir, processed_directory)

    #### Run the preprocessing workflow ####

    # Step 1.
    # Setup logging for fMRI to standard transformation
    setup_logging("transform_fmri_to_standard")

    # Define the partial function with fixed arguments
    transform_partial_fmri_to_standard = partial(
        transform_fmri_to_standard,
        root_path=root_path,
        bids_path=bids_path,
        recon_all_path=recon_all_path,
        acparams_file=acparams_file,
    )

    # Set up a multiprocessing pool to parallelize fMRI standard space transformation (`transform_fmri_to_standard`)
    with Pool(6) as pool:
        results = pool.map(transform_partial_fmri_to_standard, subjects_to_process)  # Store results in a variable

    # Identify subjects ready for coregistration to run (`execute_coregistration`), excluding those with errors from `transform_fmri_to_standard`
    coregistration_list = [subject for sublist in results for subject in sublist]

    # Step 2.
    # Setup logging for coregistration
    setup_logging("execute_coregistration")

    # Perform coregistration on the filtered list of subjects
    for subject in coregistration_list:
        results = execute_coregistration(subject, root_path, fmri2standard_folder, bids_path, coreg_EPI2T1)

    # Identify subjects needing nuisance correction to run (`extract_wm_csf_masks`), excluding those with errors from `execute_coregistration`
    extract_wm_csf_masks_list = [subject for sublist in results for subject in sublist]

    # Step 3a.
    # Setup logging for WM and CSF mask extraction
    setup_logging("extract_wm_csf_masks")

    # Apply nuisance correction initial step using multiprocessing
    transform_partial_extract_wm_csf_masks = partial(
        results=extract_wm_csf_masks,
        root_path=root_path,
        fmri2standard_folder=fmri2standard_folder,
        recon_all_path=recon_all_path,
    )

    pool.map(transform_partial_extract_wm_csf_masks, extract_wm_csf_masks_list)

    # Identify subjects needing nuisance correction to run (`run_nuisance_regression`), excluding those with errors from `extract_wm_csf_masks`
    nuisance_regression_list = [subject for sublist in results for subject in sublist]

    # Step 3b.
    # Setup logging for nuisance regression
    setup_logging("run_nuisance_regression")

    # Execute advanced nuisance regression
    for subject in nuisance_regression_list:
        results = run_nuisance_regression(subject, root_path)

    # Identify subjects needing MNI normalization to run (`mni_normalization`), excluding those with errors from `run_nuisance_regression`
    mni_normalization_list = [subject for sublist in results for subject in sublist]

    # Step 4.
    # Setup logging for MNI normalization
    setup_logging("mni_normalization")

    # Perform MNI normalization using multiprocessing
    for subject in mni_normalization_list:
        results = mni_normalization(subject, root_path, bids_path, fmri2standard_path)

    # Identify subjects needing nuisance regression removal to run (`apply_nuisance_correction`), excluding those with errors from `mni_normalization`
    regression_list = [subject for sublist in results for subject in sublist]

    # Step 5.
    # Setup logging for nuisance correction application
    setup_logging("apply_nuisance_correction")

    ###### NOTE because I divided step 5 into two parts, ask Maria if both of these parts require parallelization. Remove if not needed
    # Apply nuisance correction using multiprocessing
    transform_partial_apply_nuisance_correction = partial(results=apply_nuisance_correction, root_path=root_path)

    with Pool(8) as pool:
        pool.map(transform_partial_apply_nuisance_correction, regression_list)

    # Identify subjects needing fMRI quality control to run (`fmri_quality_control`), excluding those with errors from `apply_nuisance_correction`
    qc_list = [subject for sublist in results for subject in sublist]

    # Step 6.
    # Setup logging for fMRI quality control
    setup_logging("fmri_quality_control")

    # Perform fMRI quality control using multiprocessing
    transform_partial_fmri_quality_control = partial(
        fmri_quality_control,
        root_path=root_path,
        fmri2standard_path=fmri2standard_path,
        nuisance_correction_path=nuisance_correction_path,
        bids_path=bids_path,
        recon_all_path=recon_all_path,
    )

    with Pool(8) as pool:
        pool.map(transform_partial_fmri_quality_control, qc_list)


if __name__ == "__main__":
    main()
