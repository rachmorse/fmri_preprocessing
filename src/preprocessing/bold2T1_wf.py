#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_fmri2standard_wf(tvols, subject_id, ACQ_PARAMS):
    """Estimates transformation from Gradient Field Distortion-warped BOLD to T1.

    In general:
        BOLD is field-inhomogeneity corrected and coregistered into standard space (T1).

    The workflow performs the following steps:
    1.  Coregister SBref to SEgfmAP (fsl.FLIRT).
    2.  Realign BOLD to corrected SBref (fsl.MCFLIRT).
    3.  Estimate field inhomogeneity correction of SBref from SEfm_AP and SEfm_PA (fsl.TOPUP).
    4.  Apply field inhomogeneity correction to SBref (fsl.ApplyTOPUP).
    5.  Apply field inhomogeneity correction to BOLD (fsl.ApplyTOPUP).
    6.  Transform FreeSurfer brain mask (brain.mgz) to T1 space (freesurfer.ApplyVolTransform),
        binarize it (fsl.UnaryMaths), and extract the mask from T1 (fsl.BinaryMaths).
    7.  Coregister BOLD (field-inhomogeneity corrected) to Standard T1 (fsl.Epi2Reg).

    Args:
        tvols (list): List containing [t_initial, t_final] volumes included in the preprocess.
        subject_id (str): Subject identifier.
        ACQ_PARAMS (str): Path to txt file containing MRI acquisition parameters; needs to be specified for topup correction.

    Returns:
        nipype.pipeline.engine.Workflow: The configured workflow for BOLD to T1 transformation.
    """
    from nipype import Node, Workflow, interfaces
    from nipype.interfaces import freesurfer, fsl, utility

    print("Defining workflow...")
    wf = Workflow(name=subject_id, base_dir="")

    # Setting INPUT node...
    print("Defining input node...")
    node_input = Node(
        utility.IdentityInterface(
            fields=[
                "func_sbref_img",
                "func_segfm_ap_img",
                "func_segfm_pa_img",
                "func_bold_ap_img",
                "T1_img",
                "T1_brain_freesurfer_mask",
            ]
        ),
        name="input_node",
    )

    print(
        "Averages the three repeated Spin-Echo images with same Phase Encoding (AP or PA) for Susceptibility Correction (unwarping)..."
    )
    node_average_SEgfm = Node(fsl.maths.MeanImage(), name="Mean_SEgfm_AP")

    print("Coregister SB-ref to average SEgfm-AP")
    node_coregister_SBref2SEgfm = Node(
        fsl.FLIRT(
            dof=6  # translation and rotation only
        ),
        name="Corregister_SBref2SEgfm",
    )

    print("Eliminates first volumes.")
    node_eliminate_first_scans = Node(
        fsl.ExtractROI(
            t_min=tvols[0],  # first included volume
            t_size=tvols[1] - tvols[0],  # number of volumes from the first to the last one
        ),
        name="eliminate_first_scans",
    )

    print("Realigns fMRI BOLD volumes to SBref in SEgfm-AP space")
    node_realign_bold = Node(
        fsl.MCFLIRT(
            save_plots=True,  # save transformation matrices
        ),
        name="realign_fmri2SBref",
    )

    print("Concatenates AP and PA SEgfm volumes...")
    # Expected order: AP AP AP PA PA PA
    node_merge_ap_pa_inputs = Node(
        utility.base.Merge(
            2  # number of inputs; it concatenates lists
        ),
        name="Merge_ap_pa_inputs",
    )
    node_merge_SEgfm = Node(
        fsl.Merge(
            dimension="t"
        ),
        name="Merge_SEgfm_AP_PA",
    )

    print("Estimates TopUp inhomogeneity correction from SEfm_AP and SEfm_PA...")
    node_topup_SEgfm = Node(
        fsl.TOPUP(
            encoding_file=ACQ_PARAMS,
        ),
        name="Topup_SEgfm_estimation",
    )

    print("Applies warp from TOPUP to correct SBref...")
    node_apply_topup_to_SBref = Node(
        fsl.ApplyTOPUP(
            encoding_file=ACQ_PARAMS,
            method="jac",  # jacobian modulation
            interp="spline",  # interpolation method
        ),
        name="apply_topup_to_SBref",
    )

    print("Applies warp from TOPUP to correct realigned BOLD...")
    node_apply_topup = Node(
        fsl.ApplyTOPUP(
            encoding_file=ACQ_PARAMS,
            method="jac",  # jacobian modulation
            interp="spline",  # interpolation method
        ),
        name="apply_topup",
    )

    # BRAIN MASK
    # Registration to T1. Epireg without fieldmaps combined
    # (see https://www.fmrib.ox.ac.uk/primers/intro_primer/ExBox20/IntroBox20.html)

    # BRAIN MASK OPTION 1: BET
    print("Eliminates scalp from brain using T1 high res image")
    node_mask_T1 = Node(
        fsl.BET(
            frac=0.7,  # fractional intensity threshold
            mask=True,
        ),
        name="mask_T1",
    )

    # BRAIN MASK OPTION 1: FREESURFER
    print("Transform brain mask T1 from freesurfer space to T1 space")
    node_vol2vol_brain = Node(
        freesurfer.ApplyVolTransform(
            reg_header=True, 
            transformed_file="brainmask_warped.nii.gz",
        ),
        name="vol2vol",
    )

    print("Transform brain mask T1 to binary mask")
    node_bin_mask_brain = Node(
        fsl.UnaryMaths(  # fslmaths T1_brain -bin T1_binarized mask
            operation="bin", 
        ),
        name="binarize_mask",
    )

    node_extract_mask = Node(
        fsl.BinaryMaths(  # fslmaths T1 -mul T1_binarized_mask T1_extracted_mask
            operation="mul"  
        ),
        name="extract_mask",
    )

    print("...")
    print("Setting OUTPUT node...")
    
    node_output = Node(
        interfaces.utility.IdentityInterface(
            fields=[
                "SBref2SEgfm_mat",
                "realign_movpar_txt",
                "realign_fmri_img",
                "topup_movpar_txt",
                "topup_field_coef_img",
                "epi2str_mat",
                "epi2str_img",
                "rfmri_unwarped_imgs",
                "sb_ref_unwarped_img",
            ]
        ),
        name="output_node",
    )

    print("All nodes created; Starts creating connections")

    inputs = []
    connections = []
    ...
    node_connections = [*inputs, *connections, ...]

    # Connects nodes
    wf.connect([
        # Inputs
        (node_input, node_average_SEgfm, [("func_segfm_ap_img", "in_file")]),
        (node_input, node_coregister_SBref2SEgfm, [("func_sbref_img", "in_file")]),
        (node_input, node_eliminate_first_scans, [("func_bold_ap_img", "in_file")]),
        (node_input, node_merge_ap_pa_inputs, [("func_segfm_ap_img", "in1"), ("func_segfm_pa_img", "in2")]),
        (node_merge_ap_pa_inputs, node_merge_SEgfm, [("out", "in_files")]),
        (node_input, node_mask_T1, [("T1_img", "in_file")]),
        (node_input, node_vol2vol_brain, [("T1_brain_freesurfer_mask", "source_file")]),
        (node_input, node_vol2vol_brain, [("T1_img", "target_file")]),
        (node_input, node_extract_mask, [("T1_img", "in_file")]),
        # Connections
        (node_mask_T1, node_bin_mask_brain, [("out_file", "in_file")]),
        (node_eliminate_first_scans, node_realign_bold, [("roi_file", "in_file")]),
        (node_average_SEgfm, node_coregister_SBref2SEgfm, [("out_file", "reference")]),
        (node_coregister_SBref2SEgfm, node_realign_bold, [("out_file", "ref_file")]),
        # T1 brain mask transformations (change space / vol2vol, binarize and extract)
        (node_bin_mask_brain, node_extract_mask, [("out_file", "operand_file")]),
        (node_merge_SEgfm, node_topup_SEgfm, [("merged_file", "in_file")]),
        (node_realign_bold, node_apply_topup, [("out_file", "in_files")]),
        (
            node_topup_SEgfm,
            node_apply_topup,
            [("out_fieldcoef", "in_topup_fieldcoef"), ("out_movpar", "in_topup_movpar")],
        ),
        (
            node_topup_SEgfm,
            node_apply_topup_to_SBref,
            [("out_fieldcoef", "in_topup_fieldcoef"), ("out_movpar", "in_topup_movpar")],
        ),
        (node_coregister_SBref2SEgfm, node_apply_topup_to_SBref, [("out_file", "in_files")]),
        # Yield relevant data to output node
        (node_coregister_SBref2SEgfm, node_output, [("out_matrix_file", "SBref2SEgfm_mat")]),
        (node_realign_bold, node_output, [("par_file", "realign_movpar_txt"), ("out_file", "realign_fmri_img")]),
        (
            node_topup_SEgfm,
            node_output,
            [("out_fieldcoef", "topup_field_coef_img"), ("out_corrected", "sb_ref_unwarped_img")],
        ),
        (node_apply_topup, node_output, [("out_corrected", "rfmri_unwarped_imgs")]),
    ])
    print("All connections created")
    return wf
