#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Tue Feb  4 12:27:39 2020
@author: Dídac Macià
"""


def get_fmri2standard_wf(tvols, subject_id, ACQ_PARAMS="/home/didac/LabScripts/fMRI_preprocess/acparams_hcp.txt"):
    """Estimates transformation from Gradiend Field Distortion-warped BOLD to T1

    In general:
        BOLD is field-inhomogeneity corrected and corregistered into standard space (T1).

    To do so, the following steps are carried out:
        1)  Corregister SBref to SEgfmAP (fsl.FLIRT)
        2)  Realign BOLD to corrected SBref (fsl.MCFLIRT)
        3)  Field inhomogeneity correction estimation of SBref from SEfm_AP and SEfm_PA (fsl.TOPUP)
        4)  Apply field inhomogeneity correction to SBref (fsl.ApplyTOPUP)
        5)  Apply field inhomogeneity correction to BOLD (fsl.ApplyTOPUP)
        6)  Transform free-surfer brain mask (brain.mgz) to T1 space (freesurfer.ApplyVolTransform ;mri_vol2vol),
            then binarized (fsl.UnaryMaths) and the mask is extracted from T1 (fsl.BinaryMaths)
        7)  Corregister BOLD (field-inhomogeneity corrected) to Standard T1 (fsl.Epi2Reg)

    Parameters
    ----------
    tvols: [t_initial, t_final] volumes included in the preprocess
    ACQ_params: Path to txt file containing MRI acquisition parameters; needs to be specified for topup correction


    Returns:
    -------
    Workflow with the transformation

    """
    from nipype import Node, Workflow, interfaces
    from nipype.interfaces import freesurfer, fsl, utility

    print("defining workflow...")
    wf = Workflow(name=subject_id, base_dir="")

    # Setting INPUT node...
    print("defines input node...")
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

    print("Corregister SB-ref to average SEgfm-AP")
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
    # AP AP AP PA PA PA
    node_merge_ap_pa_inputs = Node(
        utility.base.Merge(
            2  # number of inputs; it concatenates lists
        ),
        name="Merge_ap_pa_inputs",
    )
    node_merge_SEgfm = Node(
        fsl.Merge(
            dimension="t"  # ¿?
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

    # Registration to T1. Epireg without fieldmaps combined (see https://www.fmrib.ox.ac.uk/primers/intro_primer/ExBox20/IntroBox20.html)

    # BRAIN MASK OPTION 1: BET
    print("Eliminates scalp from brain using T1 high res image")
    node_mask_T1 = Node(
        fsl.BET(
            frac=0.7,  # umbral
            mask=True,
        ),
        name="mask_T1",
    )

    # BRAIN MASK OPTION 1: FREESURFER
    print("Transform brain mask T1 from freesurfer space to T1 space")
    node_vol2vol_brain = Node(
        freesurfer.ApplyVolTransform(
            reg_header=True,  # (--regheader)
            transformed_file="brainmask_warped.nii.gz",
            # source_file --mov (INPUT; freesurfer brain.mgz)
            # transformed_file --o (OUTPUT; ...brain.nii.gz)
            # target_file --targ (REFERENCE; ...T1w.nii.gz)
        ),
        name="vol2vol",
    )

    print("Transform brain mask T1 to binary mask")
    node_bin_mask_brain = Node(
        fsl.UnaryMaths(  # fslmaths T1_brain -bin T1_binarized mask
            operation="bin",  # (-bin)
            # in_file (T1_brain)
            # out_file (T1_binarized_mask)
        ),
        name="binarize_mask",
    )
    # TODO: why are there commented out bits? clean it
    #
    #    print('Extract brain mask from T1 using binary mask'); # necesario? creo que no
    node_extract_mask = Node(
        fsl.BinaryMaths(  # fslmaths T1 -mul T1_binarized_mask T1_extracted_mask
            operation="mul"  # (-mul)
            # in_file (T1)
            # out_file (T1_extracted_mask)
            # operand_file (T1_binarized_mask)
        ),
        # TODO: why are there semicolons at the end of the lines?
        name="extract_mask",
    )

    print("...")

    # node_fmriMask.overwrite=True
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

    # TODO: the length of this makes it really hard to read and understand. If there is a way to split this stuff up,
    # it would be a lot easier to understand.

    inputs = []

    connections = []

    ...

    node_connections = [*inputs, *connections, ...]

    # Connects nodes
    wf.connect([
        # inputs
        (node_input, node_average_SEgfm, [("func_segfm_ap_img", "in_file")]),
        (node_input, node_coregister_SBref2SEgfm, [("func_sbref_img", "in_file")]),
        (node_input, node_eliminate_first_scans, [("func_bold_ap_img", "in_file")]),
        (node_input, node_merge_ap_pa_inputs, [("func_segfm_ap_img", "in1"), ("func_segfm_pa_img", "in2")]),
        (node_merge_ap_pa_inputs, node_merge_SEgfm, [("out", "in_files")]),
        (node_input, node_mask_T1, [("T1_img", "in_file")]),
        (node_input, node_vol2vol_brain, [("T1_brain_freesurfer_mask", "source_file")]),
        (node_input, node_vol2vol_brain, [("T1_img", "target_file")]),
        (node_input, node_extract_mask, [("T1_img", "in_file")]),
        # connections
        (node_mask_T1, node_bin_mask_brain, [("out_file", "in_file")]),
        (node_eliminate_first_scans, node_realign_bold, [("roi_file", "in_file")]),
        (node_average_SEgfm, node_coregister_SBref2SEgfm, [("out_file", "reference")]),
        (node_coregister_SBref2SEgfm, node_realign_bold, [("out_file", "ref_file")]),
        # T1 brain mask transformations (change space / vol2vol, binarize and extract)
        #                (node_vol2vol_brain, node_bin_mask_brain, [("transformed_file","in_file")]),
        (node_bin_mask_brain, node_extract_mask, [("out_file", "operand_file")]),
        # (node_realign_bold, node_tsnr, [("out_file", "in_file")]),
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
        # yeld relevant data to output node
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


def intermediate_files_coregSPM():
    pass
