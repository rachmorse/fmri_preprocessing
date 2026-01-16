#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""This script defines the workflow for nuisance regression in fMRI data.
It includes steps for motion correction, cosine filtering, and CompCor (WM/CSF) regression.
"""


def motion_regressors(realign_movpar_txt, output_dir, order=0, derivatives=1):
    """Compute motion regressors up to a given order and derivative.

    Calculates motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic).

    Args:
        realign_movpar_txt (str): File path to the realignment motion parameters.
        output_dir (str): Directory path where the output will be saved.
        order (int): Order of the polynomial expansion. Defaults to 0.
        derivatives (int): Number of derivatives to compute. Defaults to 1.

    Returns:
        str: The file path of the generated motion regressors text file.
    """
    import os

    import numpy as np

    params = np.genfromtxt(realign_movpar_txt)
    out_params = params

    for d in range(1, derivatives + 1):
        # Create a lagged version of the parameters to compute differences
        # Duplicates first row and put it on the top of the matrix
        cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0), params))

        # Calculate the difference between the current row and the previous ones (derivatives)
        # We take the first 6 columns (original motion parameters) and center them
        out_params = out_params[:, 0:6] - out_params[0, 0:6] * np.ones([out_params.shape[0], 1])
        out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))

    # Note: Higher order expansions (e.g., quadratic) are currently not used/implemented
    # but the structure allows for it if 'order' > 0 loop is uncommented/implemented.

    filename = os.path.join(output_dir, "motion_regressor.txt")
    np.savetxt(filename, out_params, fmt="%.10f")
    return filename


def cosine_filter_txt(timepoints, timestep, output_dir, period_cut=128):
    """Create the discrete cosine transform (DCT) basis functions for high-pass filtering.

    Args:
        timepoints (int): Number of time volumes (length of the acquisition).
        timestep (float): The fMRI TR (Repetition Time) in seconds.
        output_dir (str): Directory path where the output will be saved.
        period_cut (float): Minimum period for the cosine basis functions in seconds. Defaults to 128.

    Returns:
        str: The file path of the generated cosine filter text file.
    """
    import os

    import nipype.algorithms.confounds as cf
    import numpy as np

    frametimes = timestep * np.arange(timepoints)
    X = cf._full_rank(cf._cosine_drift(period_cut, frametimes))[0]
    non_constant_regressors = X[:, :-1] if X.shape[1] > 1 else np.array([])
    filename = os.path.join(output_dir, "cosine_filter.txt")
    np.savetxt(filename, non_constant_regressors, fmt="%.10f")
    return filename


def get_nuisance_regressors_wf(outdir, timepoints, subject_id, global_signal=False, order=0, derivatives=1, comp=3):
    """Create a Nipype workflow for nuisance correction.

    The workflow includes:
    1. Intercept
    2. Drift (cosine transform)
    3. Motion Correction parameters
    4. White Matter & CSF Nuisance Regressors (aCompCor)

    Args:
        outdir (str): Output directory for the workflow.
        timepoints (int): Number of time volumes.
        subject_id (str): Subject identifier.
        global_signal (bool): Whether to include global signal regression. Defaults to False.
        order (int): Order of motion parameter expansion. Defaults to 0.
        derivatives (int): Number of motion parameter derivatives. Defaults to 1.
        comp (int): Number of CompCor components to extract. Defaults to 3.

    Returns:
        nipype.pipeline.engine.Workflow: The configured nuisance regression workflow.
    """
    import os

    from nipype import Node, Workflow
    from nipype.algorithms import confounds
    from nipype.interfaces import fsl, utility

    gb = "_GB" if global_signal else ""

    wf_reg = Workflow(name=subject_id + gb, base_dir=outdir)

    print("Setting INPUT node...")
    node_input = Node(
        utility.IdentityInterface(
            fields=[
                "realign_movpar_txt",
                "rfmri_unwarped_imgs",
                "mask_wm",
                "mask_csf",
                "global_mask_img",
                "bold_img",
            ]
        ),
        name="input_node",
    )

    # Node to merge WM and CSF masks for CompCor
    node_merge_wm_csf = Node(utility.base.Merge(2), name="Merge_wm_csf")

    # Node for Anatomical CompCor (aCompCor)
    # Extracts principal components from WM and CSF masks to use as nuisance regressors
    node_ACompCor = Node(
        confounds.ACompCor(
            num_components=comp,
            pre_filter=False,
            repetition_time=0.8,
            merge_method="none",
        ),
        name="AcompCor_mask",
    )

    # Node for creating Cosine Filter (High-pass filter) regressors
    node_cosine_filter_reg = Node(
        utility.Function(
            input_names=["timepoints", "timestep", "period_cut", "output_dir"],
            output_names=["cosine_filter_txt"],
            function=cosine_filter_txt,
        ),
        name="cosine_filter",
    )
    node_cosine_filter_reg.inputs.output_dir = os.path.join(
        os.path.join(os.path.join(wf_reg.base_dir, wf_reg.name)), node_cosine_filter_reg.name
    )
    node_cosine_filter_reg.inputs.timepoints = timepoints
    node_cosine_filter_reg.inputs.timestep = 0.8

    # Node for calculating Motion Regressors (and derivatives)
    motion_regressors_interface = utility.Function(
        input_names=["realign_movpar_txt", "output_dir", "order", "derivatives"],
        output_names=["motion_reg_txt"],
        function=motion_regressors,
    )
    node_motion_regressors = Node(motion_regressors_interface, name="motion_regressors_txt")
    node_motion_regressors.inputs.output_dir = os.path.join(
        os.path.join(os.path.join(wf_reg.base_dir, wf_reg.name)), node_motion_regressors.name
    )

    # Node to merge all nuisance text files into one list
    node_merge_txts = Node(utility.base.Merge(4), name="Merge_txt_inputs")

    # Node to combine all regressors into a single matrix file
    node_merge_regressors = Node(
        utility.Function(
            input_names=["nuisance_txts", "output_dir"],
            output_names=["nuisance_txt"],
            function=merge_nuisance_regressors,
        ),
        name="merge_nuisance_txt",
    )
    node_merge_regressors.inputs.output_dir = os.path.join(
        os.path.join(wf_reg.base_dir, wf_reg.name), node_merge_regressors.name
    )

    # Node to apply the regression to the BOLD data
    node_filter_regressor = Node(
        fsl.FilterRegressor(
            filter_all=True,
        ),
        name="filter_regressors_bold",
    )

    node_output = Node(utility.IdentityInterface(fields=["nuisance_txt", "bold_nuisance_filtered"]), name="output_node")

    # Connect the nodes in the workflow
    wf_reg.connect([
        (node_input, node_merge_wm_csf, [("mask_wm", "in1"), ("mask_csf", "in2")]),
        (node_input, node_ACompCor, [("rfmri_unwarped_imgs", "realigned_file")]),
        (node_merge_wm_csf, node_ACompCor, [("out", "mask_files")]),
        (node_input, node_motion_regressors, [("realign_movpar_txt", "realign_movpar_txt")]),
        (node_motion_regressors, node_merge_txts, [("motion_reg_txt", "in1")]),
        (node_ACompCor, node_merge_txts, [("components_file", "in2")]),
        (node_cosine_filter_reg, node_merge_txts, [("cosine_filter_txt", "in3")]),
        (node_merge_txts, node_merge_regressors, [("out", "nuisance_txts")]),
    ])

    wf_reg.connect([
        (node_merge_regressors, node_filter_regressor, [("nuisance_txt", "design_file")]),
        (node_input, node_filter_regressor, [("bold_img", "in_file")]),
        (node_filter_regressor, node_output, [("out_file", "bold_nuisance_filtered")]),
        (node_merge_regressors, node_output, [("nuisance_txt", "nuisance_txt")]),
    ])

    return wf_reg


def merge_nuisance_regressors(nuisance_txts, output_dir, standardize=True):
    """Merge multiple nuisance regressor files into a single matrix.

    This function combines motion parameters, CompCor components, and cosine drift terms.
    It also standardizes the regressors (z-score) to avoid scaling issues.
    
    References:
        https://www.ncbi.nlm.nih.gov/pubmed/30666750
        (Combining all steps into a single linear filter)

    Args:
        nuisance_txts (list): List of file paths to the nuisance text files.
        output_dir (str): Directory path where the output will be saved.
        standardize (bool): Whether to standardize (z-score) the regressors. Defaults to True.

    Returns:
        str: The file path of the combined nuisance regressors text file.
    """
    import os

    import numpy as np

    txts_values = []
    for txt in nuisance_txts:
        txt_values = np.genfromtxt(txt)

        # Handle 1D arrays (single regressor)
        if txt_values.ndim == 1:
            txt_values = np.matrix(txt_values)
            txt_values = txt_values[~np.isnan(txt_values)].transpose()
            if standardize:
                print("hy")
                txt_values = (txt_values - txt_values.mean(axis=0)) / np.std(txt_values, axis=0)
            txts_values.append(txt_values)
        # Handle 2D arrays (multiple regressors)
        else:
            txt_values = txt_values[~np.isnan(txt_values).any(axis=1)]
            txt_values = np.matrix(txt_values)
            if standardize:
                print("h")
                txt_values = (txt_values - txt_values.mean(axis=0)) / np.std(txt_values, axis=0)
            txts_values.append(txt_values)

        out_params = np.hstack((txts_values))

    # Add a column of ones (intercept)
    out_params = np.hstack((np.ones((len(out_params), 1)), out_params))

    filename = os.path.join(output_dir, "all_nuisances.txt")
    print(filename)
    np.savetxt(filename, out_params, fmt="%.10f")
    return filename

