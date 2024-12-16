#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#creates a nuisance matrix and regresses the fMRI voxels on it and keeps the 
#residuals

import matplotlib.pyplot as plt

def motion_regressors(realign_movpar_txt, output_dir, order=0, derivatives=1):
    """Compute motion regressors upto given order and derivative
    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    Parameters
    ----------
    realign_movpar_txt: [string] file path where realignment's motion parameters are
    output_dir: [string] directory path where the output is gonna be saved
    order:                
    derivatives:
        
    Returns
    -------
    filename:
    """
    import os
    import numpy as np
    params = np.genfromtxt(realign_movpar_txt)
    out_params = params
    # TODO: why does this need to be done repeatedly? Given that you only reference the outparams once outside of the
    # loop?
    for d in range(1, derivatives + 1):

        # TODO: this would repeat d times - what does this actually do? Why? Explain this better in code
        # Duplicates first row and put it on the top of the matrix
        cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                             params))

        # TODO: This is all really unclean, and hard to follow. Why do you need the first 6 rows only?
        # Computes the difference of every row and its posterior: row(n) - row(n-1)
        # and appends it as 6 new columns on the right
        out_params = out_params[:,0:6]-out_params[0,0:6]*np.ones([out_params.shape[0],1])
        out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))

    out_params2 = out_params
    #We will never use second order 
    #for i in range(1, order + 1):
        #out_params2 = np.hstack((out_params2, np.power(out_params, i)))
    filename = os.path.join(output_dir, "motion_regressor.txt")
    np.savetxt(filename, out_params2, fmt=b"%.10f")
    return filename

def cosine_filter_txt (timepoints, timestep, output_dir,period_cut=128):
    """creates the discrete cosine transform (DCT) basis functions
    Parameters
    ----------
    period_cut:  minimum period for the cosine basis functions
    timestep : must be equal to the fmri TR
    timepoints: number of time volumes (length of the acquisition)
    
    Returns
    -------
    filename:
    """
    import nipype.algorithms.confounds as cf
    import numpy as np
    import os
    frametimes = timestep * np.arange(timepoints)
    X = cf._full_rank(cf._cosine_drift(period_cut, frametimes))[0]
    non_constant_regressors = X[:, :-1] if X.shape[1] > 1 else np.array([])
    filename = os.path.join(output_dir, "cosine_filter.txt")
    np.savetxt(filename,non_constant_regressors, fmt=b"%.10f")
    return filename


def get_nuisance_regressors_wf(outdir, timepoints, subject_id, global_signal=False, order=0, derivatives=1, comp=3):
    """Creates nipype workflow for nuisance correction composed by:
        Intercept + Drift (cosine transform) + Motion Correction + WhiteMatter&CSF Nuisance Regressors (CompCor)
    Parameters
    ----------
    outdir:
    subject_id:
    global_signal:
    timepoints:
    order:
    derivatives:
    comp:
        
    Returns
    -------
    wf_reg:
    """
    from nipype.algorithms import confounds
    from nipype import Workflow, Node
    from nipype.interfaces import utility, fsl
    import os
    
    if global_signal: 
        gb='_GB'
    else:
        gb=''
        
    wf_reg=Workflow(name=subject_id+gb,base_dir=outdir);
    
    print ("Setting INPUT node...");
    node_input = Node(utility.IdentityInterface(fields=[
           "realign_movpar_txt",        
           'rfmri_unwarped_imgs',
           'mask_wm',
           'mask_csf',
           'global_mask_img',
           'bold_img'
           ]),
            name='input_node'
    ) 
    
    #Merging wm and csf masks
    node_merge_wm_csf = Node(utility.base.Merge(2),name='Merge_wm_csf')
    
    #AcompCor
    node_ACompCor=Node(confounds.ACompCor( 
            num_components=3,
            #save_pre_filter='high_pass_filter.txt',       
            pre_filter=False,
           # high_pass_cutoff=128,
            repetition_time=0.8,
            merge_method='none',
            #use_regress_poly=False,
            #realigned_file= fMRI_BOLD_unwarped,
           # mask_files='/institut/processed_data/BBHI_func/output2/sub-41064/GetMasksInT1Space/binarize_mask/MNI152_WM_09_warp_thresh.nii.gz',
             ),
    name="AcompCor_mask")
    #node_ACompCor.inputs.save_pre_filter=os.path.join(os.path.join(os.path.join(wf_reg.base_dir,wf_reg.name),node_ACompCor.name), 'high_pass_filter.txt')  

    #cosine_filter    
    node_cosine_filter_reg=Node(utility.Function(input_names=["timepoints", "timestep","period_cut","output_dir"],
                             output_names=["cosine_filter_txt"],
                             function=cosine_filter_txt), 
                                name="cosine_filter")    
    node_cosine_filter_reg.inputs.output_dir=os.path.join(os.path.join(os.path.join(wf_reg.base_dir,wf_reg.name)),node_cosine_filter_reg.name) 
    node_cosine_filter_reg.inputs.timepoints=timepoints
    node_cosine_filter_reg.inputs.timestep=0.8
    #node_cosine_filter_reg.overwrite=True
    
    #global_signal    
#    if global_signal :
#        node_global_signal=Node(utility.Function(input_names=["timeseries_file", "label_file", "filename"],
#                                 output_names=["global_signal_txt"],
#                                 function=extract_subrois), 
#                                    name="global_signal")    
#        node_global_signal.inputs.filename=os.path.join(os.path.join(os.path.join(os.path.join(wf_reg.base_dir,wf_reg.name)),node_global_signal.name),'global_signal.txt') 
#        #node_global_signal.overwrite=True

    #motion regressors
    motion_regressors_interface = utility.Function(input_names=["realign_movpar_txt", "output_dir","order","derivatives"],
                             output_names=["motion_reg_txt"],
                             function=motion_regressors)
    node_motion_regressors=Node(motion_regressors_interface, name="motion_regressors_txt")    
    node_motion_regressors.inputs.output_dir=os.path.join(os.path.join(os.path.join(wf_reg.base_dir,wf_reg.name)),node_motion_regressors.name) 
    #node_motion_regressors.overwrite=True
    
    
    #merges all regressors     
    node_merge_txts = Node(utility.base.Merge(4),name='Merge_txt_inputs')    
    
    node_merge_regressors = Node(utility.Function(input_names=["nuisance_txts", "output_dir"],
                             output_names=["nuisance_txt"],
                             function=merge_nuisance_regressors),
    name="merge_nuisance_txt")
    node_merge_regressors.inputs.output_dir=os.path.join(os.path.join(wf_reg.base_dir,wf_reg.name),node_merge_regressors.name) 
    
    node_filter_regressor = Node(fsl.FilterRegressor(
            #design_file (-d) nuissance_txt
            filter_all=True,
            #in_file (-i) bold after SPM coregistration to T1
            #out_file
            ),
    name="filter_regressors_bold")
    
    
    node_output = Node(utility.IdentityInterface(fields=[
        'nuisance_txt', 
        'bold_nuisance_filtered'
    ]),
    name='output_node') 
    
    wf_reg.connect([ (node_input, node_merge_wm_csf, [('mask_wm','in1'),
                                                      ('mask_csf', 'in2')]),
                     (node_input, node_ACompCor,[('rfmri_unwarped_imgs', 'realigned_file')]),
                     (node_merge_wm_csf, node_ACompCor, [('out', 'mask_files')]),
                     (node_input, node_motion_regressors,[('realign_movpar_txt', 'realign_movpar_txt')]),                     
                     
                     (node_motion_regressors,node_merge_txts, [('motion_reg_txt', 'in1')]),
                     (node_ACompCor,node_merge_txts, [('components_file', 'in2')]),
                     (node_cosine_filter_reg,node_merge_txts, [('cosine_filter_txt', 'in3')]),
                     (node_merge_txts, node_merge_regressors, [('out', 'nuisance_txts')]),
                     ])   
#    if global_signal:       
#         wf_reg.connect([
#                         (node_input, node_global_signal,[('rfmri_unwarped_imgs', 'timeseries_file'),
#                                                     ('global_mask_img', 'label_file')]),    
#                        (node_global_signal, node_merge_txts, [('global_signal_txt', 'in4')])                
#                         ])
    
    wf_reg.connect([    (node_merge_regressors, node_filter_regressor, [('nuisance_txt','design_file')]),
                        (node_input, node_filter_regressor, [('bold_img','in_file')]),
                        (node_filter_regressor, node_output, [('out_file','bold_nuisance_filtered')]),
                        (node_merge_regressors, node_output,[('nuisance_txt', 'nuisance_txt')])                
                         ])
    return wf_reg

#def extract_subrois(timeseries_file, label_file, filename,indices=[1]):
#    """Extract voxel time courses for each subcortical roi index
#    Parameters
#    ----------
#    timeseries_file: a 4D Nifti file
#    label_file: a 3D file containing rois in the same space/size of the 4D file
#    indices: a list of indices for ROIs to extract.
#
#    Returns
#    -------
#    out_file: a text file containing time courses for each voxel of each roi
#        The first four columns are: freesurfer index, i, j, k positions in the
#        label file
#    """
#    import nibabel as nb
#    from nipype.utils import NUMPY_MMAP, filemanip
#    import os
#    import numpy as np
#    
#    img = nb.load(timeseries_file, mmap=NUMPY_MMAP)
#    data = img.get_data()
#    roiimg = nb.load(label_file, mmap=NUMPY_MMAP)
#    rois = roiimg.get_data()
#    prefix = filemanip.split_filename(timeseries_file)[1]
#    out_ts_file = os.path.join(os.getcwd(), 'img.nii.gz' )
#   # with open(out_ts_file, 'wt') as fp:
#    data2=data.copy() 
#    data_txt=[]
#    for fsindex in indices:
#        ijk = np.nonzero(rois == fsindex)
#        data2[ijk]=2
#        ts = np.mean(data[ijk], axis=0)
#        data_txt.append(ts)
#    t_means=np.hstack(data_txt)    
#    np.savetxt(filename, t_means, fmt=b"%.10f")
#    return filename


def merge_nuisance_regressors (nuisance_txts, output_dir, standardize=True):
    
    #https://www.ncbi.nlm.nih.gov/pubmed/30666750
    #Serial filtering introduces spurious correlations: SOLUTIONS:
    #(a) combining all steps into a single linear filter, or (b) sequential orthogonalization of covariates/linear filters performed in series.
    #our procedure is a)
    import numpy as np
    import os
    
    out_files = []
    # TODO: You're just making all these files then reading them in the next function. Why doing so much read and write?
    # TODO: consider passing the data directly
    # FYI: if you have long for loops and you want to know the progress, you can use tqdm
    txts_values = []
    for txt in nuisance_txts:
        txt_values=np.genfromtxt(txt)
        if txt_values.ndim==1:
            txt_values=np.matrix(txt_values)
            txt_values=txt_values[~np.isnan(txt_values)].transpose()
            if standardize:
                print('hy')
                txt_values=(txt_values-txt_values.mean(axis=0))/np.std(txt_values,axis=0)
            txts_values.append(txt_values)
        else:
            txt_values=txt_values[~np.isnan(txt_values).any(axis=1)]
            txt_values=np.matrix(txt_values)
            if standardize:
                print('h')
                txt_values=(txt_values-txt_values.mean(axis=0))/np.std(txt_values,axis=0)
            txts_values.append(txt_values)
        out_params = np.hstack((txts_values))
    
    #adds column of ones
    out_params=np.hstack((np.ones((len(out_params),1)),out_params))
    filename = os.path.join(output_dir, "all_nuisances.txt")
    print(filename)
    np.savetxt(filename, out_params, fmt=b"%.10f")
    return filename
