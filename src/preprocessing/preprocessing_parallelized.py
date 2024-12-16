#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 01:06:21 2022

@author: mariacabello
"""

#exec(open('/home/mariacabello/git_projects/MRI_preprocess/Functional/bold2T1_wf.py').read())
#exec(open('/home/mariacabello/git_projects/MRI_preprocess/Functional/bold_nuisance_correction_wf.py').read())
runfile('/home/mariacabello/git_projects/MRI_preprocess/Functional/bold2T1_wf.py', wdir='/home/mariacabello/git_projects/MRI_preprocess/Functional')
runfile('/home/mariacabello/git_projects/MRI_preprocess/Functional/bold_nuisance_correction_wf.py', wdir='/home/mariacabello/git_projects/MRI_preprocess/Functional')
from bold2T1_wf import get_fmri2standard_wf
from bold_nuisance_correction_wf import get_nuisance_regressors_wf


from nipype.interfaces import spm
from nipype.algorithms.confounds import FramewiseDisplacement, ComputeDVARS
import os
import datetime

from multiprocessing import Pool

root_path = '/home/mariacabello/wf_workspace/bold_preprocess_SA'

heudiconv_folder='func_anat'
fmri2standard_folder='fmri2standard'
bids_path=root_path+'/func_anat'
fmri2standard_path=root_path+'/'+fmri2standard_folder
nuisance_correction_path=root_path+'/nuisance_correction'
recon_all_path=root_path+"/recon_all"#/home/mariacabello/wf_workspace/bold_preprocess_SA/recon_all"
qc_path=root_path+'/QC'


coreg_EPI2T1 = spm.Coregister(
        # target (reference; fixed) [in .nii]
        # source (souce; moving) [in .nii]
        # apply_to_files (moving) [in .nii]
)


def prepare_data(subject_id):
    try:
        if (not os.path.isfile(recon_all_path+"/"+subject_id+"/mri/aseg.mgz")):
            print("Copying aseg.mgz and brain.mgz from institut_recon_all...")
            os.system("mkdir "+recon_all_path+"/"+subject_id)
            os.system("mkdir "+recon_all_path+"/"+subject_id+"/mri")
            os.system("cp /institut/UB/Superagers/MRI/freesurfer-reconall/"+subject_id+"/mri/aseg.mgz "+recon_all_path+"/"+subject_id+"/mri")
            os.system("cp /institut/UB/Superagers/MRI/freesurfer-reconall/"+subject_id+"/mri/brain.mgz "+recon_all_path+"/"+subject_id+"/mri")
        else:
            print("aseg.mgz and brain.mgz already copied")
    except:
        with open(root_path+'/fmri2standard/errors.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while COPYING ASEG.MGZ & BRAIN.MGZ\n")
        return(0)
    
    try:
        if (not os.path.isdir(root_path+"/"+heudiconv_folder+"/"+subject_id)):
            print("Copying bids from institut_bids...")
            os.system("mkdir "+root_path+"/"+heudiconv_folder+"/"+subject_id)
            os.system("mkdir "+root_path+"/"+heudiconv_folder+"/"+subject_id+"/ses-01")
            os.system("cp -r /institut/UB/Superagers/MRI/BIDS/"+subject_id+"/ses-01/func "+root_path+"/"+heudiconv_folder+"/"+subject_id+"/ses-01")
            os.system("cp -r /institut/UB/Superagers/MRI/BIDS/"+subject_id+"/ses-01/anat "+root_path+"/"+heudiconv_folder+"/"+subject_id+"/ses-01")
        else:
            print("Bids already copied")
    except:
        with open(root_path+'/fmri2standard/errors.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while COPYING FUNC-ANAT\n")
        return(0)

def execute_preprocessing_part1(subject_id):
        
    print("##################################################")
    print(subject_id)    
    print("##################################################")

    prepare_data(subject_id)
    
    
    print("\n\nFMRI TO STANDARD\n\n")
    try:
        #defines WF
        fmri2t1_wf=get_fmri2standard_wf([10,750], subject_id, '/home/mariacabello/git_projects/MRI_preprocess/Functional/acparams_hcp.txt')
        
        fmri2t1_wf.base_dir=root_path+'/fmri2standard'
        
        #sets necessary inputs
        fmri2t1_wf.inputs.input_node.T1_img = bids_path+'/{subject_id}/ses-01/anat/{subject_id}_ses-01_run-01_T1w.nii.gz'.format(subject_id=subject_id)
        fmri2t1_wf.inputs.input_node.func_bold_ap_img = bids_path+'/{subject_id}/ses-01/func/{subject_id}_ses-01_run-01_rest_bold_ap.nii.gz'.format(subject_id=subject_id)
        fmri2t1_wf.inputs.input_node.func_sbref_img = bids_path+'/{subject_id}/ses-01/func/{subject_id}_ses-01_run-01_rest_sbref_ap.nii.gz'.format(subject_id=subject_id)
        fmri2t1_wf.inputs.input_node.func_segfm_ap_img = bids_path+'/{subject_id}/ses-01/func/{subject_id}_ses-01_run-01_rest_sefm_ap.nii.gz'.format(subject_id=subject_id)
        fmri2t1_wf.inputs.input_node.func_segfm_pa_img = bids_path+'/{subject_id}/ses-01/func/{subject_id}_ses-01_run-01_rest_sefm_pa.nii.gz'.format(subject_id=subject_id)
        #fmri2t1_wf.inputs.input_node.T1_brain_freesurfer_mask = '/institut/BBHI/MRI/processed_data/freesurfer-reconall/{subject_id}/mri/brainmask.mgz'.format(subject_id=subject_id)
        fmri2t1_wf.inputs.input_node.T1_brain_freesurfer_mask = recon_all_path+'/{subject_id}/mri/brain.mgz'.format(subject_id=subject_id)

#    fmri2t1_wf.inputs.input_node.root_path=root_path 
#    fmri2t1_wf.inputs.input_node.fmri2standard_folder=fmri2standard_folder
#    fmri2t1_wf.inputs.input_node.heudiconv_folder=heudiconv_folder
#    fmri2t1_wf.inputs.input_node.subject_id=subject_id
#    fmri2t1_wf.inputs.input_node.mode_tonii="to_nii"
#    fmri2t1_wf.inputs.input_node.mode_fromnii="to_nii_gz"
    
    
    #writes WF graph and runs it
    #fmri2t1_wf.write_graph()
        fmri2t1_wf.run()
    except:
        with open(root_path+'/fmri2standard/errors.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while FMRI2T1_WF\n")
        return(0)


def execute_preprocessing_part2(subject_id):
    try:
#        sbref_path = root_path+'/'+fmri2standard_folder+"/{subject_id}/apply_topup_to_SBref/{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii".format(subject_id=subject_id)
#        bold_path = root_path+'/'+fmri2standard_folder+"/{subject_id}/apply_topup/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii".format(subject_id=subject_id)
           
        
        # Creating intermediate unziped .nii files
        os.system("bash intermediate-files_SPM-coregister2T1_nii-format.sh -r " + root_path + " -f " + fmri2standard_folder  + " -b " + heudiconv_folder + " -s " + subject_id + " -m to_nii")
    
        
        sbref2T1_path = root_path+'/'+fmri2standard_folder+"/{subject_id}/spm_coregister2T1_sbref/{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii".format(subject_id=subject_id)
        bold2T1_path = root_path+'/'+fmri2standard_folder+"/{subject_id}/spm_coregister2T1_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii".format(subject_id=subject_id)

        # SPM coregistration EPI to Standard T1
        coreg_EPI2T1.inputs.target = root_path+'/'+heudiconv_folder+'/{subject_id}/ses-01/anat/{subject_id}_ses-01_run-01_T1w.nii'.format(subject_id=subject_id)
        coreg_EPI2T1.inputs.source = sbref2T1_path
        coreg_EPI2T1.inputs.jobtype= "estimate"
        coreg_EPI2T1.inputs.apply_to_files = bold2T1_path

        coreg_EPI2T1.run()
        
        # Deleting intermediate unziped .nii files
        os.system("bash intermediate-files_SPM-coregister2T1_nii-format.sh -r " + root_path + " -f " + fmri2standard_folder  + " -b " + heudiconv_folder + " -s " + subject_id + " -m to_nii_gz")

    except:
         with open(root_path+'/fmri2standard/errors.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while SPM COREGISTRATION\n")
         return(0)


def execute_preprocessing_part3_1(subject_id):
    print("\n\nNUISANCE CORRECTION\n\n")
    try:
        sbref2T1_path = root_path+'/'+fmri2standard_folder+"/{subject_id}/spm_coregister2T1_sbref/{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii".format(subject_id=subject_id)
        bold2T1_path = root_path+'/'+fmri2standard_folder+"/{subject_id}/spm_coregister2T1_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii".format(subject_id=subject_id)
      
        #Extracting masks from bold (wm and csf)
        sbref2T1_path=sbref2T1_path+'.gz'
        bold2T1_path=bold2T1_path+'.gz'
        output_masks = root_path+"/nuisance_correction"+"/"+subject_id+"/masks_csf_wm"
        aseg_folder = recon_all_path+"/" + subject_id + "/mri/aseg.mgz"
        os.system("mkdir -p " + root_path+"/nuisance_correction"+"/"+subject_id)
        os.system("mkdir -p " + output_masks)

        os.system("bash extract_wm_csf_eroded_masks.sh -s " + subject_id + " -a " + aseg_folder  + " -r " + bold2T1_path + " -o " + output_masks + " -b " + bold2T1_path + " -e 2")

    except:
        with open(root_path+'/nuisance_correction/errors_wmcsfextraction.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while EXTRACTING WM AND CSF MASKS\n")
        return(0)
        

def execute_preprocessing_part3_2(subject_id): 
    try:
        #defines WF
        wf_reg=get_nuisance_regressors_wf(outdir=root_path+'/nuisance_correction', subject_id=subject_id, timepoints=740)
    
        #sets necessary inputs
        wf_reg.inputs.input_node.realign_movpar_txt = root_path+'/fmri2standard/{subject_id}/realign_fmri2SBref/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf.nii.gz.par'.format(subject_id=subject_id)    
        wf_reg.inputs.input_node.rfmri_unwarped_imgs = root_path+'/fmri2standard/{subject_id}/spm_coregister2T1_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz'.format(subject_id=subject_id)
        #wf_reg.inputs.input_node.masks_imgs = root_path+'/nuisance_correction/{subject_id}/masks_csf_wm/wm_binmask.nii.gz'.format(subject_id=subject_id)
        wf_reg.inputs.input_node.mask_wm = root_path+'/nuisance_correction/{subject_id}/masks_csf_wm/wm_binmask.nii.gz'.format(subject_id=subject_id)
        wf_reg.inputs.input_node.mask_csf = root_path+'/nuisance_correction/{subject_id}/masks_csf_wm/csf_binmask.nii.gz'.format(subject_id=subject_id)
        #CONNECT WITH fmri2standard WF  
        wf_reg.inputs.input_node.bold_img = root_path+'/fmri2standard/{subject_id}/spm_coregister2T1_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz'.format(subject_id=subject_id)
        
        #writes WF graph and runs it
        #wf_reg.write_graph()
        wf_reg.run()
    except:
        with open(root_path+'/nuisance_correction/errors_nuisancewf.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while NUISANCE WF\n")
        return(0)


def execute_preprocessing_part4(subject_id):
    print(("\n\nMNI NORMALIZATION\n\n"))
    try:
        os.system('mkdir -p '+root_path+'/normalization/'+subject_id)
        
        #SEGMENT T1
        T1_niigz=bids_path+'/{subject_id}/ses-01/anat/{subject_id}_ses-01_run-01_T1w.nii.gz'.format(subject_id=subject_id)
        T1_niigzcopy=root_path+'/normalization/{subject_id}/{subject_id}_ses-01_run-01_T1w.nii.gz'.format(subject_id=subject_id)
        T1_nii=root_path+'/normalization/{subject_id}/{subject_id}_ses-01_run-01_T1w.nii'.format(subject_id=subject_id)
        
        os.system('cp '+T1_niigz+' '+T1_niigzcopy)
        os.system('gunzip '+T1_niigzcopy)
              
        
        #APPLY DEFORMATION MNI

        bold_niigz=fmri2standard_path+'/{subject_id}/spm_coregister2T1_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz'.format(subject_id=subject_id)
        bold_niigzcopy=root_path+'/normalization/{subject_id}/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz'.format(subject_id=subject_id)
        bold_nii=root_path+'/normalization/{subject_id}/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii'.format(subject_id=subject_id)
        sbref_niigz=fmri2standard_path+'/{subject_id}/spm_coregister2T1_sbref/{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz'.format(subject_id=subject_id)
        sbref_niigzcopy=root_path+'/normalization/{subject_id}/{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz'.format(subject_id=subject_id)
        sbref_nii=root_path+'/normalization/{subject_id}/{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii'.format(subject_id=subject_id)
        os.system('cp '+bold_niigz+' '+bold_niigzcopy)
        os.system('gunzip '+bold_niigzcopy)
        os.system('cp '+sbref_niigz+' '+sbref_niigzcopy)
        os.system('gunzip '+sbref_niigzcopy)

        MNI=spm.preprocess.Normalize12()
        MNI.inputs.image_to_align = T1_nii
        MNI.inputs.apply_to_files = sbref_nii
        MNI.inputs.write_bounding_box=[[-90,-126,-72], [90,90,108]]
        MNI.run()
        
        MNI=spm.preprocess.Normalize12()
        MNI.inputs.image_to_align = T1_nii
        MNI.inputs.apply_to_files = bold_nii
        MNI.inputs.write_bounding_box=[[-90,-126,-72], [90,90,108]]
        MNI.run()
        
    except:
        with open(root_path+'/normalization/errors.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while NORMALIZATION\n")
        return(0)

def execute_preprocessing_part5(subject_id):        
    print(("\n\APPLYING NUISANCE CORRECTION\n\n"))
    try:
        nuisance_filter_bash='/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/command.txt'.format(subject_id=subject_id)
        new_name_nii = '/home/mariacabello/wf_workspace/bold_preprocess_SA/normalization/{subject_id}/w{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii'.format(subject_id=subject_id)
        
#        new_name_ = '\/home\/mariacabello\/wf_workspace\/thesis_data\/normalization\/{subject_id}\/w{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz'.format(subject_id=subject_id)
#        to_remove_ = '\/home\/mariacabello\/wf_workspace\/thesis_data\/fmri2standard\/{subject_id}\/spm_coregister2T1_bold\/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz'.format(subject_id=subject_id)
        
        nuisance_output='/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt.nii.gz'.format(subject_id=subject_id)
        native_name='/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz'.format(subject_id=subject_id)
        MNI_name='/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/{subject_id}/filter_regressors_bold/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz'.format(subject_id=subject_id)
        
        os.system("gzip "+ new_name_nii)
        os.system("rm "+new_name_nii)
        os.system("mv "+nuisance_output+" "+native_name)
#        
        #print("sed -i 's/"+to_remove_+"/"+new_name_+"/g' "+nuisance_filter_bash)
        
        #os.system("sed -i 's/"+to_remove_+"/"+new_name_+"/g' "+nuisance_filter_bash)
        #os.system("bash "+nuisance_filter_bash)
        #os.system("mv "+nuisance_output+" "+MNI_name)
        mni_pre="/home/mariacabello/wf_workspace/bold_preprocess_SA/normalization/"+subject_id+"/w"+subject_id+"_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz"
        mni_post="/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/"+subject_id+"/filter_regressors_bold/"+subject_id+"_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz"
        nuisances="/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/"+subject_id+"/merge_nuisance_txt/all_nuisances.txt"
#        smoothed="/home/mariacabello/wf_workspace/bold_preprocess_SA/smoothing/"+suj+"_ses-01_run-01_rest_bold_ap_MNI-space_smoothing-8mm.nii.gz"
        command_nuisance="fsl_regfilt -i "+mni_pre+" -o "+mni_post+" -d "+nuisances+" -f '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27'"
    
        os.system("mkdir -p /home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/"+subject_id+"/filter_regressors_bold")
        os.system(command_nuisance)
    except:        
        with open(root_path+'/nuisance_correction/errors.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while Applying nuisance mask\n")
        return(0)

       
    print("\n\nFMRI QC\n\n")
    try:
        os.system("mkdir -p "+root_path+"/QC/"+subject_id)
        fwd=FramewiseDisplacement()
        fwd.inputs.in_file = root_path+'/fmri2standard/{subject_id}/realign_fmri2SBref/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf.nii.gz.par'.format(subject_id=subject_id)
        fwd.inputs.parameter_source = 'FSL'
        fwd.inputs.out_file=root_path+"/QC/"+subject_id+"/framewise_displ.txt"
        fwd.inputs.save_plot=True
        fwd.inputs.out_figure=root_path+"/QC/"+subject_id+"/framewise_displ.pdf"
        #needs seaborn package to plot
        
        fwd.run()
        
    except:
        with open(root_path+'/QC/errors.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while FRAMEWISE_DISPL\n")
        return(0)
        
    try:
        input_=fmri2standard_path+"/"+subject_id+"/binarize_mask/"+subject_id+"_ses-01_run-01_T1w_brain_bin.nii.gz"
        ref=nuisance_correction_path+"/"+subject_id+"/filter_regressors_bold/"+subject_id+"_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz"
        omat=qc_path+"/"+subject_id+"/brain_mask/omat.mat"
        out=qc_path+"/"+subject_id+"/brain_mask/brain_mask_bin_BOLD_T1.nii.gz"
    
        os.system("mkdir "+qc_path+"/"+subject_id+"/brain_mask/")
    
        os.system("flirt -in "+input_+" -ref "+ref+" -omat "+omat)
    
        os.system("flirt -in "+input_+" -applyxfm -init "+omat+" -out "+out+" -paddingsize 0.0 -interp trilinear -ref "+ref)
    
    except:
        with open(root_path+'/QC/errors.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while BRAIN MASK 2 BOLD\n")
        return(0)
    
    try:
        from nipype import Workflow, Node
        from nipype.interfaces import utility
        wf=Workflow(name=subject_id, base_dir='/home/mariacabello/wf_workspace/bold_preprocess_SA/QC')
        
        
        node_input = Node(utility.IdentityInterface(fields=[
            'bold_T1', 
            'brain_mask'
        ]),
        name='input_node') 
        
        node_dvars = Node(ComputeDVARS(
                save_all=True
                ),
        name='dvars_node') 
        
        wf.connect([
                        (node_input,node_dvars,[("bold_T1","in_file")]),
                        (node_input,node_dvars,[("brain_mask","in_mask")])
                        
                        ])
        
        wf.inputs.input_node.bold_T1=ref
        wf.inputs.input_node.brain_mask=out
        wf.run()
        
    except:
        with open(root_path+'/QC/errors.txt', 'a') as f:
            f.write(str(datetime.datetime.now()) + "\t" +subject_id + " did not executed cleanly while DVARS\n")
        return(0)
    
    #REMOVE recon AND bids
    os.system("rm -r "+root_path+"/"+heudiconv_folder+"/"+subject_id)
    os.system("rm -r "+recon_all_path+"/"+subject_id)
    
    return(1)
    
 

def check_error(suj, error_file):   
    with open(error_file, "r") as f:
        text=f.readlines()
    return sum([suj in line for line in text])>0



## RUNNING PREPROCESSING  

todo=os.listdir("/institut/UB/Superagers/MRI/BIDS")

done=[]#os.listdir('/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp2') 
todo=list(set(todo).difference(done))
todo.remove(".heudiconv")
todo.remove("error_heurdiconv.sh")
#todo=done
dict_done=dict()
for id_ in todo:
    
    #bold native
    bold_native=os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/"+id_+"/filter_regressors_bold/"+id_+"_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz")
    
    #sbref native
    sbref_native=os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/fmri2standard/"+id_+"/spm_coregister2T1_sbref/"+id_+"_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz")
    
    #bold MNI
    bold_MNI=os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/"+id_+"/filter_regressors_bold/"+id_+"_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_MNI.nii.gz")
    
    #sbref MNI
    sbref_MNI=os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/normalization/"+id_+"/w"+id_+"_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii") or os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/normalization/"+id_+"/w"+id_+"_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz")
    
    #motion
    motion=os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/fmri2standard/"+id_+"/realign_fmri2SBref/"+id_+"_ses-01_run-01_rest_bold_ap_roi_mcf.nii.gz.par")
    #nuisance regressors
    nuisance=os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/"+id_+"/merge_nuisance_txt/all_nuisances.txt")
    #framewise displ
    framew=os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/QC/"+id_+"/framewise_displ.txt")
    
    dict_done[id_]=[motion,nuisance,bold_native,sbref_native,bold_MNI,sbref_MNI,framew]

    
dict_n_done=dict()
for id_ in dict_done.keys():
    dict_n_done[id_]=sum(dict_done[id_]) 


with open("/home/mariacabello/wf_workspace/bold_preprocess_SA/move.txt", "w") as f:
    for id_ in done:
        f.write(id_+" ")


tofix=[id_ for id_ in dict_done.keys() if dict_done[id_][:4]==[True, True, False, True]]


i=0
nopre=[]
for id_ in tofix:
    print(str(i)+": "+str(id_))
    i=i+1
    nuisances="/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/"+id_+"/merge_nuisance_txt/all_nuisances.txt"
    pre="/home/mariacabello/wf_workspace/bold_preprocess_SA/fmri2standard/"+id_+"/spm_coregister2T1_bold/"+id_+"_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz"
    post="/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/"+id_+"/filter_regressors_bold/"+id_+"_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1_regfilt_NATIVE.nii.gz"
    if os.path.exists(pre) and not os.path.exists(post):
        os.system("fsl_regfilt -i "+pre+" -o "+post+" -d "+nuisances+" -f '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27'")
    elif not os.path.exists(pre):
        nopre.append(id_)
        print("NO PRE")
    else:
        print("POST ALREADY")



todo=[s for s in dict_done.keys() if dict_n_done[s]==0]
##########
# PART 1 #   fmri2std
##########

#nypipe controls if it has been already done or not
pool = Pool(6)                         # Create a multiprocessing Pool
pool.map(execute_preprocessing_part1, todo)  # process list_subjs iterable with pool
#uncomment
#for suj in todo: 
#    execute_preprocessing_part1(suj)


##########
# PART 2 #   SPM coregistration
##########

#needs to subset the list to not repeat the computation

new_list_todo_part2=[id_ for id_ in todo if not check_error(id_,"/home/mariacabello/wf_workspace/bold_preprocess_SA/fmri2standard/errors.txt")]

#new_list_todo_part2=[id_ for id_ in new_list_todo_part2 if not os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/fmri2standard/"+id_+"/spm_coregister2T1_sbref/"+id_+"_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz") and
#                     not os.path.exists("/home/mariacabello/wf_workspace/bold_preprocess_SA/fmri2standard/"+id_+"/spm_coregister2T1_bold/"+id_+"_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz")]

#uncomment
for suj in new_list_todo_part2: 
    execute_preprocessing_part2(suj)
    
    
##########
# PART 3 #   nuisance correction (1)
##########

#nypipe controls if it has been already done or not
    
new_list_todo_part3=[id_ for id_ in todo if not check_error(id_,"/home/mariacabello/wf_workspace/bold_preprocess_SA/fmri2standard/errors.txt")]
#new_list_todo_part3=[id_ for id_ in new_list_todo_part2 if not check_error(id_,"/home/mariacabello/wf_workspace/bold_preprocess_SA/fmri2standard/errors.txt")]


todo_part3_1=new_list_todo_part3
#todo_part3_1=[id_ for id_ in new_list_todo_part3 if not len(os.listdir("/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/"+id_+"/masks_csf_wm"))>8]

#uncomment
pool = Pool(6)                         # Create a multiprocessing Pool
pool.map(execute_preprocessing_part3_1, todo_part3_1)  # process list_subjs iterable with pool

#for suj in todo_part3_1: 
#    execute_preprocessing_part3_1(suj)
    



#todo_part3_2=[id_ for id_ in todo_part3_1 if not check_error(id_,"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/errors_wmcsfextraction.txt")]
todo_part3_2=[id_ for id_ in new_list_todo_part3 if not check_error(id_,"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/errors_wmcsfextraction.txt")]

#uncomment
for suj in todo_part3_2: 
    execute_preprocessing_part3_2(suj)


todo_part3_2=[id_ for id_ in todo_part3_2 if not check_error(id_,"/home/mariacabello/wf_workspace/bold_preprocess_SA/nuisance_correction/errors_nuisancewf.txt")]

#this part is for glasser-timeseries extraction
#pool = Pool(6)                         # Create a multiprocessing Pool
#pool.map(execute_preprocessing_part3_3, todo_part3_2)  # process list_subjs iterable with pool


  
##########
# PART 4 #   SPM normalization
##########

todo_part4=todo_part3_2


#list_subjs_part4=[id_ for id_ in list_subjs_todo if not os.path.exists(root_path+'/normalization/{subject_id}/{subject_id}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii'.format(subject_id=id_)) and not os.path.exists(root_path+'/normalization/{subject_id}/{subject_id}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii'.format(subject_id=id_))]

#uncomment
for suj in todo_part4:
    execute_preprocessing_part4(suj)
 


##########
# PART 5 #   nuisance MNI and QC
##########

todo_part5=[id_ for id_ in todo_part4 if not check_error(id_,"/home/mariacabello/wf_workspace/bold_preprocess_SA/normalization/errors.txt")]

pool = Pool(8)                         # Create a multiprocessing Pool
pool.map(execute_preprocessing_part5, todo_part5)  # process list_subjs iterable with pool

#for suj in todo_part5:
#    execute_preprocessing_part5(suj)
 
