

Changes:
1. Add README
2. Change name of repo to whatever you want
3. Remove almost all of the os.system calls
4. Remove the `runfile` uses and call the functions directly
5. Consider removing the data -> text -> data pattern, and just using the data directly
6. Clean up the really long nipype.connect functions. Split up if you can


# fMRI_Preprocessing.py

## Purpose
This script executes the fMRI preprocessing workflow using Nipype and other neuroimaging tools. It is designed to facilitate the transformation, coregistration, nuisance correction, normalization, and quality control in fMRI data analysis.

## Functions
- **prepare_data**: Sets up directories and copies necessary MRI data for a subject.
- **transform_fmri_to_standard**: Aligns fMRI data to a standard space - the first volume - for each subject.
- **execute_coregistration**: Coregisters BOLD images to T1-weighted images using SPM.
- **extract_wm_csf_masks**: Extracts white matter and cerebrospinal fluid masks, needed for nuisance correction.
- **run_nuisance_regression**: Removes nuisance signals from fMRI data using the predefined masks.
- **mni_normalization**: Performs MNI normalization on T1 and fMRI images, aligning them to the standardized brain atlas.
- **apply_nuisance_correction**: Applies processing steps to regress out nuisance signals from the fMRI data.
- **fmri_quality_control**: Conducts framewise displacement calculations and DVARS computation for fMRI quality assessment.
- **initialize_preprocessing_dirs**: Sets up initial directories and retrieves a list of subjects to process.
- **get_subjects_with_errors**: Extracts subject IDs from error log files that show encountered errors during each processing steps.
- **extract_subject_id_from_error_log**: Updates the format of the extracted subject ID from the error logs.
- **main**: Initializes paths and directs the execution flow, conducting the preprocessing for all subjects.

## Notes
- The script supports parallel processing for efficiency.
- Dependencies include Nipype, SPM, FSL, and additional custom/Freesurfer scripts as included externally.