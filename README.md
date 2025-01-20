# preprocessing_parallelized.py

## Purpose
This script executes the fMRI preprocessing workflow. It is designed to facilitate preprocessing and quality control for fMRI data analysis using interleaved acquisitions. 

## Functions
- **transform_fmri_to_standard**: Aligns fMRI data to a standard space - the first volume - for each subject.
  - Also removes the first 10 scans and corrects for field inhomogeneity. 

- **execute_coregistration**: Coregisters BOLD images to T1-weighted images using SPM. 

- **extract_wm_csf_masks**: Extracts white matter and cerebrospinal fluid masks, needed for nuisance correction.

- **run_nuisance_regression**: Sets up the workflow to remove nuisance signals using the predefined masks, running a regression to remove signal originating from movement and the other morphology. Also runs a band-pass filter that removes all signal that are below the threshold for being considered a BOLD signal response.  

- **mni_normalization**: Performs MNI normalization on T1 and fMRI images, aligning them to the standardized brain atlas.

- **apply_nuisance_correction**: Applies processing steps to regress out nuisance signals from the fMRI data.

- **fmri_quality_control**: Conducts framewise displacement calculations and DVARS computation for fMRI quality assessment.

- **prepare_and_copy_preprocessed_data**: Prepares necessary directories and copies preprocessed data for each subject to the designated output location.

- **initialize_preprocessing_dirs**: Sets up initial directories and retrieves a list of subjects to process.

- **change_logger_file**: Configures the logging settings for a each processing step.

- **main**: Initializes paths and directs the execution flow, conducting the preprocessing for all subjects.

## Notes
- The script supports parallel processing for efficiency.
- Dependencies include Nipype, SPM, FSL, and Freesurfer.
