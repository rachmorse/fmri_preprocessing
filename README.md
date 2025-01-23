# preprocessing_parallelized.py

## Purpose
This script executes the fMRI preprocessing workflow. It is designed to facilitate preprocessing and quality control for fMRI data analysis using interleaved acquisitions from a ten minute run with 750 volumes.

## Functions
- **transform_fmri_to_standard**: Aligns fMRI data to a standard space - the first volume - for each subject.
  - Also removes the first 10 scans and corrects for field inhomogeneity (e.g. correcting for the anterior -> posterior and p -> a transformations).
  - Uses **get_fmri2standard_wf** (from `bold2T1_wf.py`), which constructs the workflow for transforming fMRI to standard space, including steps for field inhomogeneity correction.

- **execute_coregistration**: Coregisters BOLD images to T1-weighted images using SPM.

- **extract_wm_csf_masks**: Extracts white matter and cerebrospinal fluid masks, needed for nuisance correction.

- **run_nuisance_regression**: Sets up the workflow to remove nuisance signals using the predefined masks.
  - Uses:
    - **get_nuisance_regressors_wf** (from `bold_nuisance_correction.py`): Creates the workflow for nuisance correction.
    - **motion_regressors** (from `bold_nuisance_correction.py`): Computes motion regressors based on realignment parameters, aiding nuisance regression.
    - **cosine_filter_txt** (from `bold_nuisance_correction.py`): Creates a discrete cosine transform (DCT) basis for modeling low-frequency drifts in the data.
    - **merge_nuisance_regressors** (from `bold_nuisance_correction.py`): Combines multiple nuisance regressors into a single matrix for filtering.

- **mni_normalization**: Performs MNI normalization on T1 and fMRI images, aligning them to the standardized brain atlas.

- **apply_nuisance_correction**: Applies processing steps to regress out and remove signal originating from movement, white matter and cerebrospinal fluid.
  - Also runs a band-pass filter that removes all signal that are below the threshold for being considered a BOLD signal response.

- **fmri_quality_control**: Conducts framewise displacement calculations and DVARS computation for fMRI quality assessment.

- **prepare_and_copy_preprocessed_data**: Prepares necessary directories and copies preprocessed data for each subject to the designated output location.

- **initialize_preprocessing_dirs**: Sets up initial directories and retrieves a list of subjects to process.

- **change_logger_file**: Configures the logging settings for a each processing step.

- **main**: Initializes paths and directs the execution flow, conducting the preprocessing for all subjects.

### Additional Functions

- **get_fmri2standard_wf**: Constructs the workflow for transforming fMRI to standard space, including steps for field inhomogeneity correction.

- **motion_regressors**: Computes motion regressors based on realignment parameters, aiding nuisance regression.

- **cosine_filter_txt**: Creates a discrete cosine transform (DCT) basis for modeling low-frequency drifts in the data.

- **merge_nuisance_regressors**: Combines multiple nuisance regressors into a single matrix for filtering.

## Notes
- This pipeline does not use global signal regression or smoothing.
- The script supports parallel processing for efficiency.
- Dependencies include Nipype, SPM, FSL, and Freesurfer.
