# preprocessing_parallelized.py

## Purpose
This script executes the fMRI preprocessing workflow using Nipype, SPM, Freesurfer and fsleyes. It is designed to facilitate the transformation, coregistration, nuisance correction, normalization, and quality control in fMRI data analysis.

## Functions
- **transform_fmri_to_standard**: Aligns fMRI data to a standard space - the first volume - for each subject.
  - Uses FSL with zipped files (exports unzipped).

- **execute_coregistration**: Coregisters BOLD images to T1-weighted images using SPM.
  - Uses SPM with unzipped files (rezips the files after).

- **extract_wm_csf_masks**: Extracts white matter and cerebrospinal fluid masks, needed for nuisance correction.
  - Uses FSL & FreeSurfer with zipped files.

- **run_nuisance_regression**: Removes nuisance signals from fMRI data using the predefined masks.
  - Uses FSL with zipped files.

- **mni_normalization**: Performs MNI normalization on T1 and fMRI images, aligning them to the standardized brain atlas.
  - Uses SPM with unzipped files (unzips).

- **apply_nuisance_correction**: Applies processing steps to regress out nuisance signals from the fMRI data.
  - Uses FSL with zipped files (rezips).

- **fmri_quality_control**: Conducts framewise displacement calculations and DVARS computation for fMRI quality assessment.
  - Generates quality metrics for fMRI data with FSL tools.

- **prepare_and_copy_preprocessed_data**: Prepares necessary directories and copies preprocessed data for each subject to the designated output location.

- **initialize_preprocessing_dirs**: Sets up initial directories and retrieves a list of subjects to process.

- **change_logger_file**: Configures the logging settings for a each processing step.

- **main**: Initializes paths and directs the execution flow, conducting the preprocessing for all subjects.

## Notes
- The script supports parallel processing for efficiency.
- Dependencies include Nipype, SPM, FSL, and Freesurfer.
