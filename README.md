# fMRI preprocessing

## Getting Started

### Installation
Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Usage

Both cohorts (BBHI and BBHI senior) and all timepoints run from the same scripts on the `main` branch.

1. Activate the virtual environment:
   ```bash
   poetry shell
   ```
2. Set the constants at the top of `run_pipeline.py`. All scripts in the [workflow](#workflow) use them:
   - `COHORTS`: `"bbhi"` and/or `"bbhi_senior"`.
   - `SESSIONS`: `"ses-01"` and/or `"ses-02"`.
   - `SUBJECTS`: subjects to (re)process. Leave empty to auto-detect subjects that have all inputs but no complete output.
   - `N_JOBS` / `N_JOBS_HIGH_MEM`: cores per step.
   - `COHORT_CONFIG`: input, intermediate, output and shared output paths for each cohort.
3. Run the [workflow](#workflow).

Intermediate files are written to `<work_root>/<ses>` and the final outputs to `<output>/<subject>/<ses>`.
Logs are written to `logs/<cohort>_<ses>/`, and a summary of each run to `logs/pipeline_summary_<date>.txt`.

## Workflow

Run these in order from the repo root.

1. **Preprocess**: `python run_pipeline.py --dry-run`, then without `--dry-run`.
   Check the summary at the end for failed subjects.
2. **Scrub**: `python scrubbing.py --dry-run`, then without `--dry-run`.
   Scrubbed images and JSON sidecars are saved next to the BOLD in the local output folder. Scrubbing settings are at the top of the script.
3. **Copy to the shared folder**: `sudo "$(which python)" copy_to_shared.py --dry-run`, then without `--dry-run`.
   Writing to `/pool` needs sudo. Subjects already in the shared folder are skipped; add `--overwrite` to replace a reprocessed subject.
4. **Erase intermediate files**: `python erase_intermediate_files.py --dry-run`, then without `--dry-run`.
   Only subjects with complete final outputs are cleaned up.

---

## Preprocessing functions
The fMRI preprocessing steps are in `preprocessing/preprocessing_steps.py` which is called by `run_pipeline.py`; the nipype workflows they use are in `preprocessing/bold2T1_wf.py` and `preprocessing/bold_nuisance_correction_wf.py`. Below, the functions used in the workflow are described in more detail.

- **transform_fmri_to_standard**: Realigns the BOLD to the SBref and registers it to the T1 for each subject.
  - Also removes the first 10 scans and corrects for field inhomogeneity (e.g. correcting for the anterior -> posterior and p -> a transformations).
  - Uses **get_fmri2standard_wf** (from `bold2T1_wf.py`), which constructs the workflow for transforming fMRI to standard space, including steps for field inhomogeneity correction.
- **execute_coregistration**: Coregisters BOLD images to T1-weighted images using SPM.
- **extract_wm_csf_masks**: Extracts white matter and cerebrospinal fluid masks, needed for nuisance correction.
- **run_nuisance_regression**: Sets up the workflow to remove nuisance signals using the predefined masks.
  - Uses:
    - **get_nuisance_regressors_wf** (from `bold_nuisance_correction_wf.py`): Creates the workflow for nuisance correction.
    - **motion_regressors** (from `bold_nuisance_correction_wf.py`): Computes motion regressors based on realignment parameters, aiding nuisance regression.
    - **cosine_filter_txt** (from `bold_nuisance_correction_wf.py`): Creates a discrete cosine transform (DCT) basis for modeling low-frequency drifts in the data.
    - **merge_nuisance_regressors** (from `bold_nuisance_correction_wf.py`): Combines multiple nuisance regressors into a single matrix for filtering.
- **mni_normalization**: Performs MNI normalization on T1 and fMRI images, aligning them to the standardized brain atlas.
- **apply_nuisance_correction**: Applies processing steps to regress out and remove signal originating from movement, white matter and cerebrospinal fluid.
  - Also runs a band-pass filter that removes all signal that are below the threshold for being considered a BOLD signal response.
- **fmri_quality_control**: Conducts framewise displacement calculations and DVARS computation for fMRI quality assessment.
- **prepare_and_copy_preprocessed_data**: Copies the final native T1 and MNI outputs of each subject to the output folder.

### Additional scripts

- **scrubbing.py**: Interpolates (or cuts) BOLD volumes with framewise displacement above the threshold and saves a JSON sidecar with the scrubbing status. Also aggregates the FWD of all processed subjects into `all_fwd_<ses>.csv`.
- **copy_to_shared.py**: Copies each subject's final outputs, including the scrubbed files, from the local output folder to the shared folder.
- **erase_intermediate_files.py**: Deletes the intermediate files of subjects whose final outputs are complete.

## Notes
- Dependencies include SPM, FreeSurfer, MATLAB, and FSL.
