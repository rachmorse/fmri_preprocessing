#!/usr/bin/env python3
"""Scrub the preprocessed BOLD images by interpolating (or cutting) volumes with high framewise displacement.

Scrubbed images and their JSON sidecars are saved next to the preprocessed BOLD in the local output folder.

Usage:
    python scrubbing.py [--dry-run]
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.interpolate

from run_pipeline import COHORT_CONFIG, COHORTS, SESSIONS, SUBJECTS, processed_dirs

# Make sure that these match the values used to process any other subs that will be combined with these
THRESHOLD = 0.5  # FWD threshold in mm
METHOD = "interpolate"  # "interpolate" or "cut"
MAX_SCRUB_PERCENT = 30  # Subjects with more frames above the threshold are not scrubbed to save space
N_JOBS = 3

# Scrubbed file suffix per cohort, matching the existing shared files
SCRUBBED_SUFFIX = {
    "bbhi": "scrubbed-interp-{threshold_str}",
    "bbhi_senior": "scrubbed_{threshold}",
}

# Output folder and the space in its file names
SPACES = {
    "native_T1": "T1-space",
    "MNI_2mm": "MNI-space",
}


def aggregate_fwd_data(cohort, ses, output_file):
    """Aggregates framewise_displ.txt from the local and shared output folders into a single CSV.

    This creates a combined FWD file for all subjects who have been processed, not just subjects
    scrubbed while running this script.

    Args:
        cohort (str): Cohort identifier (e.g., "bbhi").
        ses (str): Session identifier (e.g., "ses-01").
        output_file (str or Path): Path to the output CSV file.
    """
    fwd_list = []
    processed_subjects = set()

    # Local folder first so reprocessed subjects use their new FWD
    templates = processed_dirs(cohort, ses)
    print(f"Aggregating FWD data from: {templates}")

    for template in templates:
        root_dir = template.split("{subject}")[0]
        if not os.path.exists(root_dir):
            print(f"Warning: Root directory not found: {root_dir}")
            continue

        for subject in sorted(os.listdir(root_dir)):
            if not subject.startswith("sub-"):
                continue

            # Avoid duplicates if subject exists in multiple roots (priority to first found)
            if subject in processed_subjects:
                continue

            fwd_path = Path(template.format(subject=subject)) / "native_T1" / "framewise_displ.txt"

            if fwd_path.exists():
                try:
                    fwd_data_df = pd.read_csv(fwd_path)
                    fwd_series = pd.Series(fwd_data_df["FramewiseDisplacement"].tolist(), name=subject)
                    fwd_list.append(fwd_series.to_frame().T)
                    processed_subjects.add(subject)
                except Exception as e:
                    print(f"Error reading FWD for {subject} in {root_dir}: {e}")

    if fwd_list:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        all_fwd_df = pd.concat(fwd_list)
        all_fwd_df.to_csv(output_file, index=True, header=False)
        print(f"Aggregated FWD data saved to {output_file} ({len(fwd_list)} subjects)")
    else:
        print("No FWD data found to aggregate.")


def save_json_sidecar(json_file, status, threshold, percent_scrubbed, num_bad_frames, total_frames, max_scrub_percent):
    """Saves the scrubbing status and statistics to a JSON sidecar file.

    Args:
        json_file (str): Path to the output JSON file.
        status (str): The scrubbing status (e.g., 'Scrubbed', 'NoScrubbingNeeded_LowMotion', 'SkippedScrubbing_HighMotion').
        threshold (float): The FWD threshold used.
        percent_scrubbed (float): Percentage of frames scrubbed.
        num_bad_frames (int): Count of frames scrubbed.
        total_frames (int): Total number of frames in the series.
        max_scrub_percent (float): Maximum allowed percentage of scrubbed frames.
    """
    data = {
        "Description": "Scrubbed fMRI images",
        "ExclusionCriteria": f"Subjects with >{max_scrub_percent}% frames exceeding {threshold}mm FD were excluded.",
        "ScrubbingStatus": status,
        "MotionThreshold": threshold,
        "PercentageFramesScrubbed": round(percent_scrubbed, 2),
        "NumFramesScrubbed": int(num_bad_frames),
        "TotalFrames": int(total_frames)
    }

    try:
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved scrub status JSON to {json_file}")
    except Exception as e:
        print(f"Error saving JSON file {json_file}: {e}")


def scrub(subject, bold_file, fwd_data, scrubbed_file, shared_scrubbed_file, json_file, threshold=0.5, method="interpolate", max_scrub_percent=30, overwrite=False):
    """Scrubs BOLD fMRI images by either removing or interpolating scans based on FWD threshold.

    Args:
        subject (str): Subject ID used to process the BOLD and FWD files.
        bold_file (str): Path to the BOLD image file (NIfTI format).
        fwd_data (array-like): Framewise Displacement data.
        scrubbed_file (str): Path to save the scrubbed BOLD image file (NIfTI format).
        shared_scrubbed_file (str): Path of the scrubbed BOLD image in the shared folder.
        json_file (str): Path to save the JSON sidecar file.
        threshold (float, optional): Threshold for FWD above which scans are considered moved. Defaults to 0.5 FWD.
        method (str, optional): Method for handling moved scans. Either 'cut' or 'interpolate'. Defaults to 'interpolate'.
        max_scrub_percent (float, optional): Maximum percentage of frames that can be scrubbed before the subject is skipped. Defaults to 30%.
        overwrite (bool, optional): Whether to scrub again even if a scrubbed file already exists. Defaults to False.

    Returns:
        bool: True if scrubbing was performed, False if subject was skipped due to too many frames exceeding threshold.
    """
    # Check if no frames exceed threshold, skip loading BOLD and scrubbing
    fwd_check = np.array(fwd_data)
    total_frames_est = len(fwd_check) + 1

    if not np.any(np.nan_to_num(fwd_check) >= threshold):
        print(f"No frames exceed the threshold for subject {subject}. Skipping scrubbing.")
        save_json_sidecar(json_file, "NoScrubbingNeeded_LowMotion", threshold, 0.0, 0, total_frames_est, max_scrub_percent)
        return False

    # Then check if too many frames exceed threshold, skip loading BOLD and scrubbing
    num_bad_frames = np.sum(np.nan_to_num(fwd_check) >= threshold)
    est_scrub_percent = (num_bad_frames * 100) / total_frames_est

    if est_scrub_percent > max_scrub_percent:
        print(f"WARNING: More than {max_scrub_percent}% of timepoints would be scrubbed for subject {subject}. Skipping scrubbing.")
        save_json_sidecar(json_file, "SkippedScrubbing_HighMotion", threshold, est_scrub_percent, num_bad_frames, total_frames_est, max_scrub_percent)
        return False

    # Check for existing scrubbed files (local or shared) to avoid re-scrubbing.
    # If found, make sure the JSON sidecar exists
    file_exists = False
    source_loc = ""

    if os.path.exists(scrubbed_file):
        file_exists = True
        source_loc = "local"
    elif os.path.exists(shared_scrubbed_file):
        file_exists = True
        source_loc = "shared"

    if file_exists and not overwrite:
        if not os.path.exists(json_file):
            print(f"Scrubbed file exists ({source_loc}) for {subject}, regenerating JSON sidecar only.")

            # Use fixed total frames (740) for efficiency to avoid loading BOLD header
            total_frames = 740
            fwd = np.array(fwd_data)

            # Check for array alignment
            if len(fwd) == total_frames:
                pass
            elif len(fwd) == total_frames - 1:
                fwd = np.insert(fwd, 0, 0)
            else:
                raise ValueError(f"Length mismatch: FWD has {len(fwd)} points, expected {total_frames} volumes.")

            # Identify volumes with excessive motion
            all_tps = np.arange(total_frames)
            incorrect_tps = all_tps[fwd >= threshold]

            # Calculate percentage of volumes that would be scrubbed
            scrub_percent = (len(incorrect_tps) * 100) / total_frames

            # Save JSON
            save_json_sidecar(json_file, "Scrubbed", threshold, scrub_percent, len(incorrect_tps), total_frames, max_scrub_percent)
            print(f"JSON sidecar regenerated for {subject}")
            return True
        else:
            print(f"Scrubbed file ({source_loc}) and JSON already exist for {subject}. Skipping re-scrubbing.")
            return True

    print(f"Scrubbing BOLD image for subject: {subject}")

    # Load BOLD image data
    print("Loading BOLD image from file:", bold_file)
    bold = nib.load(bold_file)
    bold_data = bold.get_fdata()
    bold_affine = bold.affine

    # Load FWD data
    fwd = np.array(fwd_data)

    # Check for array alignment
    if len(fwd) == bold_data.shape[3]:
        pass # Lengths match, no adjustment needed
    elif len(fwd) == bold_data.shape[3] - 1:
        # FWD is one shorter than BOLD, prepend 0 to align.
        fwd = np.insert(fwd, 0, 0)
    else:
        raise ValueError(f"Length mismatch: FWD has {len(fwd)} points, BOLD has {bold_data.shape[3]} volumes.")

    # Identify volumes with excessive motion
    all_tps = np.arange(bold_data.shape[3])
    correct_tps = all_tps[fwd < threshold]
    incorrect_tps = all_tps[fwd >= threshold]

    # Calculate percentage of volumes that would be scrubbed
    scrub_percent = (len(incorrect_tps) * 100) / bold_data.shape[3]

    print(
        f"{len(incorrect_tps)} out of {bold_data.shape[3]} scans ({round(scrub_percent, 2)}%) exceed the motion threshold (FWD > {threshold})."
    )

    # Extract only the correct timepoints for interpolation
    correct_bold = bold_data[:, :, :, correct_tps]

    # Start scrubbing based on the method
    if method == "cut":
        # If the method is 'cut', remove the timepoints with excessive motion. Save as scrubbed_data
        scrubbed_data = correct_bold
        print(f"Removing {len(incorrect_tps)} scans due to excessive motion.")
    elif method == "interpolate":
        # If the method is 'interpolate', replace the timepoints with excessive motion through interpolation or extrapolation
        scrubbed_data = bold_data.copy()
        # Check if the first or last timepoints are incorrect because if they are, they should be extrapolated
        if 1 not in incorrect_tps and bold_data.shape[3] - 1 not in incorrect_tps:
            print(f"Interpolating {len(incorrect_tps)} scans due to excessive motion.")
            # Perform interpolation when neither the first nor last timepoints are incorrect
            interpolator = scipy.interpolate.interp1d(correct_tps, correct_bold, axis=3, fill_value="extrapolate")
            scrubbed_data[:, :, :, incorrect_tps] = interpolator(incorrect_tps)
            print("No scans require extrapolation.")
        else:
            extrap_idx = []
            intrap_idx = list(incorrect_tps)
            extrap_text = []
            i = 1
            # Perform left extrapolation for incorrect first timepoint
            while i in incorrect_tps:
                extrap_idx.append(i)
                intrap_idx.remove(i)
                extrap_text.append("left")
                i += 1
            i = 1
            # Perform right extrapolation for incorrect last timepoint
            while bold_data.shape[3] - i in incorrect_tps:
                extrap_idx.append(bold_data.shape[3] - i)
                intrap_idx.remove(bold_data.shape[3] - i)
                extrap_text.append("right")
                i += 1

            print(f"Interpolating {len(intrap_idx)} scans due to excessive motion.")
            interpolator = scipy.interpolate.interp1d(correct_tps, correct_bold, axis=3)
            scrubbed_data[:, :, :, intrap_idx] = interpolator(intrap_idx)

            if extrap_text:
                print(
                    f"Extrapolating {len(extrap_idx)} scans in the {', '.join(extrap_text)} direction(s) due to motion."
                )
                extrapolator = scipy.interpolate.interp1d(correct_tps, correct_bold, fill_value="extrapolate", axis=3)
                scrubbed_data[:, :, :, extrap_idx] = extrapolator(extrap_idx)

    else:
        print("Unknown method specified; returning the original BOLD file.")
        scrubbed_data = bold_data  # Unchanged; return original BOLD data

    scrubbed_image = nib.Nifti1Image(scrubbed_data, bold_affine)
    os.makedirs(os.path.dirname(scrubbed_file), exist_ok=True)
    nib.save(scrubbed_image, scrubbed_file)

    # Save JSON for Performed case
    save_json_sidecar(json_file, "Scrubbed", threshold, scrub_percent, len(incorrect_tps), bold_data.shape[3], max_scrub_percent)

    print(f"Scrubbing complete for subject: {subject}")
    return True


def subject_paths(cohort, ses, subject, folder_name):
    """Builds the input and output paths for one subject, session and space.

    Args:
        cohort (str): Cohort identifier (e.g., "bbhi").
        ses (str): Session identifier (e.g., "ses-01").
        subject (str): Subject ID.
        folder_name (str): Output folder, either "native_T1" or "MNI_2mm".

    Returns:
        dict: Paths of the `fwd`, `bold`, `scrubbed`, `json` and `shared_scrubbed` files.
    """
    local_dir, shared_dir = (Path(t.format(subject=subject)) / folder_name for t in processed_dirs(cohort, ses))
    stem = f"{subject}_{ses}_run-01_rest_bold_ap_{SPACES[folder_name]}"
    suffix = SCRUBBED_SUFFIX[cohort].format(threshold=THRESHOLD, threshold_str=str(THRESHOLD).replace(".", ""))

    return {
        "fwd": str(local_dir / "framewise_displ.txt"),
        "bold": str(local_dir / f"{stem}.nii.gz"),
        "scrubbed": str(local_dir / f"{stem}_{suffix}.nii.gz"),
        "json": str(local_dir / f"{stem}_{suffix}.json"),
        "shared_scrubbed": str(shared_dir / f"{stem}_{suffix}.nii.gz"),
    }


def process_subject(subject, ses, cohort, folder_name, error_log, overwrite):
    """Processes a single subject by scrubbing the BOLD fMRI images based on the FWD.

    Saves the scrubbed BOLD file for the subject and logs errors if any occur.

    Args:
        subject (str): Subject ID used to process the BOLD and FWD files.
        ses (str): Session (timepoint).
        cohort (str): Cohort identifier (e.g., "bbhi").
        folder_name (str): Output folder, either "native_T1" or "MNI_2mm".
        error_log (str): Path to error log.
        overwrite (bool): Whether to scrub again even if a scrubbed file already exists.

    Returns:
        bool: True if processing was successful (even if skipped), False if error.
    """
    try:
        paths = subject_paths(cohort, ses, subject, folder_name)
        print(f"Processing subject: {subject}")

        fwd_data = pd.read_csv(paths["fwd"])["FramewiseDisplacement"].tolist()

        scrub(
            subject,
            paths["bold"],
            fwd_data,
            paths["scrubbed"],
            paths["shared_scrubbed"],
            paths["json"],
            threshold=THRESHOLD,
            method=METHOD,
            max_scrub_percent=MAX_SCRUB_PERCENT,
            overwrite=overwrite,
        )
        return True

    except Exception as e:
        print(f"Error processing subject {subject}: {e}")
        with open(error_log, "a") as f:
            f.write(f"Error processing subject {subject}: {e}\n")
        return False


def needs_scrubbing(cohort, ses, subject, folder_name):
    """Checks whether a subject still needs scrubbing or a missing JSON sidecar.

    Determine processing needs:
    1. Full scrub if no scrubbed data exists (local or shared).
    2. JSON regeneration if scrubbed data exists but sidecar is missing.

    Args:
        cohort (str): Cohort identifier (e.g., "bbhi").
        ses (str): Session identifier (e.g., "ses-01").
        subject (str): Subject ID.
        folder_name (str): Output folder, either "native_T1" or "MNI_2mm".

    Returns:
        bool: True if the subject should be processed.
    """
    paths = subject_paths(cohort, ses, subject, folder_name)

    if not os.path.exists(paths["fwd"]):
        print(f"No framewise_displ.txt found for {subject}")
        return False

    local_exists = os.path.exists(paths["scrubbed"])
    shared_exists = os.path.exists(paths["shared_scrubbed"])

    if not local_exists and not shared_exists:
        return True
    if not os.path.exists(paths["json"]):
        print(f"Subject {subject} - Scrubbed NIfTI exists, but JSON missing. Adding to process list.")
        return True
    return False


def main():
    """Main function to run this script.

    This function performs the following steps:

    1. Aggregates the FWD of all processed subjects for each cohort and session.
    2. Generates a list of subjects to be processed for each space (native T1 and MNI).
    3. Scrubs the BOLD images by parallel processing subjects.
    4. Saves the scrubbed BOLD images and JSON sidecars next to the preprocessed BOLD.
    5. Logs errors to `scrubbing_errors_<ses>_<folder>.txt` in the output folder.

    Subjects in `SUBJECTS` are always scrubbed again; otherwise subjects with existing scrubbed data are skipped.
    """
    parser = argparse.ArgumentParser(description="Scrub the preprocessed BOLD images.")
    parser.add_argument("--dry-run", action="store_true", help="List the subjects that would be scrubbed and exit.")
    args = parser.parse_args()

    for cohort in COHORTS:
        output_root = COHORT_CONFIG[cohort].output
        if not output_root.is_dir():
            continue

        for ses in SESSIONS:
            print(f"\n=== {cohort} {ses} ===")
            if not args.dry_run:
                aggregate_fwd_data(cohort, ses, output_root / f"all_fwd_{ses}.csv")

            for folder_name in SPACES:
                print(f"\n--- Processing {folder_name} ---")
                error_log = str(output_root / f"scrubbing_errors_{ses}_{folder_name}.txt")

                if SUBJECTS:
                    subjects = [s for s in SUBJECTS if (output_root / s / ses).is_dir()]
                else:
                    subjects = [
                        s for s in sorted(os.listdir(output_root))
                        if (output_root / s / ses).is_dir() and needs_scrubbing(cohort, ses, s, folder_name)
                    ]

                print(f"Total subjects needing processing: {len(subjects)}")
                print(f"Subjects to be processed: {subjects}")

                if args.dry_run:
                    continue

                with Pool(N_JOBS) as pool:
                    pool.starmap(
                        process_subject,
                        [(subject, ses, cohort, folder_name, error_log, bool(SUBJECTS)) for subject in subjects],
                    )

    print("Processing complete.")


if __name__ == "__main__":
    main()
