
#María Cabello, 17-03-2020

#**************** USAGE ******************
#extract_wm_csf_eroded_masks.sh -s <subject_id> -r <reference_file> -a <aseg_file> -o <output_folder> -b <bold_file> -e <erode_mm>
#bash <where_this_script_is>/extract_wm_csf_eroded_masks.sh -s -r -a -o -b -e
#*****************************************

#SUBJECT_ID=sub-103151
#REFERENCE_FILE=/home/mariacabello/wf_workspace/thesis_data/fmri2standard/$SUBJECT_ID/apply_topup_to_SBref/${SUBJECT_ID}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz
#ASEG_FILE=/institut/processed_data/BBHI_output/structural/$SUBJECT_ID/mri/aseg.mgz
#OUTPUT_FOLDER=/home/mariacabello/wf_workspace/masks_extraction/${SUBJECT_ID}_2
#BOLD_FILE=/home/mariacabello/wf_workspace/thesis_data/fmri2standard/$SUBJECT_ID/apply_topup/${SUBJECT_ID}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz
#ERODE=1

POSITIONAL=()

# Parse input arguments
while [[ $# -gt 0 ]]; do
  key="$1"

case $key in
-s|--subject_id)
    SUBJECT_ID="$2"
    shift # past argument
    shift # past value
    ;;
-e|--erode)
    ERODE="$2"
    shift # past argument
    shift # past value
    ;;
-b|--bold_file)
    BOLD_FILE="$2"
    shift # past argument
    shift # past value
    ;;
-a|--aseg_file)
    ASEG_FILE="$2"
    shift # past argument
    shift # past value
    ;;
-o|--output_folder)
    OUTPUT_FOLDER="$2"
    shift # past argument
    shift # past value
    ;;
*)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

# Validate necessary variables are set
if [[ -z $SUBJECT_ID || -z $ERODE || -z $BOLD_FILE || -z $ASEG_FILE || -z $OUTPUT_FOLDER ]]; then
  echo "Missing one or more required arguments. Aborting."
  exit 1
fi

# WM processing
echo "WM 0"
mri_binarize --i "$ASEG_FILE" --match 2 41 --o "$OUTPUT_FOLDER/wm.mgz" --erode "$ERODE"

echo "WM 1.a"
mri_vol2vol --mov "$OUTPUT_FOLDER/wm.mgz" --regheader --targ "$BOLD_FILE" --o "$OUTPUT_FOLDER/wm_sbref.mgz"

echo "WM 1.b"
mri_convert --in_type mgz --out_type nii "$OUTPUT_FOLDER/wm_sbref.mgz" "$OUTPUT_FOLDER/wm.nii.gz"

echo "WM 2"
fslmaths "$OUTPUT_FOLDER/wm.nii.gz" -bin "$OUTPUT_FOLDER/wm_binmask.nii.gz"

echo "WM 3"
fslmaths "$BOLD_FILE" -mul "$OUTPUT_FOLDER/wm_binmask.nii.gz" "$OUTPUT_FOLDER/wm_bold_extracted.nii.gz"

# CSF processing
echo "CSF 0"
mri_binarize --i "$ASEG_FILE" --match 4 5 14 15 24 43 44 --o "$OUTPUT_FOLDER/csf.mgz" --erode "$ERODE"

echo "CSF 1.a"
mri_vol2vol --mov "$OUTPUT_FOLDER/csf.mgz" --regheader --targ "$BOLD_FILE" --o "$OUTPUT_FOLDER/csf_sbref.mgz"

echo "CSF 1.b"
mri_convert --in_type mgz --out_type nii "$OUTPUT_FOLDER/csf_sbref.mgz" "$OUTPUT_FOLDER/csf.nii.gz"

echo "CSF 2"
fslmaths "$OUTPUT_FOLDER/csf.nii.gz" -bin "$OUTPUT_FOLDER/csf_binmask.nii.gz"

echo "CSF 3"
fslmaths "$BOLD_FILE" -mul "$OUTPUT_FOLDER/csf_binmask.nii.gz" "$OUTPUT_FOLDER/csf_bold_extracted.nii.gz"

echo "END"
