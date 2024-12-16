
#María Cabello, 16-03-2020

#**************** USAGE ******************
#intermediate-files_SPM-coregister2T1_nii-format.sh -r <root_path> -f <fmri2standard_folder> -b <bids_folder> -s <subject_id> -m <mode>
#bash <where_this_script_is>/intermediate-files_SPM-coregister2T1_nii-format.sh -r /home/mariacabello/wf_workspace -f OUTPUT_fmri2standard -b HEUDICONV_func -s sub-42525 -m to_nii
#bash <where_this_script_is>/intermediate-files_SPM-coregister2T1_nii-format.sh -r /home/mariacabello/wf_workspace -f OUTPUT_fmri2standard -b HEUDICONV_func -s sub-42525 -m to_nii_gz
#*****************************************

#ROOT_PATH=/home/mariacabello/wf_workspace
#FMRI2STANDARD_FOLDER=OUTPUT_fmri2standard
#HEUDICONV_FOLDER=HEUDICONV_func
#ROOT_PATH=/home/mariacabello/wf_workspace/thesis_data/
#FMRI2STANDARD_FOLDER=fmri2standard
#HEUDICONV_FOLDER=func_anat
#SUBJECT_ID=sub-42525


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -r|--root_path)
    ROOT_PATH="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--subject_id)
    SUBJECT_ID="$2"
    shift # past argument
    shift # past value
    ;;
    -m|--mode)
    UNZIP_ZIP="$2"
    shift # past argument
    shift # past value
    ;;
    -f|--fmri2standard_folder)
    FMRI2STANDARD_FOLDER="$2"
    shift # past argument
    shift # past value
    ;;
    -b|--bids_folder)
    HEUDICONV_FOLDER="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

echo $UNZIP_ZIP

if [ "$UNZIP_ZIP" = "to_nii" ] ;then

  mkdir -p $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_sbref
  echo "Created spm_coregister2T1_sbref folder"
  mkdir -p $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_bold
  echo "Created spm_coregister2T1_bold folder"

  echo "Creating copies and unziping files..."
  #SBref
  cp $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/apply_topup_to_SBref/${SUBJECT_ID}_ses-01_run-01_rest_sbref_ap_flirt_corrected.nii.gz $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_sbref/${SUBJECT_ID}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz
  gunzip $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_sbref/${SUBJECT_ID}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz
  echo "...SBref unziped as $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_sbref/${SUBJECT_ID}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii"

  #T1
  cp $ROOT_PATH/$HEUDICONV_FOLDER/${SUBJECT_ID}/ses-01/anat/${SUBJECT_ID}_ses-01_run-01_T1w.nii.gz $ROOT_PATH/$HEUDICONV_FOLDER/${SUBJECT_ID}/ses-01/anat/${SUBJECT_ID}_ses-01_run-01_T1w_copy.nii.gz
  gunzip $ROOT_PATH/$HEUDICONV_FOLDER/${SUBJECT_ID}/ses-01/anat/${SUBJECT_ID}_ses-01_run-01_T1w_copy.nii.gz
  cp $ROOT_PATH/$HEUDICONV_FOLDER/${SUBJECT_ID}/ses-01/anat/${SUBJECT_ID}_ses-01_run-01_T1w_copy.nii $ROOT_PATH/$HEUDICONV_FOLDER/${SUBJECT_ID}/ses-01/anat/${SUBJECT_ID}_ses-01_run-01_T1w.nii
  rm -f $ROOT_PATH/$HEUDICONV_FOLDER/${SUBJECT_ID}/ses-01/anat/${SUBJECT_ID}_ses-01_run-01_T1w_copy.nii
  echo "...T1 unziped as $ROOT_PATH/$HEUDICONV_FOLDER/${SUBJECT_ID}/ses-01/anat/${SUBJECT_ID}_ses-01_run-01_T1w.nii"

  #BOLD
  cp $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/apply_topup/${SUBJECT_ID}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected.nii.gz $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_bold/${SUBJECT_ID}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz
  gunzip $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_bold/${SUBJECT_ID}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz
  echo "...BOLD unziped as $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_bold/${SUBJECT_ID}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii"

elif [ "$UNZIP_ZIP" = "to_nii_gz" ] ;then

  echo "Deleting copies and compressing results from SPM-Coregister..."
  #SBref
  gzip $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_sbref/${SUBJECT_ID}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii
  #if unziped version already exists, remove
  rm -f $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_sbref/${SUBJECT_ID}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii
  echo "...SBref compressed as $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_sbref/${SUBJECT_ID}_ses-01_run-01_rest_sbref_ap_flirt_corrected_coregistered2T1.nii.gz"

  #T1
  rm -f $ROOT_PATH/$HEUDICONV_FOLDER/${SUBJECT_ID}/ses-01/anat/${SUBJECT_ID}_ses-01_run-01_T1w.nii
  echo "...T1 unziped removed"

  #BOLD
  gzip $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_bold/${SUBJECT_ID}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii
  echo "...BOLD compressed as $ROOT_PATH/$FMRI2STANDARD_FOLDER/${SUBJECT_ID}/spm_coregister2T1_bold/${SUBJECT_ID}_ses-01_run-01_rest_bold_ap_roi_mcf_corrected_coregistered2T1.nii.gz"

else
  echo "Error! Unknow mode after flag -m --mode. Possible values are: 'to_nii' or 'to_nii_gz'"
  exit 1
fi

