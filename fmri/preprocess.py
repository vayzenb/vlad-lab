'''
A script to create a BIDS structure, convert dicom to nifti, perform skull stripping on anatomical images,
and generate motion outlier detection for fMRI data using FSL tools.
'''

project_name = 'vlad-lab' #specify your project name here
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
#add git_dir to path
sys.path.append(git_dir)

import subprocess
import json
from pathlib import Path
import pdb
import argparse
from glob import glob as glob



base_dir = f'/zpool/vladlab/data_drive'
raw_data_dir = f'{base_dir}/mri_raw_data'

project_name = 'scofi'
subject_id = ['scofi1001']
ses = ['01'] #session list in format ['01','02']
anat_acq = 6 #acquistion number for anatomical images

func_acq = [14,22,29] #acquistion numbers for functional images
seq_name = 'FACE'
task_name = 'catloc' #specify your task name here, e.g., 'rest', 'catloc', this will need to be run seperately for each task
task_runs = [1,2,3] #specify your task runs here, e.g., [1,2,3], this will need to be run seperately for each task

skull_strip = True #whether to run skull stripping on anatomical images

motion_outliers = True #whether to generate fsl_motion_outliers for functional images

overwrite = False #whether to overwrite existing files

for sub, ses in zip(subject_id, ses):
    print(f'Processing subject {sub}, session {ses}')

    #create BIDS structure
    os.makedirs(f'{base_dir}/{project_name}/sub-{sub}/ses-{ses}', exist_ok=True) #create subject and session directory for project data
    
    #glob target dicom directories
    dicom_dir = glob(f'{raw_data_dir}/{project_name}/*{sub}/scans/')[0]


    '''
    if anat acquistion number is specified, run dicom to nifti conversion for anatomical images
    '''
    if anat_acq:

        #check if file already exists, if overwrite is False, skip conversion, else delete existing file and reconvert
        anat_nifti_file = f'{base_dir}/{project_name}/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_T1w.nii.gz'
        if os.path.exists(anat_nifti_file) and not overwrite:
            print(f'Anatomical nifti file already exists for subject {sub}, session {ses}, skipping conversion')
            
        elif os.path.exists(anat_nifti_file) and overwrite:
            print(f'Anatomical nifti file already exists for subject {sub}, session {ses}, overwriting existing file')
            os.remove(anat_nifti_file)

            print('Converting anatomical dicoms to nifti')
            #make anat nifti directory
            anat_nifti_dir = f'{base_dir}/{project_name}/sub-{sub}/ses-{ses}/anat'
            os.makedirs(anat_nifti_dir, exist_ok=True)
            #set dicom and nifti file paths
            anat_dicom_dir = glob(f'{dicom_dir}/*T1w_MPR/resources/DICOM/files/')[0]
            #remove trailing slash if exists
            if anat_dicom_dir.endswith('/'):
                anat_dicom_dir = anat_dicom_dir[:-1]
            
            
            #convert dicom to nifti using dcm2niix
            #cmd = ['dcm2niix', '-z', 'y', '-f', f'sub-{sub}_ses-{ses}_T1w', '-o', anat_nifti_dir, anat_dicom_dir]
            bash_cmd = f'dcm2niix -o {anat_nifti_dir} -f sub-{sub}_ses-{ses}_T1w -z y {anat_dicom_dir}'
            #pdb.set_trace()
            subprocess.run(bash_cmd, check=True, shell=True)

    '''
    Convert functional dicoms to nifti
    '''
    if func_acq:
        #make func nifti directory
        func_nifti_dir = f'{base_dir}/{project_name}/sub-{sub}/ses-{ses}/func'
        os.makedirs(func_nifti_dir, exist_ok=True)
        for run, acq in zip(task_runs, func_acq):
            #check if file already exists, if overwrite is False, skip conversion, else delete existing file and reconvert
            func_nifti_file = f'{func_nifti_dir}/sub-{sub}_ses-{ses}_task-{task_name}_run-{run}_bold.nii.gz'
            if os.path.exists(func_nifti_file) and not overwrite:
                print(f'Functional nifti file already exists for subject {sub}, session {ses}, task {task_name} run {run}, skipping conversion')
                
            elif os.path.exists(func_nifti_file) and overwrite:
                print(f'Functional nifti file already exists for subject {sub}, session {ses}, task {task_name} run {run}, overwriting existing file')
                os.remove(func_nifti_file)
                
                print(f'Converting functional dicoms to nifti for {task_name} run {run}')
                #pdb.set_trace()
                #set dicom and nifti file paths
                func_dicom_dir = glob(f'{dicom_dir}/{acq}*/resources/DICOM/files/')[0]
                
                
                
                func_nifti_file = f'{func_nifti_dir}/sub-{sub}_ses-{ses}_task-{task_name}_run-0{run}_bold.nii.gz'
                #convert dicom to nifti using dcm2niix
                #cmd = ['dcm2niix', '-z', 'y', '-f', f'sub-{sub}_ses-{ses}_task-{task_name}_run-{run}_bold', '-o', func_nifti_dir, func_dicom_dir]
                bash_cmd = f'dcm2niix -f sub-{sub}_ses-{ses}_task-{task_name}_run-{run}_bold -o {func_nifti_dir} -z y {func_dicom_dir}'
                subprocess.run(bash_cmd, check=True, shell=True)

    '''
    Run skull stripping on anatomical images using FSL BET
    '''
    if skull_strip:
        print('Running skull stripping on anatomical image')
        #add -f parameter to adjust the fractional intensity threshold if needed
        bash_cmd = f'bet {anat_nifti_file} {anat_nifti_file[:-7]}_brain.nii.gz -R' #-f 0.5 -g 0 -B
        subprocess.run(bash_cmd, check=True, shell=True)

    '''generate fsl_motion_outliers for functional images'''
    if motion_outliers:
        for run in task_runs:
            print(f'Generating motion outliers for functional image, task {task_name} run {run}')
            func_nifti_file = f'{func_nifti_dir}/sub-{sub}_ses-{ses}_task-{task_name}_run-{run}_bold.nii.gz'
            outlier_file = f'{func_nifti_dir}/sub-{sub}_ses-{ses}_task-{task_name}_run-{run}_motion_outliers.tsv'
            bash_cmd = f'fsl_motion_outliers -i {func_nifti_file} -o {outlier_file}'
            pdb.set_trace()
            subprocess.run(bash_cmd, check=True, shell=True)
'''

def create_bids_structure(base_dir, subject_id, project_name):
    """Create BIDS folder structure."""
    bids_dir = Path(base_dir) / project_name
    sub_dir = bids_dir / f"sub-{subject_id}"
    anat_dir = sub_dir / "anat"
    func_dir = sub_dir / "func"
    
    anat_dir.mkdir(parents=True, exist_ok=True)
    func_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "bids_root": str(bids_dir),
        "sub_dir": str(sub_dir),
        "anat_dir": str(anat_dir),
        "func_dir": str(func_dir)
    }

def skull_strip(input_img, output_img):
    """Run FSL BET for skull stripping."""
    cmd = ["bet", input_img, output_img, "-B"]
    try:
        subprocess.run(cmd, check=True)
        print(f"Skull stripping completed: {output_img}")
    except subprocess.CalledProcessError as e:
        print(f"Error during skull stripping: {e}")

def generate_motion_outliers(fmri_file, output_file):
    """Generate motion outlier detection using fsl_motion_outliers."""
    cmd = ["fsl_motion_outliers", "-i", fmri_file, "-o", output_file, "-s"]
    try:
        subprocess.run(cmd, check=True)
        print(f"Motion outliers generated: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating motion outliers: {e}")

def main(base_dir, subject_id, project_name, anat_image, func_image):
    """Main pipeline."""
    # Create BIDS structure
    dirs = create_bids_structure(base_dir, subject_id, project_name)
    
    # Skull strip anatomical image
    anat_output = os.path.join(dirs["anat_dir"], f"sub-{subject_id}_T1w_brain.nii.gz")
    skull_strip(anat_image, anat_output)
    
    # Generate motion outliers
    outliers_output = os.path.join(dirs["func_dir"], f"sub-{subject_id}_motion_outliers.txt")
    generate_motion_outliers(func_image, outliers_output)
    
    print(f"Pipeline completed for subject {subject_id}")

if __name__ == "__main__":
    main(
        base_dir="/data",
        subject_id="001",
        project_name="my_project",
        anat_image="/path/to/anat.nii.gz",
        func_image="/path/to/func.nii.gz"
    )

'''