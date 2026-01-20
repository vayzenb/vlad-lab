'''
A script to create a BIDS structure, convert dicom to nifti, perform skull stripping on anatomical images,
and generate motion outlier detection for fMRI data using FSL tools.
'''
import os
import subprocess
from pathlib import Path
import pdb
from glob import glob as glob

#project_name = 'vlad-lab' #specify your project name here
#get current working directory
#cwd = os.getcwd()
#git_dir = cwd.split(project_name)[0] + project_name

#add git_dir to path
#sys.path.append(git_dir)

base_dir = f'/zpool/vladlab/data_drive'
raw_data_dir = f'{base_dir}/mri_raw_data'

project_name = 'scofi'
subject_id = ['scofi1001']
ses = ['01'] #session list in format ['01','02']

# ---- Put ALL tasks here ----
TASKS = [
    dict(
        task_name="catloc",
        seq_name="FACE",
        runs=[1, 2, 3],
        func_acq=[14, 22, 29],
    ),
    dict(
        task_name="topheavy",     # goes into BIDS filename: task-topheavy
        seq_name="TOPHEAVY",      
        runs=[1, 2, 3],
        func_acq=[16, 20, 25],    # acquisition numbers corresponding to runs
    ),
    dict(
        task_name="emfl",
        seq_name="EMFL",
        runs=[1, 2, 3],
        func_acq=[18, 27, 31],
    ),
]


anat_acq = 24 #acquistion number for anatomical images 

#skull_strip = True #whether to run skull stripping on anatomical images
skull_strip = False

motion_outliers = True #whether to generate fsl_motion_outliers for functional images

anat_overwrite = False #whether to overwrite existing nifti files for anatomical images
func_overwrite = False #whether to overwrite existing nifti files for functional images

for sub, ses in zip(subject_id, ses):
    
    print(f'Processing subject {sub}, session {ses}')

    #create BIDS structure
    sub_folder = f'{base_dir}/{project_name}/sub-{sub}/ses-{ses}'
    os.makedirs(sub_folder, exist_ok=True) #create subject and session directory for project data
    
    #glob target dicom directories
    dicom_dir = glob(f'{raw_data_dir}/{project_name}/*{sub}/scans/')[0]

    '''
    if anat acquistion number is specified, run dicom to nifti conversion for anatomical images
    '''
    if anat_acq:
        
        #make anat nifti directory
        anat_nifti_dir = f'{base_dir}/{project_name}/sub-{sub}/ses-{ses}/anat'
        os.makedirs(anat_nifti_dir, exist_ok=True)
        # set anat nifti file name
        anat_nifti_file = f'{base_dir}/{project_name}/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_T1w.nii.gz'
        
        # Decide whether to convert
        anat_convert = True
        if os.path.exists(anat_nifti_file):
            if anat_overwrite: #if exist and ovewrite, delete existing file and reconvert later
                print(f'Anatomical nifti file already exists for subject {sub}, session {ses}, overwriting existing file')
                os.remove(anat_nifti_file)
            else: #if exist and not overwrite, skip conversion
                print(f'Anatomical nifti file already exists for subject {sub}, session {ses}, skipping conversion')
                anat_convert = False

        if anat_convert:
            print('Converting anatomical dicoms to nifti')

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
        Run skull stripping on anatomical images using FSL BET
        '''
        if skull_strip:
            print('Running skull stripping on anatomical image')
            #add -f parameter to adjust the fractional intensity threshold if needed
            bash_cmd = f'bet {anat_nifti_file} {anat_nifti_file[:-7]}_brain.nii.gz -R' #-f 0.5 -g 0 -B
            subprocess.run(bash_cmd, check=True, shell=True)

    '''
    If task information is specified, convert functional dicoms to nifti
    '''
    if TASKS:
        
        #make func nifti directory
        func_nifti_dir = f'{base_dir}/{project_name}/sub-{sub}/ses-{ses}/func'
        os.makedirs(func_nifti_dir, exist_ok=True)
        
        for task in TASKS:
            
            task_name = task['task_name']
            seq_name = task['seq_name']
            task_runs = task['runs']
            func_acq = task['func_acq']

            for run, acq in zip(task_runs, func_acq):
                #check if file already exists, if overwrite is False, skip conversion, else delete existing file and reconvert
                func_output_prefix = f'sub-{sub}_ses-{ses}_task-{task_name}_run-{run}_bold'
                func_nifti_file = f'{func_nifti_dir}/{func_output_prefix}.nii.gz'
                
                func_convert = True
                if os.path.exists(func_nifti_file):
                    if func_overwrite: #if exist and ovewrite, delete existing file and reconvert later
                        print(f'Functional nifti file already exists for subject {sub}, session {ses}, task {task_name} run {run}, overwriting existing file')
                        os.remove(func_nifti_file)
                    else: #if exist and not overwrite, skip conversion
                        func_convert = False
                        print(f'Functional nifti file already exists for subject {sub}, session {ses}, task {task_name} run {run}, skipping conversion')
                    
                if func_convert:
                    print(f'Converting functional dicoms to nifti for {task_name} run {run}')
                    #pdb.set_trace()
                    #set dicom and nifti file paths
                    func_dicom_dir = glob(f'{dicom_dir}/{acq}*/resources/DICOM/files/')[0]
 
                    #convert dicom to nifti using dcm2niix
                    #cmd = ['dcm2niix', '-z', 'y', '-f', f'sub-{sub}_ses-{ses}_task-{task_name}_run-{run}_bold', '-o', func_nifti_dir, func_dicom_dir]
                    bash_cmd = f'dcm2niix -f {func_output_prefix} -o {func_nifti_dir} -z y {func_dicom_dir}'
                    subprocess.run(bash_cmd, check=True, shell=True)

                '''generate fsl_motion_outliers for functional images in each run of each task'''
                if motion_outliers:
                    
                    outlier_file = f'{func_nifti_dir}/{func_output_prefix[:-5]}_motion_outliers.tsv'

                    if os.path.exists(outlier_file):
                        print(f'Motion outlier file already exists for subject {sub}, session {ses}, task {task_name} run {run}, skipping motion outlier generation')
                    else:
                        print(f'Generating motion outliers for functional image, task {task_name} run {run}')
                        bash_cmd = f'fsl_motion_outliers -i {func_nifti_file} -o {outlier_file}'
                        subprocess.run(bash_cmd, check=True, shell=True)
