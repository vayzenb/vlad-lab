# Getting started on the workstation


### Overview

Overview text here

contact info: vayzenb@temple.edu

### Connecting to the workstation

Using a terminal:
ssh your_tuid@cla19779.tu.temple.edu

e.g., ssh tuc66980@cla19779.tu.temple.edu

Using remote desktop:
1. Download chrome remote desktop: https://remotedesktop.google.com/
2. Select Setup via ssh


Using VSCode:
1. Download VSCode: 
2. 


### Organization of the workstation

The workstation is split into **/zpool/vladlab** and **/zpool/olsonlab** directories

Within each lab directory is a **data_drive** and a **active_drive** folder. 

The **data_drive** is intended to be primarily a read-only drive that stores large data files that are frequently read, but not modified, such as MRI data.
The **active_drive** is intended to support data and tasks that are frequently updated, such as analysis scripts. 

This split into data and active drives is designed optimize back up schedules for different data types so that the backups don't take up too much space. 

Each drive is backed up on a regular basis using different schedules. Because **data_drive** files should change less frequently, backups are made less regularly but are stored over longer periods (weekly/monthly/yearly).  By contrast, because **active_drive** files may change many times a day, these files are backed up more regularly, but stored for shorter periods of time (hourly/daily/weekly). 

See below for an example file structure for a hypothetical fMRI study 'project1'. The raw and preprocessed fMRI data would be stored in data_drive, ideally in [BIDS format](https://bids.neuroimaging.io/index.html), and the code for analyzing the data would be stored in personal folder in active_drive, ideally in a folder that syncs to [github](https://docs.github.com/en/get-started/git-basics).

.  
└── /zpool/  
    ├── vladlab/  
    │   ├── data_drive/  
    │   │   ├── fmri_project1/  
    │   │   │   ├── sub-001  
    │   │   │   ├── sub-002  
    │   │   │   └── ...  
    │   │   ├── fmri_project2  
    │   │   └── image_sets  
    │   └── active_drive/  
    │       └── vlad/  
    │           └── git_repos/  
    │               ├── project1_code/  
    │               │   ├── analyses  
    │               │   ├── tasks  
    │               │   ├── figures  
    │               │   └── results  
    │               └── project2_code  
    └── olsonlab/  

Note, these are just recommendedations given typical backup needs. There is not a hard constraint on which files can go where.

### Programs already installed on the workstation

Coding: anaconda, Matlab_R2025a, R 

MRI: FSL, Freesurfer 8.0.0 & 7.4.1, AFNI, ANTS, wb_workbench

*To install new software or different versions of existing software email Vlad
*Any python-supported packages should be installed in your own conda environment


