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

Each drive is backed up on a regular basis use different schedules. Because **data_drive** files should change less frequently, backups are made less regularly but are stored over longer periods (weekly/monthly/yearly). By contrast, because **active_drive** files may change many times a day, these files are backed up more regularly, but stored for shorter periods of time (hourly/daily/weekly).

See below for an example file structure for a hypothetical fMRI study 'project1'. The raw and preprocessed fMRI data would be stored in data

.
└── /zpool/
    ├── vladlab/
    │   ├── data_drive/
    │   │   ├── project1/
    │   │   │   ├── sub-001
    │   │   │   ├── sub-002
    │   │   │   └── ...
    │   │   ├── project2
    │   │   └── image_sets
    │   └── active_drive/
    │       └── vlad/
    │           └── git_repos/
    │               └── project1_code/
    │                   ├── analyses
    │                   ├── tasks
    │                   ├── figures
    │                   └── results
    └── olsonlab/
        ├── data_drive
        └── active_drive

### Programs already installed on the workstation

Coding: anaconda, Matlab_R2025a, R 

MRI: FSL, Freesurfer 8.0.0, AFNI, ANTS, wb_workbench

*To install new software or different versions of existing software email Vlad
*Any python-supported packages should be installed in your own conda environment


