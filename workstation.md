# Welcome to ARTOO

ARTOO is the vlad and olson lab shared work station. 

It contains a 96 core AMD Threadripper CPU, 2 NVIDIA RTX 6000 Ada GPUs, and 512 GB of memory

contact info: vayzenb@temple.edu

## Before you connect

1. Reach out to Vlad (vayzenb@temple.edu) to request access. Include your TU Username (e.g., tuc66980)
2. If you plan to connect from home, request VPN access via TUHelp
3. If you are connecting from a Windows computer, install [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)

## Connecting to the Artoo

### Using a terminal:
```bash
ssh your_tuid@cla19779.tu.temple.edu
# for example, ssh tuc66980@cla19779.tu.temple.edu
```

### Using VScode
1. [Download VSCode ](https://code.visualstudio.com/)
2. Select extensions on the left-hand side
3. Install Remote- SSH and Remote Viewer (basically all the Microsoft remote apps)
4. Open the command pallete (ctrl/cmd + shift + P)
5. Start typing Remote-SSH connect to host
6. Paste your connection info: your_tuid@cla19779.tu.temple.edu

### Using chrome remote desktop
Contact Vlad or George about setting this up

### Organization of the workstation

Artoo is split into **/zpool/vladlab** and **/zpool/olsonlab** directories

Within each lab directory is a **data_drive** and a **active_drive** folder. 

The **data_drive** is intended to be primarily a read-only drive that stores large data files that are frequently read, but not modified, such as MRI data.
The **active_drive** is intended to support data and tasks that are frequently updated, such as analysis scripts. 

This split into data and active drives is designed optimize back up schedules for different data types so that the backups don't take up too much space. 

Each drive is backed up on a regular basis using different schedules. Because **data_drive** files should change less frequently, backups are made less regularly but are stored over longer periods (weekly/monthly/yearly).  By contrast, because **active_drive** files may change many times a day, these files are backed up more regularly, but stored for shorter periods of time (hourly/daily/weekly). 

See below for an example file structure for a hypothetical fMRI study 'project1'. The raw and preprocessed fMRI data would be stored in data_drive, ideally in [BIDS format](https://bids.neuroimaging.io/index.html), and the code for analyzing the data would be stored in personal folder in active_drive, ideally in a folder that syncs to [github](https://docs.github.com/en/get-started/git-basics).

![image](files/directory_structure.png)

Note, these are just recommendedations given typical backup needs. There are no hard constraint on which files can go where.

## Programs already installed on the workstation

Coding: anaconda, Matlab_R2025a, R 

MRI: FSL, Freesurfer 8.0.0 & 7.4.1*, AFNI, ANTS, wb_workbench

*By default, version 8.0.0 of freesurfer is the one loaded at startup for each user. Talk to vlad if you need a different version or need to be able to switch between versions

### Managing different packages and coding libraries

Different tasks may require different code libraries or packages. Each might have their own dependancies and installing too many of these in the same place can cause them to interfere with eachother. 

To address this, the workstation has [anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/cheatsheet.html) installed system wide. Anaconda allows users to create seperate coding environments for differnt where packages or libraries can be installed. For example, you may want one environment for fmri analysis packages (e.g., fmriprep, nilearn), machine learning (e.g., pytorch, scikit-learn), or to try out a some risky looking code base that might otherwise break everything.

**You will not be able to install things into the base environment**

## General Recommendations

Take a look at the [bash tutorail file](tutorials/bash_startup.md) for some commonly used bash commands

The best way to work on ARTOO is via the terminal or VSCode. VSCode supports both data and figure visualization via Jupyter notebooks and you can even visualize neural data using nilearn. 

If you need to browse through files or need a GUI for something like FSL, freesurfer etc. use chrome remote desktop.

I **highly** recommend you organize your code into git hub repositories, or repos. For instructions on getting your github credentials set up on ARTOO, [see here](tutorials/git_setup.md).









