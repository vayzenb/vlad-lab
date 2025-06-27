# Getting Started With the workstation


### Login with your username and password through the terminal
```bash
ssh username@cla19779.tu.temple.edu


#after vlad adds you to the groups do
exec bash

# open chrome remote desktop and follow instructions to add your account
```

### Check that you have access to your lab's data and active drives
If you are in the vlad lab you should have access to /zpool/vladlab/data_drive and /zpool/vladlab/active_drive

If you are in the olson lab you should have access to /zpool/olsonlab/data_drive and /zpool/olsonlab/active_drive

```bash
#try making a personal folder in your active drive
cd /zpool/vladlab/active_drive
mkdir your_username #e.g., mkdir vayzenb
```

## Let's create a conda environment for fMRI analysis


```bash
# Create a new conda environment named 'fmri'
# this instantiates a new conda environment with the name 'fmri' and installs Python
conda create --name fmri python 
conda activate fmri

# Check that the correct Python version is being used
which python
# It should point to something like /home/your_username/anaconda3/envs/fmri/bin/python

#let's install some common packages
conda install numpy 
# let's install a few more
conda install pandas seaborn 

# install fMRI specific packages
pip install nilearn fmriprep-docker
```

[See here, for a more detailed guide on how to use anaconda](anaconda_tutorial.md)

### Let's test that the environment works
```bash
#check that you have access to docker
docker --version
fmriprep-docker --help
#hit yes to the prompt about installing the docker image

# Check that python is working and packages were installed
python
# Import the packages we installed
import nilearn
print(nilearn.__version__)
```

## Connect to the workstation throuhgh VSCode
1. Open VSCode
2. Select extensions on the left-hand side and install the following extensions:
   - Remote-SSH
   - Jupyter
   - Python
3. Open the command palette (ctrl/cmd + shift + P)
4. Start typing Remote-SSH connect to host
5. Paste your connection info: username@cla19779.tu.temple.edu
6. Open the folder you created in your active drive: /zpool/vladlab/active_drive/your_username
7. Open an integrated terminal in VSCode (View > Terminal)

### Let's create a jupyter notebook to look at some data
1. Open the command palette (ctrl/cmd + shift + P)
2. Type "Jupyter: Create New Notebook"

```python 
# Import the packages we installed
import pandas as pd
import seaborn as sns
from nilearn import plotting, image


# load sample data
df = pd.read_csv('/zpool/olsonlab/data_drive/sample_data/penguins.csv')
# Look at the data under jupyer

# Create a simple scatter plot
sns.scatterplot(data=df, x='flipper_length_mm', y='body_mass_g', hue='species')

# load a sample fMRI image
zstat = image.load_img('/zpool/olsonlab/data_drive/sample_data/sample_data.nii.gz')

# Plot the fMRI image
plotting.plot_stat_map(zstat, title='Sample fMRI Image', threshold=3.0)

# View iamge interactively
plotting.view_img(zstat, title='Sample fMRI Image', threshold=3.0)

```

## Connect to the workstation through Chrome Remote Desktop

1. Open Chrome Remote Desktop
2. Select the computer you want to connect to. By default it is called CLA19779.tu.temple.edu
3. Enter the PIN you set up when you added your account
4. Select Ubuntu as your desktop environment

### Setting up your desktop environment on Remote Desktop

First lets set up a shortcut to your lab's drive
1. Open the folder browser on the left-hand side
2. Select + Other Locations
3. Navigate to Computer > zpool 
4. Drag the vladlab or olsonlab folder to the left hand side to bookmark it

Let's also set up some shortcuts to commonly used apps
1. Open the apps menu in the bottom left corner (9 dots)
2. Search for terminal and drag it to the left-hand side to pin it
3. Search for VSCode and drag it to the left-hand side to pin it
4. Search for surfer and drag it to the left-hand side to pin it

Next, let's test that we can run some common neuroimaging software
Open the terminal and test the following commands:

```bash
fsl
fsleyes
freeview
suma
```
Let's install mrtix into your fmri conda environment
```bash
conda activate fmri
conda install -c conda-forge -c MRtrix3 mrtrix3 libstdcxx-ng
```
Test mrtrix
```bash
mrview
```



