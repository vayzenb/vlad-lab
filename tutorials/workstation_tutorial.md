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
conda create -n fmri
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

### Let's test that the environment works
```bash
#check that you have access to docker
docker --version
fmriprep-docker --help

# Check that python is working and packages were installed
python
# Import the packages we installed
import nilearn
print(nilearn.__version__)
```


## Connect to the workstation throuhgh VSCode
1. Open VSCode
2. Select extensions on the left-hand side
3. Install Remote- SSH and Remote Viewer (basically all the Microsoft remote apps)
4. Open the command palette (ctrl/cmd + shift + P)
5. Start typing Remote-SSH connect to host
6. Paste your connection info: username@cla19779.tu.temple.edu
7. Open the folder you created in your active drive: /zpool/vladlab/active_drive/your_username
8. Open an integrated terminal in VSCode (View > Terminal)

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

1. Check folder browser
2. Test some GUIs ine at a time

```bash
fsl
fsleyes
freeview
afni
suma
```
3. Let's install mrtix into your fmri conda environment
```bash
conda activate fmri
conda install -c conda-forge -c MRtrix3 mrtrix3 libstdcxx-ng
```
4. Test mrtrix
```bash
mrview
```



