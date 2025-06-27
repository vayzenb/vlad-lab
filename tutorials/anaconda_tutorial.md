# Anaconda Tutorial for Bash Beginners

A beginner-friendly guide to using Anaconda for Python package and environment management.

## What is Anaconda?

Anaconda is a distribution of Python that comes with:
- **Python interpreter** - The core Python language
- **Conda package manager** - Tool to install and manage software packages
- **Pre-installed packages** - Common data science libraries (NumPy, Pandas, etc.)
- **Environment management** - Keep different projects separate

Think of it as a complete Python toolkit that makes managing different projects easy.


## Understanding Environments

### What are Environments?

Think of environments as separate workspaces for different projects:
- **Project A** might need Python 3.8 and specific package versions
- **Project B** might need Python 3.11 and different packages
- Environments keep them separate so they don't conflict

### The Base Environment

When you install Anaconda, you get a "base" environment:

```bash
# Check which environment you're in
conda info --envs

# You'll see something like:
# base                  *  /home/username/anaconda3
# The * shows your current environment
```

## Basic Conda Commands


## Working with Environments

### Creating New Environments

```bash
# Create environment with default Python version
conda create --name myproject

# Create environment with specific Python version
conda create --name myproject python=3.9

# Create environment with packages
conda create --name dataproject python=3.9 pandas numpy matplotlib


```

### Activating and Deactivating Environments

```bash
# Activate an environment
conda activate myproject

# Your prompt will change to show the environment:
# (myproject) username@computer:~$

# Deactivate current environment (go back to base)
conda deactivate

# Your prompt returns to normal:
# username@computer:~$
```

### Installing Packages

```bash
# Install a single package
conda install numpy

# Install multiple packages
conda install pandas matplotlib seaborn

# Install specific version
conda install python=3.9

# Install from specific channel (repository)
conda install -c conda-forge package_name
```

### Getting Information

```bash
# Show conda version
conda --version

# Show all environments
conda info --envs
conda env list                    # Same thing, shorter command

# Show packages in current environment
conda list

# Show information about a specific package
conda info numpy
```


## Practical Examples

### Example 1: Setting Up a fMRI Project

```bash
# Create a new environment for data science
conda create --name fmri python=3.9

# Activate the environment
conda activate data_analysis

# Install data science packages
conda install pandas numpy matplotlib seaborn jupyter

#Install fMRI specific packages
pip install fmriprep nilearn

# Verify packages are installed
conda list

# Start working on your project
jupyter notebook

# When done, deactivate
conda deactivate
```
