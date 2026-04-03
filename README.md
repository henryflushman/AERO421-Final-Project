# AERO 421 Final Project

## Team Members
- Henry Flushman
- Jackson William Mehiel
- Nick Schaeffer

## Project Description
AERO 421 Final Project, using the Python based `control-systems` library

## Installation Guide
The setup process of this repository should be fairly straight-forward if you already have conda installed, otherwise it might be slightly difficult. For this reason I will only describe the setup process for users with conda already installed. The process for installing Anaconda will be in another file titled `ANACONDAINSTALL.md`

### Copying the Repo to Your Computer
1. Open Anaconda Prompt
2. Change directory to the desired location of the repository using:
```
cd {repository location}
```
3. Input these commands into Anaconda Prompt:
```
git clone https://github.com/henryflushman/AERO421-Final-Project.git
cd AERO421-Final-Project
```
### Creating a new conda environment
Now that the repository is properly placed in your local device, you'll need to create a new conda environment to house the required dependencies for this repo.
1. Open Anaconda Prompt once again
2. Input this command into Anaconda Prompt:
```
conda create --name AERO421_venv python=3.11
conda activate AERO421_venv
```
3. At this point, you should see in the terminal:
```
(AERO421_venv) C:\{Your directory}>
```
If you do not see this, you have either not properly created the conda environment, or the environment has not been properly activated
4. When you've activated your conda environment, you can now begin downloading dependencies using this command:
```
conda install -c conda-forge control slycot
```
5. Verify that this installation has processed, run this command:
```
python -c "import slycot"
```

## Files
- .vscode/settings.json
    - *Configures the repository to automatically activate your conda environment*
- README.md
    - *This file, it offers a brief description as well as installation guide on how to install the proper dependencies.*