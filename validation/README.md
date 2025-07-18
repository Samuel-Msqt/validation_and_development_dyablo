# Readme
#### Folders, codes and readme made by Samuel Mesquita 2025

In this readme I will explain what are the different important files and folders for future usage.  
I have made all of those during my 2025 6-month internship.  
Here we aim to validate structure formation in Dyablo.

## The most important codes here are : 
### my_hmf_cmf.ipynb
Notebook that aimed at making sure I was able to confidently code the HMF (Halo Mass Function) and CMF (Conditional Mass Function).
### my_hmf_cmf_lib.py 
Python library that contains all the functions that I have coded in the aforementioned notebook and more. Those functions can be use to easily compute values like the fraction of collapsed in a region, or plot the theoretical vs empirical CMF...
### hmf_cmf_dyablo_snap.ipynb
Notebook that allows the study of a simulation snapshot from Dyablo.
### tri_z_R0_delta_fit.py
Script running PySR (Cranmer, 2023) to try and find a fit to the fraction of collapsed mass in 3D (redshift, radius of the region, density of the region). Its output is fits.txt.   
The 2D fit I found by setting R0 to 1 Mpc seemed correct. I have a harder time finding a 3D fit that works correctly.

NB : Here stylesheet.mplstyle is simply my way to use matplotlib.

## The folders here are :
### [datastageM2](datastageM2)
Symbolic link to the data of one simulation that I've studied. It contains snapshots of the simulation at different and also some metadata.

### [outputs](outputs)
Folder automatically created by PySR when trying to find a fit. Not very important.

### [previous_codes](previous_codes)
All the scripts and notebooks I have made during my internship. Most of them are not useful and the rest were created for me to able to easily make plots for my internship report/oral.

### [saved_results](saved_results)
Here you may find images of my results or data that comes from some computation :  
[saved_results](saved_results/)  
├── [data](saved_results/data)  
│   ├── [R_vary](saved_results/data/R_vary) -  csv files for the computation of the fraction of collapsed mass for different value of the radius of the subregions.  
│   └── [Particle data](saved_results/data/ParticleData) -  Save of the data I used to my studies (in case the symbolic link breaks).  
├── [misc](saved_results/misc) - Miscellaneous images that don't belong in z_vary_results.  
└── [z_vary_results](saved_results/z_vary_results) - Images (svg+pdf) for the results of my study of the simulation. There is : the Halo Mass function, the Conditional Mass Function depending on the environment ; the comparison between the empirical et theoretical fraction of collapsed mass.