# Readme
#### Folders, codes and readme made by Samuel Mesquita 2025

In this readme I will explain what are the different important files and folders for future usage.  
I have made all of those during my 2025 6-month internship.  

# Validation of structure formation

Here we aim to validate structure formation in Dyablo.  
Check folder **validation**.

## The most important codes here are : 
### my_hmf_cmf.ipynb
Notebook that aimed at making sure I was able to confidently code the HMF (Halo Mass Function) and CMF (Conditional Mass Function).
### my_hmf_cmf_lib.py 
Python library that contains all the functions that I have coded in the aforementioned notebook and more. Those functions can be use to easily compute values like the fraction of collapsed in a region, or plot the theoretical vs empirical CMF...
### hmf_cmf_dyablo_snap.ipynb
Notebook that allows the study of a simulation snapshot from Dyablo.
### tri_z_R0_delta_fit.py
Script running PySR (Cranmer, 2023) to try and find a fit to the fraction of collapsed mass in 3D (redshift, radius of the region, density of the region). Its output is fits.txt (python3 tri_z_R0_delta_fit.py > fits.txt).   
The 2D fit I found by setting R0 to 1 Mpc seemed correct. I have a harder time finding a 3D fit that works correctly.

NB : Here stylesheet.mplstyle is simply my way to use matplotlib.

## The folders here are :

### [datastageM2](validation/datastageM2)
Symbolic link to the data of one simulation that I've studied. It contains snapshots of the simulation at different and also some metadata.

### [outputs](validation/outputs)
Folder automatically created by PySR when trying to find a fit. Not very important.

### [previous_codes](validation/previous_codes)
All the scripts and notebooks I have made during my internship. Most of them are not useful and the rest were created for me to able to easily make plots for my internship report/oral.

### [saved_results](validation/saved_results)
Here you may find images of my results or data that comes from some computation :  
[saved_results](validation/saved_results/)  
├── [data](validation/saved_results/data)  
│   ├── [R_vary](validation/saved_results/data/R_vary) -  csv files for the computation of the fraction of collapsed mass for different value of the radius of the subregions.  
│   └── [Particle data](validation/saved_results/data/ParticleData) -  Save of the data I used to my studies (in case the symbolic link breaks).  
├── [misc](validation/saved_results/misc) - Miscellaneous images that don't belong in z_vary_results.  
└── [z_vary_results](validation/saved_results/z_vary_results) - Images (svg+pdf) for the results of my study of the simulation. There is : the Halo Mass function, the Conditional Mass Function depending on the environment ; the comparison between the empirical et theoretical fraction of collapsed mass.

# Improving Dyablo's ionization module

Here we aim to improve Dyablo's ionization module.  
Check folder **development**.

## SourceUpdate_Ionization_Chem.cpp

Dyablo file that I've modified with my fit. I've mostly modified lines 71 to 103. Also at the beginning of the function with auxialiary values like redshift, fstar, dnl...

## ProbeSnape.ipynb

Allows the study of a simulation, notably the SFR and the evolution of the average ionization fraction and density of ionized gas in the simulation.

## cosmo_units.ini

The initial conditions I have used to run my Dyablo simulations. 

The important variables here are :

```cpp
ndot                 = 1.5e48
rho_crit             = 16e-28
write_variables      = xe,rho,zre
```
ndot is the stellar emissivity which is the number of photons emitted by second by solar mass.

rho_crit was used before but serve no purpose anymore since I have added my code.

write_variables, zre was previously used to store the value of the redshift for the end of reionization. Now the purpose of zre is to **store de stellar mass density in each cell**.

We also made it so the size of each cell is 1 Mpc to work well with the fit.
