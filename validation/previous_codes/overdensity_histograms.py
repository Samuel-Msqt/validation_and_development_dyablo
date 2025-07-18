from tqdm import tqdm
import numpy as np
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys, platform, os
from CosmoUtils import *
import my_hmf_cmf_lib as mycmf

### CONSTANTS
Ncut = 32

L = mycmf.L
h = mycmf.h

grid_size = [L/h, L/h, L/h]

### OPENING FILE

# nb_iter, zsnap = ["0000000",26.1272]
# nb_iter, zsnap = ["0000250",2.47551]
nb_iter, zsnap = ["0002850",0.0]

filename = f"cosmo_particles_particles_iter{nb_iter}.h5"

fpart = h5py.File(f'datastageM2/{filename}', 'r')

positions = np.array(fpart['coordinates'])
print(np.shape(positions))

x=positions[:,0]
y=positions[:,1]
z=positions[:,2]
Npart=np.size(x)

x_mpc = mycmf.norm2dist(x, grid_size)
y_mpc = mycmf.norm2dist(y, grid_size)
z_mpc = mycmf.norm2dist(z, grid_size)
part_pos_mpc = np.column_stack((x_mpc, y_mpc, z_mpc))

### SETTING UP

idx_halos_paved = []
density_paved = []
delta_NL_paved = []
delta_L_paved = []
cmf_emp = []
err_emp = []
bcen_emp = []
cmf_th = []
bins_emp = []
nb_halos = []


paved_centers, paved_radius = mycmf.paving_domain(Ncut, grid_size)
nb_sphere = len(paved_centers)

print(f"Computing values for {nb_sphere} spheres of radius {paved_radius:.2f} Mpc...")
current_nb_part_tot=0
for current_center in tqdm(paved_centers):

    current_density = mycmf.density_in_shape(current_center, paved_radius, grid_size, part_pos_mpc, mycmf.compute_mpart(mycmf.rho_0, Npart, L ), shape="cube")
    # densite_moy = current_nb_part / (1./Ncut)**3
    # delta_NL = (densite_moy - (Npart/1**3))/ (Npart/1**3)
    delta_NL = (current_density - mycmf.rho_0_Msun_Mpc3)  / mycmf.rho_0_Msun_Mpc3 #non-linear
    # delta_L = mycmf.compute_delta_linear(delta_NL) #delta_L = delta_0
    
    # current_halos = idx_halos(current_center, paved_radius, grid_size, halo_pos_mpc, method="Manual", shape="cube")
    # mh_in_sph = mh[current_halos]
    # halos_found = len(current_halos)


    
    density_paved.append(current_density)
    delta_NL_paved.append(delta_NL)
    # delta_L_paved.append(delta_L)
    # idx_halos_paved.append(current_halos)
    # nb_halos.append(halos_found)
    
    # if halos_found > 0 :
    #     lightest_halo = np.min(mh_in_sph)
    #     heaviest_halo = np.max(mh_in_sph)
        
    #     current_bins=np.logspace(np.log10(lightest_halo), np.log10(heaviest_halo),num=int(np.sqrt(halos_found)))
    #     current_bcen=0.5*(current_bins[1:]+current_bins[:-1])  
    #     current_db=current_bins[1:]-current_bins[:-1]
    #     current_cmf,current_bin_edges = np.histogram(mh_in_sph*h,bins=current_bins) 
        
    #     Vsphere = 4/3*np.pi*(paved_radius*h)**3
    #     current_dndm=current_cmf/current_db/Vsphere
        
    #     current_dndm_upper_err=current_dndm+3*np.sqrt(current_cmf)/current_db/Vsphere #3 sigma dispersion
    #     current_dndm_lower_err=current_dndm-3*np.sqrt(current_cmf)/current_db/Vsphere #3 sigma dispersion
    #     dndm_err = [current_dndm_upper_err, current_dndm_lower_err]
        
    #     ### MY CMF COMPUTATION
    #     cST_list = [mycmf.cmfcalc(M_list[i], kh, pk, paved_radius/h, delta_L, "cST") for i in range(len(M_list))]
        
    #     bins_emp.append(current_bins)
    #     bcen_emp.append(current_bcen)
    #     cmf_emp.append(current_dndm)
    #     cmf_th.append(cST_list)
    #     err_emp.append(dndm_err)
    # else : 
    #     bins_emp.append(None)
    #     bcen_emp.append(None)
    #     cmf_emp.append(None)
    #     cmf_th.append(None)
    #     err_emp.append(None)

save_dir = "./saved_results/overdensity_histograms"
os.makedirs(save_dir, exist_ok=True)

# Save the results as CSV
csv_path = os.path.join(save_dir, f"results_Ncut_is_{Ncut}.csv")
np.savetxt(csv_path, np.column_stack((density_paved, delta_NL_paved)), delimiter=",", header="density_paved,delta_NL_paved")

print(f"Results saved to {csv_path}")