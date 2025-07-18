import my_hmf_cmf_lib as mycmf
import numpy as np
import h5py
import yt
import numpy as np
import matplotlib.pyplot as plt
import sys, platform, os
from CosmoUtils import *
from tqdm import tqdm
import matplotlib.cm as cm
import pandas as pd

plt.style.use('stylesheet.mplstyle')

Ncut = 8

Ncut = [6,10,14]
Ncut = np.array(Ncut)

#Delta_NL range
indiv_range = 0.5
lower_bound = -1
upper_bound = 1.6
dNL_range = np.arange(lower_bound, upper_bound, indiv_range)
count_cut = 128
method_sub_cmf = "range"
if method_sub_cmf == "range":
    arg_sub_cmf = dNL_range
elif method_sub_cmf == "number":
    arg_sub_cmf =  count_cut


corr_iter_z = [("0000000", 26.1272),
 ("0000250", 2.47551),
 ("0000500", 1.65828),
 ("0000750", 1.30188),
 ("0001000", 1.04726),
 ("0001250", 0.841156),
 ("0001500", 0.660872),
 ("0001750", 0.509201),
 ("0002000", 0.360715),
 ("0002250", 0.24084),
 ("0002500", 0.140594),
 ("0002750", 0.0389838),
 ("0002850", 0.0)]

corr_iter_z = [("0002850", 0.0)]



## CONSTANTS
L=164 #Mpc/h
H0org=67.00 # km/s/Mpc
om=0.3175 # total density parameter
H0=H0org*1e3/3.086e22
h=H0org/100.
rhoc=3*H0**2/(8*np.pi*6.67e-11) #kg/m^3
rho0 = rhoc * om
rho_MsunMpc3 = rho0 / 2e30 * 3.086e22**3 #Msun/Mpc^3
grid_size = [L/h, L/h, L/h]

radius_list = []
fcoll_emp_list = []
fcoll_th_list = []

print(Ncut)
print(L/h/Ncut/2)

for nb_iter, zsnap in tqdm(corr_iter_z):
    for Ncut in tqdm(Ncut):
        age = mycmf.compute_age_Gyr(zsnap) #Gyr
        print(f"\n\n\n   ----- Ncut={Ncut} - R={L/h/Ncut/2:.1f} Mpc - z={zsnap:.2f} - age={age:.2f} Gyr -----\n\n\n")

        
        filename = f"cosmo_particles_particles_iter{nb_iter}.h5"

        ## FINDING Npart
        fpart = h5py.File(f'datastageM2/{filename}', 'r')

        positions = np.array(fpart['coordinates'])

        x=positions[:,0]
        y=positions[:,1]
        z=positions[:,2]
        Npart=np.size(x)
        mpart=rho0*(L/h*3.086e22)**3/Npart #kg #either rhoc * om or rho0

        ##LOADING HALOS
        masses=np.ones(Npart) # dummy particle masses array as a set of unit masses
        idx=np.arange(Npart) # monotonic indexes for particles

        data = dict(
            particle_position_x=x,
            particle_position_y=y,
            particle_position_z=z,
            particle_velocity_x=x,#dummy with no vel
            particle_velocity_y=y,#dummy with no vel
            particle_velocity_z=z,#dummy with no vel
            particle_mass=masses,
            particle_index=idx
        )
        ds = yt.load_particles(data,length_unit=L*3.086e24,periodicity=(True,True,True),mass_unit=mpart*1e3)
        
        fhalo = h5py.File(f'saved_results/ParticleData/ParticleData{nb_iter}.h5', 'r') # upload Hop results
        
        xh=np.array(fhalo['particle_position_x']) #HOP halo positions
        yh=np.array(fhalo['particle_position_y']) #HOP halo positions
        zh=np.array(fhalo['particle_position_z']) #HOP halo positions
        mh=np.array(fhalo['particle_mass']) #HOP halo mass
        
        if len(mh)==0:
            print(f"\n\n\n   ----- No halo found for {filename} -----\n\n\n")
            continue
        
        x_mpc = mycmf.norm2dist(x, grid_size)
        y_mpc = mycmf.norm2dist(y, grid_size)
        z_mpc = mycmf.norm2dist(z, grid_size)
        part_pos_mpc = np.column_stack((x_mpc, y_mpc, z_mpc))

        xh_mpc = mycmf.norm2dist(xh, grid_size)
        yh_mpc = mycmf.norm2dist(yh, grid_size)
        zh_mpc = mycmf.norm2dist(zh, grid_size)
        halo_pos_mpc = np.column_stack((xh_mpc, yh_mpc, zh_mpc))
        
        M_list = np.logspace(11,16,200) #Msun/h
        
        ######################## HMF #############################
        ## Quick kh and pk compuattion
        pars,results,s8_fid = mycmf.compute_init_power_spectrum(zsnap, H0org, mycmf.TCMB, om, mycmf.ob, h, mycmf.ns, mycmf.As)
        kh, _, pk = mycmf.compute_matter_power_spectrum(zsnap, pars, s8_fid)
        
        n_ST_list = [mycmf.hmfcalc(M_list[i], kh, pk, "ST") for i in range(len(M_list))]
        plt.figure()
        plt.plot(M_list, n_ST_list)
        plt.yscale('log')
        plt.xscale('log')

        ###### DOING A HMF TO FIND Mmin 
        bins_hmf=np.logspace(11,16,num=128)
        bcen_hmf=0.5*(bins_hmf[1:]+bins_hmf[:-1])
        db_hmf=bins_hmf[1:]-bins_hmf[:-1]
        # Halo mass function using Hop mass estimate, note that masses must be given in Msol/h
        # Note : hop mass is slightly underestimated compared to proper M200 calculation
        myhmf,bmf=np.histogram(mh*h,bins=bins_hmf) #<<< msol/h

        mydndm_hmf=myhmf/db_hmf/L**3
        mydndm1_hmf=mydndm_hmf+3*np.sqrt(myhmf)/db_hmf/L**3 #3 sigma dispersion
        mydndm2_hmf=mydndm_hmf-3*np.sqrt(myhmf)/db_hmf/L**3 #3 sigma dispersion

        hmf_over_bins = [mycmf.hmfcalc(bcen_hmf[i], kh, pk, "ST") for i in range(len(bcen_hmf))]
        Mmin = mycmf.Mmin_finder(bcen_hmf, mydndm1_hmf, mydndm2_hmf, hmf_over_bins) #Msun/h
        
        plt.errorbar(bcen_hmf, mydndm_hmf,
            yerr = [mydndm_hmf - mydndm2_hmf,
            mydndm1_hmf - mydndm_hmf],
            fmt='o-', markersize=5, alpha=0.7)
        
        plt.xlabel(r"Mass $[h^{-1}M_\odot]$")
        plt.ylabel(r"dn/dM $[h^{4}{\rm Mpc}^{-3}M_\odot^{-1}]$");
        plt.xlim([2e12,5e14])
        plt.ylim([1e-22,1e-15])
        plt.legend()
        plt.title(f"HMF - z={zsnap:.2f} ($\simeq$ {age:.2f} Gyr)")
        plt.savefig(f"./saved_results/z_vary_results/hmf_{nb_iter}.pdf", bbox_inches='tight')
        
        print(f"\n   ----- HMF plot saved as hmf_{nb_iter} -----")
        print("   ----- Doing the CMF computation now... -----\n")
        
        shape="sphere"
        nbins = 50

        subreg_list = mycmf.analyze_subregions(Ncut, shape ,grid_size, halo_pos_mpc, part_pos_mpc, mpart, rho_MsunMpc3, mh, kh, pk, M_list, model = "QcST", nbins=nbins)
        
        radius_sr = subreg_list[0].radius #Mpc
        radius_list.append(radius_sr)
        delta_NL_paved = [subregion.delta_NL for subregion in subreg_list]
        idx_sort_delta_NL = np.argsort(delta_NL_paved)
        subreg_sorted = subreg_list[idx_sort_delta_NL]
        delta_NL_paved = np.array([subregion.delta_NL for subregion in subreg_sorted])

        print("   ----- Doing the fcoll computation now... -----\n")
        
        ######################### FOR FCOLL ###########################
        
        ### COMPUTING FCOLL EMP
        fcoll_emp, delta_NL_filter = mycmf.compute_fcoll_emp(subreg_sorted, Mmin, mh, mpart)
        fcoll_emp_list.append(fcoll_emp)
        
        ### COMPUTING FCOLL TH
        fcoll_th, delta_NL_filter = mycmf.compute_fcoll_th(subreg_sorted, Mmin, kh, pk, mh, mpart)
        fcoll_th_list.append(fcoll_th)

        ### save
        fcoll_df = pd.DataFrame({
            'fcoll_emp': fcoll_emp,
            'fcoll_th': fcoll_th,
            'delta_NL_filter': delta_NL_filter,
            'zsnap': [zsnap] * len(fcoll_emp),
            'Ncut': [Ncut] * len(fcoll_emp),
            'R': [subreg_sorted[0].radius] * len(fcoll_emp),
            'Mmin': [Mmin] * len(fcoll_emp),
        })
        fcoll_df.to_csv(f'./saved_results/data/R_vary/fcoll_results_N{Ncut}_{nb_iter}.csv', index=False)


### PLOTTING RESULTS
plt.figure()

plt.scatter(radius_list, fcoll_emp_list, s=10, alpha=0.7, label=r'$f_{coll}^{emp}$')
plt.scatter(radius_list, fcoll_th_list, s=10, alpha=0.7, label=r'$f_{coll}^{th}$')

plt.title(f"f_coll - z={zsnap:.2f} ($\simeq$ {age:.2f} Gyr)")
plt.xlabel(r"Radius of the subregion $[Mpc]$")
plt.ylabel(r"$f_{coll}$")
plt.legend()

plt.savefig(f"./saved_results/R_vary/fcoll_{nb_iter}.pdf", bbox_inches='tight')