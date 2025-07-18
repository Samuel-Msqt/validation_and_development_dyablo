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
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.style.use('stylesheet.mplstyle')

Ncut = 8

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


corr_iter_z = [
    ("0002850", 0.0),
    ("0002750", 0.0389838),
    ("0002500", 0.140594),
    ("0002250", 0.24084),
    ("0002000", 0.360715),
    ("0001750", 0.509201),
    ("0001500", 0.660872),
    ("0001250", 0.841156),
    ("0001000", 1.04726),
    ("0000750", 1.30188),
    ("0000500", 1.65828),
    ("0000250", 2.47551),
    ("0000000", 26.1272)
]


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

for nb_iter, zsnap in tqdm(corr_iter_z):
    age = mycmf.compute_age_Gyr(zsnap) #Gyr
    print(f"\n\n\n   ----- z={zsnap:.2f} - age={age:.2f} Gyr -----\n\n\n")
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
    plt.plot(M_list, n_ST_list, lw=2, label="Theoretical (ST)")
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
    
    # plt.errorbar(bcen_hmf, mydndm_hmf,
    #     yerr = [mydndm_hmf - mydndm2_hmf,
    #     mydndm1_hmf - mydndm_hmf],
    #     fmt='o-', markersize=5, alpha=0.5, color="black")
    
    plt.scatter(bcen_hmf, mydndm_hmf, s=10, color="black", alpha=0.5, label="Empirical")
    plt.fill_between(bcen_hmf,mydndm1_hmf,mydndm2_hmf,color='black',alpha=0.3, label="3$\sigma$ dispersion")
    
    plt.xlabel(r"Mass $[h^{-1}M_\odot]$")
    plt.ylabel(r"dn/dM $[h^{4}{\rm Mpc}^{-3}M_\odot^{-1}]$");
    plt.xlim([1.5e12,5e14])
    plt.ylim([1e-22,1e-15])
    plt.legend()
    # plt.title(f"z={zsnap:.2f} ($\simeq$ {age:.2f} Gyr)")
    plt.savefig(f"./saved_results/z_vary_results/hmf_{nb_iter}.svg", bbox_inches='tight')
    
    print(f"\n   ----- HMF plot saved as hmf_{nb_iter} -----")
    print("   ----- Doing the CMF computation now... -----\n")
    
    shape="sphere"
    nbins = 50

    subreg_list = mycmf.analyze_subregions(Ncut, shape ,grid_size, halo_pos_mpc, part_pos_mpc, mpart, rho_MsunMpc3, mh, kh, pk, M_list, model = "QcST", nbins=nbins)
    
    delta_NL_paved = [subregion.delta_NL for subregion in subreg_list]
    idx_sort_delta_NL = np.argsort(delta_NL_paved)
    subreg_sorted = subreg_list[idx_sort_delta_NL]
    delta_NL_paved = np.array([subregion.delta_NL for subregion in subreg_sorted])
    
    ######################## CMF #############################
    savename = f"./saved_results/z_vary_results/subcmf_{nb_iter}.svg"
    mycmf.plot_every_sub_cmf2(subreg_sorted, arg_sub_cmf, delta_NL_paved, Ncut, mh, nbins, M_list, method=method_sub_cmf, showfig=False, z=zsnap, savename=savename)
    
    print(f"\n   ----- CMF plot saved as {savename} -----")
    print("   ----- Doing the fcoll computation now... -----\n")
    
    ######################## FOR FCOLL ###########################
    
    ### COMPUTING FCOLL EMP
    fcoll_emp, delta_NL_filter = mycmf.compute_fcoll_emp(subreg_sorted, Mmin, mh, mpart)

    ### COMPUTING FCOLL TH
    fcoll_th, delta_NL_filter = mycmf.compute_fcoll_th(subreg_sorted, Mmin, kh, pk, mh, mpart)

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
    fcoll_df.to_csv(f'./saved_results/z_vary_results/fcoll_results_R{subreg_sorted[0].radius:.0f}_{nb_iter}.csv', index=False)
    
    ###
    max_fcoll = np.max([np.max(fcoll_emp), np.max(fcoll_th)])
    
    ## PLOTTING RESULTS
    # plt.figure()
    
    # cmap = cm.viridis

    # norm = plt.Normalize(vmin=np.round(delta_NL_filter[0],1), vmax=np.round(delta_NL_filter[-1],1))
    # color = cmap(norm(delta_NL_filter))

    # plt.plot(np.linspace(0,max_fcoll+0.02,100), np.linspace(0,max_fcoll+0.02,100), '--', color="black", alpha=0.5, label="x=y")

    # for i in range(len(fcoll_emp)):
    #     plt.scatter(fcoll_th[i], fcoll_emp[i], color=color[i], alpha=1)

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm)
    # cbar.set_label(f'$\delta_{{NL}}$')

    # plt.xlabel(r"Theoretical $f_{coll}^{th}$")
    # plt.ylabel(r"Empirical $f_{coll}^{emp}$")
    # plt.title(f"z={zsnap:.2f} ($\simeq$ {age:.2f} Gyr) - $M_{{min}}$={Mmin:.2e} $h^{{-1}}M_\odot$")
    # plt.legend()
    # plt.savefig(f"./saved_results/z_vary_results/fcoll_emp_vs_th_{nb_iter}.svg", bbox_inches='tight')
    
    
    # print(f"\n   ----- fcoll plot saved as fcoll_emp_vs_th_{nb_iter} -----\n\n")
    
    
    fig, ax = plt.subplots()

    cmap = cm.viridis
    norm = plt.Normalize(vmin=np.round(delta_NL_filter[0], 1), vmax=np.round(delta_NL_filter[-1], 1))
    color = cmap(norm(delta_NL_filter))

    ax.plot(np.linspace(0, max_fcoll + 0.02, 100), np.linspace(0, max_fcoll + 0.02, 100), '--', color="black", alpha=0.5, label="x=y")

    for i in range(len(fcoll_emp)):
        ax.scatter(fcoll_th[i], fcoll_emp[i], color=color[i], alpha=1)

    # Create colorbar right next to the main axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02)  # pad controls spacing
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r'$\delta_{NL}$')

    ax.set_xlabel(r"Theoretical $f_{coll}^{th}$")
    ax.set_ylabel(r"Empirical $f_{coll}^{emp}$")
    # ax.set_title(f"z={zsnap:.2f} ($\simeq$ {age:.2f} Gyr) - $M_{{min}}$={Mmin:.2e} $h^{{-1}}M_\odot$")
    ax.legend()

    plt.savefig(f"./saved_results/z_vary_results/fcoll_emp_vs_th_{nb_iter}.svg", bbox_inches='tight')
    print(f"\n   ----- fcoll plot saved as fcoll_emp_vs_th_{nb_iter} -----\n\n")
