import my_hmf_cmf_lib as mycmf

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

corr_iter_z = corr_iter_z[-3:]

for i in range(len(corr_iter_z)):
    print(f"\n[{i+1}/{len(corr_iter_z)}] Computing halos for zsnap = ", corr_iter_z[i][1])
    print("")
    nb_iter = corr_iter_z[i][0]
    mycmf.halo_finder(nb_iter, save_dir="./saved_results/")