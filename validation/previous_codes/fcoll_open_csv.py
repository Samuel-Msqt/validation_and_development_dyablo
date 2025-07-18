import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('stylesheet.mplstyle')

corr_iter_z = [
    ("0002850", 0.0),
    # ("0002750", 0.0389838),
    # ("0002500", 0.140594),
    # ("0002250", 0.24084),
    # ("0002000", 0.360715),
    # ("0001750", 0.509201),
    # ("0001500", 0.660872),
    # ("0001250", 0.841156),
    # ("0001000", 1.04726),
    # ("0000750", 1.30188),
    # ("0000500", 1.65828),
    # ("0000250", 2.47551),
    # ("0000000", 26.1272)
]

# Load CSV
R_val = 15

for nb_iter, zsnap in corr_iter_z:
    fcoll_df = pd.read_csv(f'./saved_results/z_vary_results/fcoll_results_R{R_val:.0f}_{nb_iter}.csv')

    # Extract data
    fcoll_emp = fcoll_df['fcoll_emp'].values
    fcoll_th = fcoll_df['fcoll_th'].values
    delta_NL_filter = fcoll_df['delta_NL_filter'].values
    zsnap = fcoll_df['zsnap'].iloc[0]
    Mmin = fcoll_df['Mmin'].iloc[0]

    max_fcoll = np.max([np.max(fcoll_emp), np.max(fcoll_th)])

    # Plot
    fig, ax = plt.subplots()
    cmap = cm.viridis
    norm = plt.Normalize(vmin=np.round(delta_NL_filter[0], 1), vmax=np.round(delta_NL_filter[-1], 1))
    color = cmap(norm(delta_NL_filter))

    ax.plot(np.linspace(0, max_fcoll + 0.02, 100), np.linspace(0, max_fcoll + 0.02, 100), '--', color="black", alpha=0.5, label="x=y")

    for i in range(len(fcoll_emp)):
        ax.scatter(fcoll_th[i], fcoll_emp[i], color=color[i], alpha=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.01)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r'$\delta_{NL}$')

    ax.set_xlabel(r"Theoretical $f_{coll}^{th}$")
    ax.set_ylabel(r"Empirical $f_{coll}^{emp}$")
    # ax.set_title(f"z={zsnap:.2f} - $M_{{min}}$={Mmin:.2e} $h^{{-1}}M_\odot$")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"./saved_results/z_vary_results/fcoll_emp_vs_th_{nb_iter}.pdf", bbox_inches='tight')
