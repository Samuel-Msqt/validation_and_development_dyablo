import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.stats import linregress

import my_hmf_cmf_lib as mycmf

plt.style.use('stylesheet.mplstyle')

z_list = [0., 1, 2, 3, 4, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8, 8.5, 9, 9.5, 10.0, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15.0, 15.5, 16, 17, 18, 19, 20.0, 22, 24, 26 ,28, 30.0]
delta_NL_list = np.linspace(-1,3,30)
delta_NL_list[0] += 0.0001
R_list = [2,1,0.6,0.4,0.2,0.1] #Mpc

z_list = np.array(z_list)
delta_NL_list = np.array(delta_NL_list)
R_list = np.array(R_list)


Mmin = 1e8 #Msun/h

delta_L = mycmf.compute_delta_linear(delta_NL_list)

h = mycmf.h
Vreg = (R_list*h)**3 * 4.0/3.0 * np.pi #(Mpc/h)^3

def compute_Mreg(dnl, R):
    Vreg = (R*h)**3 * 4.0/3.0 * np.pi #(Mpc/h)^3
    Mreg = mycmf.rho_0_h2_Msun_Mpc3 * (1 + dnl) * Vreg #Msun/h
    return Mreg


fcoll_3d = []
# Compute "true" values
for z in tqdm(z_list):
    pars, results, s8_fid = mycmf.compute_init_power_spectrum(z, mycmf.H0, mycmf.TCMB, mycmf.om, mycmf.ob, mycmf.h, mycmf.ns, mycmf.As)
    kh, _, pk = mycmf.compute_matter_power_spectrum(z, pars, s8_fid)
    fcoll_2d = []
    for R_fixed in R_list:
        fcoll_1r = []
        for dnl in delta_NL_list:
            Mreg = compute_Mreg(dnl, R_fixed)
            dl = mycmf.compute_delta_linear(dnl)
            fcoll_val = mycmf.fcoll(Mmin, Mreg, kh, pk, R_fixed * h, dl, dnl, 'QcST')
            fcoll_1r.append(fcoll_val)
        fcoll_2d.append(fcoll_1r)
    
    fcoll_3d.append(fcoll_2d)

# fcoll_3d[z_idx][r_idx][dnl_idx]


from pysr import PySRRegressor

# Flatten the 3D data
X = []
y = []

for iz, z in enumerate(z_list):
    for idnl, dnl in enumerate(delta_NL_list):
        for iR, R in enumerate(R_list):
            X.append([z, R, dnl])
            y.append(fcoll_3d[iz][iR][idnl])
            # y.append(fcoll_2d[iz][idnl])

X = np.array(X)
y = np.array(y)


model = PySRRegressor(
    niterations=5000,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=[
        "exp", "log", "sqrt", "abs", "sin", "cos", "tan", "tanh", "sinh", "cosh",
        "sign"
    ],
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 3, # stop after 3h at max
    # constraints={ '^': (-1, 1)},  ###maybe remove this
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    complexity_of_constants=2,
    select_k_features=3,
)

model.fit(X, y)


print("Best equation:", model.get_best())


def compute_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_mean = np.mean(y_true)
    total_sum_of_squares = np.sum((y_true - y_mean) ** 2)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    
    return r2


equations_df = model.equations_

print("Equation candidates with R² scores:\n")

for i, row in equations_df.iterrows():
    expr = row['sympy_format']
    complexity = row['complexity']
    func = row['lambda_format'] 

    y_pred = func(X)  
    r2 = compute_r2(y, y_pred)

    print(f"Equation {i}:")
    print(f"  Expression : {expr}")
    print(f"  Complexity : {complexity}")
    print(f"  R² Score   : {r2:.6f}\n")






