import numpy as np
import numpy.linalg as la
import sys
sys.path.append("../")
from sim_config import real_dtype, ghost_size
import matplotlib.pyplot as plt
from util.plotset import plotset
from single_soliton_case import soliton, single_soliton_case
from matplotlib.pyplot import cm

N = 256
non_linear_coeff_range = [0.02, 0.01, 0.005, 0.0025]
non_linear_coeff_range = np.array(non_linear_coeff_range)
dx = real_dtype(1 / N)
x_0 = real_dtype(0.3)
initial_mag = real_dtype(1.0)
# soliton speed
c = real_dtype(initial_mag / 3)
t_end = real_dtype(0.5 / c)
L2_error = []
Linf_error = []

plotset()
plt.figure()
color = cm.rainbow(np.linspace(0, 1, non_linear_coeff_range.shape[0]))

for idx, non_linear_coeff in enumerate(non_linear_coeff_range):
    X, initial_field, field = single_soliton_case(
        num_grid_pts=N,
        x_0=x_0,
        initial_mag=initial_mag,
        non_linear_coeff=non_linear_coeff,
        t_end=t_end,
        #timestepper="zabusky_kruskal",
    )
    x_f = x_0 + c * t_end
    final_analytical_field = np.zeros_like(field)
    final_analytical_field[ghost_size:-ghost_size] = soliton(
        X=X, x_0=x_f, initial_mag=initial_mag, non_linear_coeff=non_linear_coeff
    )
    error_field = (
        final_analytical_field[ghost_size:-ghost_size] - field[ghost_size:-ghost_size]
    )
    L2_error.append(la.norm(error_field) * dx**0.5)
    Linf_error.append(np.amax(np.fabs(error_field)))
    plt.plot(
        X,
        initial_field[ghost_size:-ghost_size],
        "-",
        c=color[idx],
        label=f"Soliton:(initial t), delta:{non_linear_coeff}",
    )
    plt.plot(
        X,
        field[ghost_size:-ghost_size],
        "--",
        c=color[idx],
        label=f"Numerical:(final t), delta:{non_linear_coeff}",
    )
    plt.plot(
        X,
        final_analytical_field[ghost_size:-ghost_size],
        "-o",
        c=color[idx],
        label=f"Analytical:(final t), delta:{non_linear_coeff}",
    )
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend(ncol=1, loc="upper left")
plt.savefig("soliton_non_linearity_var.jpg", dpi=300)

plt.figure(2)
plt.semilogy(non_linear_coeff_range, np.array(L2_error), '-o', label="2 norm error")
plt.semilogy(non_linear_coeff_range, np.array(Linf_error), '-s', label="inf norm error")
plt.xlabel("non linearity coeff")
plt.ylabel("error")
plt.savefig("non_linearity_error_var.jpg", dpi=300)
plt.show()
