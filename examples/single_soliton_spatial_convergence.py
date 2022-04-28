import numpy as np
import sys
sys.path.append("../")
import numpy.linalg as la
from sim_config import real_dtype, ghost_size
import matplotlib.pyplot as plt
from util.plotset import plotset
from single_soliton_case import soliton, single_soliton_case


N_range = [32, 64, 128, 256]
N_range = np.array(N_range)
dx_range = real_dtype(1 / N_range)
x_0 = real_dtype(0.3)
initial_mag = real_dtype(1.0)
non_linear_coeff = real_dtype(0.02)
# soliton speed
c = real_dtype(initial_mag / 3)
t_end = real_dtype(0.5 / c)
L2_error = []
Linf_error = []

for idx, N in enumerate(N_range):
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
    error_field = final_analytical_field[ghost_size:-ghost_size] - field[ghost_size:-ghost_size]
    L2_error.append(la.norm(error_field) * dx_range[idx] ** 0.5)
    Linf_error.append(np.amax(np.fabs(error_field)))

plotset()
plt.loglog(dx_range, np.array(L2_error), '-o', label="2 norm error")
plt.loglog(dx_range, np.array(Linf_error), '-s', label="inf norm error")
plt.loglog(dx_range, 1e2 * dx_range ** 2, '--', label="2nd order")
plt.loglog(dx_range, 4e2 * dx_range ** 2, '--', label="2nd order")
plt.xlabel("dx")
plt.ylabel("Error")
plt.legend(ncol=1, loc="upper left")
plt.savefig("spatial_error_convg.jpg", dpi=300)
plt.show()
