import numpy as np
import numpy.linalg as la
import sys
sys.path.append("../")
from sim_config import real_dtype, ghost_size
import matplotlib.pyplot as plt
from util.plotset import plotset
from single_soliton_case import soliton, single_soliton_case


N = 256
dt_prefac_range = [0.32, 0.16, 0.08, 0.04]
dt_prefac_range = np.array(dt_prefac_range)
dx = real_dtype(1 / N)
non_linear_coeff = real_dtype(0.02)
dt_range = dt_prefac_range * dx ** 3 / (non_linear_coeff ** 2)
x_0 = real_dtype(0.3)
initial_mag = real_dtype(1.0)
# soliton speed
c = real_dtype(initial_mag / 3)
t_end = real_dtype(0.5 / c)
L2_error = []
Linf_error = []

for idx, dt_prefac in enumerate(dt_prefac_range):
    X, initial_field, field = single_soliton_case(
        num_grid_pts=N,
        x_0=x_0,
        initial_mag=initial_mag,
        non_linear_coeff=non_linear_coeff,
        t_end=t_end,
        dt_prefac=dt_prefac,
        #timestepper="zabusky_kruskal",

    )
    x_f = x_0 + c * t_end
    final_analytical_field = np.zeros_like(field)
    final_analytical_field[ghost_size:-ghost_size] = soliton(
        X=X, x_0=x_f, initial_mag=initial_mag, non_linear_coeff=non_linear_coeff
    )
    error_field = final_analytical_field[ghost_size:-ghost_size] - field[ghost_size:-ghost_size]
    L2_error.append(la.norm(error_field) * dx ** 0.5)
    Linf_error.append(np.amax(np.fabs(error_field)))

plotset()
plt.loglog(dt_range, np.array(L2_error), '-o', label="2 norm error")
plt.loglog(dt_range, np.array(Linf_error), '-s', label="inf norm error")
plt.loglog(dt_range, 1e10 * dt_range ** 2, '--', label="2nd order")
plt.loglog(dt_range, 1e13 * dt_range ** 3, '--', label="3rd order")
plt.xlabel("dt")
plt.ylabel("Error")
plt.legend(ncol=1, loc="upper left")
plt.savefig("temporal_error_convg.jpg", dpi=300)
plt.show()
