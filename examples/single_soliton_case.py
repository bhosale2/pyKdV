import numpy as np
import sys
sys.path.append("../")
from kdv_kernels.timestepper import euler_forward_timestep, RK2_timestep, zabusky_kruskal_timestep
from sim_config import real_dtype, ghost_size
import matplotlib.pyplot as plt
from util.plotset import plotset


def soliton(X, x_0, initial_mag, non_linear_coeff):
    # single soliton at x = x_0, for reference refer to Zabusky 1965
    delta = real_dtype(non_linear_coeff / np.sqrt(initial_mag / 12))
    return (
        initial_mag
        * np.cosh(np.minimum(((X - x_0) + 1), np.fabs(X - x_0)) / delta) ** -2
    ).astype(real_dtype)


def single_soliton_case(
    num_grid_pts,
    x_0,
    initial_mag,
    non_linear_coeff,
    t_end,
    dt_prefac=0.04,
    timestepper="RK2",
):
    # number of nodes
    N = num_grid_pts
    dx = real_dtype(1 / N)
    # domain of [0, 1]
    X = np.linspace(dx / 2, 1 - dx / 2, N).astype(real_dtype)

    initial_field = np.zeros((N + 2 * ghost_size,))
    initial_field[ghost_size:-ghost_size] = soliton(
        X=X, x_0=x_0, initial_mag=initial_mag, non_linear_coeff=non_linear_coeff
    )
    # timestep
    field = initial_field.copy()
    t = 0
    dt = dt_prefac * dx**3 / (non_linear_coeff**2)

    if timestepper == "RK2":
        while t <= t_end:
            RK2_timestep(field=field, non_linear_coeff=non_linear_coeff, dx=dx, dt=dt)
            t += dt
    elif timestepper == "zabusky_kruskal":
        # bootstrap for first step
        old_field = field.copy()
        RK2_timestep(field=field, non_linear_coeff=non_linear_coeff, dx=dx, dt=dt)
        while t <= t_end:
            zabusky_kruskal_timestep(field, old_field, non_linear_coeff, dx, dt)
            t += dt
    else:
        while t <= t_end:
            euler_forward_timestep(
                field=field, non_linear_coeff=non_linear_coeff, dx=dx, dt=dt
            )
            t += dt

    return X, initial_field, field


if __name__ == "__main__":
    N = 256
    x_0 = real_dtype(0.3)
    initial_mag = real_dtype(1.0)
    non_linear_coeff = real_dtype(0.02)
    # soliton speed
    c = real_dtype(initial_mag / 3)
    t_end = real_dtype(0.5 / c)
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

    plotset()
    plt.plot(X, initial_field[ghost_size:-ghost_size], label="Soliton:(initial time)")
    plt.plot(X, field[ghost_size:-ghost_size], label="Numerical:(final time)")
    plt.plot(
        X,
        final_analytical_field[ghost_size:-ghost_size],
        "o",
        label="Analytical:(final time)",
    )
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend(ncol=1, loc="upper left")
    plt.savefig("soliton_illustration.jpg", dpi=300)
    plt.show()
