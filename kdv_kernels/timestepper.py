from sim_config import real_dtype
from numba import njit
from kdv_kernels.total_flux import total_flux, total_flux_zabusky_kruskal


@njit(cache=True)
def euler_forward_timestep(field, non_linear_coeff, dx, dt):
    field[...] = field + dt * total_flux(field, non_linear_coeff, dx)


@njit(cache=True)
def RK2_timestep(field, non_linear_coeff, dx, dt):
    midstep_field = field + real_dtype(dt / 2) * total_flux(field, non_linear_coeff, dx)
    field[...] = field + dt * total_flux(midstep_field, non_linear_coeff, dx)


@njit(cache=True)
def zabusky_kruskal_timestep(field, old_field, non_linear_coeff, dx, dt):
    """
    leap frogging timestep as proposed by Zabusky and Kruskal.
    """
    new_field = old_field + real_dtype(2 * dt) * total_flux_zabusky_kruskal(
        field, non_linear_coeff, dx
    )
    old_field[...] = field
    field[...] = new_field
