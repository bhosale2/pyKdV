from numba import njit
from kdv_kernels.advection_flux import (
    advection_flux_conservative_ENO3,
    advection_flux_zabusky_kruskal,
)
from kdv_kernels.dispersion_flux import dispersion_flux_FD2
from kdv_kernels.periodic_ghost_comm import periodic_ghost_comm


@njit(cache=True)
def total_flux(field, non_linear_coeff, dx):
    periodic_ghost_comm(field)
    return advection_flux_conservative_ENO3(field, dx) + dispersion_flux_FD2(
        field, non_linear_coeff, dx
    )


@njit(cache=True)
def total_flux_zabusky_kruskal(field, non_linear_coeff, dx):
    periodic_ghost_comm(field)
    return advection_flux_zabusky_kruskal(field, dx) + dispersion_flux_FD2(
        field, non_linear_coeff, dx
    )
