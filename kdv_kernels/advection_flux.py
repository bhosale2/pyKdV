import numpy as np
from numba import njit
from sim_config import real_dtype, ghost_size


@njit(cache=True)
def advection_flux_conservative_ENO3(field, dx):
    """
    Computes 1D conservative advection flux using
    3rd order ENO stencil.
    """
    kernel_w = ghost_size
    advection_flux = np.zeros_like(field)
    face_velocity_x_east = real_dtype(0.5) * (
        field[kernel_w:-kernel_w] + field[kernel_w + 1 : -kernel_w + 1]
    )
    face_velocity_x_west = real_dtype(0.5) * (
        field[kernel_w:-kernel_w] + field[kernel_w - 1 : -kernel_w - 1]
    )
    nodal_flux = field * field / real_dtype(2)
    upwind_switch = face_velocity_x_east > 0
    advection_flux_east = (
        real_dtype(1 / 3) * nodal_flux[kernel_w + 1 : -kernel_w + 1]
        + real_dtype(5 / 6) * nodal_flux[kernel_w:-kernel_w]
        - real_dtype(1 / 6) * nodal_flux[kernel_w - 1 : -kernel_w - 1]
    ) * upwind_switch + (real_dtype(1) - upwind_switch) * (
        real_dtype(1 / 3) * nodal_flux[kernel_w:-kernel_w]
        + real_dtype(5 / 6) * nodal_flux[kernel_w + 1 : -kernel_w + 1]
        - real_dtype(1 / 6) * nodal_flux[(2 * kernel_w) :]
    )
    upwind_switch[...] = face_velocity_x_west > 0
    advection_flux_west = (
        real_dtype(1 / 3) * nodal_flux[kernel_w:-kernel_w]
        + real_dtype(5 / 6) * nodal_flux[kernel_w - 1 : -kernel_w - 1]
        - real_dtype(1 / 6) * nodal_flux[: -(2 * kernel_w)]
    ) * upwind_switch + (real_dtype(1) - upwind_switch) * (
        real_dtype(1 / 3) * nodal_flux[kernel_w - 1 : -kernel_w - 1]
        + real_dtype(5 / 6) * nodal_flux[kernel_w:-kernel_w]
        - real_dtype(1 / 6) * nodal_flux[kernel_w + 1 : -kernel_w + 1]
    )
    advection_flux[kernel_w:-kernel_w] = real_dtype(-1 / dx) * (
        advection_flux_east - advection_flux_west
    )
    return advection_flux


@njit(cache=True)
def advection_flux_zabusky_kruskal(field, dx):
    """
    Computes 1D conservative advection flux as proposed
    by Zabusky and Kruskal.
    """
    kernel_w = ghost_size
    advection_flux = np.zeros_like(field)
    advection_flux[kernel_w:-kernel_w] = (
        real_dtype(-1 / 6 / dx)
        * (
            field[kernel_w:-kernel_w]
            + field[kernel_w + 1 : -kernel_w + 1]
            + field[kernel_w - 1 : -kernel_w - 1]
        )
        * (field[kernel_w + 1 : -kernel_w + 1] - field[kernel_w - 1 : -kernel_w - 1])
    )
    return advection_flux
