import numpy as np
from numba import njit
from sim_config import real_dtype, ghost_size


@njit(cache=True)
def dispersion_flux_FD2(field, non_linear_coeff, dx):
    """
    Computes 1D dispersion flux using
    2rd order finite difference stencil.
    """
    kernel_w = ghost_size
    dispersion_flux = np.zeros_like(field)
    dispersion_flux[kernel_w:-kernel_w] = (
        real_dtype(-0.5 / dx**3)
        * (non_linear_coeff**2)
        * (
            field[(2 * kernel_w) :]
            - 2 * field[kernel_w + 1 : -kernel_w + 1]
            + 2 * field[kernel_w - 1 : -kernel_w - 1]
            - field[: -(2 * kernel_w)]
        )
    )
    return dispersion_flux
