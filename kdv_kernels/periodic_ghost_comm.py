from sim_config import ghost_size
from numba import njit


@njit(cache=True)
def periodic_ghost_comm(field):
    field[:ghost_size] = field[-(2 * ghost_size) : -ghost_size]
    field[-ghost_size:] = field[ghost_size : (2 * ghost_size)]
