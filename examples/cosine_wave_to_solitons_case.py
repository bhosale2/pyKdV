import numpy as np
import sys

sys.path.append("../")
from kdv_kernels.timestepper import RK2_timestep, zabusky_kruskal_timestep
from sim_config import real_dtype, ghost_size
import matplotlib.pyplot as plt
from util.plotset import plotset
import os


SAVE_VIDEO = True
# number of nodes
N = 256
x_0 = real_dtype(0.3)
dx = real_dtype(1 / N)
# domain of [0, 1]
X = np.linspace(dx / 2, 1 - dx / 2, N).astype(real_dtype)

initial_field = np.zeros((N + 2 * ghost_size,))
initial_field[ghost_size:-ghost_size] = np.cos(2 * np.pi * X).astype(real_dtype)
# timestep
field = initial_field.copy()
non_linear_coeff = real_dtype(0.022)
dt_prefac = real_dtype(0.04)
t = 0
dt = dt_prefac * dx**3 / (non_linear_coeff**2)
t_end = 1.0
it = 0
save_rate = 1000
plotset()
# bootstrap for first step
old_field = field.copy()
RK2_timestep(field=field, non_linear_coeff=non_linear_coeff, dx=dx, dt=dt)
while t <= t_end:
    zabusky_kruskal_timestep(field, old_field, non_linear_coeff, dx, dt)
    t += dt
    it += 1
    if it % save_rate == 0:
        print(t)
        plt.plot(X, initial_field[ghost_size:-ghost_size], "--")
        plt.plot(X, field[ghost_size:-ghost_size])
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.savefig(
            "snap_" + str("%0.6d" % (t * 10000)) + ".png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()
        plt.close("all")


plt.plot(X, initial_field[ghost_size:-ghost_size], "--", label="(initial time)")
plt.plot(X, field[ghost_size:-ghost_size], label="(final time)")
plt.savefig("cosine_wave_to_solitons_ill.jpg", dpi=300)
plt.show()
if SAVE_VIDEO:
    os.system(
        f"ffmpeg -r 16 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' -vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' cosine_wave_to_soliton.mp4"
    )
    os.system("rm -f *png")
