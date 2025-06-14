import h5py
import numpy as np
import matplotlib.pyplot as plt

# === 参数设置 ===
with h5py.File("graphite_simulation.h5", "r") as f:
    vel_data = f["velocity_traj"][:]  # [id, vx, vy, vz, t]


particle_ids = vel_data[:, 0].astype(int)
velocities = vel_data[:, 1:4]   
times = vel_data[:, 4]

unique_times = np.unique(times)
unique_ids = np.unique(particle_ids)
N_steps = len(unique_times)
N_particles = len(unique_ids)


velocities_reshaped = velocities.reshape((N_steps, N_particles, 3))


max_lag = N_steps // 2
vacf = np.zeros(max_lag)

for lag in range(max_lag):
    dot_products = []
    for t in range(N_steps - lag):
        v_t = velocities_reshaped[t]
        v_t_lag = velocities_reshaped[t + lag]
        dots = np.einsum('ij,ij->i', v_t, v_t_lag)  # 每个粒子 v·v
        dot_products.append(np.mean(dots))
    vacf[lag] = np.mean(dot_products)


vacf /= vacf[0]

# === 绘图 ===
time_step = unique_times[1] - unique_times[0]  # 假设等间隔
time_ps = np.arange(max_lag) * time_step * 1e12  # ps

plt.figure(figsize=(8, 5))
plt.plot(time_ps, vacf)
plt.xlabel("Time Lag (ps)")
plt.ylabel("Normalized VACF")
plt.title("Velocity Autocorrelation Function")
plt.grid(True)
plt.tight_layout()
plt.savefig("vacf_plot.png")
plt.show()
