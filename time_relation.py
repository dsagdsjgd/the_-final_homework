import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 常数与参数 ===
kB = 1.380649e-23  # J/K
T = 100            # K
mass_amu = 12.0
amu = 1.66053906660e-27
mass_kg = mass_amu * amu

v_rms = np.sqrt(3 * kB * T / mass_kg)
v_max = 5 * v_rms   # m/s

# === 读取数据 ===
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

# === VACF计算（剔除异常速度） ===
max_lag = N_steps // 2
vacf = np.zeros(max_lag)

for lag in tqdm(range(max_lag)):
    dot_products = []
    for t in range(N_steps - lag):
        v_t = velocities_reshaped[t]
        v_t_lag = velocities_reshaped[t + lag]

        # --- 剔除任意时刻超速的粒子 ---
        v1_mag = np.linalg.norm(v_t, axis=1)
        v2_mag = np.linalg.norm(v_t_lag, axis=1)
        valid_mask = (v1_mag <= v_max) & (v2_mag <= v_max)

        if np.count_nonzero(valid_mask) == 0:
            dot_products.append(0.0)
        else:
            dots = np.einsum('ij,ij->i', v_t[valid_mask], v_t_lag[valid_mask])
            dot_products.append(np.mean(dots))

    vacf[lag] = np.mean(dot_products)

# === 归一化 ===
vacf /= vacf[0] if vacf[0] != 0 else 1.0

# === 绘图 ===
time_step = unique_times[1] - unique_times[0]  # s
time_ps = np.arange(max_lag) * time_step * 1e12  # ps

plt.figure(figsize=(8, 5))
plt.plot(time_ps, vacf)
plt.xlabel("Time Lag (ps)")
plt.ylabel("Normalized VACF")
plt.title("Velocity Autocorrelation Function (filtered)")
plt.grid(True)
plt.tight_layout()
plt.savefig("vacf_filtered.png")
plt.show()
