import numpy as np
import matplotlib.pyplot as plt

with open("msd.txt", "r") as f:
    msd_list = np.loadtxt(f, skiprows=1)
# === 计算均方位移 (MSD) ===
plt.plot(msd_list, label='MSD')
plt.title('Mean Squared Displacement (MSD) vs Time Step')
plt.xlabel('Time Step (10fs)')
plt.ylabel('Mean Squared Displacement (Å²)')
plt.savefig("msd.png")
plt.show()

# === 计算自扩散系数 ===
time_steps = np.arange(1,len(msd_list))
diffusion_coefficient = msd_list[1:] / (6 * time_steps)  # 6D for 3D space
plt.plot(time_steps, diffusion_coefficient, label='Diffusion Coefficient')
plt.title('Diffusion Coefficient vs Time Step')
plt.xlabel('Time Step (10fs)')
plt.ylabel('Diffusion Coefficient (0.1Å²/fs)')
plt.savefig("diffusion_coefficient.png")
plt.show()
# === 计算平均自扩散系数 ===
average_diffusion_coefficient = np.mean(diffusion_coefficient)
print(f"Average Diffusion Coefficient: {average_diffusion_coefficient:.4f} (0.1 Å²/fs)")
# 1.9477 (0.1 Å²/fs)