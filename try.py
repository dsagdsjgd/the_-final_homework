import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
#模拟石墨晶体

NUM_LAYERS = 3
ATOMS_PER_LAYER = 100
LATTICE_CONST = 1.42  # nm
INTERLAYER_DIST = 3.35  # nm
MASS = 12.01  # 原子质量单位 (amu)
EPSILON = 2.964e-3  # eV
SIGMA = 0.34  # nm
CUTOFF = 2.5 * SIGMA
DT = 0.001  # ps
STEPS = 1000

def initialize_graphite(n_layers, n_atoms_per_layer, a, d):
    positions = []
    n_side = int(np.ceil(np.sqrt(n_atoms_per_layer)))

    for k in tqdm(range(n_layers)):
        for i in range(n_side):
            for j in range(n_side):
                if (i + j) % 2 == 0:
                    x = i * a
                    y = j * a * np.sqrt(3) / 2
                    z = k * d
                    positions.append([x, y, z])
                if len(positions) >= n_layers * n_atoms_per_layer:
                    return np.array(positions), np.zeros((len(positions), 3))
    return np.array(positions), np.zeros((len(positions), 3))


def compute_lj_forces(pos, sigma, epsilon, rc):
    N = len(pos)
    forces = np.zeros_like(pos)

    for i in tqdm(range(N - 1)):
        for j in range(i + 1, N):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < rc and r > 1e-6:
                sr6 = (sigma / r) ** 6
                sr12 = sr6 ** 2
                f_scalar = 24 * epsilon / r * (2 * sr12 - sr6)
                fij = f_scalar * (rij / r)
                forces[i] += fij
                forces[j] -= fij
    return forces



def run_md():
    pos, vel = initialize_graphite(NUM_LAYERS, ATOMS_PER_LAYER, LATTICE_CONST, INTERLAYER_DIST)

    for step in tqdm(range(STEPS)):
        forces = compute_lj_forces(pos, SIGMA, EPSILON, CUTOFF)

        # Velocity Verlet 积分
        vel += 0.5 * DT * forces / MASS
        pos += DT * vel
        forces_new = compute_lj_forces(pos, SIGMA, EPSILON, CUTOFF)
        vel += 0.5 * DT * forces_new / MASS

        # 可视化每100步
        if step % 100 == 0:
            print(f"Step {step}")
            plot_atoms(pos, step)



def plot_atoms(pos, step):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='black', s=10)
    ax.set_title(f"Step {step}")
    ax.set_xlabel("X [nm]")
    ax.set_ylabel("Y [nm]")
    ax.set_zlabel("Z [nm]")
    ax.set_box_aspect([1, 1, 0.5])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_md()
