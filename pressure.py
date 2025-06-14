import numpy as np
import h5py
import subprocess
from LJ import temperature, nx, ny, nz, a1, a2, a3, sigma, epsilon  # 直接从LJ.py导入参数

# 单位转换常数
eV_per_angstrom3_to_GPa = 160.21766  # 1 eV/Å³ = 160.21766 GPa
kB = 8.617333262145e-5  # eV/K

T_start = 1
T_end = 11
T_number = 5

# 运行模拟
def run_simulation(temperature):
    """运行LJ.py模拟并返回能量列表"""
    # 修改LJ.py中的温度参数（使用UTF-8编码）
    with open("LJ.py", "r", encoding='utf-8') as f:  # 明确指定编码
        lines = f.readlines()
    
    # 找到并修改温度参数
    for i, line in enumerate(lines):
        if "temperature =" in line:
            lines[i] = f"temperature = {temperature}  # Kelvin\n"
            break
    
    # 写回文件（同样使用UTF-8）
    with open("LJ.py", "w", encoding='utf-8') as f:
        f.writelines(lines)
    
    # 运行模拟
    process = subprocess.run(['python', 'LJ.py'], check=True)
    

def calculate_pt_curve(h5_file_path, temperature_range):
    """
    计算P-T曲线
    参数:
        h5_file_path: 轨迹文件路径
        temperature_range: 温度数组(K)
    返回:
        (temperatures, pressures) 元组
    """
    

    # 计算P-T曲线
    pressures = []
    for T in temperature_range:
        # 0.预先跑一下模拟
        run_simulation(T)

        # 1. 读取轨迹数据
        with h5py.File(h5_file_path, "r") as f:
            pos_data = f["trajetory"][:]  # 注意原拼写错误
            vel_data = f["velocity_traj"][:]
        
        # 2. 计算盒子体积 (Å³)
        box_volume = np.abs(np.dot(a1, np.cross(a2, a3))) * nx * ny * nz
        N_particles = len(np.unique(pos_data[:, 0]))
        rho = N_particles / box_volume  # 数密度(Å⁻³)

        # 3. 预处理轨迹数据
        positions = pos_data[:, 1:4]  # Å
        velocities = vel_data[:, 1:4] * 1e10  # m/s → Å/s
        N_steps = len(np.unique(pos_data[:, 4]))
        
        # 4. 计算维里项（与温度无关的部分）
        virial_sum = 0.0
        for step in range(N_steps):
            pos = positions[step*N_particles:(step+1)*N_particles]
            for i in range(N_particles):
                for j in range(i+1, N_particles):
                    rij = pos[i] - pos[j]
                    r = np.linalg.norm(rij)
                    if r < 4.0:  # 截断半径
                        r6 = (sigma/r)**6
                        r12 = r6**2
                        dphi_dr = 24 * epsilon * (2*r12 - r6) / r  # eV/Å
                        virial_sum += np.dot(rij, rij) * dphi_dr / r
        avg_virial = virial_sum / (3 * N_particles * N_steps)
        # 动能项 (2/3N * KE = k_B T)
        P_ideal = rho * kB * T
        P_virial = rho * avg_virial
        pressure = (P_ideal + P_virial) * eV_per_angstrom3_to_GPa
        pressures.append(pressure)
        print(f"计算完毕！pressure数值为 {pressure:.4e} ")
    return temperature_range, np.array(pressures)

def plot_pt_curve(temperatures, pressures):
    """绘制P-T曲线"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, pressures, 'o-', lw=2)
    plt.xlabel("Temperature (K)", fontsize=12)
    plt.ylabel("Pressure (GPa)", fontsize=12)
    plt.title("Graphite P-T Curve", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("graphite_pt_curve.png")
    plt.show()

if __name__ == "__main__":
    # 示例：计算100-500K的P-T曲线
    temperature_range = np.linspace(T_start, T_end, T_number)
    T, P = calculate_pt_curve(
        h5_file_path="graphite_simulation.h5",
        temperature_range=temperature_range
    )
    
    # 保存结果
    np.savetxt("pt_curve.dat", np.column_stack((T, P)), 
               header="Temperature(K) Pressure(GPa)", fmt="%.1f %.3f")
    
    # 绘图
    plot_pt_curve(T, P)