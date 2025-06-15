import numpy as np
import sys

def extract_poteng_from_log(logfile, npyfile):
    energies = []
    read_data = False

    with open(logfile, 'r') as f:
        for line in f:
            line = line.strip()
            # 发现表头，开始读取
            if line == "Step Temp PotEng":
                read_data = True
                continue

            if read_data:
                if not line or not line[0].isdigit():
                    # 遇到空行或非数据行，停止读取该段数据
                    read_data = False
                    continue
                parts = line.split()
                if len(parts) < 3:
                    read_data = False
                    continue
                try:
                    poteng = float(parts[2])
                    energies.append(poteng)
                except ValueError:
                    read_data = False

    if energies:
        np.save(npyfile, np.array(energies, dtype=np.float32))
        print(f"成功提取 {len(energies)} 条 PotEng 数据，保存为 {npyfile}")
    else:
        print("未找到任何有效的 PotEng 数据。")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_poteng.py input_log.lammps output_energy.npy")
    else:
        extract_poteng_from_log(sys.argv[1], sys.argv[2])
