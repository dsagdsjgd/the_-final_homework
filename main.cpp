#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
// TODO:t可以是int，单位为dt

const double epsilon = 0.01, sigma = 3.4, mass = 12.0, cutoff = 8.5; // LJ参数

struct Atom {
    int id; // 原子id
    double r[3]; // 位置
    double v[3]={0,0,0}; // 速度
    double a[3]={0,0,0}; // 加速度
    vector<int> neighbors; // 邻居原子id列表
    Atom() : id(-1) {} // 默认构造函数
};

class Atom_r {
public:
    int id; // 原子id
    double r[3]; // 位置

    // // 使用Atom构造Atom_r
    // Atom_r(const Atom& atom) {
    //     id = atom.id;
    //     for (int d = 0; d < 3; ++d) {
    //         r[d] = atom.r[d];
    //     }
    // }
};

vector<Atom_r> get_atoms_r(const vector<Atom>& atoms) {
    vector<Atom_r> atom_rs;
    for (const auto& atom : atoms) {
        atom_rs.emplace_back(atom); // 使用Atom构造Atom_r
    }
    return atom_rs;
}

// class Atom_r{
//     public:
//         int id; // 原子id
//         double r[3]; // 位置
//     };
//     vector<Atom_r> get_atoms_r(vector<Atom> atoms){
//         vector<Atom_r> atom_rs;
//         for (const auto& atom : atoms) {
//             Atom_r ar;
//             for (int d = 0; d < 3; ++d) {
//                 ar.r[d] = atom.r[d];
//             }
//             atom_rs.push_back(ar);
//         }
//         return atom_rs;
//     }

class frame { //视频的一帧
public:
    double t; // 时间
    vector<Atom_r> atom_rs; // 存储原子位置
    frame() : t(0.0) {} // 默认构造函数
    frame(double _t,vector<Atom_r> _atom_rs) : t(_t), atom_rs(_atom_rs) {}
};

class simulation {
public:
    vector<Atom> atoms; // 使用一维数组存储所有原子
    // TODO:需要合适的数据结构，可以更好地表征两个原子的相邻特征，这样在计算LJ的时候更适合截断
    // 我设计的结构是nx,ny,nz,atom_id这样的结构。但它的问题是会导致原子位置的重复存储
    // 可能需要一个邻居列表来存储每个原子的邻居原子
    // 对，我同意邻居列表是合适的选择。atom_id是0-99，再加一个vector<int>来存储邻居原子id

    double t;           // 当前时间
    int nx, ny, nz;     // 网格大小
    double a1[3]={2.456,0,0}, 
    a2[3]={-1.228,2.126958,0}, 
    a3[3]={0,0,7}; // 晶格矢量
    vector<frame> frames; // 存储每一帧的原子位置

    simulation(int nx, int ny, int nz) : nx(nx), ny(ny), nz(nz), t(0.0) {
        // // 初始化晶格矢量
        // a1[0] = 2.456; a1[1] = 0.0; a1[2] = 0.0;
        // a2[0] = -1.228; a2[1] = 2.126958; a2[2] = 0.0;
        // a3[0] = 0.0; a3[1] = 0.0; a3[2] = 7.0;

        // 基元原子（分数坐标）
        double basis_frac[4][3] = {
            {0.0, 0.0, 0.25},
            {0.0, 0.0, 0.75},
            {1.0 / 3, 2.0 / 3, 0.25},
            {2.0 / 3, 1.0 / 3, 0.75}
        };
        double basis_real[4][3];
        for (int b = 0; b < 4; ++b) {
            for (int d = 0; d < 3; ++d) {
            basis_real[b][d] = basis_frac[b][0] * a1[d] +
                       basis_frac[b][1] * a2[d] +
                       basis_frac[b][2] * a3[d];
            }
        }

        // 初始化原子位置
        int N = 0; //循环结束时N=原子总数
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < int(ny/(sqrt(3)/2)); ++j) {
                for (int i = int(-0.5*ny/(sqrt(3)/2)); i < nx; ++i) {
                    double shift[3] = {
                        i * a1[0] + j * a2[0] + k * a3[0],
                        i * a1[1] + j * a2[1] + k * a3[1],
                        i * a1[2] + j * a2[2] + k * a3[2]
                    };
                    for (int b = 0; b < 4; ++b) {
                        Atom atom;
                        for (int d = 0; d < 3; ++d) {
                            atom.r[d] = basis_real[b][d] + shift[d];
                            // atom.v[d] = 0.0; // 初始速度为0
                            // atom.a[d] = 0.0; // 初始加速度为0
                        }
                        // if atom not in atoms{
                        atom.id = N++; // 分配原子id
                        atoms.push_back(atom);
                        // }
                    }
                }
            }
        }

        //验证N==atoms.size()
        if (N != atoms.size()) {
            cerr << "Error: N != atoms.size()" << endl;
            exit(1);
        }

        // 写出相邻矩阵
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                double dx = atoms[i].r[0] - atoms[j].r[0];
                double dy = atoms[i].r[1] - atoms[j].r[1];
                double dz = atoms[i].r[2] - atoms[j].r[2];
                // // 周期性边界条件（这啥？看不懂）
                // dx -= round(dx / a1[0]) * a1[0];
                // dy -= round(dy / a2[1]) * a2[1];
                // dz -= round(dz / a3[2]) * a3[2];
                double dist_sq = dx * dx + dy * dy + dz * dz;
                if (dist_sq < cutoff^2) {
                    atoms[i].neighbors.push_back(atoms[j].id);
                    atoms[j].neighbors.push_back(atoms[i].id);
                }
            }
        }
        //这个方法不更新neighbors，所以它的假设是原子的运动不足以使两个原子从cutoff外变为cutoff内
    }
    void verlrt() {
        // TODO:MD更新
        for (auto& atom : atoms) {
            for (int neighbor_id : atom.neighbors) {
                Atom& neighbor_atom = atoms[neighbor_id]; // 获取邻居原子
                double rij[3]; // 计算原子间距离向量
                for (int d = 0; d < 3; ++d) {
                    rij[d] = atom.r[d] - neighbor_atom.r[d];
                }
                double r = sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]); // 计算距离
                if (r < cutoff) { // 如果在截断距离内
                    double r6 = pow(sigma / r, 6); // 计算LJ势能
                    double r12 = r6 * r6;
                    double force_scalar = 24 * epsilon * (2 * r12 - r6) / (r * r); // 力的标量值
                    for (int d = 0; d < 3; ++d) {
                        double force_vector = force_scalar * rij[d]; // 力的向量值
                        atom.a[d] += force_vector / mass; // 更新加速度
                    }
                }
            }
        }
        // 注意：周期性边界条件（这里先不要了，否则neighbors也需要更正）
        double dt = 0.01; //TODO:dt的值
        t += dt;
        frame f(t, get_atoms_r(atoms));
        frames.push_back(f);
    }
    void gen_video(){
        for (const auto& f : frames) {
            // TODO:plot 每一帧，文件名为时间戳
            // TODO:C++我不会画图，整个程序需要转成python
            // 这样连续播放就能获得视频效果            
        }
    }
};

int main() {
    simulation graphite(5,5,1);
    int nsteps = 10; // 模拟步数
    for (int i = 0; i < nsteps; ++i) {
        graphite.verlrt();
    }
    graphite.gen_video();
    return 0;
}