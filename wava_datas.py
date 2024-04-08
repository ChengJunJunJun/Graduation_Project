import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fftn, fftfreq, ifftn, fftshift, ifftshift


# 时间采样帧数、采样间隔
T = 60 #(自己根据需要设置)
delta_t = 1
# 频率划分份数
N_omg = 30
# 角度划分份数
N_theta = 30
# 频率间隔
start_omega = 0.45
end_omega = 1.5
delta_omega0 = (end_omega-start_omega)/N_omg
# 角度间隔
start_theta = -np.pi/2
end_theta = np.pi/2
delta_theta = (end_theta-start_theta)/N_theta
# 重力加速度
g = 9.8
# 径向分辨率、径向范围、径向采样数目
r_delta = 10
r_end = 1000
r_num = int(r_end/r_delta)


def gennerate_data():
    # 随机频率、传播角、相位角
    omega_ran = np.linspace(start_omega, end_omega, N_omg)
    theta_ran = np.linspace(start_theta, end_theta, N_theta)
    epsilon_ran = np.linspace(0, 2*np.pi, N_omg*N_theta)
    np.random.shuffle(epsilon_ran)

    # 周期和有义波高
    T0 = 3.5 #（根据需要设置）
    H0 = 4 #（根据需要设置）

    eta00 = np.zeros([T, r_num])
    for n in range(T):
        t = delta_t*(n+1)
        z = np.zeros([r_num])
        r = np.linspace(0, r_end, r_num)
        num = 0
        # 叠加波分量
        for i in range(N_omg):
            for j in range(N_theta):
                omega_i = omega_ran[i]
                k_i = omega_i**2/g
                theta_j = theta_ran[j]
                epsilon_ij = epsilon_ran[num]
                num += 1
                a_ij = np.sqrt(2*(173*H0**2*T0**(-4)*omega_i**(-5)*np.exp(-691*omega_i**(-4)*T0**(-4))*2/np.pi*(np.cos(theta_j))**2)*delta_omega0*delta_theta)
                z += a_ij*np.cos(omega_i*t-k_i*r*np.cos(theta_j)+epsilon_ij)
        eta00[n, :] = z
    
    return eta00

combined_data = np.empty((0, 100))
N_num = 1000
for i in range(N_num):
    data = gennerate_data()

    combined_data = np.concatenate((combined_data,data), axis=0)

    print(f"已经完成{1 + i} / {N_num}")


print(combined_data.shape)


np.save(r'D:/data/chengjun.npy', combined_data)

print('原始波浪场仿真完毕')
