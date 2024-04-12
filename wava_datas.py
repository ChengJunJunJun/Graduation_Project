import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import random

def range_omega(H0, T0, start_omega, end_omega):
    """
    get range of omega
    :param H0:
    :param T0:
    :return:
    """
    w = np.linspace(start_omega, end_omega, 100)
    sw = 173 * H0 ** 2 * T0 ** (-4) * w ** (-5) * np.exp(-691 * w ** (-4) * T0 ** (-4))
    # sw = (1 / 4 * math.pi) * (H0 ** 2) * (((2 * math.pi)/T0) ** 4) * w ** (-5) * np.exp((-1 / math.pi) * (((2 * math.pi)/T0) ** 4) * w ** (-4))
    plt.plot(w, sw)
    plt.show()

def s(w,T0,H0):
    sw = 173 * H0 ** 2 * T0 ** (-4) * w ** (-5) * np.exp(-691 * w ** (-4) * T0 ** (-4))
    return sw

T = 20000
H0 = 4
T0 = 3.5
Tz = T0 / 1.408
start_omega = 0.8
end_omega = 4
g = 9.8
# # start_omega = (-3.11/((H0 ** 2) * math.log(10, mu))) ** (1/4)
# # end_omega = (-3.11/((H0 ** 2) * math.log(10, (1 - mu)))) ** (1/4)

# start_omega = (1 / T0) * (-1605.3/math.log(10, mu)) ** (1/4)
# end_omega = (1 / T0) * (-1605.3/math.log(10, (1 - mu))) ** (1/4)



# range_omega(H0, T0, start_omega, end_omega)
eta00 = np.zeros([T, 100])

M = 60
delta_w = (end_omega - start_omega) / M
w = np.linspace(start_omega, end_omega, M)

for t in range(T):
    z = np.zeros([100])
    for i in range(M-1):
        x = np.linspace(1, 100, 100)
        w_i = random.uniform(w[i], w[i + 1])
        phi_i = random.uniform(0, 2 * math.pi)
        k_i = w_i**2/g
        a_i = (2 * delta_w * s(w_i, T0, H0)) ** 0.5
        z += a_i * np.cos(k_i * x - w_i * t + phi_i)
    eta00[t, :] = z

    print(f"已经完成{1 + t} / {T}个时间步")

np.save(r'D:/data/chengjun.npy', eta00)