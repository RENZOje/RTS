import matplotlib.pyplot as plt
import random
import math
from time import time

start = time()


def signal(n, omega, N, min_value=0, max_value=1):
    """Return signal function."""
    A = [min_value + (max_value - min_value) * random.random() for _ in range(n)]
    phi = [min_value + (max_value - min_value) * random.random() for _ in range(n)]

    def f(t):
        x = 0
        for i in range(n):
            x += A[i]*math.sin(omega/n*t*i + phi[i])
        return x
    return f



def get_m_D(x):
    """Return expected value(m) and dispersion."""
    m = sum(x)/len(x)
    return m, sum([(i - m) ** 2 for i in x]) / (len(x) - 1)



def get_m(x):
    """Return expected value."""
    return sum(x)/len(x)



def get_D(x, m=None):
    """return dispersion."""
    if m is None:
        m = get_m(x)
    return sum([(i - m) ** 2 for i in x]) / (len(x) - 1)


def get_R(x, y, m=None):
    """Return correlation."""
    start = time()
    m_x, D_x = get_m_D(x)
    m_y, D_y = get_m_D(y)
    if m is not None:
        m_x = m
        m_y = m
    N = min(len(x), len(y))
    R = [(x[i]-m_x)*(y[i]-m_y)/(N-1) for i in range(N)]
    print(time() - start)
    return R



def get_R_by_gen(x_gen, y_gen, N):
    R = [0]*N
    m_x = get_m([x_gen(i) for i in range(N)])
    m_y = get_m([y_gen(i) for i in range(N)])
    for tau in range(N):
        for t in range(N):
            R[tau] += (x_gen(t)-m_x)*(y_gen(t+tau)-m_y)/(N-1)

    return R



def r_tau(x_gen, y_gen, N, tau=0):
    x = [x_gen(i) for i in range(N)]
    y = [y_gen(i+tau) for i in range(N)]
    m_x, D_x = get_m_D(x)
    m_y, D_y = get_m_D(y)

    R = 0
    for i in range(N//2-1):
        R += (x[i] - m_x)*(y[i+tau] - m_y)/(N//2-1)
    R /= (D_x*D_y)**(1/2)
    return R

n = 12
omega = 1100
N = 256

range_min = 0
range_max = 1

x_gen = signal(n, omega, N, range_min, range_max)
y_gen = signal(n, omega, N, range_min, range_max)

x = [x_gen(i) for i in range(N)]
y = [y_gen(i) for i in range(N)]


plt.plot(range(N), x, label='x')
plt.plot(range(N), y, label='y')
plt.legend()
plt.show()
m = get_m(x)
D = get_D(x, m)

print(f'm: {m}\nD: {D}')


r_tau_X = [r_tau(x_gen, y_gen, N, tau) for tau in range(N//2)]
r_tau_Y = [r_tau(x_gen, x_gen, N, tau) for tau in range(N//2)]
plt.plot(range(N//2), r_tau_X, label='Rxx_tauX')
plt.plot(range(N//2), r_tau_Y, label='Rxx_tauY')


plt.legend()
plt.show()


print(f'Total time: `{time() - start}`')