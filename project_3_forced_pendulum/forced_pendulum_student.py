import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def forced_pendulum_ode(t, state, l, g, C, Omega):
    """
    受驱单摆的常微分方程
    state: [theta, omega]
    返回: [dtheta/dt, domega/dt]
    """
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g / l) * np.sin(theta) + C * np.cos(Omega * t)*np.sin(Omega*t)
    return [dtheta_dt, domega_dt]


def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0, 100), y0=[0, 0]):
    """
    求解受迫单摆运动方程
    返回: t, theta
    """
    sol = solve_ivp(forced_pendulum_ode, t_span, y0,
                    args=(l, g, C, Omega),
                    t_eval=np.linspace(t_span[0], t_span[1], 2000),
                    rtol=1e-6, atol=1e-9)
    return sol.t, sol.y[0]


def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_span=(0, 200), y0=[0, 0]):
    """
    寻找共振频率
    返回: Omega_range, amplitudes
    """
    if Omega_range is None:
        Omega_range = np.linspace(0.1, 20, 50)

    amplitudes = []
    for Omega in Omega_range:
        t, theta = solve_pendulum(l, g, C, Omega, t_span, y0)
        # 取最后1/2的数据计算振幅，避免瞬态响应

        steady_idx = t > t_span[0] + (t_span[1] - t_span[0]) / 2
        amplitude = np.max(np.abs(theta[steady_idx]))
        amplitudes.append(amplitude)

    return Omega_range, np.array(amplitudes)


def plot_results(t, theta, title):
    """绘制结果"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()


def plot_resonance(Omega_range, amplitudes):
    """绘制共振曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(Omega_range, amplitudes, 'o-', label='Amplitude')
    plt.axvline(np.sqrt(9.81 / 0.1), color='r', linestyle='--',
                label='Natural frequency (small angle)')
    plt.title('Resonance Curve: Amplitude vs Driving Frequency')
    plt.xlabel('Driving Frequency Ω (rad/s)')
    plt.ylabel('Steady-state Amplitude (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """主函数"""
    # 任务1: 特定参数下的数值解与可视化
    t, theta = solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0, 50))
    plot_results(t, theta, 'Forced Pendulum with Ω=5 rad/s')

    # 任务2: 探究共振现象
    Omega_range, amplitudes = find_resonance()
    plot_resonance(Omega_range, amplitudes)

    # 找到共振频率并绘制共振情况
    resonance_idx = np.argmax(amplitudes)
    resonance_Omega = Omega_range[resonance_idx]
    print(f"Resonance frequency: {resonance_Omega:.3f} rad/s")

    t_res, theta_res = solve_pendulum(Omega=resonance_Omega)
    plot_results(t_res, theta_res,
                 f'Forced Pendulum at Resonance (Ω={resonance_Omega:.3f} rad/s)')


if __name__ == '__main__':
    main()
