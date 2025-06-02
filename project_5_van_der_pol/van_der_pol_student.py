import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
plt.rcParams['font.sans-serif']=['KaiTi']

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """
    van der Pol振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间
        mu: float, 非线性阻尼参数
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dx_dt = v
    dv_dt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dx_dt, dv_dt])

def rk4_step(ode_func: Callable, t: float, state: np.ndarray, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        t: float, 当前时间
        state: np.ndarray, 当前状态
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    # 调整参数顺序，确保正确传递给ode_func
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = ode_func(state + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = ode_func(state + dt * k3, t + dt, **kwargs)
    
    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float],
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。

    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数

    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt) + 1
    t_points = np.linspace(t_start, t_end, num_steps)
    state_dim = len(initial_state)

    states = np.zeros((num_steps, state_dim))
    states[0] = initial_state

    for i in range(num_steps - 1):
        states[i + 1] = rk4_step(ode_func, states[i], t_points[i], dt, **kwargs)

    return t_points, states


def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。

    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], 'b-', label='位置 x')
    plt.plot(t, states[:, 1], 'r-', label='速度 v')
    plt.title(title)
    plt.xlabel('时间 t')
    plt.ylabel('状态')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。

    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], 'g-')
    plt.title(title)
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """
    计算van der Pol振子的能量。

    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        omega: float, 角频率

    返回:
        float: 系统的能量
    """
    x, v = state
    return 0.5 * (v ** 2 + omega ** 2 * x ** 2)


def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）。

    参数:
        states: np.ndarray, 状态数组

    返回:
        Tuple[float, float]: (振幅, 周期)
    """
    x_values = states[:, 0]
    max_amplitude = np.max(np.abs(x_values))

    # 寻找峰值来估计周期
    peaks = []
    for i in range(1, len(x_values) - 1):
        if x_values[i] > x_values[i - 1] and x_values[i] > x_values[i + 1]:
            peaks.append(i)

    if len(peaks) >= 2:
        periods = np.diff(peaks)
        avg_period = np.mean(periods)
    else:
        avg_period = np.nan

    return max_amplitude, avg_period


def main():
    # 设置基本参数
    omega = 1.0
    t_span = (0, 20)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])

    # 任务1 - 基本实现
    mu = 1.0
    t_points, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t_points, states, f'van der Pol方程时间演化 (μ={mu})')

    # 任务2 - 参数影响分析
    mu_values = [1.0, 2.0, 4.0]
    amplitudes = []
    periods = []

    plt.figure(figsize=(12, 8))
    for i, mu in enumerate(mu_values):
        _, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        amplitude, period = analyze_limit_cycle(states)
        amplitudes.append(amplitude)
        periods.append(period)

        plt.subplot(2, 2, i + 1)
        plt.plot(states[:, 0], states[:, 1], label=f'μ={mu}')
        plt.title(f'相空间轨迹 (μ={mu})')
        plt.xlabel('位置 x')
        plt.ylabel('速度 v')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

    # 打印参数分析结果
    print("参数影响分析:")
    for mu, amp, per in zip(mu_values, amplitudes, periods):
        print(f"μ={mu}: 振幅={amp:.3f}, 周期={per:.3f}")

    # 任务3 - 不同初始条件的相空间分析
    plt.figure(figsize=(10, 10))
    initial_conditions = [
        (1.0, 0.0),  # 初始条件1
        (2.0, 0.0),  # 初始条件2
        (0.0, 1.0),  # 初始条件3
        (-1.0, 0.5)  # 初始条件4
    ]

    mu = 2.0  # 选择一个中等非线性参数值
    for ic in initial_conditions:
        ic_state = np.array(ic)
        _, states = solve_ode(van_der_pol_ode, ic_state, t_span, dt, mu=mu, omega=omega)
        plt.plot(states[:, 0], states[:, 1], label=f'初始条件: x={ic[0]}, v={ic[1]}')

    plt.title(f'不同初始条件下的相空间轨迹 (μ={mu})')
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 任务4 - 能量分析
    mu = 2.0
    t_points, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)

    energies = np.array([calculate_energy(state, omega) for state in states])

    plt.figure(figsize=(10, 6))
    plt.plot(t_points, energies, 'm-')
    plt.title(f'系统能量随时间的变化 (μ={mu})')
    plt.xlabel('时间 t')
    plt.ylabel('能量 E')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
