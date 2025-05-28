import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dxdt = v
    dvdt = -omega**2 * x
    return np.array([dxdt, dvdt])

def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    dxdt = v
    dvdt = -omega**2 * x**3
    return np.array([dxdt, dvdt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state np:.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + dt/2 * k1, t + dt/2, **kwargs)
    k3 = ode_func(state + dt/2 * k2, t + dt/2, **kwargs)
    k4 = ode_func(state + dt * k3, t + dt, **kwargs)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

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
    t = np.linspace(t_start, t_end, num_steps)
    states = np.zeros((num_steps, len(initial_state)))
    states[0] = initial_state
    
    for i in range(1, num_steps):
        states[i] = rk4_step(ode_func, states[i-1], t[i-1], dt, **kwargs)
    
    return t, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='位移 x(t)')
    plt.plot(t, states[:, 1], label='速度 v(t)', linestyle='--')
    plt.title(title)
    plt.xlabel('时间 t')
    plt.ylabel('位移 x 和速度 v')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])
    plt.title(title)
    plt.xlabel('位移 x')
    plt.ylabel('速度 v')
    plt.grid(False)
    plt.axis('equal')
    plt.show()

def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
    
    返回:
        float: 估计的振动周期
    """
    displacement = states[:, 0]
    peaks, _ = find_peaks(displacement)
    
    if len(peaks) < 2:
        return None
    
    periods = t[peaks[1:]] - t[peaks[:-1]]
    return np.mean(periods)

def main():
    # 设置参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    
    # 任务1：简谐振子的数值求解
    print("任务1：简谐振子的数值求解")
    initial_state_harm = np.array([1.0, 0.0])
    t_harm, states_harm = solve_ode(harmonic_oscillator_ode, initial_state_harm, t_span, dt, omega=omega)
    plot_time_evolution(t_harm, states_harm, '简谐振子的位移和速度随时间变化')
    plot_phase_space(states_harm, '简谐振子的相空间轨迹')
    
    # 任务2：振幅对周期的影响分析（简谐振子）
    print("\n任务2：简谐振子振幅对周期的影响分析")
    amplitudes = [1.0, 2.0, 3.0, 4.0, 5.0]
    periods_harm = []
    
    for amp in amplitudes:
        initial_state = np.array([amp, 0.0])
        _, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t_harm, states)
        periods_harm.append(period)
        print(f"振幅 {amp}: 周期 {period:.4f}")
    
    # 绘制简谐振子振幅-周期关系图
    plt.figure(figsize=(10, 6))
    plt.plot(amplitudes, periods_harm, 'o-', label='简谐振子')
    plt.title('简谐振子振幅与周期关系')
    plt.xlabel('振幅')
    plt.ylabel('周期')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 任务3：非谐振子的数值分析
    print("\n任务3：非谐振子的数值分析")
    initial_state_anharm = np.array([1.0, 0.0])
    t_anharm, states_anharm = solve_ode(anharmonic_oscillator_ode, initial_state_anharm, t_span, dt, omega=omega)
    plot_time_evolution(t_anharm, states_anharm, '非谐振子的位移和速度随时间变化')
    plot_phase_space(states_anharm, '非谐振子的相空间轨迹')
    
    分 #析非谐振子振幅对周期的影响
    print("\n非谐振子振幅对周期的影响分析")
    amplitudes_anharm = [0.5, 1.0, 2.0, 3.0, 4.0]
    periods_anharm = []
    
    for amp in amplitudes_anharm:
        initial_state = np.array([amp, 0.0])
        _, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t_anharm, states)
        periods_anharm.append(period)
        print(f"振幅 {amp}: 周期 {period:.4f}")
    
    # 绘制非谐振子振幅-周期关系图
    plt.figure(figsize=(10, 6))
    plt.plot(amplitudes_anharm, periods_anharm, 'o-', label='非谐振子')
    plt.title('非谐振子振幅与周期关系')
    plt.xlabel('振幅')
    plt.ylabel('周期')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 任务4：相空间分析
    print("\n任务4：相空间分析")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_phase_space(states_harm, '简谐振子相空间轨迹')
    
    plt.subplot(1, 2, 2)
    plot_phase_space(states_anharm, '非谐振子相空间轨迹')
    plt.suptitle('简谐振子与非谐振子相空间轨迹对比')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
