import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

# 定义简谐振子的微分方程组
def harmonic_oscillator_ode(state, t, omega=1.0):
    """
    简谐振子的一阶微分方程组。
    
    参数:
        state: 当前状态，包含位置x和速度v
        t: 时间（未使用）
        omega: 角频率，默认为1.0
    
    返回:
        微分方程的导数向量 [dx/dt, dv/dt]
    """
    x, v = state  # 解包状态向量为位置和速度
    dxdt = v  # 位置的变化率是速度
    dvdt = -omega**2 * x  # 速度的变化率由回复力决定
    return np.array([dxdt, dvdt])  # 返回导数向量

# 定义非谐振子的微分方程组
def anharmonic_oscillator_ode(state, t, omega=1.0):
    """
    非谐振子的一阶微分方程组。
    
    参数:
        state: 当前状态，包含位置x和速度v
        t: 时间（未使用）
        omega: 角频率，默认为1.0
    
    返回:
        微分方程的导数向量 [dx/dt, dv/dt]
    """
    x, v = state  # 解包状态向量为位置和速度
    dxdt = v  # 位置的变化率是速度
    dvdt = -omega**2 * x**3  # 非线性回复力导致速度的变化率与位置的立方成正比
    return np.array([dxdt, dvdt])  # 返回导数向量

# 实现四阶龙格-库塔方法进行数值积分
def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: 微分方程函数
        state: 当前状态
        t: 当前时间
        dt: 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        下一步的状态
    """
    k1 = ode_func(state, t, **kwargs)  # 第一个斜率估计
    k2 = ode_func(state + 0.5*dt*k1, t + 0.5*dt, **kwargs)  # 第二个斜率估计（中点）
    k3 = ode_func(state + 0.5*dt*k2, t + 0.5*dt, **kwargs)  # 第三个斜率估计（中点）
    k4 = ode_func(state + dt*k3, t + dt, **kwargs)  # 第四个斜率估计（终点）
    # 使用四个斜率的加权平均更新状态
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# 求解常微分方程组的函数
def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: 微分方程函数
        initial_state: 初始状态
        t_span: 时间范围 (t_start, t_end)
        dt: 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        时间点数组和状态数组
    """
    t_start, t_end = t_span  # 获取时间范围
    t = np.arange(t_start, t_end + dt, dt)  # 创建时间点数组
    states = np.zeros((len(t), len(initial_state)))  # 初始化状态数组
    states[0] = initial_state  # 设置初始状态
    
    # 使用RK4方法逐步求解
    for i in range(1, len(t)):
        states[i] = rk4_step(ode_func, states[i-1], t[i-1], dt, **kwargs)
    
    return t, states  # 返回时间和状态数组

# 绘制状态随时间的演化图
def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: 时间点数组
        states: 状态数组
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.plot(t, states[:, 0], label='Position x(t)')  # 绘制位置随时间变化
    plt.plot(t, states[:, 1], label='Velocity v(t)')  # 绘制速度随时间变化
    plt.xlabel('Time t')  # 设置X轴标签
    plt.ylabel('State Variables')  # 设置Y轴标签
    plt.title(title)  # 设置图表标题
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例
    plt.show()  # 显示图形

# 绘制相空间轨迹图
def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: 状态数组
        title: 图表标题
    """
    plt.figure(figsize=(8, 8))  # 设置图形大小
    plt.plot(states[:, 0], states[:, 1])  # 绘制速度vs位置
    plt.xlabel('Position x')  # 设置X轴标签
    plt.ylabel('Velocity v')  # 设置Y轴标签
    plt.title(title)  # 设置图表标题
    plt.grid(True)  # 显示网格
    plt.axis('equal')  # 保持比例
    plt.show()  # 显示图形

# 分析振动周期的函数
def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。
    
    参数:
        t: 时间点数组
        states: 状态数组
    
    返回:
        估计的振动周期
    """
    x = states[:, 0]  # 获取位置数据
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(t[i])  # 记录位置极大值对应的时间
    
    if len(peaks) < 2:  # 如果极大值点不足两个，无法计算周期
        return np.nan
    
    # 计算相邻极大值之间的时间差的平均值
    periods = np.diff(peaks)
    return np.mean(periods)

# 主函数，执行所有任务
def main():
    # 设置参数
    omega = 1.0  # 角频率
    t_span = (0, 50)  # 时间范围
    dt = 0.01  # 时间步长
    
    # 任务1：简谐振子的数值求解
    print("任务1：简谐振子的数值求解")
    initial_state = np.array([1.0, 0.0])  # 初始状态：位置1.0，速度0.0
    t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_time_evolution(t, states, 'Time Evolution of Harmonic Oscillator')
    # 图片内容：显示简谐振子的位置和速度随时间变化的曲线，位置为正弦波，速度为余弦波
    period = analyze_period(t, states)
    print(f'Harmonic Oscillator Period: {period:.4f} (Theoretical: {2*np.pi/omega:.4f})')
    
    # 任务2：简谐振子振幅对周期的影响分析
    print("\n任务2：简谐振子振幅对周期的影响分析")
    amplitudes = [0.5, 1.0, 2.0]  # 不同的初始振幅
    periods = []
    
    for A in amplitudes:
        initial_state = np.array([A, 0.0])  # 设置初始状态
        t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)  # 分析周期
        periods.append(period)
        print(f'Amplitude {A}: Period = {period:.4f}')
    
    # 任务3：非谐振子的数值分析
    print("\n任务3：非谐振子的数值分析")
    for A in amplitudes:
        initial_state = np.array([A, 0.0])  # 设置初始状态
        t, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)  # 分析周期
        print(f'Anharmonic Oscillator - Amplitude {A}: Period = {period:.4f}')
        plot_time_evolution(t, states, f'Time Evolution of Anharmonic Oscillator (Amplitude={A})')
        # 图片内容：显示非谐振子的位置和速度随时间变化的曲线，波形非正弦，呈现非线性特征
    
    # 任务4：相空间分析
    print("\n任务4：相空间分析")
    initial_state = np.array([1.0, 0.0])  # 初始状态：位置1.0，速度0.0
    t, states_harmonic = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_harmonic, 'Phase Space Trajectory of Harmonic Oscillator')
    # 图片内容：显示简谐振子的相空间轨迹，为完美椭圆
    
    t, states_anharmonic = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    plot_phase_space(states_anharmonic, 'Phase Space Trajectory of Anharmonic Oscillator')
    # 图片内容：显示非谐振子的相空间轨迹，为变形的闭合曲线

if __name__ == "__main__":
    main()  # 执行主函数
