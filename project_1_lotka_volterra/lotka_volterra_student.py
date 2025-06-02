#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 - 学生代码模板

学生姓名：[段林焱]
学号：[20231050098]
完成日期：[2025/5/28]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def lotka_volterra_system(state: np.ndarray, t: float, alpha: float, beta: float,
                          gamma: float, delta: float) -> np.ndarray:
    """
    Lotka-Volterra方程组的右端函数

    方程组：
    dx/dt = α*x - β*x*y  (猎物增长率 - 被捕食率)
    dy/dt = γ*x*y - δ*y  (捕食者增长率 - 死亡率)

    参数:
        state: np.ndarray, 形状为(2,), 当前状态向量 [x, y]
        t: float, 时间（本系统中未显式使用，但保持接口一致性）
        alpha: float, 猎物自然增长率
        beta: float, 捕食效率
        gamma: float, 捕食者从猎物获得的增长效率
        delta: float, 捕食者自然死亡率

    返回:
        np.ndarray, 形状为(2,), 导数向量 [dx/dt, dy/dt]
    """
    x, y = state

    # 计算猎物和捕食者的导数
    dxdt = alpha * x - beta * x * y  # 猎物增长率减去被捕食率
    dydt = gamma * x * y - delta * y  # 捕食者因捕食增长减去自然死亡

    return np.array([dxdt, dydt])


def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float],
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    欧拉法求解常微分方程组

    参数:
        f: 微分方程组的右端函数，签名为 f(y, t, *args)
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        *args: 传递给f的额外参数

    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组，形状为 (len(t), len(y0))
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)

    y = np.zeros((n_steps, n_vars))
    y[0] = y0

    # 使用欧拉法迭代更新解
    for i in range(n_steps - 1):
        y[i + 1] = y[i] + dt * f(y[i], t[i], *args)  # y_{n+1} = y_n + dt * f(y_n, t_n)

    return t, y


def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float],
                          dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    改进欧拉法（2阶Runge-Kutta法）求解常微分方程组

    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
        *args: 传递给f的额外参数

    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)

    y = np.zeros((n_steps, n_vars))
    y[0] = y0

    # 使用改进欧拉法迭代
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)  # 第一步斜率
        k2 = dt * f(y[i] + k1, t[i] + dt, *args)  # 第二步斜率
        y[i + 1] = y[i] + (k1 + k2) / 2  # 取平均值更新

    return t, y


def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float],
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    4阶龙格-库塔法求解常微分方程组

    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        *args: 传递给f的额外参数

    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组，形状为 (len(t), len(y0))
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)

    y = np.zeros((n_steps, n_vars))
    y[0] = y0

    # 使用4阶龙格-库塔法迭代
    for i in range(n_steps - 1):
        k1 = dt * f(y[i], t[i], *args)  # 第一步斜率
        k2 = dt * f(y[i] + k1 / 2, t[i] + dt / 2, *args)  # 第二步斜率
        k3 = dt * f(y[i] + k2 / 2, t[i] + dt / 2, *args)  # 第三步斜率
        k4 = dt * f(y[i] + k3, t[i] + dt, *args)  # 第四步斜率
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # 加权平均更新

    return t, y


def solve_lotka_volterra(alpha: float, beta: float, gamma: float, delta: float,
                         x0: float, y0: float, t_span: Tuple[float, float],
                         dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用4阶龙格-库塔法求解Lotka-Volterra方程组

    参数:
        alpha: float, 猎物自然增长率
        beta: float, 捕食效率
        gamma: float, 捕食者从猎物获得的增长效率
        delta: float, 捕食者自然死亡率
        x0: float, 初始猎物数量
        y0: float, 初始捕食者数量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长

    返回:
        t: np.ndarray, 时间数组
        x: np.ndarray, 猎物种群数量数组
        y: np.ndarray, 捕食者种群数量数组
    """
    # 构造初始条件向量
    y0_vec = np.array([x0, y0])
    # 调用4阶龙格-库塔法求解
    t, y = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    # 提取猎物和捕食者数量
    x, y = y[:, 0], y[:, 1]

    return t, x, y


def compare_methods(alpha: float, beta: float, gamma: float, delta: float,
                    x0: float, y0: float, t_span: Tuple[float, float],
                    dt: float) -> dict:
    """
    比较三种数值方法求解Lotka-Volterra方程组

    参数:
        alpha, beta, gamma, delta: 模型参数
        x0, y0: 初始条件
        t_span: 时间范围
        dt: 时间步长

    返回:
        dict: 包含三种方法结果的字典，格式为：
        {
            'euler': {'t': t_array, 'x': x_array, 'y': y_array},
            'improved_euler': {'t': t_array, 'x': x_array, 'y': y_array},
            'rk4': {'t': t_array, 'x': x_array, 'y': y_array}
        }
    """
    y0_vec = np.array([x0, y0])
    params = (alpha, beta, gamma, delta)

    # 使用欧拉法求解
    t_euler, y_euler = euler_method(lotka_volterra_system, y0_vec, t_span, dt, *params)
    # 使用改进欧拉法求解
    t_improved, y_improved = improved_euler_method(lotka_volterra_system, y0_vec, t_span, dt, *params)
    # 使用4阶龙格-库塔法求解
    t_rk4, y_rk4 = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, *params)

    # 构造结果字典
    results = {
        'euler': {'t': t_euler, 'x': y_euler[:, 0], 'y': y_euler[:, 1]},
        'improved_euler': {'t': t_improved, 'x': y_improved[:, 0], 'y': y_improved[:, 1]},
        'rk4': {'t': t_rk4, 'x': y_rk4[:, 0], 'y': y_rk4[:, 1]}
    }

    return results


def plot_population_dynamics(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                             title: str = "Lotka-Volterra种群动力学") -> None:
    """
    绘制种群动力学图

    参数:
        t: np.ndarray, 时间数组
        x: np.ndarray, 猎物种群数量
        y: np.ndarray, 捕食者种群数量
        title: str, 图标题
    """
    plt.figure(figsize=(12, 5))

    # 子图1：时间序列图
    plt.subplot(1, 2, 1)
    plt.plot(t, x, label='猎物 (x)')
    plt.plot(t, y, label='捕食者 (y)')
    plt.xlabel('时间')
    plt.ylabel('种群数量')
    plt.title('种群数量随时间变化')
    plt.legend()
    plt.grid(True)

    # 子图2：相空间轨迹图
    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.xlabel('猎物数量 (x)')
    plt.ylabel('捕食者数量 (y)')
    plt.title('相空间轨迹')
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_method_comparison(results: dict) -> None:
    """
    绘制不同数值方法的比较图

    参数:
        results: dict, compare_methods函数的返回结果
    """
    plt.figure(figsize=(15, 8))
    methods = ['euler', 'improved_euler', 'rk4']
    titles = ['欧拉法', '改进欧拉法', '4阶龙格-库塔法']

    # 上排：时间序列图
    for i, method in enumerate(methods):
        plt.subplot(2, 3, i + 1)
        plt.plot(results[method]['t'], results[method]['x'], label='猎物 (x)')
        plt.plot(results[method]['t'], results[method]['y'], label='捕食者 (y)')
        plt.xlabel('时间')
        plt.ylabel('种群数量')
        plt.title(titles[i])
        plt.legend()
        plt.grid(True)

    # 下排：相空间图
    for i, method in enumerate(methods):
        plt.subplot(2, 3, i + 4)
        plt.plot(results[method]['x'], results[method]['y'])
        plt.xlabel('猎物数量 (x)')
        plt.ylabel('捕食者数量 (y)')
        plt.title(f'{titles[i]} - 相空间')
        plt.grid(True)

    plt.suptitle('不同数值方法比较')
    plt.tight_layout()
    plt.show()


def analyze_parameters() -> None:
    """
    分析不同参数对系统行为的影响

    分析内容：
    1. 不同初始条件的影响
    2. 守恒量验证
    """
    # 基本参数
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    t_span = (0, 30)
    dt = 0.01

    # 测试不同初始条件
    initial_conditions = [(2.0, 2.0), (3.0, 1.0), (1.0, 3.0)]
    plt.figure(figsize=(12, 5))

    for i, (x0, y0) in enumerate(initial_conditions):
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)

        # 时间序列图
        plt.subplot(1, 2, 1)
        plt.plot(t, x, label=f'x0={x0}, y0={y0} (猎物)')
        plt.plot(t, y, '--', label=f'x0={x0}, y0={y0} (捕食者)')

        # 相空间图
        plt.subplot(1, 2, 2)
        plt.plot(x, y, label=f'x0={x0}, y0={y0}')

    plt.subplot(1, 2, 1)
    plt.xlabel('时间')
    plt.ylabel('种群数量')
    plt.title('不同初始条件下的种群动态')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.xlabel('猎物数量 (x)')
    plt.ylabel('捕食者数量 (y)')
    plt.title('不同初始条件下的相空间轨迹')
    plt.legend()
    plt.grid(True)

    plt.suptitle('参数分析 - 初始条件影响')
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：演示Lotka-Volterra模型的完整分析

    执行步骤：
    1. 设置参数并求解基本问题
    2. 比较不同数值方法
    3. 分析参数影响
    4. 输出数值统计结果
    """
    # 参数设置（根据题目要求）
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t_span = (0, 30)
    dt = 0.01

    print("=== Lotka-Volterra捕食者-猎物模型分析 ===")
    print(f"参数: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    print(f"初始条件: x0={x0}, y0={y0}")
    print(f"时间范围: {t_span}, 步长: {dt}")

    # 1. 基本求解
    print("\n1. 使用4阶龙格-库塔法求解...")
    t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    plot_population_dynamics(t, x, y)

    # 2. 方法比较
    print("\n2. 比较不同数值方法...")
    results = compare_methods(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    plot_method_comparison(results)

    # 3. 参数分析
    print("\n3. 分析参数影响...")
    analyze_parameters()

    # 4. 数值结果统计
    print("\n4. 数值结果统计:")
    print(f"猎物种群最大值: {max(x):.2f}, 最小值: {min(x):.2f}")
    print(f"捕食者种群最大值: {max(y):.2f}, 最小值: {min(y):.2f}")


if __name__ == "__main__":
    main()
