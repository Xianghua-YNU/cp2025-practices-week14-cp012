#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
洛伦兹方程求解与混沌系统分析
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from typing import Tuple, List, Union


class LorenzSystem:
    """洛伦兹系统类，封装方程定义与求解逻辑"""
    def __init__(self, sigma: float = 10.0, r: float = 28.0, b: float = 8/3):
        """
        初始化系统参数
        
        参数:
            sigma: 普朗特数相关参数 (默认10)
            r: 瑞利数相关参数 (默认28)
            b: 几何参数 (默认8/3)
        """
        self.sigma = sigma
        self.r = r
        self.b = b

    def get_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        计算洛伦兹方程的导数
        
        参数:
            t: 当前时间（需保留，尽管未使用）
            state: 状态向量 [x, y, z]
            
        返回:
            导数向量 [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        dx_dt = self.sigma * (y - x)
        dy_dt = self.r * x - y - x * z
        dz_dt = x * y - self.b * z
        return np.array([dx_dt, dy_dt, dz_dt])

    def solve(self, 
              initial_state: Union[List[float], np.ndarray] = [0.1, 0.1, 0.1],
              t_span: Tuple[float, float] = (0, 50),
              dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        数值求解洛伦兹方程
        
        参数:
            initial_state: 初始状态向量 [x0, y0, z0]
            t_span: 时间区间 (t_start, t_end)
            dt: 输出时间步长
            
        返回:
            t: 时间点数组
            y: 状态解数组 (形状为(3, n_points))
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)
        solution = solve_ivp(
            fun=self.get_derivatives,
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method='RK45',
            atol=1e-6,
            rtol=1e-6
        )
        return solution.t, solution.y


class LorenzVisualizer:
    """洛伦兹系统可视化类"""
    @staticmethod
    def plot_attractor(t: np.ndarray, y: np.ndarray, title: str = "洛伦兹吸引子") -> None:
        """
        绘制3D吸引子轨迹
        
        参数:
            t: 时间点数组
            y: 状态解数组
            title: 图像标题
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        ax.plot(y[0], y[1], y[2], lw=0.6, color='#1f77b4', alpha=0.9)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(False)
        
        plt.show()

    @staticmethod
    def compare_trajectories(t: np.ndarray, 
                             y1: np.ndarray, 
                             y2: np.ndarray, 
                             title: str = "初始条件敏感性分析") -> None:
        """
        对比不同初始条件的轨迹差异
        
        参数:
            t: 共享时间数组
            y1: 状态解数组1
            y2: 状态解数组2
            title: 图像标题
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 绘制各分量对比
        ax1.plot(t, y1[0], 'b-', label='X1', alpha=0.7)
        ax1.plot(t, y2[0], 'r--', label='X2', alpha=0.7)
        ax1.plot(t, y1[1], 'g-', label='Y1', alpha=0.7)
        ax1.plot(t, y2[1], 'm--', label='Y2', alpha=0.7)
        ax1.set_xlabel('时间', fontsize=12)
        ax1.set_ylabel('状态值', fontsize=12)
        ax1.set_title('状态分量对比', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制轨迹距离（对数尺度）
        distance = np.linalg.norm(y1 - y2, axis=0)
        ax2.plot(t, distance, 'k-', alpha=0.9)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_ylabel('欧氏距离（对数尺度）', fontsize=12)
        ax2.set_title('轨迹分离速度', fontsize=14)
        ax2.set_yscale('log')
        ax2.grid(True, which='both', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()


# 保留顶层函数引用，保持与测试用例的兼容性
def lorenz_system(state, sigma=10.0, r=28.0, b=8/3):
    """兼容旧版接口的洛伦兹系统函数"""
    return LorenzSystem(sigma, r, b).get_derivatives(0, state)

def solve_lorenz_equations(sigma=10.0, r=28.0, b=8/3,
                          x0=0.1, y0=0.1, z0=0.1,
                          t_span=(0, 50), dt=0.01):
    """兼容旧版接口的求解函数"""
    system = LorenzSystem(sigma, r, b)
    return system.solve([x0, y0, z0], t_span, dt)


def main():
    """主函数，执行系统求解与分析"""
    # 初始化系统
    lorenz = LorenzSystem()
    
    # 任务A: 求解标准初始条件
    t, y = lorenz.solve()
    
    # 任务B: 绘制吸引子
    LorenzVisualizer.plot_attractor(t, y)
    
    # 任务C: 初始条件敏感性测试
    ic1 = [0.1, 0.1, 0.1]
    ic2 = [0.1 + 1e-5, 0.1, 0.1]  # 微小扰动
    
    # 求解两个初始条件
    t1, y1 = lorenz.solve(initial_state=ic1)
    t2, y2 = lorenz.solve(initial_state=ic2)
    
    # 确保时间数组一致（分别对每个维度进行插值）
    y2_common = np.array([np.interp(t1, t2, y2[i]) for i in range(3)])
    
    # 对比分析
    LorenzVisualizer.compare_trajectories(t1, y1, y2_common)


if __name__ == '__main__':
    main()
