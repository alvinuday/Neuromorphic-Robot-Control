
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ArmAnimator:
    def __init__(self, arm_model, dt=0.02):
        self.arm = arm_model
        self.dt = dt
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        self.line, = self.ax.plot([], [], '-o', linewidth=3, color='blue')
        self.target_point, = self.ax.plot([], [], 'rx', markersize=10)
        self.time_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes)

    def init_plot(self):
        self.line.set_data([], [])
        self.target_point.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.target_point, self.time_text

    def update(self, frame, trajectory, target_traj=None):
        """
        frame: index
        trajectory: (T, state_dim) array of states
        target_traj: (T, state_dim) array of reference states (optional)
        """
        x = trajectory[frame]
        theta = x[:2]
        pts = self.arm.forward_kinematics(theta)
        
        self.line.set_data(pts[0, :], pts[1, :])
        
        if target_traj is not None:
            if frame < len(target_traj):
                x_ref = target_traj[frame]
                theta_ref = x_ref[:2]
                pts_ref = self.arm.forward_kinematics(theta_ref)
                # Just plot the end-effector of target
                self.target_point.set_data([pts_ref[0,-1]], [pts_ref[1,-1]])
        
        self.time_text.set_text(f'Time = {frame * self.dt:.2f} s')
        return self.line, self.target_point, self.time_text

    def animate(self, trajectory, target_traj=None, save_path=None):
        ani = FuncAnimation(self.fig, self.update, frames=len(trajectory),
                            fargs=(trajectory, target_traj),
                            init_func=self.init_plot, blit=True, interval=self.dt*1000)
        
        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=int(1/self.dt))
            else:
                ani.save(save_path, writer='ffmpeg', fps=int(1/self.dt))
        else:
            plt.show()

def plot_trajectory_static(trajectory, target_traj=None, dt=0.02):
    """
    Static plot of joint angles and velocities.
    """
    # trajectory: [x0, x1, ..., xT] (T+1)
    # target_traj: [r0, r1, ..., r_T-1] (T)
    T = len(target_traj) if target_traj is not None else len(trajectory)
    time = np.arange(T) * dt
    
    # Slice trajectory if needed
    traj = trajectory[:T] if len(trajectory) > T else trajectory
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Joint Angles
    axes[0].plot(time, traj[:, 0], label='q1')
    axes[0].plot(time, traj[:, 1], label='q2')
    if target_traj is not None:
        axes[0].plot(time, target_traj[:, 0], '--', label='q1_ref', alpha=0.7)
        axes[0].plot(time, target_traj[:, 1], '--', label='q2_ref', alpha=0.7)
    axes[0].set_ylabel('Angle (rad)')
    axes[0].legend()
    axes[0].set_title('Joint Angles')
    axes[0].grid(True)
    
    # Joint Velocities
    axes[1].plot(time, traj[:, 2], label='dq1')
    axes[1].plot(time, traj[:, 3], label='dq2')
    if target_traj is not None:
        axes[1].plot(time, target_traj[:, 2], '--', label='dq1_ref', alpha=0.7)
        axes[1].plot(time, target_traj[:, 3], '--', label='dq2_ref', alpha=0.7)
    axes[1].set_ylabel('Velocity (rad/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend()
    axes[1].set_title('Joint Velocities')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
