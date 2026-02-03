
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver
from src.solver.sho_solver import SHOSolver
import os

class InteractiveArm:
    def __init__(self):
        self.arm = Arm2DOF()
        self.mpc = MPCBuilder(self.arm, N=15)
        
        # Solvers
        self.osqp_solver = OSQPSolver()
        self.sho_solver = SHOSolver(n_bits=4, rho=100.0) 
        self.current_solver_name = 'OSQP'
        
        self.dt = 0.05
        self.reset()
        
        self.setup_plot()
        
        # Interaction state
        self.paused = False

    def reset(self):
        self.x = np.array([0.0, 0.0, 0.0, 0.0])
        self.target_pos = np.array([0.5, 0.5])
        self.target_theta_guess = np.array([0.0, 0.0])

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title("Click to set goal! Arm follows via MPC.")
        
        self.line, = self.ax.plot([], [], '-o', lw=4, markersize=8, color='blue')
        self.goal_dot, = self.ax.plot([], [], 'rx', ms=12, mew=3)
        self.status_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes)
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # UI Buttons
        ax_reset = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(lambda event: self.reset())
        
        ax_radio = plt.axes([0.05, 0.02, 0.15, 0.15], facecolor='#e0e0e0')
        self.radio = RadioButtons(ax_radio, ('OSQP', 'SHO'))
        self.radio.on_clicked(self.change_solver)

    def change_solver(self, label):
        self.current_solver_name = label

    def on_click(self, event):
        if event.inaxes != self.ax: return
        self.target_pos = np.array([event.xdata, event.ydata])
        self.goal_dot.set_data([self.target_pos[0]], [self.target_pos[1]])
        self.fig.canvas.draw_idle()
        
    def solve_ik_crude(self, pos):
        x, y = pos
        l1, l2 = self.arm.l1, self.arm.l2
        d2 = x**2 + y**2
        
        # Reachability check
        if d2 > (l1+l2)**2:
            d2 = (l1+l2)**2 - 1e-4
            scale = np.sqrt(d2 / (x**2 + y**2))
            x *= scale
            y *= scale
            
        c2 = (d2 - l1**2 - l2**2) / (2 * l1 * l2)
        c2 = np.clip(c2, -1, 1)
        s2 = np.sqrt(1 - c2**2)
        # Elbow down/up solution (use one fixed for now)
        th2 = np.arctan2(s2, c2)
        
        k1 = l1 + l2 * c2
        k2 = l2 * s2
        th1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        return np.array([th1, th2, 0, 0])

    def update(self, i):
        # 1. Check stability
        if np.any(np.abs(self.x) > 100):
            self.reset()
            self.status_text.set_text("Unstable! Resetting...")
            return self.line, self.goal_dot, self.status_text

        # 2. Get theta goal
        x_goal = self.solve_ik_crude(self.target_pos)
        
        # 3. Build MPC
        ref_traj = self.mpc.build_reference_trajectory(self.x, x_goal)
        qp_matrices = self.mpc.build_qp(self.x, ref_traj)
        
        # 4. Solve
        u = np.zeros(self.arm.nu)
        try:
            if self.current_solver_name == 'OSQP':
                z = self.osqp_solver.solve(qp_matrices)
            else:
                z = self.sho_solver.solve(qp_matrices, x_min_val=-5.0, x_max_val=5.0)
                
            if z is not None:
                u = z[self.arm.nx : self.arm.nx + self.arm.nu]
            else:
                print("Solver failed.")
                
        except Exception as e:
            print(f"Solver Error: {e}")
            u = np.zeros(self.arm.nu) # Fallback to zero

        # Safety clamp
        u = np.clip(u, self.mpc.tau_min, self.mpc.tau_max)

        # 5. Integrate
        self.x = self.arm.step_dynamics(self.x, u, self.dt)

        # 6. Draw
        pts = self.arm.forward_kinematics(self.x[:2])
        self.line.set_data(pts[0], pts[1])
        
        # Info
        pos_err = np.linalg.norm(pts[:, -1] - self.target_pos)
        self.status_text.set_text(f"Solver: {self.current_solver_name} | Err: {pos_err:.3f}")
        
        return self.line, self.goal_dot, self.status_text

    def run(self):
        from matplotlib.animation import FuncAnimation
        self.ani = FuncAnimation(self.fig, self.update, interval=50, blit=False) # blit=False for text update
        plt.show()

if __name__ == "__main__":
    app = InteractiveArm()
    app.run()
