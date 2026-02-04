
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
        self.dt = 0.02
        self.arm = Arm2DOF()
        self.mpc = MPCBuilder(self.arm, N=20, dt=self.dt)
        
        # Solvers
        self.osqp_solver = OSQPSolver()
        self.sho_solver = SHOSolver(n_bits=4, rho=100.0) 
        self.current_solver_name = 'OSQP'
        
        self.reset()
        
        # PD Control gains for fallback
        self.Kp = np.array([50.0, 30.0])
        self.Kd = np.array([10.0, 5.0])
        
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
        # Coordinate shift: forward_kinematics uses y negative for down.
        # But for IK we usually assume standard quadrants. 
        # Actually our FK has y1 = -l1*cos(th1), x1 = l1*sin(th1).
        # This is polar with th=0 being straight down.
        # Let's adjust IK to match this convention.
        
        l1, l2 = self.arm.l1, self.arm.l2
        
        # In our FK:
        # x = l1*sin(th1) + l2*sin(th1+th2)
        # y = -l1*cos(th1) - l2*cos(th1+th2)
        
        # Rotate to standard (x_std, y_std) where th1=0 is x-axis
        # x_std = l1*cos(th1_std) + l2*cos(th1_std+th2_std)
        # y_std = l1*sin(th1_std) + l2*sin(th1_std+th2_std)
        # Mapping: x_std = -y, y_std = x
        
        x_std, y_std = -y, x
        d2 = x_std**2 + y_std**2
        
        if d2 > (l1+l2)**2:
            d2 = (l1+l2)**2 - 1e-4
            scale = np.sqrt(d2 / (x_std**2 + y_std**2))
            x_std *= scale
            y_std *= scale
            
        c2 = (d2 - l1**2 - l2**2) / (2 * l1 * l2)
        c2 = np.clip(c2, -1, 1)
        s2 = np.sqrt(1 - c2**2) # elbow up
        th2 = np.arctan2(s2, c2)
        
        k1 = l1 + l2 * c2
        k2 = l2 * s2
        th1_std = np.arctan2(y_std, x_std) - np.arctan2(k2, k1)
        
        # Convert back: th1 = th1_std
        return np.array([th1_std, th2, 0, 0])

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
                solver_status = "OPTIMAL"
            else:
                # PD Fallback with Gravity Comp
                theta_err = x_goal[:2] - self.x[:2]
                # Shortest path for angles
                theta_err = (theta_err + np.pi) % (2 * np.pi) - np.pi
                dtheta_err = x_goal[2:] - self.x[2:]
                
                # Gravity comp
                m1, m2, l1, l2, g = self.arm.m1, self.arm.m2, self.arm.l1, self.arm.l2, self.arm.g
                th1, th2 = self.x[0], self.x[1]
                G1 = (m1*l1/2 + m2*l1)*g*np.sin(th1) + m2*l2/2*g*np.sin(th1+th2)
                G2 = m2*l2/2*g*np.sin(th1+th2)
                
                u = self.Kp * theta_err + self.Kd * dtheta_err + np.array([G1, G2])
                solver_status = "FALLBACK"
                
        except Exception as e:
            print(f"Control Loop Error: {e}")
            u = np.zeros(self.arm.nu)
            solver_status = "ERROR"

        # Safety clamp
        u = np.clip(u, self.mpc.tau_min, self.mpc.tau_max)

        # 5. Integrate
        self.x = self.arm.step_dynamics(self.x, u, self.dt)

        # 6. Draw
        pts = self.arm.forward_kinematics(self.x[:2])
        self.line.set_data(pts[0], pts[1])
        
        # Info
        pos_err = np.linalg.norm(pts[:, -1] - self.target_pos)
        self.status_text.set_text(f"Solver: {self.current_solver_name} | Status: {solver_status} | Err: {pos_err:.3f}")
        
        return self.line, self.goal_dot, self.status_text

    def run(self):
        from matplotlib.animation import FuncAnimation
        self.ani = FuncAnimation(self.fig, self.update, interval=50, blit=False) # blit=False for text update
        plt.show()

if __name__ == "__main__":
    app = InteractiveArm()
    app.run()
