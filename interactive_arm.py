
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver
from src.solver.sho_solver import SHOSolver
import os
from collections import deque

class InteractiveArm:
    def __init__(self, g=9.81):
        self.dt = 0.02
        self.g_val = g
        self.gravity_on = (g > 0)
        self.mpc_on = True             # False → zero torque, arm swings under gravity only

        self.theta_min = np.array([-np.pi*0.75, -np.pi*0.8])
        self.theta_max = np.array([ np.pi*0.75,  np.pi*0.8])

        self._build_arm_and_mpc()

        # Solvers
        self.osqp_solver = OSQPSolver()
        self.sho_solver = SHOSolver(n_bits=8, rho=100.0)
        self.current_solver_name = 'OSQP'

        # Streak state
        self.streak_len = 10
        self.ee_streak = deque(maxlen=self.streak_len)

        self.reset()

        # PD Control gains for fallback
        self.Kp = np.array([50.0, 30.0])
        self.Kd = np.array([10.0, 5.0])

        self.setup_plot()

        # Interaction state
        self.paused = False

    def _build_arm_and_mpc(self):
        """(Re-)instantiate Arm2DOF and MPCBuilder with current gravity setting."""
        g_use = self.g_val if self.gravity_on else 0.0
        self.arm = Arm2DOF(g=g_use)
        self.mpc = MPCBuilder(self.arm, N=20, dt=self.dt, bounds={
            'theta_min': self.theta_min,
            'theta_max': self.theta_max,
            'tau_min': np.array([-50.0, -50.0]),
            'tau_max': np.array([ 50.0,  50.0])
        })

    def reset(self):
        self.x = np.array([0.0, 0.0, 0.0, 0.0])
        self.target_pos = np.array([0.5, 0.5])
        self.target_theta_guess = np.array([0.0, 0.0])
        try:
            self.ee_streak.clear()
        except AttributeError:
            pass  # not yet created

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        plt.subplots_adjust(bottom=0.22)

        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self._update_title()

        self.line, = self.ax.plot([], [], '-o', lw=4, markersize=8, color='blue', zorder=5)
        self.goal_dot, = self.ax.plot([], [], 'rx', ms=12, mew=3, zorder=6)

        # Streak plot (end-effector trail)
        self.streak_plots = [
            self.ax.plot([], [], 'o', color='blue', alpha=(i+1)/self.streak_len, markersize=4)[0]
            for i in range(self.streak_len)
        ]

        self.status_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontweight='bold')

        self._plot_workspace_boundary()

        self.ax.text(
            0.02, 0.02,
            f"Constraints:\nTh1: [{np.rad2deg(self.theta_min[0]):.0f}, {np.rad2deg(self.theta_max[0]):.0f}]°\n"
            f"Th2: [{np.rad2deg(self.theta_min[1]):.0f}, {np.rad2deg(self.theta_max[1]):.0f}]°",
            transform=self.ax.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # ── Solver radio buttons ──────────────────────────────────────
        ax_radio = plt.axes([0.05, 0.02, 0.12, 0.14], facecolor='#e0e0e0')
        self.radio = RadioButtons(ax_radio, ('OSQP', 'SHO'))
        self.radio.on_clicked(self.change_solver)

        # ── Reset button ──────────────────────────────────────────────
        ax_reset = plt.axes([0.82, 0.05, 0.08, 0.065])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(lambda e: self.reset())

        # ── Gravity toggle ────────────────────────────────────────────
        ax_grav = plt.axes([0.22, 0.05, 0.12, 0.065])
        lbl_grav = 'Grav ON' if self.gravity_on else 'Grav OFF'
        self.btn_gravity = Button(ax_grav, lbl_grav,
                                  color='#d4f0d4' if self.gravity_on else '#f0d4d4')
        self.btn_gravity.on_clicked(self.toggle_gravity)

        # ── MPC ON/OFF toggle ─────────────────────────────────────────
        ax_mpc = plt.axes([0.36, 0.05, 0.10, 0.065])
        self.btn_mpc = Button(ax_mpc, 'MPC' if self.mpc_on else 'No MPC',
                             color='#d4e4f0' if self.mpc_on else '#f0e4d4')
        self.btn_mpc.on_clicked(self.toggle_mpc)

    def _update_title(self):
        g_str = 'Gravity' if self.gravity_on else 'No gravity'
        mpc_str = 'MPC' if self.mpc_on else 'No MPC'
        self.ax.set_title(f'Click to set goal  |  {g_str}  |  {mpc_str}')

    def toggle_gravity(self, event):
        self.gravity_on = not self.gravity_on
        self._build_arm_and_mpc()
        self.btn_gravity.label.set_text('Grav ON' if self.gravity_on else 'Grav OFF')
        self.btn_gravity.color = '#d4f0d4' if self.gravity_on else '#f0d4d4'
        self.btn_gravity.hovercolor = '#c0e8c0' if self.gravity_on else '#e8c0c0'
        self._update_title()
        # Do not reset — arm keeps current state, just physics (g) changes
        self.fig.canvas.draw_idle()
        print(f'Gravity → {"ON" if self.gravity_on else "OFF"} (g={self.arm.g}), arm not reset')

    def toggle_mpc(self, event):
        self.mpc_on = not self.mpc_on
        self.btn_mpc.label.set_text('MPC' if self.mpc_on else 'No MPC')
        self.btn_mpc.color = '#d4e4f0' if self.mpc_on else '#f0e4d4'
        self._update_title()
        # Do not reset — arm keeps current state; No MPC = zero torque, natural swing
        self.fig.canvas.draw_idle()
        print(f'MPC → {"ON" if self.mpc_on else "OFF (zero torque)"}')

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
        
        # 1. Check if d2 is feasible for arm length
        if d2 > (l1+l2)**2:
            d2 = (l1+l2)**2 - 1e-4
            scale = np.sqrt(d2 / (x_std**2 + y_std**2))
            x_std *= scale
            y_std *= scale
            
        c2 = (d2 - l1**2 - l2**2) / (2 * l1 * l2)
        c2 = np.clip(c2, -1, 1)
        s2 = np.sqrt(1 - c2**2) # elbow up
        th2_std = np.arctan2(s2, c2)
        
        k1 = l1 + l2 * c2
        k2 = l2 * s2
        th1_std = np.arctan2(y_std, x_std) - np.arctan2(k2, k1)
        
        # Convert to our convention: th2 is the relative angle
        th1 = th1_std
        th2 = th2_std
        
        # 2. Clip to joint constraints
        th1 = np.clip(th1, self.theta_min[0], self.theta_max[0])
        th2 = np.clip(th2, self.theta_min[1], self.theta_max[1])
        
        # 3. Final target pos update to match clipped IK (optional but helpful for visual)
        # self.target_pos = self.arm.forward_kinematics([th1, th2])[:, -1]
        
        return np.array([th1, th2, 0, 0])

    def _plot_workspace_boundary(self):
        """Pre-compute and plot the reachable workspace based on joint limits."""
        l1, l2 = self.arm.l1, self.arm.l2
        
        # Sample joint limits densely to find the boundary
        # The workspace is defined by theta1 in [min, max] and theta2 in [min, max]
        
        # Inner and outer arcs are defined by th2 fixed at min/max/zero
        # But specifically, for each th1, the range of reachable positions is a circular arc of th2.
        
        res = 60
        th1s = np.linspace(self.theta_min[0], self.theta_max[0], res)
        
        # Outer boundary (th2 = 0 if within bounds)
        th2_outer = 0.0 if (self.theta_min[1] <= 0 <= self.theta_max[1]) else self.theta_min[1]
        
        # We need to trace the perimeter.
        # 1. Sweep th1 at th2_outer
        # 2. Sweep th2 at th1_max
        # 3. Sweep th1 (rev) at th2_inner (?) 
        # Actually it's easier to just compute the grid and find the hull or trace the four sides of the theta rectangle
        
        pts = []
        # Side 1: th1 sweep, th2 = min
        for t1 in np.linspace(self.theta_min[0], self.theta_max[0], res):
            pts.append(self.arm.forward_kinematics([t1, self.theta_min[1]])[:, -1])
            
        # Side 2: th1 = max, th2 sweep
        for t2 in np.linspace(self.theta_min[1], self.theta_max[1], res):
            pts.append(self.arm.forward_kinematics([self.theta_max[0], t2])[:, -1])
            
        # Side 3: th1 sweep (rev), th2 = max
        for t1 in np.linspace(self.theta_max[0], self.theta_min[0], res):
            pts.append(self.arm.forward_kinematics([t1, self.theta_max[1]])[:, -1])
            
        # Side 4: th1 = min, th2 sweep (rev)
        for t2 in np.linspace(self.theta_max[1], self.theta_min[1], res):
            pts.append(self.arm.forward_kinematics([self.theta_min[0], t2])[:, -1])
            
        pts = np.array(pts)
        self.ax.fill(pts[:, 0], pts[:, 1], color='green', alpha=0.15, label='Reachable', zorder=1)
        self.ax.plot(pts[:, 0], pts[:, 1], 'g--', lw=1.5, alpha=0.4, zorder=2)
        
        # Mark inaccessible "dead zone" inner circle if any
        # (For 2-DOF, if th2 min/max are far from 180, there's always an inner circle)
        inner_r = np.sqrt(l1**2 + l2**2 + 2*l1*l2*np.cos(max(abs(self.theta_min[1]), abs(self.theta_max[1]))))
        circ = plt.Circle((0,0), inner_r, color='white', zorder=3, alpha=1.0)
        self.ax.add_artist(circ)
        self.ax.plot(inner_r*np.cos(np.linspace(0, 2*np.pi, 100)), inner_r*np.sin(np.linspace(0, 2*np.pi, 100)), 'r--', lw=0.5, alpha=0.3, zorder=4)

    def _gravity_comp(self):
        """Compute pure gravity-compensation torque at current state."""
        m1, m2, l1, l2, g = self.arm.m1, self.arm.m2, self.arm.l1, self.arm.l2, self.arm.g
        th1, th2 = self.x[0], self.x[1]
        G1 = (m1*l1/2 + m2*l1)*g*np.sin(th1) + m2*l2/2*g*np.sin(th1+th2)
        G2 = m2*l2/2*g*np.sin(th1+th2)
        return np.array([0, 0]) # No gravity compensation for now

    def update(self, i):
        # 1. Check stability
        if np.any(np.abs(self.x) > 100):
            self.reset()
            self.status_text.set_text("Unstable! Resetting...")
            return self.line, self.goal_dot, self.status_text

        # 2. Get theta goal
        x_goal = self.solve_ik_crude(self.target_pos)

        u = np.zeros(self.arm.nu)
        solver_status = "N/A"

        if not self.mpc_on:
            # Zero torque — arm swings under gravity (or stays if g=0)
            u = np.zeros(self.arm.nu)
            solver_status = "No MPC"
        else:
            # ── MPC path ──────────────────────────────────────────────
            # 3. Build MPC
            ref_traj    = self.mpc.build_reference_trajectory(self.x, x_goal)
            qp_matrices = self.mpc.build_qp(self.x, ref_traj)

            # 4. Solve
            try:
                if self.current_solver_name == 'OSQP':
                    z = self.osqp_solver.solve(qp_matrices)
                else:
                    z = self.sho_solver.solve(qp_matrices, x_min_val=-5.0, x_max_val=5.0)

                if z is not None:
                    u = z[self.arm.nx : self.arm.nx + self.arm.nu]
                    solver_status = "OPTIMAL"
                else:
                    # PD fallback (no gravity comp)
                    theta_err = x_goal[:2] - self.x[:2]
                    theta_err = (theta_err + np.pi) % (2 * np.pi) - np.pi
                    dtheta_err = x_goal[2:] - self.x[2:]
                    u = self.Kp * theta_err + self.Kd * dtheta_err
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
        
        # Update streak
        self.ee_streak.append(pts[:, -1])
        for idx, pos in enumerate(self.ee_streak):
            self.streak_plots[idx].set_data([pos[0]], [pos[1]])
        for idx in range(len(self.ee_streak), self.streak_len):
            self.streak_plots[idx].set_data([], [])
        
        # Info
        pos_err = np.linalg.norm(pts[:, -1] - self.target_pos)
        self.status_text.set_text(f"Solver: {self.current_solver_name} | Status: {solver_status} | Err: {pos_err:.3f}")
        
        return self.line, self.goal_dot, self.status_text, *self.streak_plots

    def run(self):
        from matplotlib.animation import FuncAnimation
        self.ani = FuncAnimation(self.fig, self.update, interval=50, blit=False) # blit=False for text update
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Interactive 2-DOF Arm with MPC')
    parser.add_argument('--gravity', type=float, default=9.81,
                        help='Gravitational acceleration (use 0.0 for gravity-off)')
    args = parser.parse_args()
    app = InteractiveArm(g=args.gravity)
    app.run()
