"""
Franka Panda Motion Planning with Pinocchio Integration
========================================================
7-DOF arm + 2-DOF gripper smooth trajectory generation
"""

import numpy as np

try:
    import pinocchio as pin
except Exception:
    pin = None

class FrankaMotionPlanning:
    """Professional trajectory planner for Franka Panda using Pinocchio"""
    
    def __init__(self, urdf_path=None, home=None, grasp_target=None,
                 move_time=6.0, hold_time=2.0, lift_height=0.1):
        """
        Initialize Franka motion planner.
        
        Args:
            urdf_path: Path to Franka URDF/XML (pinocchio will handle conversion)
            home: Home configuration [7] for arm (default: safe config)
            grasp_target: Target grasp config [7]  (default: forward reach)
            move_time: Approach duration (seconds)
            hold_time: Grasp hold duration (seconds)
            lift_height: Height to lift object (meters)
        """
        self.move_time = move_time
        self.hold_time = hold_time
        self.lift_height = lift_height
        
        # Safe home configuration for Franka (all zeros)
        self.home = home if home is not None else np.array([
            0.0,      # joint1
            0.0,      # joint2
            0.0,      # joint3
            -np.pi/2, # joint4 (natural elbow down)
            0.0,      # joint5
            np.pi/2,  # joint6 (natural wrist)
            np.pi/4   # joint7 (natural rotation)
        ])
        
        # Default grasp target (forward reach)
        self.grasp_target = grasp_target if grasp_target is not None else np.array([
            0.0,       # joint1
            -np.pi/4,  # joint2 (shoulder angle)
            0.0,       # joint3
            -np.pi/2,  # joint4 (elbow down for reaching)
            0.0,       # joint5
            np.pi/2,   # joint6
            np.pi/4    # joint7
        ])
        self.lift_target = self.grasp_target.copy()
        self.place_target = self.grasp_target.copy()
        
        # Gripper states (0=open, 0.04=closed)
        self.gripper_open = 0.04
        self.gripper_closed = 0.001
        
        # Load Pinocchio model if URDF provided
        self.pin_model = None
        self.pin_data = None
        self.ee_frame_id = None
        if urdf_path:
            try:
                self.pin_model = self._build_pin_model(urdf_path)
                self.pin_data = self.pin_model.createData()
                self.ee_frame_id = self._resolve_ee_frame_id()
                print(f"[Franka] Pinocchio model loaded: {self.pin_model.nq} DOF")
            except Exception as e:
                print(f"[Franka] Warning: Could not load Pinocchio model: {e}")
                print(f"[Franka] Using analytical kinematics fallback")

    def _build_pin_model(self, model_path):
        """Build a Pinocchio model from URDF or MJCF path."""
        if pin is None:
            raise RuntimeError("pinocchio is not available in this interpreter")

        lower = str(model_path).lower()
        if lower.endswith(".urdf"):
            return pin.buildModelFromUrdf(model_path)
        return pin.buildModelFromMJCF(model_path)

    def _resolve_ee_frame_id(self):
        """Find a robust end-effector frame id across common Franka model names."""
        if self.pin_model is None:
            return None

        preferred_names = [
            "panda_hand",
            "hand",
            "hand_fixed",
            "left_finger",
            "right_finger",
        ]
        for frame_name in preferred_names:
            try:
                frame_id = self.pin_model.getFrameId(frame_name)
                if frame_id < len(self.pin_model.frames):
                    return frame_id
            except Exception:
                continue
        return len(self.pin_model.frames) - 1
    
    def quintic_trajectory(self, q_start, q_end, t_phase, t_in_phase):
        """
        Quintic polynomial trajectory: zero velocity and acceleration at endpoints.
        
        Args:
            q_start: [7] starting configuration
            q_end:   [7] ending configuration  
            t_phase: total phase duration (seconds)
            t_in_phase: current time within phase (seconds)
        
        Returns:
            q_ref: [7] reference position
            dq_ref: [7] reference velocity
        """
        if t_in_phase >= t_phase:
            return q_end.copy(), np.zeros(7)
        
        # Normalized time [0, 1]
        tau = t_in_phase / t_phase
        
        # Quintic polynomial: α(τ) = 10τ³ - 15τ⁴ + 6τ⁵
        alpha = 10*tau**3 - 15*tau**4 + 6*tau**5
        d_alpha = (30*tau**2 - 60*tau**3 + 30*tau**4) / t_phase
        
        # Interpolate
        q_ref = q_start + alpha * (q_end - q_start)
        dq_ref = d_alpha * (q_end - q_start)
        
        return q_ref, dq_ref
    
    def get_reference(self, t_sim, total_cycle_time=None):
        """
        Get reference trajectory for pick-and-place task.
        
        Phases (cycling):
        1. APPROACH (0 - move_time):     HOME → GRASP_TARGET
        2. GRASP (move_time - move_time+1):    Hold at GRASP_TARGET, close gripper
        3. LIFT (move_time+1 - move_time+3):   Lift object
        4. PLACE (move_time+3 - move_time+5):  Move to place location
        5. RELEASE (move_time+5 - move_time+6): Open gripper
        6. RETRACT (move_time+6 - cycle):      Return HOME
        
        Args:
            t_sim: Current simulation time (seconds)
            total_cycle_time: Total cycle time (auto-computed if None)
        
        Returns:
            q_ref: [9] reference state (7 arm + 2 gripper)
            dq_ref: [9] reference velocity
        """
        t_approach = self.move_time
        t_grasp = self.hold_time
        t_lift = 1.6
        t_place = 1.8
        t_release = 0.8
        t_retract = 2.0

        if total_cycle_time is None:
            total_cycle_time = t_approach + t_grasp + t_lift + t_place + t_release + t_retract

        # Phase times
        t_approach_end = t_approach
        t_grasp_end = t_approach_end + t_grasp
        t_lift_end = t_grasp_end + t_lift
        t_place_end = t_lift_end + t_place
        t_release_end = t_place_end + t_release
        t_retract_end = t_release_end + t_retract
        
        # Normalize time to cycle
        t_cycle = t_sim % total_cycle_time
        
        # Initialize references
        q_ref_arm = self.home.copy()
        dq_ref_arm = np.zeros(7)
        gripper_ref = self.gripper_open
        
        # PHASE 1: APPROACH (HOME → GRASP_TARGET)
        if t_cycle < t_approach_end:
            q_ref_arm, dq_ref_arm = self.quintic_trajectory(
                self.home, self.grasp_target, self.move_time, t_cycle
            )
            gripper_ref = self.gripper_open  # Keep open during approach
        
        # PHASE 2: GRASP (Hold and close)
        elif t_cycle < t_grasp_end:
            q_ref_arm = self.grasp_target
            dq_ref_arm = np.zeros(7)
            # Smoothly close gripper
            tau_close = (t_cycle - t_approach_end) / max(t_grasp_end - t_approach_end, 1e-9)
            gripper_ref = self.gripper_open - tau_close * (self.gripper_open - self.gripper_closed)
        
        # PHASE 3: LIFT
        elif t_cycle < t_lift_end:
            t_in_lift = t_cycle - t_grasp_end
            q_ref_arm, dq_ref_arm = self.quintic_trajectory(
                self.grasp_target, self.lift_target, t_lift_end - t_grasp_end, t_in_lift
            )
            gripper_ref = self.gripper_closed  # Maintain grip
        
        # PHASE 4: PLACE (move to target location)
        elif t_cycle < t_place_end:
            t_in_place = t_cycle - t_lift_end
            q_ref_arm, dq_ref_arm = self.quintic_trajectory(
                self.lift_target, self.place_target, t_place_end - t_lift_end, t_in_place
            )
            gripper_ref = self.gripper_closed  # Keep grip
        
        # PHASE 5: RELEASE (open gripper)
        elif t_cycle < t_release_end:
            q_ref_arm = self.place_target
            dq_ref_arm = np.zeros(7)
            tau_release = (t_cycle - t_place_end) / max(t_release_end - t_place_end, 1e-9)
            gripper_ref = self.gripper_closed + tau_release * (self.gripper_open - self.gripper_closed)
        
        # PHASE 6: RETRACT (return HOME)
        elif t_cycle < t_retract_end:
            t_retract = t_cycle - t_release_end
            q_ref_arm, dq_ref_arm = self.quintic_trajectory(
                self.place_target, self.home, t_retract_end - t_release_end, t_retract
            )
            gripper_ref = self.gripper_open
        
        # Combine arm and gripper references
        q_ref = np.concatenate([q_ref_arm, np.array([gripper_ref, gripper_ref])])
        dq_ref = np.concatenate([dq_ref_arm, np.array([0.0, 0.0])])
        
        return q_ref, dq_ref
    
    def forward_kinematics_pinocchio(self, q):
        """
        Compute end-effector position using Pinocchio.
        
        Args:
            q: [7] arm configuration
        
        Returns:
            ee_pos: [3] end-effector position in world frame
            ee_rot: [3,3] end-effector rotation matrix
        """
        if self.pin_model is None or self.pin_data is None:
            return None, None
        
        try:
            q_full = np.zeros(self.pin_model.nq)
            q_full[:min(len(q), self.pin_model.nq)] = q[:min(len(q), self.pin_model.nq)]
            if self.pin_model.nq >= 9:
                q_full[7:9] = np.array([0.04, 0.04])

            pin.forwardKinematics(self.pin_model, self.pin_data, q_full)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            
            ee_frame_id = self.ee_frame_id if self.ee_frame_id is not None else self._resolve_ee_frame_id()
            ee_transform = self.pin_data.oMf[ee_frame_id]
            
            return ee_transform.translation, ee_transform.rotation
        except Exception as e:
            return None, None
    
    def jacobian_pinocchio(self, q):
        """
        Compute Jacobian at configuration using Pinocchio.
        
        Args:
            q: [7] arm configuration
        
        Returns:
            J: [3, 7] geometric Jacobian (position only)
        """
        if self.pin_model is None or self.pin_data is None:
            return None
        
        try:
            q_full = np.zeros(self.pin_model.nq)
            q_full[:min(len(q), self.pin_model.nq)] = q[:min(len(q), self.pin_model.nq)]
            if self.pin_model.nq >= 9:
                q_full[7:9] = np.array([0.04, 0.04])

            pin.forwardKinematics(self.pin_model, self.pin_data, q_full)
            pin.computeJointJacobians(self.pin_model, self.pin_data, q_full)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            
            ee_frame_id = self.ee_frame_id if self.ee_frame_id is not None else self._resolve_ee_frame_id()
            J = pin.getFrameJacobian(self.pin_model, self.pin_data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
            
            # Return only position part [3, 7]
            return J[:3, :7]
        except Exception as e:
            return None

    def solve_ik_position(self, target_pos, q_init, max_iters=120, tol=3e-3,
                          damping=1e-4, step_size=0.65):
        """Solve 3D position IK with damped least-squares on the arm joints."""
        if self.pin_model is None or self.pin_data is None:
            return np.array(q_init, dtype=float), False

        q = np.array(q_init, dtype=float).copy()
        target = np.array(target_pos, dtype=float).reshape(3)

        for _ in range(max_iters):
            ee_pos, _ = self.forward_kinematics_pinocchio(q)
            if ee_pos is None:
                return np.array(q_init, dtype=float), False

            err = target - ee_pos
            if np.linalg.norm(err) < tol:
                return q, True

            j = self.jacobian_pinocchio(q)
            if j is None:
                return np.array(q_init, dtype=float), False

            jj_t = j @ j.T + damping * np.eye(3)
            dq = j.T @ np.linalg.solve(jj_t, err)
            q = q + step_size * dq

        return q, False
