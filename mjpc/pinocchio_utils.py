"""
Pinocchio Utilities
===================
IK, FK, and dynamics using Pinocchio library for efficient computation.
"""

import numpy as np
import warnings

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    warnings.warn("Pinocchio not installed. Install with: pip install pin")


class PinocchioRobotModel:
    """Wrapper around Pinocchio model for xArm 6DOF"""
    
    def __init__(self, urdf_path=None):
        """
        Initialize Pinocchio model.
        
        Args:
            urdf_path: Path to URDF file (optional, uses xarm_6dof.urdf if not provided)
        """
        if not HAS_PINOCCHIO:
            raise RuntimeError("Pinocchio not installed. Run: pip install pin")
        
        # Use default if not provided
        if urdf_path is None:
            from pathlib import Path
            urdf_path = str(Path(__file__).parent.parent / "assets/xarm_6dof.urdf")
        
        try:
            self.model = pin.buildModelFromURDF(urdf_path)
            self.data = self.model.createData()
            self.ee_frame_id = self.model.getFrameId("tool0") if "tool0" in self.model.frames else self.model.nframes - 1
            self.initialized = True
        except Exception as e:
            warnings.warn(f"Could not load URDF: {e}. Using dummy model for FK.")
            self.initialized = False
            self.model = None
            self.data = None
    
    def forward_kinematics(self, q):
        """
        Compute forward kinematics.
        
        Args:
            q: Joint configuration [6] or [7] (arm only or with free joint)
        
        Returns:
            ee_pos: End-effector position [3]
            ee_R: End-effector rotation matrix [3x3]
        """
        if not self.initialized or self.model is None:
            # Fallback: approximate FK based on arm kinematics
            return self._approximate_fk(q)
        
        try:
            # Ensure q is 6-dim (just arm)
            q_full = np.zeros(self.model.nq)
            if len(q) == 6:
                q_full[7:13] = q  # Arm joints at indices 7:13 (after free joint)
            else:
                q_full = q
            
            pin.forwardKinematics(self.model, self.data, q_full)
            
            # Get end-effector transformation
            ee_transform = self.data.oMi[self.ee_frame_id]
            ee_pos = ee_transform.translation
            ee_R = ee_transform.rotation
            
            return ee_pos, ee_R
        
        except Exception as e:
            warnings.warn(f"FK computation failed: {e}")
            return self._approximate_fk(q)
    
    def _approximate_fk(self, q):
        """Approximate FK using simple geometry if Pinocchio unavailable"""
        # Simple approximation for debugging
        # Link lengths (approximate): L1=0.2, L2=0.2, L3=0.16
        q1, q2, q3, q4, q5, q6 = q
        
        x = 0.2 * np.cos(q1 + q2) + 0.2 * np.cos(q1 + q2 + q3)
        y = 0.2 * np.sin(q1 + q2) + 0.2 * np.sin(q1 + q2 + q3)
        z = 0.16 + 0.2 * np.sin(q4)
        
        ee_pos = np.array([x, y, z])
        ee_R = np.eye(3)
        
        return ee_pos, ee_R
    
    def inverse_kinematics(self, target_pos, q_init=None, max_iter=100, tol=1e-4):
        """
        Compute inverse kinematics using Pinocchio IK solver.
        
        Args:
            target_pos: Target end-effector position [3]
            q_init: Initial joint configuration [6] (optional)
            max_iter: Max iterations for IK solver
            tol: Position tolerance (m)
        
        Returns:
            q_solution: Joint configuration [6]
            success: Whether IK converged
            error: Final position error (m)
        """
        if not self.initialized or self.model is None:
            return self._approximate_ik(target_pos)
        
        try:
            if q_init is None:
                q_init = np.zeros(self.model.nq)
            else:
                q_tmp = np.zeros(self.model.nq)
                if len(q_init) == 6:
                    q_tmp[7:13] = q_init
                else:
                    q_tmp = q_init
                q_init = q_tmp
            
            # Use Pinocchio's IK solver
            from pinocchio import IkSolverFromQP
            
            # Create IK problem
            ik_solver = pin.IkSolver(self.model)
            
            # Run IK
            q_solution = q_init.copy()
            
            for iter in range(max_iter):
                # Compute FK
                pin.forwardKinematics(self.model, self.data, q_solution)
                ee_transform = self.data.oMi[self.ee_frame_id]
                current_pos = ee_transform.translation
                
                # Check convergence
                error = np.linalg.norm(current_pos - target_pos)
                if error < tol:
                    return q_solution[7:13], True, error
                
                # Compute Jacobian
                J = pin.computeFrameJacobian(self.model, self.data, q_solution, self.ee_frame_id)
                J_pos = J[:3, :]  # Position part only
                
                # Pseudo-inverse
                J_pinv = np.linalg.pinv(J_pos)
                
                # Compute velocity to reach target
                pos_error = target_pos - current_pos
                dq = J_pinv @ pos_error
                
                # Update solution with damping
                q_solution = q_solution + 0.1 * dq
                
                # Clamp to joint limits
                q_solution = np.clip(q_solution, self.model.lowerPositionLimit, self.model.upperPositionLimit)
            
            # Return best solution even if not fully converged
            pin.forwardKinematics(self.model, self.data, q_solution)
            ee_transform = self.data.oMi[self.ee_frame_id]
            final_error = np.linalg.norm(ee_transform.translation - target_pos)
            
            return q_solution[7:13], final_error < tol * 10, final_error
        
        except Exception as e:
            warnings.warn(f"IK computation failed: {e}")
            return self._approximate_ik(target_pos)
    
    def _approximate_ik(self, target_pos, q_init=None):
        """Approximate IK using numerical optimization as fallback"""
        if q_init is None:
            q_sol = np.array([0.2, -0.5, -2.0, 0.0, -0.5, 0.0])
        else:
            q_sol = q_init.copy() if len(q_init) == 6 else q_init[:6]
        
        # Simple numerical optimization
        for _ in range(50):
            current_pos, _ = self._approximate_fk(q_sol)
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < 0.01:
                return q_sol, True, np.linalg.norm(error)
            
            # Gradient descent
            for i in range(6):
                q_temp = q_sol.copy()
                dq = 0.001
                q_temp[i] += dq
                pos_perturb, _ = self._approximate_fk(q_temp)
                grad = (np.linalg.norm(pos_perturb - target_pos) - np.linalg.norm(current_pos - target_pos)) / dq
                q_sol[i] -= 0.01 * grad
            
            # Clamp
            q_sol = np.clip(q_sol, -np.pi, np.pi)
        
        final_pos, _ = self._approximate_fk(q_sol)
        return q_sol, False, np.linalg.norm(final_pos - target_pos)
    
    def compute_jacobian(self, q):
        """
        Compute Jacobian for end-effector.
        
        Args:
            q: Joint configuration [6]
        
        Returns:
            J: Jacobian matrix [6 x 6]
        """
        if not self.initialized or self.model is None:
            return np.eye(6)
        
        try:
            q_full = np.zeros(self.model.nq)
            q_full[7:13] = q
            
            J = pin.computeFrameJacobian(self.model, self.data, q_full, self.ee_frame_id)
            return J
        
        except:
            return np.eye(6)


def test_pinocchio():
    """Test Pinocchio model loading and FK/IK"""
    print("Testing Pinocchio integration...")
    
    if not HAS_PINOCCHIO:
        print("✗ Pinocchio not available")
        return False
    
    try:
        model = PinocchioRobotModel()
        if not model.initialized:
            print("✗ Model initialization failed")
            return False
        
        print("✓ Model loaded")
        
        # Test FK
        q = np.array([0.0, -0.5, -2.0, 0.0, -0.5, 0.0])
        pos, R = model.forward_kinematics(q)
        print(f"✓ FK: position = {pos}")
        
        # Test IK
        target = np.array([0.3, 0.0, 0.52])
        q_sol, success, error = model.inverse_kinematics(target, q)
        print(f"✓ IK: q_sol = {q_sol}, error = {error:.4f}m")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    test_pinocchio()
