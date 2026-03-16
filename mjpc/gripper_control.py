"""
Gripper Control Module
======================
Manages gripper open/close actions with force control and contact detection.
"""

import numpy as np


class GripperController:
    """Control gripper with discrete phases: IDLE, CLOSING, HOLDING, OPENING"""
    
    def __init__(self, close_torque=10.0, hold_torque=5.0, contact_threshold=0.1):
        """
        Args:
            close_torque: Torque to apply when closing fingers (N·m)
            hold_torque: Torque to maintain grip (N·m)
            contact_threshold: Force threshold to detect contact (N)
        """
        self.close_torque = close_torque
        self.hold_torque = hold_torque
        self.contact_threshold = contact_threshold
        
        self.state = "IDLE"  # IDLE, CLOSING, HOLDING, OPENING
        self.phase_start_time = None
        self.is_grasping = False
        self.contact_force = 0.0
    
    def update(self, t_sim, phase_command="IDLE", contact_force=None):
        """
        Update gripper state and compute control torque.
        
        Args:
            t_sim: Current simulation time (seconds)
            phase_command: Desired phase ("IDLE", "CLOSING", "HOLDING", "OPENING")
            contact_force: Measured contact force magnitude (N, optional)
        
        Returns:
            tau_gripper: Torque command for gripper motors [2] (N·m)
        """
        
        # Update contact detection
        if contact_force is not None:
            self.contact_force = contact_force
            if contact_force > self.contact_threshold:
                self.is_grasping = True
        
        # State machine for gripper phases
        if phase_command != self.state:
            self.state = phase_command
            self.phase_start_time = t_sim
        
        # Compute torque based on current phase
        if self.state == "IDLE":
            tau = np.array([0.0, 0.0])
        
        elif self.state == "CLOSING":
            # Apply strong closing torque
            tau = np.array([self.close_torque, self.close_torque])
            
            # If contact detected, transition to HOLDING
            if self.is_grasping:
                elapsed = t_sim - self.phase_start_time
                if elapsed > 0.1:  # Wait 100ms to confirm contact
                    self.state = "HOLDING"
        
        elif self.state == "HOLDING":
            # Maintain grip with reduced torque
            tau = np.array([self.hold_torque, self.hold_torque])
        
        elif self.state == "OPENING":
            # Drop grip (zero torque lets springs open fingers)
            tau = np.array([0.0, 0.0])
            self.is_grasping = False
        
        else:
            tau = np.array([0.0, 0.0])
        
        return tau
    
    def close_gripper(self):
        """Command gripper to CLOSING phase"""
        self.state = "CLOSING"
    
    def hold_object(self):
        """Command gripper to HOLDING phase"""
        self.state = "HOLDING"
    
    def open_gripper(self):
        """Command gripper to OPENING phase"""
        self.state = "OPENING"
    
    def idle(self):
        """Command gripper to IDLE phase"""
        self.state = "IDLE"


class PickAndPlaceSequence:
    """Multi-phase pick and place controller"""
    
    def __init__(self):
        self.gripper = GripperController(close_torque=10.0, hold_torque=5.0)
        self.phase = "HOME"
        self.phase_start_time = 0.0
        
        # Timing for each phase (seconds)
        self.phase_timing = {
            'HOME_TO_APPROACH': 2.0,      # Move to grasp position
            'CLOSE_GRIPPER': 1.5,          # Close gripper
            'GRASP_CONFIRM': 0.5,          # Wait for contact
            'LIFT': 1.0,                   # Lift object
            'MOVE_TO_PLACE': 3.0,          # Move to place location
            'LOWER': 1.0,                  # Lower to place height
            'OPEN_GRIPPER': 1.0,           # Release object
            'RETREAT': 0.5,                # Lift arm away
            'RETURN_HOME': 2.0,            # Return to home
        }
        
        self.phase_sequence = [
            'HOME_TO_APPROACH',
            'CLOSE_GRIPPER',
            'GRASP_CONFIRM',
            'LIFT',
            'MOVE_TO_PLACE',
            'LOWER',
            'OPEN_GRIPPER',
            'RETREAT',
            'RETURN_HOME',
        ]
    
    def get_gripper_command(self, t_sim, block_grasped=False):
        """
        Get gripper control torque for current phase.
        
        Args:
            t_sim: Simulation time
            block_grasped: Whether block is in gripper contact
        
        Returns:
            tau_gripper [2]: Torque commands
            current_phase: Name of current phase
        """
        
        elapsed = t_sim - self.phase_start_time
        phase_duration = self.phase_timing[self.phase]
        
        # Transition to next phase if time exceeded
        if elapsed >= phase_duration:
            phase_idx = self.phase_sequence.index(self.phase)
            if phase_idx < len(self.phase_sequence) - 1:
                self.phase = self.phase_sequence[phase_idx + 1]
                self.phase_start_time = t_sim
                elapsed = 0.0
            else:
                # Cycle complete, return to first phase
                self.phase = self.phase_sequence[0]
                self.phase_start_time = t_sim
                elapsed = 0.0
        
        # Command gripper based on current phase
        contact_force = 1.0 if block_grasped else 0.0
        
        if 'CLOSE' in self.phase:
            tau_gripper = self.gripper.update(t_sim, 'CLOSING', contact_force)
        elif 'CONFIRM' in self.phase:
            tau_gripper = self.gripper.update(t_sim, 'HOLDING', contact_force)
        elif 'LIFT' in self.phase or 'MOVE' in self.phase or 'LOWER' in self.phase:
            tau_gripper = self.gripper.update(t_sim, 'HOLDING', contact_force)
        elif 'OPEN' in self.phase or 'RETREAT' in self.phase:
            tau_gripper = self.gripper.update(t_sim, 'OPENING', contact_force)
        else:  # HOME phases
            tau_gripper = self.gripper.update(t_sim, 'IDLE', contact_force)
        
        return tau_gripper, self.phase, elapsed / phase_duration


def detect_block_contact(model, data, gripper_body_id=14, contact_threshold=0.01):
    """
    Detect if block is in contact with gripper.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        gripper_body_id: Body ID of gripper (typically 14 for fingers)
        contact_threshold: Force threshold (N)
    
    Returns:
        is_grasping: Boolean, True if contact detected
        contact_force: Maximum contact force magnitude
    """
    
    max_force = 0.0
    
    # Check all contacts
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        
        # Check if contact involves gripper
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        # Get body IDs from geometry
        body1_id = model.geom_bodyid[geom1_id]
        body2_id = model.geom_bodyid[geom2_id]
        
        # If either body is the gripper, measure contact force
        if body1_id == gripper_body_id or body2_id == gripper_body_id:
            # Contact force magnitude
            force = np.linalg.norm(contact.H)
            max_force = max(max_force, force)
    
    is_grasping = max_force > contact_threshold
    
    return is_grasping, max_force


if __name__ == "__main__":
    print("✓ Gripper module loaded")
    gripper = GripperController()
    print(f"  - Gripper state: {gripper.state}")
    print(f"  - Is grasping: {gripper.is_grasping}")
