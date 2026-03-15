"""Mock VLA for testing without model weights."""
import numpy as np
import time


class MockVLAServer:
    """
    Deterministic mock VLA for testing.
    
    CRITICAL CONTRACT:
        Every output dict MUST contain "source": "MOCK".
        This prevents mock metrics from being confused with real SmolVLA metrics.
    """
    
    RESPONSE_TIME_MS = 5.0  # Realistic mock latency
    
    def __init__(self):
        self.call_count = 0
    
    def predict(
        self,
        rgb: np.ndarray,          # [H, W, 3] uint8 (not used by mock)
        state: np.ndarray,        # [6] arm joint angles
        instruction: str = "pick up the red block",
    ) -> dict:
        """
        Returns deterministic 7-D action from state.
        
        Args:
            rgb: Image (ignored by mock)
            state: [6] joint angles
            instruction: Language instruction (ignored by mock)
        
        Returns:
            dict with action, status, and source="MOCK"
        """
        self.call_count += 1
        t0 = time.perf_counter()
        
        s = np.asarray(state[:6], dtype=np.float64)
        
        # Deterministic target: slowly move joints toward a "reach" pose
        reach_pose = np.array([0.3, -0.5, 0.8, -0.2, 0.1, 0.0])
        action_6 = s + 0.05 * (reach_pose - s)  # 5% of error per step
        action_7 = np.append(action_6, 0.0)  # gripper open
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 + self.RESPONSE_TIME_MS
        
        return {
            'action': action_7.tolist(),
            'action_std': [0.01] * 7,
            'latency_ms': elapsed_ms,
            'success': True,
            'source': 'MOCK',  # CRITICAL: always include this
        }
    
    def reset(self):
        """Reset call counter."""
        self.call_count = 0
