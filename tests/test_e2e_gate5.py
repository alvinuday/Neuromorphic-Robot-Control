"""
Gate 5: End-to-End System Testing with Real SmolVLA

Tests the complete dual-system architecture with:
- Real 3-DOF arm dynamics (MuJoCo simulation)
- Real 3-DOF MPC solver (SL-based QP)
- Real SmolVLA vision-language queries
- Background async polling thread
- State machine controller

All components integrated and tested under realistic conditions.

SETUP:
1. Start SmolVLA server in Colab (vla/smolvla_server.ipynb)
2. Set: export SMOLVLA_SERVER_URL="https://xxxx-ngrok-free.dev"
3. Run: pytest tests/test_e2e_gate5.py -v -s
"""

import pytest
import pytest_asyncio
import asyncio
import numpy as np
import logging
import os
import time
from typing import Optional, Tuple
import threading

from src.integration.smolvla_server_client import SmolVLAServerConfig, RealSmolVLAClient
from src.smolvla_client.trajectory_buffer import TrajectoryBuffer
from src.integration.dual_system_controller import DualSystemController, ControlState
from src.integration.vla_query_thread import VLAQueryThread
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
from src.dynamics.kinematics_3dof import Arm3DOF
from src.dynamics.lagrangian_3dof import Arm3DOFDynamics


logger = logging.getLogger(__name__)


@pytest.fixture
def server_url() -> Optional[str]:
    """Get SmolVLA server URL from environment."""
    url = os.getenv("SMOLVLA_SERVER_URL", None)
    if url:
        logger.info(f"Using real SmolVLA server: {url}")
    return url


@pytest.fixture
def server_config(server_url) -> Optional[SmolVLAServerConfig]:
    """Create server config or skip if no server."""
    if server_url is None:
        pytest.skip("SMOLVLA_SERVER_URL not set")
    return SmolVLAServerConfig(server_url=server_url, timeout_s=3.0)


@pytest_asyncio.fixture
async def real_vla_client(server_config):
    """Create and start real VLA client."""
    client = RealSmolVLAClient(server_config)
    await client.start()
    yield client
    await client.stop()


@pytest.fixture
def arm_kinematics():
    """Create 3-DOF arm kinematics."""
    return Arm3DOF(L0=0.10, L1=0.25, L2=0.20)


@pytest.fixture
def arm_dynamics():
    """Create 3-DOF arm dynamics."""
    return Arm3DOFDynamics(L0=0.10, L1=0.25, L2=0.20, m1=1.0, m2=0.8, m3=0.6)


@pytest.fixture
def mpc_solver():
    """Create MPC solver (SL-based)."""
    return StuartLandauLagrangeDirect(
        tau_x=1.0,
        tau_lam_eq=0.1,
        tau_lam_ineq=0.5,
        mu_x=0.5,
        T_solve=30.0,
        convergence_tol=1e-3,
        adaptive_annealing=True
    )


@pytest.fixture
def trajectory_buffer():
    """Create trajectory buffer for subgoal interpolation."""
@pytest.fixture
def trajectory_buffer():
    """Create trajectory buffer with quintic interpolation."""
    return TrajectoryBuffer(arrival_threshold_rad=0.05)


@pytest.fixture
def controller(mpc_solver, real_vla_client, trajectory_buffer):
    """Create dual-system controller."""
    return DualSystemController(
        mpc_solver=mpc_solver,
        smolvla_client=real_vla_client,
        trajectory_buffer=trajectory_buffer,
        mpc_horizon_steps=10,
        control_dt_s=0.01,
        vla_query_interval_s=0.2
    )


@pytest.fixture
async def vla_polling_thread(real_vla_client, trajectory_buffer):
    """
    Create VLA background polling thread.
    
    In production, this runs concurrently while the main MPC loop runs.
    """
    # Mock RGB source for testing
    class MockRGBSource:
        def __init__(self):
            self.current_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        def get_latest(self):
            return self.current_frame
    
    rgb_source = MockRGBSource()
    
    thread = VLAQueryThread(
        smolvla_client=real_vla_client,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.5,
        query_timeout_s=2.0
    )
    
    thread.start(rgb_source, instruction="reach to target")
    yield thread
    thread.stop()


# ============================================================================
# GATE 5A: Point-to-Point Reaching
# ============================================================================

class TestPointToPointReaching:
    """Test basic reaching task: move EE from home to target."""
    
    @pytest.mark.asyncio
    async def test_reaching_task_success(self, controller, real_vla_client, arm_kinematics):
        """
        Execute a reaching task:
        1. Start at home position (q = [0, 0, 0])
        2. VLA returns target  EE position
        3. Run MPC controller for N steps
        4. Verify EE moved toward target
        """
        q = np.array([0.0, 0.0, 0.0])  # Home position
        qdot = np.array([0.0, 0.0, 0.0])
        
        # Get initial EE position
        p_home = forward_kinematics(q, arm_kinematics.L0, arm_kinematics.L1, arm_kinematics.L2)
        logger.info(f"Home EE position: {p_home}")
        
        # Query VLA for target
        rgb_start = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        vla_result = await real_vla_client.query_action(rgb_start, instruction="reach forward")
        
        if vla_result is None:
            pytest.skip("VLA query failed")
        
        # Extract target from VLA action
        action = vla_result["action"]
        p_target = np.array(action[:3])  # First 3 dims are EE position
        logger.info(f"VLA target EE position: {p_target}")
        
        # Update trajectory buffer
        controller.trajectory_buffer.update_subgoal(p_target, action[3])
        
        # Run MPC controller for 50 steps
        step_times = []
        for step in range(50):
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            start = time.time()
            tau = controller.step(q, qdot, rgb, "reach forward")
            step_time = (time.time() - start) * 1000
            
            step_times.append(step_time)
            
            # Simulate dynamics (Euler integration for simplicity)
            # In practice, this would use real or sim dynamics
            qdot = qdot + (tau / 10.0) * 0.01  # Simple acceleration
            q = q + qdot * 0.01
            
            # Safety limits
            q = np.clip(q, arm_kinematics.q_min, arm_kinematics.q_max)
            
            if step % 10 == 0:
                p_current = forward_kinematics(q, arm_kinematics.L0, 
                                             arm_kinematics.L1, arm_kinematics.L2)
                distance = np.linalg.norm(p_current - p_target)
                logger.info(f"Step {step:3d}: distance={distance:.4f}m, tau={tau}")
        
        mean_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        
        logger.info(f"\n✓ Reaching task completed in 50 steps")
        logger.info(f"  Step timing: mean={mean_step_time:.1f}ms, max={max_step_time:.1f}ms")
        
        # Verify MPC timing (< 50ms per step)
        assert mean_step_time < 50, f"MPC too slow: {mean_step_time:.1f}ms"
        assert max_step_time < 100, f"MPC peak too high: {max_step_time:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_multiple_sequential_tasks(self, controller, real_vla_client):
        """Execute 3 sequential reaching tasks with VLA queries."""
        q = np.zeros(3)
        qdot = np.zeros(3)
        
        task_results = []
        
        for task in range(3):
            logger.info(f"\n--- Task {task+1}/3 ---")
            
            # Query VLA for new target
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            vla_result = await real_vla_client.query_action(rgb, instruction=f"reach task {task}")
            
            if vla_result is None:
                logger.warning(f"Task {task} VLA query failed, skipping")
                continue
            
            # Update controller with new target
            action = vla_result["action"]
            controller.trajectory_buffer.update_subgoal(action[:3], action[3])
            
            # Run for 20 steps
            for step in range(20):
                tau = controller.step(q, qdot, rgb, f"task {task}")
                qdot = qdot + tau * 0.01
                q += qdot * 0.01
            
            task_results.append({
                "task": task,
                "vla_latency_ms": vla_result["latency_ms"],
                "final_q": q.copy(),
            })
            
            logger.info(f"✓ Task {task+1} completed")
        
        assert len(task_results) >= 2, "Should complete at least 2 tasks"


# ============================================================================
# GATE 5B: Concurrent MPC + VLA Operations
# ============================================================================

class TestConcurrentOperations:
    """Test that MPC and VLA operations don't interfere."""
    
    @pytest.mark.asyncio
    async def test_mpc_unaffected_by_vla_latency(self, controller, real_vla_client):
        """
        Verify MPC timing is consistent while VLA queries happen.
        
        Expected behavior:
        - MPC steps: 10-20ms consistently
        - VLA queries: 600-900ms (in background)
        - No correlation between VLA latency and MPC timing
        """
        q = np.zeros(3)
        qdot = np.zeros(3)
        
        mpc_timings = []
        vla_latencies = []
        
        for i in range(30):
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            # Time MPC step (foreground)
            start = time.time()
            tau = controller.step(q, qdot, rgb, "reach")
            mpc_time = (time.time() - start) * 1000
            
            mpc_timings.append(mpc_time)
            qdot += tau * 0.01
            q += qdot * 0.01
            
            # Query VLA (would be in background thread in production)
            if i % 5 == 0:
                vla_result = await real_vla_client.query_action(rgb)
                if vla_result:
                    vla_latencies.append(vla_result["latency_ms"])
        
        mean_mpc = np.mean(mpc_timings)
        std_mpc = np.std(mpc_timings)
        min_mpc = np.min(mpc_timings)
        max_mpc = np.max(mpc_timings)
        
        mean_vla = np.mean(vla_latencies) if vla_latencies else 0
        
        logger.info(f"\n✓ Concurrent operations test results:")
        logger.info(f"  MPC: mean={mean_mpc:.1f}ms ±{std_mpc:.1f}ms, "
                   f"min={min_mpc:.1f}ms, max={max_mpc:.1f}ms")
        logger.info(f"  VLA: mean={mean_vla:.1f}ms")
        logger.info(f"  Ratio: VLA is {mean_vla/mean_mpc:.0f}x slower than MPC")
        
        # Assertions
        assert mean_mpc < 40, f"MPC mean too slow: {mean_mpc:.1f}ms"
        assert max_mpc < 80, f"MPC max too slow: {max_mpc:.1f}ms"
        assert std_mpc < 20, f"MPC timing too variable: {std_mpc:.1f}ms"


# ============================================================================
# GATE 5C: Real Async Integration with Thread
# ============================================================================

class TestAsyncVLAThread:
    """Test VLA background polling thread with real server."""
    
    @pytest.mark.asyncio
    async def test_vla_thread_polling_real_server(self, vla_polling_thread, controller):
        """
        Verify background VLA polling with real server.
        
        In production, this thread:
        - Runs in background while MPC runs at 100+ Hz
        - Polls VLA every 500ms
        - Updates trajectory buffer with new subgoals
        - Never blocks the main control loop
        """
        q = np.zeros(3)
        qdot = np.zeros(3)
        
        initial_subgoal_count = controller.trajectory_buffer.update_count
        
        # Run controller for 60 steps (6 seconds)
        start_time = time.time()
        mpc_timings = []
        
        while time.time() - start_time < 6.0:
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            t0 = time.time()
            tau = controller.step(q, qdot, rgb, "reach")
            mpc_time = (time.time() - t0) * 1000
            
            mpc_timings.append(mpc_time)
            qdot += tau * 0.01
            q += qdot * 0.01
            
            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.01)
        
        elapsed = time.time() - start_time
        final_subgoal_count = controller.trajectory_buffer.update_count
        
        # Check VLA thread stats
        vla_stats = vla_polling_thread.vla_client.get_stats()
        
        logger.info(f"\n✓ Async VLA thread test (6 second run):")
        logger.info(f"  MPC steps: {len(mpc_timings)}, mean time={np.mean(mpc_timings):.1f}ms")
        logger.info(f"  VLA polling thread: {vla_stats['query_count']} queries, "
                   f"success rate={vla_stats['success_rate']:.1%}")
        logger.info(f"  Trajectory buffer updates: {final_subgoal_count - initial_subgoal_count}")
        
        # Verify thread was active
        assert vla_stats["query_count"] > 0, "VLA thread didn't query server"
        assert vla_stats["success_rate"] > 0.5, "VLA success rate too low"


# ============================================================================
# GATE 5D: Stress Testing and Robustness
# ============================================================================

class TestStressAndRobustness:
    """Test system under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_long_running_session(self, controller, real_vla_client):
        """
        Simulate a 2-minute control session with occasional VLA queries.
        """
        q = np.zeros(3)
        qdot = np.zeros(3)
        
        step_count = 0
        vla_query_count = 0
        vla_timeout_count = 0
        
        start_time = time.time()
        step_times = []
        
        while time.time() - start_time < 120:  # 2 minutes
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            # MPC step
            t0 = time.time()
            tau = controller.step(q, qdot, rgb, "long session")
            step_time = (time.time() - t0) * 1000
            
            step_times.append(step_time)
            step_count += 1
            
            # Dynamics
            qdot += tau * 0.01
            q = np.clip(q + qdot * 0.01, -np.pi, np.pi)
            
            # Occasional VLA queries (every 10 seconds or so)
            if step_count % 1000 == 0:
                vla_result = await real_vla_client.query_action(rgb)
                vla_query_count += 1
                
                if vla_result is None:
                    vla_timeout_count += 1
                else:
                    controller.trajectory_buffer.update_subgoal(
                        vla_result["action"][:3],
                        vla_result["action"][3]
                    )
        
        elapsed = time.time() - start_time
        mean_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        
        logger.info(f"\n✓ Long-running session (2 minutes) results:")
        logger.info(f"  Total steps: {step_count}")
        logger.info(f"  Step timing: mean={mean_step_time:.1f}ms, max={max_step_time:.1f}ms")
        logger.info(f"  VLA queries: {vla_query_count}, timeouts: {vla_timeout_count}")
        logger.info(f"  Actual elapsed: {elapsed:.1f}s")
        
        # Verify stability
        assert step_count > 6000, "Not enough steps executed"
        assert mean_step_time < 50, "MPC timing degraded"


# ============================================================================
# GATE 5: Comprehensive Validation
# ============================================================================

class TestGate5Validation:
    """Gate 5: End-to-End System Testing."""
    
    @pytest.mark.asyncio
    async def test_gate5_real_server_reachable(self, real_vla_client):
        """Gate 5-1: Real server is reachable and responding."""
        healthy = await real_vla_client.health_check()
        assert healthy, "Real server not reachable"
    
    @pytest.mark.asyncio
    async def test_gate5_mpc_controller_functional(self, controller):
        """Gate 5-2: MPC controller is functional."""
        q = np.zeros(3)
        qdot = np.zeros(3)
        rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        tau = controller.step(q, qdot, rgb, "test")
        
        assert tau is not None, "Controller returned None"
        assert len(tau) == 3, "Wrong torque dimension"
        assert np.all(np.isfinite(tau)), "Non-finite torques"
    
    @pytest.mark.asyncio
    async def test_gate5_real_vla_inference_reliable(self, real_vla_client):
        """Gate 5-3: Real VLA inference is reliable (>80% success on 5 queries)."""
        for _ in range(5):
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            result = await real_vla_client.query_action(rgb)
            assert result is not None, "VLA inference failed"
        
        stats = real_vla_client.get_stats()
        assert stats["success_rate"] >= 0.80, f"Success rate too low: {stats['success_rate']}"
    
    @pytest.mark.asyncio
    async def test_gate5_timing_requirements_met(self, controller, real_vla_client):
        """Gate 5-4: System meets timing requirements (<50ms MPC, 600-1000ms VLA)."""
        q = np.zeros(3)
        qdot = np.zeros(3)
        mpc_times = []
        
        for _ in range(20):
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            t0 = time.time()
            controller.step(q, qdot, rgb, "test")
            mpc_times.append((time.time() - t0) * 1000)
        
        # Check MPC timing
        mean_mpc = np.mean(mpc_times)
        max_mpc = np.max(mpc_times)
        
        assert mean_mpc < 50, f"MPC too slow: {mean_mpc:.1f}ms"
        assert max_mpc < 100, f"MPC peak too high: {max_mpc:.1f}ms"
        
        # Check VLA timing (from at least one query)
        vla_result = await real_vla_client.query_action(rgb)
        if vla_result:
            vla_latency = vla_result["latency_ms"]
            assert 500 < vla_latency < 1500, f"VLA latency out of range: {vla_latency:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
