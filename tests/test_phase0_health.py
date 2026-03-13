"""Phase 0: Component Health Checks.

These tests verify that all modules import correctly, dependencies are installed,
and basic initialization works.
"""

import pytest
import sys
import importlib
import logging
from pathlib import Path

# Test that all core modules import without errors
class TestImportsAllModules:
    """Verify all src/ submodules can be imported."""
    
    def test_import_dynamics(self):
        """Test dynamics module imports."""
        from src.dynamics.kinematics_3dof import Arm3DOF
        from src.dynamics.lagrangian_3dof import Arm3DOFDynamics
        assert Arm3DOF is not None
        assert Arm3DOFDynamics is not None
    
    def test_import_mpc(self):
        """Test MPC module imports."""
        from src.mpc.linearize_3dof import linearize_continuous
        assert linearize_continuous is not None
    
    def test_import_solver(self):
        """Test solver module imports."""
        from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
        assert StuartLandauLagrangeDirect is not None
    
    def test_import_integration(self):
        """Test integration module imports."""
        from src.integration.smolvla_server_client import RealSmolVLAClient
        from src.integration.dual_system_controller import DualSystemController
        from src.smolvla_client.trajectory_buffer import TrajectoryBuffer
        assert RealSmolVLAClient is not None
        assert DualSystemController is not None
        assert TrajectoryBuffer is not None
    
    def test_import_utils(self):
        """Test utils imports."""
        from src.utils.data_collector import DataCollector
        assert DataCollector is not None
    
    def test_import_environments(self):
        """Test environments module imports."""
        from src.environments.mujoco_3dof_env import MuJoCo3DOFEnv
        assert MuJoCo3DOFEnv is not None


class TestDependenciesInstalled:
    """Verify required dependencies are available."""
    
    def test_mujoco_installed(self):
        """Check mujoco is installed and importable."""
        try:
            import mujoco
            import mujoco.viewer
            assert mujoco is not None
        except ImportError:
            pytest.skip("mujoco not installed (optional)")
    
    def test_numpy_installed(self):
        """Check numpy is available."""
        import numpy as np
        assert np is not None
    
    def test_scipy_installed(self):
        """Check scipy is available."""
        import scipy
        assert scipy is not None
    
    def test_pandas_installed(self):
        """Check pandas is available."""
        try:
            import pandas as pd
            assert pd is not None
        except ImportError:
            pytest.skip("pandas not installed (optional)")
    
    def test_matplotlib_installed(self):
        """Check matplotlib is available."""
        try:
            import matplotlib
            assert matplotlib is not None
        except ImportError:
            pytest.skip("matplotlib not installed (optional)")


class TestComponentInitialization:
    """Test basic component initialization without errors."""
    
    def test_kinematics_3dof_init(self):
        """Kinematics model initializes."""
        from src.dynamics.kinematics_3dof import Arm3DOF
        kin = Arm3DOF()
        assert kin is not None
    
    def test_dynamics_3dof_init(self):
        """Dynamics model initializes."""
        from src.dynamics.lagrangian_3dof import Arm3DOFDynamics
        dyn = Arm3DOFDynamics()
        assert dyn is not None
    
    def test_qp_builder_init(self):
        """QP builder initializes."""
        import numpy as np
        # Test that we can build a QP problem
        from src.mpc.linearize_3dof import linearize_continuous
        assert linearize_continuous is not None
    
    def test_sl_solver_init(self):
        """Stuart-Landau solver initializes."""
        from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
        solver = StuartLandauLagrangeDirect(tau_x=1.0, tau_lam_eq=0.1)
        assert solver is not None
    
    def test_trajectory_buffer_init(self):
        """TrajectoryBuffer initializes."""
        from src.smolvla_client.trajectory_buffer import TrajectoryBuffer
        buf = TrajectoryBuffer(arrival_threshold_rad=0.05)
        assert buf is not None
    
    def test_smolvla_client_init(self):
        """RealSmolVLAClient initializes with test config."""
        from src.integration.smolvla_server_client import RealSmolVLAClient
        from src.integration.smolvla_server_client import SmolVLAServerConfig
        
        config = SmolVLAServerConfig(
            server_url="http://localhost:8000",
            timeout_s=1.0,
        )
        client = RealSmolVLAClient(config)
        assert client is not None
    
    def test_data_collector_init(self):
        """DataCollector initializes."""
        from src.utils.data_collector import DataCollector
        collector = DataCollector(task_name='test')
        assert collector is not None


class TestDataCollectorFunctionality:
    """Test basic DataCollector recording functionality."""
    
    def test_record_control_step(self):
        """DataCollector can record control steps."""
        import numpy as np
        from src.utils.data_collector import DataCollector
        
        collector = DataCollector(task_name='test_step')
        
        q = np.array([0.1, 0.2, 0.3])
        qdot = np.array([0.01, 0.02, 0.03])
        tau = np.array([1.0, 2.0, 3.0])
        ee_pos = np.array([0.5, 0.6, 0.7])
        
        collector.record_step(
            step=0,
            q=q,
            qdot=qdot,
            tau=tau,
            ee_pos=ee_pos,
            mpc_cost=1.5,
            mpc_time_ms=15.3,
        )
        
        assert len(collector.step_data) == 1
        assert collector.step_data[0]['step'] == 0
        assert collector.step_data[0]['mpc_cost'] == 1.5
    
    def test_record_vla_query(self):
        """DataCollector can record VLA queries."""
        import numpy as np
        from src.utils.data_collector import DataCollector
        
        collector = DataCollector(task_name='test_vla')
        
        action = np.array([0.5, 0.6, 0.7, 0.8])
        collector.record_vla_query(
            step=10,
            instruction="pick up the cube",
            rgb_shape=(224, 224, 3),
            action=action,
            latency_ms=650.5,
            success=True,
        )
        
        assert len(collector.vla_queries) == 1
        assert collector.vla_queries[0]['success'] is True
    
    def test_get_summary(self):
        """DataCollector can generate summary statistics."""
        import numpy as np
        from src.utils.data_collector import DataCollector
        
        collector = DataCollector(task_name='test_summary')
        
        # Record some steps
        for i in range(5):
            collector.record_step(
                step=i,
                q=np.array([0.1, 0.2, 0.3]),
                qdot=np.array([0.01, 0.02, 0.03]),
                tau=np.array([1.0, 2.0, 3.0]),
                ee_pos=np.array([0.5, 0.6, 0.7]),
                mpc_cost=1.5,
                mpc_time_ms=15.0 + i,
            )
        
        summary = collector.get_summary()
        assert summary['total_steps'] == 5
        assert 'mpc_timing_ms' in summary
        assert summary['mpc_timing_ms']['mean'] > 15.0


class TestAssetFilesExist:
    """Check that required asset files exist."""
    
    def test_arm3dof_xml_exists(self):
        """Check arm3dof.xml model file exists."""
        asset_dir = Path(__file__).parent.parent / "assets"
        xml_file = asset_dir / "arm3dof.xml"
        assert xml_file.exists(), f"Model not found: {xml_file}"
    
    def test_requirements_txt_exists(self):
        """Check requirements.txt exists."""
        req_file = Path(__file__).parent.parent.parent / "requirements.txt"
        assert req_file.exists(), f"requirements.txt not found"


class TestNoRegressions:
    """Quick regression check: all existing tests still pass."""
    
    def test_kinematics_basic_op(self):
        """Kinematics forward kinematics works."""
        import numpy as np
        from src.dynamics.kinematics_3dof import Arm3DOF
        
        kin = Arm3DOF()
        q = np.array([0.0, 0.0, 0.0])
        
        # Should not raise
        ee_pos = kin.forward_kinematics(q)
        assert ee_pos is not None
        assert ee_pos.shape == (3,)
    
    def test_dynamics_basic_op(self):
        """Dynamics mass matrix computation works."""
        import numpy as np
        from src.dynamics.lagrangian_3dof import Arm3DOFDynamics
        
        dyn = Arm3DOFDynamics()
        q = np.array([0.0, 0.0, 0.0])
        
        # Should not raise
        M = dyn.M(q)
        assert M is not None
        assert M.shape == (3, 3)
        
        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)


def _is_mujoco_available() -> bool:
    """Check if mujoco is installed."""
    try:
        import mujoco
        return True
    except ImportError:
        return False


class TestEnviromentSetup:
    """Test MuJoCo environment setup (skip if mujoco not installed)."""
    
    @pytest.mark.skipif(
        not _is_mujoco_available(),
        reason="mujoco not installed"
    )
    def test_mujoco_env_init(self):
        """MuJoCo environment initializes."""
        from src.environments.mujoco_3dof_env import MuJoCo3DOFEnv
        
        env = MuJoCo3DOFEnv(render_mode=None, headless=True)
        assert env is not None
        env.close()
    
    @pytest.mark.skipif(
        not _is_mujoco_available(),
        reason="mujoco not installed"
    )
    def test_mujoco_env_reset(self):
        """MuJoCo environment can reset."""
        from src.environments.mujoco_3dof_env import MuJoCo3DOFEnv
        
        env = MuJoCo3DOFEnv(render_mode=None, headless=True)
        obs, info = env.reset()
        assert obs is not None
        assert 'state' in obs
        env.close()
    
    @pytest.mark.skipif(
        not _is_mujoco_available(),
        reason="mujoco not installed"
    )
    def test_mujoco_env_step(self):
        """MuJoCo environment can step."""
        import numpy as np
        from src.environments.mujoco_3dof_env import MuJoCo3DOFEnv
        
        env = MuJoCo3DOFEnv(render_mode=None, headless=True)
        env.reset()
        
        action = np.array([0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert reward is not None
        env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
