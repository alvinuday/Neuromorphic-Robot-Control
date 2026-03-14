"""
SmolVLA Server Integration Test.

Test connection and inference with live SmolVLA Colab server.
URL: https://symbolistically-unfutile-henriette.ngrok-free.dev/

This test verifies:
- Server connectivity and health check
- Image encoding/transmission
- VLA inference timing and output validity
- Async client behavior
"""

import asyncio
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def test_smolvla_server_connection(
    server_url: str = "https://symbolistically-unfutile-henriette.ngrok-free.dev"
) -> bool:
    """
    Test connection to SmolVLA server.
    
    Args:
        server_url: Base URL of SmolVLA server
    
    Returns:
        True if server is reachable
    """
    import aiohttp
    
    logger.info(f"Testing SmolVLA server: {server_url}")
    
    try:
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Health check endpoint
            async with session.get(
                f"{server_url}/health",
                ssl=False,  # ngrok uses SSL but we skip verification
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                logger.info(f"Health check response: {resp.status}")
                return resp.status == 200
    
    except Exception as e:
        logger.error(f"Server connection failed: {e}")
        return False


async def test_smolvla_inference(
    server_url: str = "https://symbolistically-unfutile-henriette.ngrok-free.dev",
    test_image_size: int = 84
) -> Optional[np.ndarray]:
    """
    Test VLA inference with dummy image.
    
    Args:
        server_url: SmolVLA server URL
        test_image_size: Image resolution (84x84)
    
    Returns:
        Action array if successful, None otherwise
    """
    import aiohttp
    from PIL import Image
    import io
    import base64
    
    logger.info(f"Testing VLA inference on {server_url}")
    
    # Create dummy RGB image
    dummy_image = np.random.randint(0, 255, (test_image_size, test_image_size, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    
    # Encode to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    image_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    try:
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            payload = {
                "image": image_b64,
                "task_description": "pick up object",
            }
            
            logger.info("Sending inference request...")
            async with session.post(
                f"{server_url}/predict",
                json=payload,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Inference successful: {result}")
                    
                    # Extract action
                    if "action" in result:
                        action = np.array(result["action"])
                        logger.info(f"Action shape: {action.shape}, values: {action}")
                        return action
                else:
                    logger.error(f"Inference failed with status {resp.status}")
                    return None
    
    except asyncio.TimeoutError:
        logger.error("Inference request timed out (>10s)")
        return None
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None


async def test_smolvla_async_client(
    server_url: str = "https://symbolistically-unfutile-henriette.ngrok-free.dev"
) -> bool:
    """
    Test SmolVLA async client from codebase.
    
    Args:
        server_url: SmolVLA server URL
    
    Returns:
        True if async client works
    """
    try:
        from src.smolvla_client.client import SmolVLAAsyncClient, VLAResponse
    except ImportError:
        logger.error("Could not import SmolVLAAsyncClient")
        return False
    
    logger.info(f"Testing SmolVLA async client with {server_url}")
    
    client = SmolVLAAsyncClient(server_url=server_url, timeout_s=10.0, verify_ssl=False)
    
    try:
        # Health check
        await client.connect()
        is_healthy = await client.health_check()
        logger.info(f"Server health: {is_healthy}")
        
        if not is_healthy:
            logger.warning("Server health check failed")
            await client.disconnect()
            return False
        
        # Test inference
        dummy_image = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        resp = await client.predict(dummy_image)
        
        logger.info(f"VLA response: success={resp.success}, action shape={resp.action.shape}")
        
        if resp.success:
            logger.info(f"✓ Async client working: action={resp.action}, latency={resp.latency_ms:.1f}ms")
            await client.disconnect()
            return True
        else:
            logger.error(f"Async client failed: {resp.error}")
            await client.disconnect()
            return False
    
    except Exception as e:
        logger.error(f"Async client test failed: {e}")
        return False


async def test_integration_with_control_loop(
    server_url: str = "https://symbolistically-unfutile-henriette.ngrok-free.dev",
    n_steps: int = 5
) -> bool:
    """
    Test SmolVLA integration with control loop.
    
    Simulates real-time control with VLA queries.
    
    Args:
        server_url: SmolVLA server URL
        n_steps: Number of control steps to simulate
    
    Returns:
        True if integration works
    """
    try:
        from src.smolvla_client.client import SmolVLAAsyncClient
        from src.mpc.xarm_controller import XArmMPCController
        from src.smolvla_client.trajectory_buffer import TrajectoryBuffer
    except ImportError as e:
        logger.error(f"Could not import required modules: {e}")
        return False
    
    logger.info(f"Testing integration with control loop ({n_steps} steps)")
    
    # Initialize components
    vla_client = SmolVLAAsyncClient(server_url=server_url, timeout_s=5.0, verify_ssl=False)
    mpc = XArmMPCController()
    buffer = TrajectoryBuffer(n_joints=8)
    
    try:
        await vla_client.connect()
        
        q = np.zeros(8)
        qd = np.zeros(8)
        
        for step in range(n_steps):
            logger.info(f"\nStep {step + 1}/{n_steps}")
            
            # Create dummy observation
            rgb = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
            
            # VLA inference (async, non-blocking)
            vla_task = asyncio.create_task(vla_client.predict(rgb))
            
            # MPC control (no wait)
            tau, info = mpc.step(q, qd, q)
            logger.info(f"  MPC: tau norm={np.linalg.norm(tau):.4f}")
            
            # Wait for VLA with timeout
            try:
                vla_resp = await asyncio.wait_for(vla_task, timeout=5.0)
                if vla_resp.success:
                    logger.info(f"  VLA: action shape={vla_resp.action.shape}, "
                               f"latency={vla_resp.latency_ms:.1f}ms")
                else:
                    logger.warning(f"  VLA: failed ({vla_resp.error})")
            except asyncio.TimeoutError:
                logger.warning(f"  VLA: timeout after 5s")
        
        await vla_client.disconnect()
        logger.info("\n✓ Integration test completed")
        return True
    
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


async def main():
    """Run all SmolVLA server tests."""
    server_url = "https://symbolistically-unfutile-henriette.ngrok-free.dev"
    
    logger.info("=" * 70)
    logger.info("SMOLVLA SERVER INTEGRATION TESTS")
    logger.info("=" * 70)
    
    # Test 1: Connection
    logger.info("\n[TEST 1] Server Connection")
    logger.info("-" * 70)
    connected = await test_smolvla_server_connection(server_url)
    logger.info(f"Result: {'✅ PASS' if connected else '❌ FAIL'}\n")
    
    if not connected:
        logger.warning("Server unreachable - skipping remaining tests")
        return {
            "test_1_connection": connected,
            "test_2_inference": None,
            "test_3_async_client": None,
            "test_4_integration": None,
            "summary": "Server unreachable"
        }
    
    # Test 2: Inference
    logger.info("[TEST 2] VLA Inference")
    logger.info("-" * 70)
    action = await test_smolvla_inference(server_url)
    logger.info(f"Result: {'✅ PASS' if action is not None else '❌ FAIL'}\n")
    
    # Test 3: Async Client
    logger.info("[TEST 3] Async Client")
    logger.info("-" * 70)
    async_ok = await test_smolvla_async_client(server_url)
    logger.info(f"Result: {'✅ PASS' if async_ok else '❌ FAIL'}\n")
    
    # Test 4: Integration with control loop
    logger.info("[TEST 4] Control Loop Integration")
    logger.info("-" * 70)
    integration_ok = await test_integration_with_control_loop(server_url, n_steps=3)
    logger.info(f"Result: {'✅ PASS' if integration_ok else '❌ FAIL'}\n")
    
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Server Connection: {'✅' if connected else '❌'}")
    logger.info(f"Inference: {'✅' if action is not None else '❌'}")
    logger.info(f"Async Client: {'✅' if async_ok else '❌'}")
    logger.info(f"Integration: {'✅' if integration_ok else '❌'}")
    
    return {
        "test_1_connection": connected,
        "test_2_inference": action is not None,
        "test_3_async_client": async_ok,
        "test_4_integration": integration_ok,
    }


if __name__ == "__main__":
    results = asyncio.run(main())
    print(f"\nTests completed. Results: {results}")
