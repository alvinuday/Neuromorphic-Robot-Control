"""
Background VLA Query Thread: Non-blocking asynchronous VLA queries.

Runs in separate thread with asyncio event loop.
Polls VLA every 200ms, updates TrajectoryBuffer with new subgoals.
Never blocks the main control loop.
"""

import asyncio
import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VLAQueryThread:
    """
    Manager for background VLA query thread.

    Starts a separate thread that:
    1. Runs asyncio event loop
    2. Polls VLA every N seconds
    3. Updates TrajectoryBuffer with new subgoals
    4. Handles connection failures gracefully
    """

    def __init__(
        self,
        smolvla_client,
        trajectory_buffer,
        poll_interval_s: float = 0.2,
        query_timeout_s: float = 2.0,
    ):
        """
        Initialize background VLA query thread manager.

        Args:
            smolvla_client: SmolVLAClient instance for async queries
            trajectory_buffer: TrajectoryBuffer for storing subgoals
            poll_interval_s: How often to query VLA (seconds)
            query_timeout_s: Timeout for each VLA query (seconds)
        """
        self.vla_client = smolvla_client
        self.trajectory_buffer = trajectory_buffer
        self.poll_interval = poll_interval_s
        self.query_timeout = query_timeout_s

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Statistics
        self.query_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_query_time = 0.0

    def start(
        self,
        rgb_source,
        instruction: str,
    ) -> bool:
        """
        Start background VLA query thread.

        Args:
            rgb_source: Callable that returns current RGB frame [H, W, 3]
            instruction: Task instruction string

        Returns:
            True if thread started successfully, False otherwise
        """
        if self._running:
            logger.warning("[VLAQueryThread] Already running")
            return False

        self._running = True
        self._stop_event.clear()

        # Start thread
        self._thread = threading.Thread(
            target=self._thread_loop,
            args=(rgb_source, instruction),
            daemon=True,
        )
        self._thread.start()

        logger.info("[VLAQueryThread] Started background polling thread")
        return True

    def stop(self) -> None:
        """Stop background VLA query thread."""
        if not self._running:
            logger.warning("[VLAQueryThread] Not running")
            return

        self._running = False
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        logger.info(
            f"[VLAQueryThread] Stopped. "
            f"Queries: {self.query_count}, "
            f"Success: {self.success_count}, "
            f"Failures: {self.failure_count}"
        )

    def _thread_loop(self, rgb_source, instruction: str) -> None:
        """
        Main loop for background thread.

        Args:
            rgb_source: Callable that returns RGB frame
            instruction: Task instruction
        """
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(
                self._poll_vla_async(rgb_source, instruction)
            )
        except Exception as e:
            logger.error(f"[VLAQueryThread] Unexpected error in loop: {e}")
        finally:
            loop.close()
            self._running = False

    async def _poll_vla_async(self, rgb_source, instruction: str) -> None:
        """
        Async polling loop.

        Args:
            rgb_source: Callable that returns RGB frame
            instruction: Task instruction
        """
        while self._running and not self._stop_event.is_set():
            try:
                # Get current RGB frame from source
                try:
                    rgb = rgb_source()
                    if rgb is None:
                        await asyncio.sleep(self.poll_interval)
                        continue
                except Exception as e:
                    logger.warning(f"[VLAQueryThread] Failed to get RGB: {e}")
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Query VLA with timeout
                try:
                    t0 = time.perf_counter()
                    response = await asyncio.wait_for(
                        self.vla_client.query_action(
                            rgb=rgb,
                            instruction=instruction,
                            current_joints=None,
                        ),
                        timeout=self.query_timeout,
                    )

                    elapsed = (time.perf_counter() - t0) * 1000

                    if response is not None:
                        # Extract subgoal (end-effector position as proxy for joint angles)
                        # For now, use action chunk if available
                        if hasattr(response, "action_chunk"):
                            # Use first action's first 3 elements as joint angles
                            q_goal = response.action_chunk[0, :3].astype(
                                np.float32
                            )
                            self.trajectory_buffer.update_subgoal(q_goal)
                            self.success_count += 1
                            logger.debug(
                                f"[VLAQueryThread] Query {self.query_count} "
                                f"success in {elapsed:.1f}ms: {q_goal}"
                            )
                        else:
                            logger.warning(
                                f"[VLAQueryThread] Invalid response structure"
                            )
                            self.failure_count += 1
                    else:
                        logger.warning(
                            f"[VLAQueryThread] Query {self.query_count} "
                            f"returned None (timeout or error)"
                        )
                        self.failure_count += 1

                except asyncio.TimeoutError:
                    logger.warning(
                        f"[VLAQueryThread] Query {self.query_count} timed out "
                        f"(> {self.query_timeout}s)"
                    )
                    self.failure_count += 1
                except Exception as e:
                    logger.warning(
                        f"[VLAQueryThread] Query {self.query_count} failed: {e}"
                    )
                    self.failure_count += 1

                self.query_count += 1
                self.last_query_time = time.perf_counter()

                # Sleep before next query
                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"[VLAQueryThread] Unexpected error in polling: {e}")
                await asyncio.sleep(self.poll_interval)

    def get_stats(self) -> dict:
        """
        Return thread statistics.

        Returns:
            dict: Statistics including query counts and success rate
        """
        success_rate = (
            (self.success_count / self.query_count * 100)
            if self.query_count > 0
            else 0
        )
        return {
            "running": self._running,
            "query_count": self.query_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate_percent": success_rate,
            "last_query_time": self.last_query_time,
        }


def poll_vla_background(
    smolvla_client,
    trajectory_buffer,
    rgb_source,
    instruction: str,
    poll_interval_s: float = 0.2,
    query_timeout_s: float = 2.0,
) -> VLAQueryThread:
    """
    Convenience function to start and return VLA polling thread.

    Args:
        smolvla_client: SmolVLAClient instance
        trajectory_buffer: TrajectoryBuffer instance
        rgb_source: Callable that returns current RGB frame
        instruction: Task instruction
        poll_interval_s: Polling interval in seconds
        query_timeout_s: Query timeout in seconds

    Returns:
        VLAQueryThread manager instance
    """
    manager = VLAQueryThread(
        smolvla_client=smolvla_client,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=poll_interval_s,
        query_timeout_s=query_timeout_s,
    )
    manager.start(rgb_source, instruction)
    return manager
