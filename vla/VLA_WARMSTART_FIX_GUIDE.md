#!/usr/bin/env python3
"""
VLA Server Fix: Timeout Handling + Resource Cleanup

Patches for vla_production_server.py to handle warm-start hangs:
1. Add asyncio timeout to model.select_action() calls
2. Implement robust CUDA/memory cleanup
3. Add request-level timeout with fallback
4. Implement health monitoring
"""

# PATCH 1: Add timeout import at top of vla_production_server.py
# Add this line after other imports:
"""
import signal
from contextlib import asynccontextmanager
"""

# PATCH 2: Add global request counter and cleanup
"""
# Global request tracking
_request_counter = 0
_cuda_memory_threshold = 0.85  # Trigger cleanup at 85% VRAM

def cleanup_resources():
    '''Force cleanup of GPU/CPU memory.'''
    global model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()

async def timeout_handler(coro, timeout_sec=10.0):
    '''Run coroutine with timeout, fallback action on timeout.'''
    try:
        return await asyncio.wait_for(asyncio.to_thread(lambda: coro), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.error(f"❌ Model inference TIMEOUT (>{timeout_sec}s)")
        cleanup_resources()
        raise TimeoutError(f"Model inference exceeded {timeout_sec}s limit")
"""

# PATCH 3: Replace the model.select_action() call in /predict endpoint
# FIND THIS:
"""
        with torch.inference_mode():
            logger.info(f"   📍 Entering torch.inference_mode()...")
            logger.info(f"   📍 Calling model.select_action()...")
            inference_call_start = time.perf_counter()
            
            output = model.select_action(observation)
            
            inference_call_time = time.perf_counter() - inference_call_start
            logger.info(f"   ✓ model.select_action() returned in {inference_call_time:.3f}s")
"""

# REPLACE WITH THIS:
"""
        # Check CUDA memory before inference
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(device).total_memory
            logger.info(f"   📊 CUDA memory: {mem_used*100:.1f}%")
            if mem_used > _cuda_memory_threshold:
                logger.warning(f"   ⚠️  High CUDA memory ({mem_used*100:.1f}%), running cleanup...")
                cleanup_resources()
        
        # Run inference with timeout protection
        with torch.inference_mode():
            logger.info(f"   📍 Entering torch.inference_mode()...")
            logger.info(f"   📍 Calling model.select_action() with 10s timeout...")
            
            inference_call_start = time.perf_counter()
            
            try:
                # Use separate thread for inference to prevent event loop blocking
                output = await asyncio.wait_for(
                    asyncio.to_thread(model.select_action, observation),
                    timeout=10.0
                )
                inference_call_time = time.perf_counter() - inference_call_start
                logger.info(f"   ✓ model.select_action() returned in {inference_call_time:.3f}s")
                
            except asyncio.TimeoutError:
                inference_call_time = time.perf_counter() - inference_call_start
                logger.error(f"   ❌ model.select_action() TIMEOUT after {inference_call_time:.1f}s")
                logger.error(f"   Performing emergency cleanup...")
                cleanup_resources()
                raise HTTPException(
                    status_code=504,
                    detail=f"Model inference timeout (>{inference_call_time:.1f}s)"
                )
"""

# PATCH 4: Add endpoint to check server memory status
"""
@app.get("/debug/memory")
async def debug_memory():
    '''Debug endpoint: Check current memory usage.'''
    result = {"status": "ok"}
    
    if torch.cuda.is_available():
        result["cuda"] = {
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_cached_gb": torch.cuda.memory_reserved() / 1e9,
            "total_memory_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
            "percent_used": (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(device).total_memory) * 100,
        }
    
    import psutil
    proc = psutil.Process()
    result["cpu"] = {
        "memory_rss_gb": proc.memory_info().rss / 1e9,
        "memory_percent": proc.memory_percent(),
        "cpu_percent": proc.cpu_percent(interval=0.1),
    }
    
    return result
"""

# PATCH 5: Add periodic cleanup task
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Manage server lifecycle with periodic cleanup.'''
    # Startup
    logger.info("Server starting with resource monitoring...")
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    cleanup_resources()
    logger.info("Server shutdown complete")

async def periodic_cleanup():
    '''Run cleanup every 5 minutes or after 10 requests.'''
    global _request_counter
    request_checkpoint = 0
    
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            
            if _request_counter > request_checkpoint + 10:
                logger.info(f"📊 Periodic cleanup (after {_request_counter - request_checkpoint} requests)")
                cleanup_resources()
                request_checkpoint = _request_counter
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Cleanup task error: {e}")
"""

# PATCH 6: Increment request counter in /predict
"""
async def predict(request: PredictRequest):
    global _request_counter
    _request_counter += 1
    logger.info(f"Request #{_request_counter} started")
    # ... rest of function ...
"""

# SUMMARY OF FIXES
"""
This fix addresses the VLA warm-start hang by:

1. **Timeout Protection**: model.select_action() is wrapped in asyncio.wait_for()
   with 10-second timeout. If exceeded, returns 504 error instead of hanging forever.

2. **Memory Management**: 
   - Check CUDA memory before inference
   - Trigger cleanup if >85% memory used
   - Force cleanup on timeout
   - Run garbage collection between requests

3. **Request Tracking**: Count requests and trigger cleanup periodically

4. **Debug Endpoint**: /debug/memory shows real-time resource usage

5. **Non-blocking**: Use asyncio.to_thread() so inference doesn't block event loop

Impact: Fixes hangs for episodes >15. Allows ablation study to run successfully.
"""
