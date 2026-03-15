#!/usr/bin/env python3
"""
Production SmolVLA Server
Properly handles model caching and inference
"""

import asyncio
import base64
import gc
import io
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer

# Patch torch.xpu for CPU-only builds (Intel GPU support not available)
if not hasattr(torch, 'xpu'):
    class DummyXPU:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def empty_cache():
            pass
    torch.xpu = DummyXPU()

# Setup paths and logging
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Global model state
MODEL_ID = "lerobot/smolvla_base"
model = None
tokenizer = None
device = None
model_ready = False

# Global request tracking for resource management
_request_counter = 0
_cuda_memory_threshold = 0.75  # Trigger cleanup at 75% VRAM (more aggressive)
_last_cleanup_time = 0
_min_cleanup_interval = 5.0  # Cleanup at most every 5 seconds
_consecutive_timeouts = 0
_max_consecutive_timeouts = 3

def cleanup_resources(aggressive=False):
    """Force cleanup of GPU/CPU memory.
    
    Args:
        aggressive: If True, also clears model weights from CUDA
    """
    global model, _last_cleanup_time
    
    current_time = time.time()
    if current_time - _last_cleanup_time < _min_cleanup_interval and not aggressive:
        return  # Skip cleanup if done recently
    
    try:
        if torch.cuda.is_available():
            # More aggressive CUDA cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Force Python garbage collection
        gc.collect()
        
        if aggressive and model is not None:
            # Move model to CPU and clear CUDA
            logger.warning("   ⚠️  AGGRESSIVE cleanup: moving model to CPU temporarily")
            try:
                model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # Move back to original device
                if device != "cpu":
                    model.to(device)
            except Exception as e:
                logger.warning(f"   ⚠️  Could not move model: {e}")
        
        _last_cleanup_time = current_time
        logger.info("   🧹 Resource cleanup complete")
    except Exception as e:
        logger.error(f"   ❌ Cleanup failed: {e}")

# ============================================================================
# Data Models
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    model_id: str
    device: str
    ready: bool


class PredictRequest(BaseModel):
    rgb_image_b64: str
    task: str = "reaching"
    instruction: str = ""
    state: Optional[List[float]] = None  # Optional joint state vector


class PredictResponse(BaseModel):
    action: List[float]
    action_std: List[float]
    latency_ms: float
    success: bool


# ============================================================================
# Model Loading
# ============================================================================

def load_model_on_startup():
    """Load SmolVLA model once on server start."""
    global model, device, model_ready, tokenizer
    
    logger.info("=" * 70)
    logger.info(f"Loading SmolVLA Model: {MODEL_ID}")
    logger.info("=" * 70)
    
    try:
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"✓ Using Apple Metal (MPS)")
        else:
            device = torch.device("cpu")
            logger.info(f"✓ Using CPU")
        
        # Use LeRobot's SmolVLAPolicy (proper way to load SmolVLA)
        logger.info(f"Importing LeRobot SmolVLAPolicy...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        
        logger.info(f"Loading model from HuggingFace: {MODEL_ID}...")
        model = SmolVLAPolicy.from_pretrained(MODEL_ID)
        
        # Load tokenizer for language encoding
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                logger.info(f"✓ Tokenizer loaded from {MODEL_ID}")
            except Exception as e:
                logger.warning(f"Could not load tokenizer from {MODEL_ID}: {e}")
                # Fall back to BERT tokenizer (common for vision-language models)
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                logger.info(f"✓ Tokenizer loaded (using bert-base-uncased fallback)")
        except Exception as e:
            logger.warning(f"Could not load any tokenizer: {e}")
            tokenizer = None
        
        # Move to device
        model = model.to(device).eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Parameters: {total_params:.1f}M")
        logger.info(f"  Device: {device}")
        
        # Verify model structure
        logger.info(f"  Model type: {type(model).__name__}")
        logger.info(f"  Has forward: {hasattr(model, 'forward')}")
        logger.info(f"  Has select_action: {hasattr(model, 'select_action')}")
        
        # Warmup the model with a dummy inference
        logger.info("\n🔥 WARMING UP MODEL (this will take 3-5 minutes on first startup)...")
        try:
            warmup_image = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
            warmup_state = torch.zeros(1, 7, device=device, dtype=torch.float32)
            
            # Use char-level tokenization for warmup (consistent with fallback)
            warmup_instruction = "reach forward"
            warmup_tokens = torch.tensor([[min(ord(c), 50000) for c in warmup_instruction]], dtype=torch.long, device=device)
            warmup_attention = torch.ones((1, warmup_tokens.shape[1]), dtype=torch.bool, device=device)
            
            warmup_obs = {
                "observation.images.camera1": warmup_image,
                "observation.images.camera2": warmup_image,
                "observation.images.camera3": warmup_image,
                "observation.state": warmup_state,
                "observation.language.tokens": warmup_tokens,
                "observation.language.attention_mask": warmup_attention,
            }
            
            logger.info("   📍 Running warmup inference...")
            warmup_start = time.perf_counter()
            with torch.inference_mode():
                _ = model.select_action(warmup_obs)
            warmup_time = time.perf_counter() - warmup_start
            logger.info(f"   ✓ Warmup complete in {warmup_time:.1f}s")
            logger.info("   ✓ Model compiled and ready for fast inference")
        except Exception as e:
            logger.warning(f"⚠️  Warmup inference failed (non-fatal): {e}")
            logger.warning(f"   First real inference may still take 3-5 minutes")
        
        model_ready = True
        logger.info("\n✅ Model ready for inference")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        model_ready = False
        return False


# ============================================================================
# FastAPI Startup
# ============================================================================

app = FastAPI(
    title="SmolVLA Production Server",
    description="Production inference server for SmolVLA robot control model",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup."""
    load_model_on_startup()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Server health check endpoint."""
    return HealthResponse(
        status="ok",
        model_id=MODEL_ID,
        device=str(device) if device else "not initialized",
        ready=model_ready
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Get 7-D action prediction from RGB image.
    
    The model takes an RGB image and optionally task description,
    and outputs a 7-dimensional action for xArm 7-DOF robotic arm.
    """
    
    start_time = time.perf_counter()
    
    logger.info("=" * 80)
    logger.info("🔷 PREDICT API CALLED")
    logger.info("=" * 80)
    
    if not model_ready or model is None:
        logger.error("❌ Model not ready")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be initializing."
        )
    
    try:
        # Step 1: Decode image
        logger.info("📸 STEP 1: Decoding base64 image...")
        decode_start = time.perf_counter()
        image_data = base64.b64decode(request.rgb_image_b64)
        logger.info(f"   ✓ Decoded {len(image_data)} bytes")
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        decode_time = time.perf_counter() - decode_start
        logger.info(f"   ✓ Image opened: {pil_image.size}, took {decode_time:.3f}s")
        
        # Step 2: Preprocess image
        logger.info("🖼️  STEP 2: Preprocessing image...")
        preprocess_start = time.perf_counter()
        pil_image = pil_image.resize((256, 256), Image.LANCZOS)
        image_array = np.array(pil_image, dtype=np.uint8)
        logger.info(f"   ✓ Resized to 256x256, array shape: {image_array.shape}")
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        preprocess_time = time.perf_counter() - preprocess_start
        logger.info(f"   ✓ Converted to tensor: {image_tensor.shape}, took {preprocess_time:.3f}s")
        
        # Step 3: Build observation dict
        logger.info("📋 STEP 3: Building observation dictionary...")
        obs_start = time.perf_counter()
        
        # Add multi-camera images
        observation = {
            "observation.images.camera1": image_tensor,
            "observation.images.camera2": image_tensor,
            "observation.images.camera3": image_tensor,
        }
        logger.info(f"   ✓ Added 3x camera images")
        
        # Add state
        if hasattr(request, 'state') and request.state is not None:
            state_array = np.array(request.state, dtype=np.float32)
            state_tensor = torch.from_numpy(state_array).float().to(device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            observation["observation.state"] = state_tensor
            logger.info(f"   ✓ Added state: {observation['observation.state'].shape}")
        else:
            state_tensor = torch.zeros(1, 7, dtype=torch.float32, device=device)
            observation["observation.state"] = state_tensor
            logger.info(f"   ✓ Added default state (zeros): {observation['observation.state'].shape}")
        
        # Add language tokens
        instruction = request.instruction if hasattr(request, 'instruction') and request.instruction else "reach"
        if tokenizer is not None:
            try:
                tokens_dict = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, max_length=128)
                observation["observation.language.tokens"] = tokens_dict["input_ids"].to(device)
                # Convert attention mask to boolean for the model
                attention_mask = tokens_dict["attention_mask"].to(device).bool()
                observation["observation.language.attention_mask"] = attention_mask
                logger.info(f"   ✓ Tokenized instruction '{instruction}'")
                logger.info(f"      - tokens: {observation['observation.language.tokens'].shape}, dtype: {observation['observation.language.tokens'].dtype}")
                logger.info(f"      - attention_mask: {observation['observation.language.attention_mask'].shape}, dtype: {observation['observation.language.attention_mask'].dtype}")
            except Exception as e:
                logger.warning(f"   ⚠️  Tokenization failed: {e}, using char-level encoding")
                # Character-level fallback tokenization (instruction -> ASCII values)
                char_tokens = [min(ord(c), 50000) for c in instruction[:128]]
                while len(char_tokens) < 10:
                    char_tokens.append(0)
                tokens_tensor = torch.tensor([char_tokens], dtype=torch.long, device=device)
                observation["observation.language.tokens"] = tokens_tensor
                attention_tensor = torch.ones((1, len(char_tokens)), dtype=torch.bool, device=device)
                observation["observation.language.attention_mask"] = attention_tensor
                logger.info(f"   ✓ Using char-level encoding ({len(char_tokens)} tokens)")
        else:
            logger.warning(f"   ⚠️  No tokenizer loaded, using char-level encoding for '{instruction}'")
            # Character-level tokenization fallback
            char_tokens = [min(ord(c), 50000) for c in instruction[:128]]
            while len(char_tokens) < 10:
                char_tokens.append(0)
            tokens_tensor = torch.tensor([char_tokens], dtype=torch.long, device=device)
            observation["observation.language.tokens"] = tokens_tensor
            attention_tensor = torch.ones((1, len(char_tokens)), dtype=torch.bool, device=device)
            observation["observation.language.attention_mask"] = attention_tensor
            logger.info(f"   ✓ Using char-level encoding ({len(char_tokens)} tokens)")
        
        obs_time = time.perf_counter() - obs_start
        logger.info(f"   ✓ Observation dict complete: {list(observation.keys())}, took {obs_time:.3f}s")
        
        # Step 4: Run inference
        logger.info("🧠 STEP 4: Running model inference...")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(f"   Device: {device}")
        
        # PROACTIVE cleanup before inference
        _request_counter_val = _request_counter + 1
        if _request_counter_val % 5 == 0:  # Every 5 requests, do full cleanup
            logger.info(f"   📍 Request #{_request_counter_val}: Running proactive cleanup...")
            cleanup_resources()
        
        # Check CUDA memory before inference
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated()
            mem_reserved = torch.cuda.memory_reserved()
            props = torch.cuda.get_device_properties(device)
            mem_used = mem_allocated / props.total_memory
            logger.info(f"   📊 CUDA memory: allocated={mem_allocated/1e6:.0f}MB, reserved={mem_reserved/1e6:.0f}MB, total={props.total_memory/1e6:.0f}MB ({mem_used*100:.1f}%)")
            
            if mem_used > _cuda_memory_threshold:
                logger.warning(f"   ⚠️  High CUDA memory ({mem_used*100:.1f}%), running cleanup...")
                cleanup_resources(aggressive=mem_used > 0.90)
        
        inference_start = time.perf_counter()
        try:
            with torch.inference_mode():
                logger.info(f"   📍 Entering torch.inference_mode()...")
                logger.info(f"   📍 Calling model.select_action() with 45s timeout...")
                inference_call_start = time.perf_counter()
                
                # Run inference in separate thread with timeout protection
                try:
                    def run_select_action():
                        return model.select_action(observation)
                    
                    # Use 45s timeout (reasonable for model inference including warmup)
                    output = await asyncio.wait_for(
                        asyncio.to_thread(run_select_action),
                        timeout=45.0
                    )
                    inference_call_time = time.perf_counter() - inference_call_start
                    logger.info(f"   ✓ model.select_action() returned in {inference_call_time:.3f}s")
                    logger.info(f"   Output type: {type(output)}")
                    
                    # Reset timeout counter on success
                    _consecutive_timeouts = 0
                    
                except asyncio.TimeoutError:
                    inference_call_time = time.perf_counter() - inference_call_start
                    logger.error(f"   ❌ model.select_action() TIMEOUT after {inference_call_time:.1f}s")
                    logger.error(f"   Performing emergency cleanup...")
                    
                    # Track consecutive timeouts
                    _consecutive_timeouts += 1
                    cleanup_resources(aggressive=True)
                    
                    if _consecutive_timeouts >= _max_consecutive_timeouts:
                        logger.critical(f"   🔴 {_consecutive_timeouts} consecutive timeouts detected, may need server restart")
                    
                    raise HTTPException(
                        status_code=504,
                        detail=f"Model inference timeout (>{inference_call_time:.1f}s)"
                    )
                
                if isinstance(output, torch.Tensor):
                    action_tensor = output
                    logger.info(f"   ✓ Output is tensor: {action_tensor.shape}, dtype: {action_tensor.dtype}")
                elif isinstance(output, dict):
                    logger.info(f"   Output is dict with keys: {output.keys()}")
                    if "action" in output:
                        action_tensor = output["action"]
                        logger.info(f"   ✓ Extracted 'action': {action_tensor.shape}")
                    else:
                        action_tensor = next(
                            (v for v in output.values() if isinstance(v, torch.Tensor)),
                            None
                        )
                        if action_tensor is None:
                            raise ValueError(f"No tensor found in model output: {output.keys()}")
                        logger.info(f"   ✓ Extracted first tensor: {action_tensor.shape}")
                else:
                    action_tensor = torch.tensor(output, dtype=torch.float32, device=device)
                    logger.info(f"   ✓ Converted output to tensor: {action_tensor.shape}")
        except Exception as e:
            inference_call_time = time.perf_counter() - inference_start
            logger.error(f"❌ Inference failed after {inference_call_time:.3f}s")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}", exc_info=True)
            raise
        
        inference_time = time.perf_counter() - inference_start
        logger.info(f"   ✓ Inference complete, took {inference_time:.3f}s")
        
        # Step 5: Extract action
        logger.info("✂️  STEP 5: Extracting and formatting action...")
        extract_start = time.perf_counter()
        
        if action_tensor is None:
            raise ValueError("No action tensor generated from model")
        
        action_np = action_tensor.cpu().numpy()
        logger.info(f"   ✓ Action array shape: {action_np.shape}, dtype: {action_np.dtype}")
        
        if action_np.ndim > 1:
            action_np = action_np[0]  # Remove batch dimension
            logger.info(f"   ✓ Removed batch dimension: {action_np.shape}")
        
        action = action_np[:7].tolist() if len(action_np) >= 7 else action_np.tolist()
        logger.info(f"   ✓ Extracted first 7 values: {len(action)} values")
        
        # Ensure exactly 7 values
        while len(action) < 7:
            action.append(0.0)
        action = action[:7]
        
        # Clip to bounds
        action = [float(np.clip(a, -1.0, 1.0)) for a in action]
        logger.info(f"   ✓ Final action (clipped): {[round(a, 4) for a in action]}")
        
        extract_time = time.perf_counter() - extract_start
        logger.info(f"   ✓ Extraction complete, took {extract_time:.3f}s")
        
        # Step 6: Return response
        total_time = time.perf_counter() - start_time
        logger.info("✅ SUCCESS")
        logger.info(f"   Total latency: {total_time*1000:.1f}ms")
        logger.info("=" * 80)
        
        return PredictResponse(
            action=action,
            action_std=[0.1] * 7,
            latency_ms=total_time * 1000,
            success=True
        )
        
    except Exception as e:
        total_time = time.perf_counter() - start_time
        logger.error("❌ PREDICTION FAILED")
        logger.error(f"   Total time before failure: {total_time*1000:.1f}ms")
        logger.error(f"   Error: {type(e).__name__}: {str(e)}")
        logger.error("=" * 80)
        
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )


@app.get("/")
async def root():
    """API documentation and status."""
    return {
        "name": "SmolVLA Production Server",
        "version": "1.0.0",
        "status": "running",
        "model": MODEL_ID,
        "model_ready": model_ready,
        "device": str(device) if device else "not initialized",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("SmolVLA Production Server Starting")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Cache: {CACHE_DIR}")
    logger.info(f"Server: http://0.0.0.0:8000")
    logger.info(f"Docs: http://localhost:8000/docs")
    logger.info(f"Hot-reload: ENABLED (restarts on file changes)")
    logger.info("=" * 70 + "\n")
    
    uvicorn.run(
        "vla_production_server:app",  # Module:app format for reload to work
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=True,  # Enable auto-reload on file changes,
        reload_dirs=["vla"],  # Watch the vla/ directory for changes
    )
