#!/usr/bin/env python3
"""
Production SmolVLA Server
Properly handles model caching and inference
"""

import asyncio
import base64
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
device = None
model_ready = False

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
    global model, device, model_ready
    
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
        
        model_ready = True
        logger.info("✅ Model ready for inference")
        
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
    
    if not model_ready or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be initializing."
        )
    
    try:
        # Decode image from base64
        logger.debug("Decoding image...")
        image_data = base64.b64decode(request.rgb_image_b64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess image
        logger.debug(f"Input image size: {pil_image.size}")
        pil_image = pil_image.resize((256, 256), Image.LANCZOS)
        image_array = np.array(pil_image, dtype=np.uint8)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        logger.debug(f"Tensor shape: {image_tensor.shape}")
        
        # Run inference
        logger.debug("Running inference...")
        with torch.inference_mode():
            # Try different inference methods
            if hasattr(model, "select_action"):
                # Method 1: use select_action if available
                output = model.select_action({"observation": image_tensor})
                if isinstance(output, torch.Tensor):
                    action_tensor = output
                else:
                    action_tensor = torch.tensor(output, dtype=torch.float32)
            else:
                # Method 2: use forward/call
                output = model(image_tensor)
                if isinstance(output, torch.Tensor):
                    action_tensor = output
                elif isinstance(output, (list, tuple)):
                    action_tensor = torch.tensor(output[0] if len(output) > 0 else output, dtype=torch.float32)
                else:
                    raise ValueError(f"Unexpected model output type: {type(output)}")
        
        # Extract action (7-D for xArm)
        action_np = action_tensor.cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np[0]  # Remove batch dimension
        
        action = action_np [:7].tolist()  # Take first 7 dims
        
        # Ensure we have exactly 7 values
        while len(action) < 7:
            action.append(0.0)
        action = action[:7]
        
        # Clip to reasonable bounds
        action = [float(np.clip(a, -1.0, 1.0)) for a in action]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"✓ Prediction successful (latency: {latency_ms:.1f}ms)")
        logger.info(f"  Action: {[round(a, 4) for a in action]}")
        
        return PredictResponse(
            action=action,
            action_std=[0.1] * 7,
            latency_ms=latency_ms,
            success=True
        )
        
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Prediction failed: {type(e).__name__}: {e}")
        
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
    logger.info("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
