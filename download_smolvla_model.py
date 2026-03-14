#!/usr/bin/env python3
"""
Download and cache SmolVLA model from Hugging Face.
This script will:
1. Download the model once
2. Cache it locally
3. Verify it can be loaded
4. Test inference
"""

import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directory
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)

MODEL_ID = "lerobot/smolvla_base"

def download_model():
    """Download and cache SmolVLA model."""
    logger.info("=" * 70)
    logger.info(f"Downloading SmolVLA Model: {MODEL_ID}")
    logger.info("=" * 70)
    
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
    
    try:
        # Download image processor
        logger.info(f"\n[1/3] Downloading image processor...")
        processor = AutoImageProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=str(HF_CACHE)
        )
        logger.info(f"✓ Image processor cached")
        
        # Download model
        logger.info(f"\n[2/3] Downloading model (this may take a few minutes)...")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=str(device),
            cache_dir=str(HF_CACHE)
        )
        model = model.eval()
        logger.info(f"✓ Model cached and loaded")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
        
        # Test inference
        logger.info(f"\n[3/3] Testing inference with dummy image...")
        
        # Create dummy image
        dummy_image = Image.new('RGB', (256, 256), color='red')
        
        # Process image
        inputs = processor(dummy_image, return_tensors="pt")
        logger.info(f"  Input keys: {inputs.keys()}")
        logger.info(f"  Pixel values shape: {inputs['pixel_values'].shape}")
        
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Run inference
        with torch.inference_mode():
            outputs = model(**inputs)
        
        logger.info(f"✓ Inference successful")
        logger.info(f"  Output type: {type(outputs)}")
        
        # Check if outputs is a tuple or dict
        if isinstance(outputs, tuple):
            logger.info(f"  Output is tuple with {len(outputs)} elements")
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    logger.info(f"    [{i}] shape: {out.shape}, dtype: {out.dtype}")
                else:
                    logger.info(f"    [{i}] type: {type(out)}")
        else:
            logger.info(f"  Output structure: {outputs}")
        
        # Check model config
        if hasattr(model, 'config'):
            logger.info(f"\nModel config:")
            logger.info(f"  Model type: {model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'}")
            if hasattr(model.config, '__dict__'):
                for key in list(model.config.__dict__.keys())[:5]:
                    logger.info(f"    {key}: {getattr(model.config, key)}")
        
        # Cache location
        cache_path = HF_CACHE / f"models--lerobot--smolvla_base"
        if cache_path.exists():
            total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
            logger.info(f"\n✅ Model cached at: {cache_path}")
            logger.info(f"   Total size: {total_size / 1e9:.2f} GB")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ SmolVLA model ready for production use!")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download/load model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
