#!/usr/bin/env python3
"""Quick test of SmolVLA model loading and caching"""

import os
import torch
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

print("Starting SmolVLA model test...")
print(f"Cache dir: {os.environ['HF_HOME']}")

try:
    print("\n[1] Loading processor...")
    processor = AutoImageProcessor.from_pretrained("lerobot/smolvla_base", trust_remote_code=True)
    print("✓ Processor loaded")
    
    print("\n[2] Loading model (this may take a moment)...")
    model = AutoModel.from_pretrained(
        "lerobot/smolvla_base",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print("✓ Model loaded")
    print(f"  Type: {type(model).__name__}")
    print(f"  Has select_action: {hasattr(model, 'select_action')}")
    
    # List key methods
    methods = [a for a in dir(model) if not a.startswith('_') and callable(getattr(model, a))]
    print(f"\n  Sample methods: {methods[:15]}")
    
    # Check device
    print(f"\n[3] Testing forward pass...")
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_img = Image.new('RGB', (256, 256), color='red')
    
    # Process
    inputs = processor(dummy_img, return_tensors="pt")
    print(f"  Inputs: {list(inputs.keys())}")
    print(f"  Pixel values shape: {inputs['pixel_values'].shape}")
    
    # Forward
    with torch.inference_mode():
        outputs = model(**inputs)
    
    print(f"✓ Forward pass successful")
    print(f"  Output type: {type(outputs).__name__}")
    if hasattr(outputs, 'shape'):
        print(f"  Output shape: {outputs.shape}")
    elif isinstance(outputs, (tuple, list)):
        print(f"  Output has {len(outputs)} elements")
        for i, o in enumerate(outputs[:3]):
            if hasattr(o, 'shape'):
                print(f"    [{i}] shape: {o.shape}")
            else:
                print(f"    [{i}] type: {type(o).__name__}")
    elif isinstance(outputs, dict):
        print(f"  Output keys: {list(outputs.keys())}")
    
    print("\n" + "="*70)
    print("✅ SmolVLA model ready!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
