#!/usr/bin/env python3
"""Test SmolVLA model from lerobot directly"""

import os
import torch
from pathlib import Path

os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

print("Testing SmolVLA model loading...")
print(f"Cache: {os.environ['HF_HOME']}\n")

try:
    # Try direct lerobot import
    print("[1] Attempting to load via lerobot...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    print("  Loading lerobot SmolVLAPolicy...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    print(f"✓ Policy loaded: {type(policy).__name__}")
    print(f"  Has select_action: {hasattr(policy, 'select_action')}")
    
    # Check config
    if hasattr(policy, 'config'):
        print(f"\n  Config type: {type(policy.config).__name__}")
        print(f"  Input shapes expected:")
        for key in ['observation_shape', 'action_shape', 'input_shape']:
            if hasattr(policy.config, key):
                print(f"    {key}: {getattr(policy.config, key)}")
    
    print("\n✅ SmolVLA ready via lerobot!")
    
except ImportError as e:
    print(f"  ❌ lerobot import failed: {e}")
    print("\n[2] Attempting direct transformers load...")
    
    try:
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained(
            "lerobot/smolvla_base",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        print(f"✓ Model loaded: {type(model).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
    except Exception as e2:
        print(f"  ❌ Transformers load failed: {e2}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
