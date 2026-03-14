#!/usr/bin/env python3
"""Quick test of model loading"""
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'MPS: {torch.backends.mps.is_available()}')

try:
    from transformers import AutoModel
    print('✓ Transformers imported')
    
    print('Loading SmolVLA model...')
    model = AutoModel.from_pretrained(
        'lerobot/smolvla_base',
        trust_remote_code=True,
        device_map='cpu',
        torch_dtype=torch.float32
    )
    print(f'✓ Model loaded: {type(model).__name__}')
    print(f'  Ready for inference')
    
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
