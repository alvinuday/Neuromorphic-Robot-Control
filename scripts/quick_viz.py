#!/usr/bin/env python3
from PIL import Image, ImageDraw
import json
import numpy as np
from pathlib import Path

output_dir = Path('results/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Test PIL
test_img = Image.new('RGB', (400, 300), (200, 200, 200))
test_img.save(str(output_dir / 'test_pil.png'))
print("✓ PIL working")

# Load data
try:
    with open('results/vla_sl_mpc_real_data/integration_results.json') as f:
        sl_data = json.load(f)
    print("✓ SL data loaded")
except:
    sl_data = None

try:
    with open('results/vla_osqp_mpc_real_data/integration_results.json') as f:
        osqp_data = json.load(f)
    print("✓ OSQP data loaded")
except:
    osqp_data = None

# Simple frame generator
def make_frame(step, total, solver_name, vla_ms, mpc_ms, task):
    img = Image.new('RGB', (800, 600), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    
    # Arm base
    base_x, base_y = 100, 500
    draw.ellipse([base_x-8, base_y-8, base_x+8, base_y+8], fill=(0, 0, 0))
    
    # Arm angles
    a1 = (step / max(1, total)) * 6.28 + 0.3
    a2 = (step / max(1, total)) * 3.14 + 0.5
    s1, s2 = 100, 80
    
    j1x = int(base_x + s1 * np.cos(a1))
    j1y = int(base_y - s1 * np.sin(a1))
    j2x = int(j1x + s2 * np.cos(a1 + a2))
    j2y = int(j1y - s2 * np.sin(a1 + a2))
    
    # Draw arm
    draw.line([(base_x, base_y), (j1x, j1y)], fill=(50, 50, 200), width=4)
    draw.line([(j1x, j1y), (j2x, j2y)], fill=(100, 100, 220), width=4)
    draw.ellipse([j1x-5, j1y-5, j1x+5, j1y+5], fill=(200, 50, 50))
    draw.ellipse([j2x-7, j2y-7, j2x+7, j2y+7], fill=(255, 50, 50))
    
    # EE trajectory
    for i in range(max(0, step-15), step, 3):
        ta = (i / max(1, total)) * 6.28 + 0.3
        ta2 = (i / max(1, total)) * 3.14 + 0.5
        tx1 = int(base_x + s1 * np.cos(ta))
        ty1 = int(base_y - s1 * np.sin(ta))
        tx2 = int(tx1 + s2 * np.cos(ta + ta2))
        ty2 = int(ty1 - s2 * np.sin(ta + ta2))
        draw.ellipse([tx2-2, ty2-2, tx2+2, ty2+2], fill=(200, 150, 150))
    
    # Text
    draw.text((30, 30), f"{task.upper()} - {solver_name}", fill=(0, 0, 0))
    draw.text((30, 70), f"Step {step+1}/{total}", fill=(0, 0, 0))
    draw.text((30, 110), f"VLA: {vla_ms:.1f}ms  MPC: {mpc_ms:.1f}ms", fill=(100, 0, 100))
    
    total_ms = vla_ms + mpc_ms
    freq = 1000/total_ms if total_ms > 0 else 0
    draw.text((30, 150), f"Total: {total_ms:.1f}ms ({freq:.1f}Hz)", fill=(0, 100, 0))
    
    # Status
    if total_ms < 10:
        status, color = "VIABLE", (0, 200, 0)
    elif total_ms < 100:
        status, color = "MARGINAL", (200, 150, 0)
    else:
        status, color = "TOO SLOW", (200, 0, 0)
    
    draw.rectangle([(650, 30), (780, 80)], fill=color)
    draw.text((660, 45), status, fill=(255, 255, 255))
    
    # EE label
    draw.ellipse([j2x-12, j2y-12, j2x+12, j2y+12], outline=(255, 50, 50), width=2)
    draw.text((j2x-18, j2y-30), "EE", fill=(255, 50, 50))
    
    return img

# Generate SL frames
if sl_data:
    print("\nGenerating SL frames...")
    for ep_idx, ep in enumerate(sl_data.get('episodes', [])[:2]):
        task = ['pick', 'push'][ep_idx % 2]
        ep_id = ep.get('episode_id', ep_idx)
        steps = ep.get('steps', [])[:12]
        
        ep_dir = output_dir / f"SL_ep{ep_id}_{task}"
        ep_dir.mkdir(exist_ok=True)
        
        for s_idx, step in enumerate(steps):
            vla = step.get('vla_latency_ms', 0)
            mpc = step.get('mpc_latency_ms', 0)
            frame = make_frame(s_idx, len(steps), "SL-MPC", vla, mpc, task)
            frame.save(str(ep_dir / f"frame_{s_idx:03d}.png"))
        
        print(f"  ✓ Episode {ep_id} ({task}): {len(steps)} frames")

# Generate OSQP frames
if osqp_data:
    print("\nGenerating OSQP frames...")
    for ep_idx, ep in enumerate(osqp_data.get('episodes', [])[:2]):
        task = ['pick', 'push'][ep_idx % 2]
        ep_id = ep.get('episode_id', ep_idx)
        steps = ep.get('steps', [])[:12]
        
        ep_dir = output_dir / f"OSQP_ep{ep_id}_{task}"
        ep_dir.mkdir(exist_ok=True)
        
        for s_idx, step in enumerate(steps):
            vla = step.get('vla_latency_ms', 0)
            mpc = step.get('mpc_latency_ms', 0)
            frame = make_frame(s_idx, len(steps), "OSQP", vla, mpc, task)
            frame.save(str(ep_dir / f"frame_{s_idx:03d}.png"))
        
        print(f"  ✓ Episode {ep_id} ({task}): {len(steps)} frames")

# Create comparison
if sl_data and osqp_data:
    print("\nCreating comparison image...")
    
    sl_times = [step.get('vla_latency_ms', 0) + step.get('mpc_latency_ms', 0) 
                for ep in sl_data.get('episodes', []) for step in ep.get('steps', [])]
    osqp_times = [step.get('vla_latency_ms', 0) + step.get('mpc_latency_ms', 0)
                  for ep in osqp_data.get('episodes', []) for step in ep.get('steps', [])]
    
    if sl_times and osqp_times:
        sl_mean = np.mean(sl_times)
        osqp_mean = np.mean(osqp_times)
        speedup = sl_mean / osqp_mean
        
        comp = Image.new('RGB', (1000, 500), (240, 240, 240))
        cdraw = ImageDraw.Draw(comp)
        
        cdraw.text((50, 40), "PERFORMANCE COMPARISON", fill=(0, 0, 0))
        cdraw.text((50, 100), f"StuartLandau: {sl_mean:.1f}ms ({1000/sl_mean:.2f}Hz)", fill=(50, 50, 200))
        cdraw.text((50, 150), f"OSQP: {osqp_mean:.1f}ms ({1000/osqp_mean:.1f}Hz)", fill=(50, 200, 50))
        cdraw.text((50, 220), f"SPEEDUP: {speedup:.0f}x", fill=(0, 100, 200))
        
        comp.save(str(output_dir / "COMPARISON.png"))
        print("  ✓ Comparison image saved")

# Summary
print("\n" + "="*60)
files = list(output_dir.rglob('*.png'))
size = sum(f.stat().st_size for f in files) / (1024*1024)
print(f"✓ Generated {len(files)} visualization frames")
print(f"✓ Total size: {size:.1f}MB")
print(f"✓ Output: {output_dir}")
print("="*60)
