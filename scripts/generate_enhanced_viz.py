#!/usr/bin/env python3
"""Enhanced visualization with more frames and metrics"""

from PIL import Image, ImageDraw
import json
import numpy as np
from pathlib import Path

output_dir = Path('results/visualizations/enhanced')
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating enhanced visualizations...")

# Load data
sl_data = None
osqp_data = None

try:
    with open('results/vla_sl_mpc_real_data/integration_results.json') as f:
        sl_data = json.load(f)
except:
    pass

try:
    with open('results/vla_osqp_mpc_real_data/integration_results.json') as f:
        osqp_data = json.load(f)
except:
    pass

def draw_frame(step, total, task, solver, vla_ms, mpc_ms, width=1000, height=700):
    """Create detailed visualization frame"""
    img = Image.new('RGB', (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    
    # Main canvas area
    canvas_x1, canvas_y1 = 50, 100
    canvas_x2, canvas_y2 = 750, 650
    draw.rectangle([canvas_x1, canvas_y1, canvas_x2, canvas_y2], outline=(100, 100, 100), width=2)
    
    # Robot base
    base_x = int(canvas_x1 + (canvas_x2 - canvas_x1) * 0.1)
    base_y = int(canvas_y1 + (canvas_y2 - canvas_y1) * 0.8)
    
    # Arm animation
    progress = step / max(1, total)
    angle1 = progress * 6.28 + 0.5
    angle2 = (1 - progress) * 3.14 + 0.3
    
    seg1_len = 100
    seg2_len = 80
    
    j1_x = int(base_x + seg1_len * np.cos(angle1))
    j1_y = int(base_y - seg1_len * np.sin(angle1))
    j2_x = int(j1_x + seg2_len * np.cos(angle1 + angle2))
    j2_y = int(j1_y - seg2_len * np.sin(angle1 + angle2))
    
    # Draw base pedestal
    draw.rectangle([base_x-12, base_y-5, base_x+12, base_y+15], fill=(80, 80, 80))
    
    # Draw arm links with thickness
    for offset in range(-2, 3):
        draw.line([(base_x, base_y+offset), (j1_x, j1_y+offset)], fill=(60, 100, 180), width=2)
        draw.line([(j1_x+offset, j1_y), (j2_x+offset, j2_y)], fill=(80, 120, 200), width=2)
    
    # Draw joints
    draw.ellipse([base_x-6, base_y-6, base_x+6, base_y+6], fill=(40, 40, 40), outline=(0,0,0), width=1)
    draw.ellipse([j1_x-6, j1_y-6, j1_x+6, j1_y+6], fill=(180, 60, 60), outline=(0,0,0), width=1)
    draw.ellipse([j2_x-8, j2_y-8, j2_x+8, j2_y+8], fill=(255, 80, 80), outline=(0,0,0), width=2)
    
    # Draw EE trajectory (history)
    trail_points = []
    for i in range(max(0, step-15), step+1):
        t_progress = i / max(1, total)
        t_a1 = t_progress * 6.28 + 0.5
        t_a2 = (1 - t_progress) * 3.14 + 0.3
        tx1 = int(base_x + seg1_len * np.cos(t_a1))
        ty1 = int(base_y - seg1_len * np.sin(t_a1))
        tx2 = int(tx1 + seg2_len * np.cos(t_a1 + t_a2))
        ty2 = int(ty1 - seg2_len * np.sin(t_a1 + t_a2))
        trail_points.append((tx2, ty2))
    
    # Draw trajectory trail with gradient effect
    for i in range(len(trail_points)-1):
        x1, y1 = trail_points[i]
        x2, y2 = trail_points[i+1]
        alpha = int(100 + (i / max(1, len(trail_points))) * 100)
        draw.line([(x1, y1), (x2, y2)], fill=(255, 150, 150), width=2)
    
    # EE target point
    draw.ellipse([j2_x-12, j2_y-12, j2_x+12, j2_y+12], outline=(255, 100, 100), width=3)
    
    # Top info panel
    panel_y = 30
    draw.text((60, panel_y), f"Episode - {task.title()} Task", fill=(0, 0, 0))
    draw.text((350, panel_y), f"Solver: {solver}", fill=(0, 80, 180))
    draw.text((600, panel_y), f"Step {step+1}/{total}", fill=(0, 0, 0))
    
    # Metrics panel (right side)
    metrics_x = 780
    metrics_y = 120
    
    total_ms = vla_ms + mpc_ms
    freq = 1000 / total_ms if total_ms > 0 else 0
    
    # Draw metrics box
    draw.rectangle([metrics_x, metrics_y, metrics_x+200, metrics_y+150], 
                  fill=(240, 240, 240), outline=(100, 100, 100), width=1)
    
    my = metrics_y + 10
    draw.text((metrics_x+10, my), "PERFORMANCE", fill=(0, 0, 0))
    my += 25
    draw.text((metrics_x+10, my), f"VLA: {vla_ms:.1f}ms", fill=(100, 0, 100))
    my += 20
    draw.text((metrics_x+10, my), f"MPC: {mpc_ms:.1f}ms", fill=(100, 0, 100))
    my += 20
    draw.text((metrics_x+10, my), f"Total: {total_ms:.1f}ms", fill=(0, 0, 0))
    my += 20
    draw.text((metrics_x+10, my), f"Freq: {freq:.1f}Hz", fill=(0, 100, 0) if freq > 100 else (255, 100, 0))
    
    # Status indicator
    if total_ms < 10:
        status_color, status_text = (100, 220, 100), "VIABLE"
    elif total_ms < 100:
        status_color, status_text = (255, 200, 50), "MARGINAL"
    else:
        status_color, status_text = (255, 100, 100), "SLOW"
    
    draw.rectangle([metrics_x, metrics_y+160, metrics_x+200, metrics_y+210], 
                  fill=status_color, outline=(0, 0, 0), width=2)
    draw.text((metrics_x+40, metrics_y+175), status_text, fill=(255, 255, 255))
    
    # EE label
    draw.text((j2_x-15, j2_y-25), "EE", fill=(255, 80, 80))
    
    # Task completion bar
    bar_y = 670
    bar_w = 700
    bar_progress = (step + 1) / max(1, total)
    bar_color = (80, 180, 80) if freq >= 100 else (200, 100, 50)
    
    draw.rectangle([50, bar_y, 50+bar_w, bar_y+15], outline=(100, 100, 100), width=1)
    draw.rectangle([50, bar_y, 50+bar_w*bar_progress, bar_y+15], fill=bar_color)
    
    return img

# Generate for both solvers
if sl_data and osqp_data:
    print("\nGenerating comparison sequence...")
    
    # Pick task from both
    sl_ep = sl_data['episodes'][0]
    osqp_ep = osqp_data['episodes'][0]
    task = 'pick'
    
    steps_to_use = min(10, len(sl_ep['steps']), len(osqp_ep['steps']))
    
    print(f"  Generating {steps_to_use} comparison frames...")
    
    for step_idx in range(steps_to_use):
        # SL frame
        sl_step = sl_ep['steps'][step_idx]
        sl_frame = draw_frame(
            step_idx, steps_to_use, task, 'SL-MPC',
            sl_step['vla_latency_ms'], sl_step['mpc_latency_ms']
        )
        sl_frame.save(str(output_dir / f"SL_frame_{step_idx:02d}.png"))
        
        # OSQP frame  
        osqp_step = osqp_ep['steps'][step_idx]
        osqp_frame = draw_frame(
            step_idx, steps_to_use, task, 'OSQP',
            osqp_step['vla_latency_ms'], osqp_step['mpc_latency_ms']
        )
        osqp_frame.save(str(output_dir / f"OSQP_frame_{step_idx:02d}.png"))
    
    print(f"  ✓ Generated {steps_to_use*2} detailed frames")

# Create metrics comparison chart
if sl_data and osqp_data:
    print("\nGenerating metrics chart...")
    
    sl_times = [s['vla_latency_ms'] + s['mpc_latency_ms'] 
                for ep in sl_data['episodes'] for s in ep['steps']]
    osqp_times = [s['vla_latency_ms'] + s['mpc_latency_ms']
                  for ep in osqp_data['episodes'] for s in ep['steps']]
    
    chart = Image.new('RGB', (1200, 600), (255, 255, 255))
    draw = ImageDraw.Draw(chart)
    
    # Title
    draw.text((50, 30), "CONTROL LATENCY DISTRIBUTION", fill=(0, 0, 0))
    
    # Draw histogram-style comparison
    chart_x, chart_y = 100, 150
    chart_w, chart_h = 1000, 350
    
    draw.rectangle([chart_x, chart_y, chart_x+chart_w, chart_y+chart_h], 
                  outline=(100, 100, 100), width=2)
    
    # Y-axis labels (time in ms)
    for i, ms in enumerate([0, 250, 500, 750, 1000]):
        y = chart_y + chart_h - (i * chart_h // 4)
        draw.text((20, y), f"{ms}ms", fill=(0, 0, 0))
    
    # Plot SL times
    if sl_times:
        max_val = max(sl_times)
        for i, val in enumerate(sl_times[:50]):  # First 50 measurements
            x = chart_x + (i * chart_w // 50)
            h = (val / max(1, max_val)) * chart_h
            draw.rectangle([x, chart_y+chart_h-h, x+8, chart_y+chart_h], 
                          fill=(100, 150, 255), outline=(50, 100, 200), width=1)
    
    # Plot OSQP times (offset for visibility)
    if osqp_times:
        max_val = max(sl_times + osqp_times)
        for i, val in enumerate(osqp_times[:50]):
            x = chart_x + (i * chart_w // 50) + 12
            h = (val / max(1, max_val)) * chart_h
            draw.rectangle([x, chart_y+chart_h-h, x+8, chart_y+chart_h],
                          fill=(100, 255, 150), outline=(50, 200, 100), width=1)
    
    # Legend
    draw.rectangle([chart_x, chart_y+chart_h+20, chart_x+20, chart_y+chart_h+40],
                  fill=(100, 150, 255))
    draw.text((chart_x+30, chart_y+chart_h+20), "SL-MPC", fill=(0, 0, 0))
    
    draw.rectangle([chart_x+150, chart_y+chart_h+20, chart_x+170, chart_y+chart_h+40],
                  fill=(100, 255, 150))
    draw.text((chart_x+180, chart_y+chart_h+20), "OSQP", fill=(0, 0, 0))
    
    chart.save(str(output_dir / "metrics_chart.png"))
    print("  ✓ Metrics chart created")

# Summary
print("\n" + "="*60)
files = list(output_dir.glob('*.png'))
size = sum(f.stat().st_size for f in files) / 1024
print(f"✓ Generated {len(files)} enhanced frames")
print(f"✓ Total size: {size:.0f}KB")
print(f"✓ Output: {output_dir}")
print("="*60)
