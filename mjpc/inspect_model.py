"""Inspect model to find block body ID"""
import mujoco

xml_path = "assets/xarm_6dof.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"Total bodies: {model.nbody}")
print(f"Bodies in xpos: {len(data.xpos)}")

print("\nBody list:")
for i in range(model.nbody):
    print(f"  {i:2d}: {model.body(i).name}")

# Find block
for i in range(model.nbody):
    name = model.body(i).name.lower()
    if any(s in name for s in ['red_block', 'block', 'box', 'target']):
        print(f"\n✓ Block found: ID={i}, name={model.body(i).name}")
