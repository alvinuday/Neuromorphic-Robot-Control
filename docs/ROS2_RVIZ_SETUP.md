# ROS2 and RViz Setup for 2-DOF Planar Arm with MPC

This guide fixes the RViz2 `libfreetype.6.dylib` error on macOS, builds the ROS2 visualization **inside this repo**, and explains how the pieces work so you can use and extend them.

---

## Quick start (everything in this repo)

From the **repo root** (Neuromorphic-Robot-Control):

```bash
# 0) One-time: use the same Python env for ROS2 and the project (e.g. ros_env)
micromamba activate ros_env
pip install -r requirements.txt   # casadi, osqp, numpy, etc. for the MPC node

# 1) One-time: install freetype for RViz (if you haven’t)
micromamba install -y -c conda-forge freetype

# 2) Build the ROS2 package (creates install/ in the repo)
export PATH="/Users/alvin/y/envs/ros_env/bin:$PATH"
colcon build --packages-select ros2_arm_viz

# 3) Run (launches robot_state_publisher + MPC node + RViz)
./scripts/run_ros2_arm_viz.sh
```

If you see **“No tf data”** in RViz: the launch script starts the nodes that publish TF; use **Fixed Frame** = `base_link` (our config does this). If you opened RViz by hand before the nodes were running, set Fixed Frame to `base_link` and run the launch script.

---

## 1. Fix RViz2 “Library not loaded: libfreetype.6.dylib” (macOS + conda)

RViz2 loads Ogre, which looks for `libfreetype.6.dylib`. In a conda env that path is often missing.

**One-time:**

```bash
micromamba activate ros_env
micromamba install -y -c conda-forge freetype
```

**Each terminal (or add to `~/.zshrc` after activating ros_env):**

```bash
export DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${DYLD_LIBRARY_PATH:-}"
```

The run script sets this for you when you use `./scripts/run_ros2_arm_viz.sh`.

---

## 2. Workspace = this repo

The **ROS2 workspace is the repo itself**. The package lives under `src/ros2_arm_viz/`.

- **Build** (from repo root):

  ```bash
  export PATH="/Users/alvin/y/envs/ros_env/bin:$PATH"
  colcon build --packages-select ros2_arm_viz
  ```

  This creates `install/` and `build/` in the repo.

- **Source** (you must source **base ROS2** first, then the workspace):

  - **From bash** (recommended):

    ```bash
    source /Users/alvin/y/envs/ros_env/setup.bash
    source install/setup.bash
    export NEUROMORPHIC_ARM_PROJECT_ROOT="$(pwd)"
    ```

  - **From zsh**: `source install/setup.bash` under zsh can resolve paths incorrectly. Use:

    ```bash
    bash scripts/source_ros2_workspace.sh
    ```

    then in that same (bash) shell run `ros2 launch ...`, or use the run script (which uses bash).

- **Run** (easiest):

  ```bash
  ./scripts/run_ros2_arm_viz.sh
  ```

  This script: ensures the package is built, sources base ROS2 + workspace, sets `NEUROMORPHIC_ARM_PROJECT_ROOT` and `DYLD_LIBRARY_PATH`, and launches with RViz. Use `./scripts/run_ros2_arm_viz.sh --no-rviz` to start only the nodes (e.g. you open RViz yourself).

---

## 3. How it works (concepts)

### What you see in RViz

- **RobotModel** display: shows the 2-DOF arm. You should see: **grey box** (base), **blue cylinder** (link1), **green cylinder** (link2), **yellow sphere** (end-effector). The arm lies in the **XY plane** (links along -Y when joints are 0). Use **Orbit**: click in the 3D view and **left-drag** to rotate so you see the arm from the side; when MPC runs, the arm moves toward the goal and the yellow sphere moves.
- **TF** display (if enabled): shows coordinate axes (red/green/blue = X/Y/Z) for `base_link`, `link1`, `link2`, `ee_link`. RViz does not draw link names in the 3D view—you identify parts by color and by the TF axes at each frame.
- **Toolbar (top of 3D view):** You typically see **Interact** and **Move Camera** first. **Interact** is for clicking in the scene; **Move Camera** (or **Orbit**) is for rotating/panning when you click in the 3D view. There is often a **+** or more tools (Select, Focus Camera, Measure, etc.) in the same bar; use **Focus Camera** to frame the robot, and **Orbit** / left-drag in the view to rotate.

### What’s actually happening (and why this URDF is correct)

When you run the launch:

1. **joint_state_mpc_node** runs a loop at 50 Hz: it takes the current state `x = [q1, q2, dq1, dq2]`, builds the MPC QP (same math as `src/main.py`), solves it with OSQP, applies the first control, steps the **same 2-DOF dynamics** as in `src/dynamics/arm2dof.py` (l1=0.5 m, l2=0.5 m, gravity, etc.), and publishes the new joint positions/velocities to `/joint_states`.
2. **robot_state_publisher** reads the URDF (our `planar_arm_2dof.urdf`) and `/joint_states`, computes the TF for each link, and publishes `/tf`.
3. **RViz** subscribes to `/tf` and `/robot_description` and draws the robot at those poses.

The URDF is **correct for this project** because it matches the dynamics: two revolute joints, link lengths 0.5 m each, same axis (Z), base at origin, link1 along -Y when q1=0, link2 along -Y when q2=0. That is exactly the 2-DOF planar arm in `arm2dof.py`. Using a random URDF from the web would be wrong unless it were the same robot (same kinematics); otherwise the picture would not match what the MPC is controlling.

### How to interact and test the MPC

- **Change goal at runtime (interact):** The node subscribes to **`/goal_joint_position`**. Send a new goal (joint angles in radians) and the MPC will drive the arm toward it; the step counter resets so you get another full trajectory.
  ```bash
  # Terminal 1: run the launch (RViz + nodes)
  ./scripts/run_ros2_arm_viz.sh

  # Terminal 2: must use the SAME ROS2 env and workspace so it sees the same nodes
  cd /path/to/Neuromorphic-Robot-Control
  source /Users/alvin/y/envs/ros_env/setup.bash && source install/setup.bash
  ros2 topic pub --once /goal_joint_position std_msgs/Float64MultiArray "{data: [0.5, 0.8]}"
  ```
  **Easier:** from repo root run `./scripts/send_goal.sh 0.5 0.8` (or `./scripts/send_goal.sh 0 0`, etc.). The script sources the workspace and publishes the goal so discovery works.
  If you see "Waiting for at least 1 matching subscription(s)..." then Terminal 2 is not seeing the node: (1) ensure the launch is running in Terminal 1, (2) in Terminal 2 run from repo root and source the same env, e.g. `source /Users/alvin/y/envs/ros_env/setup.bash && source install/setup.bash`, then try again. Run `ros2 topic list`; you should see `/goal_joint_position` when the launch is running.
  In RViz you should see the arm move toward the new goal.

- **Change goal / initial state at launch:** Use parameters when starting the launch (see §4 in this doc), e.g. `goal:="[0.5, 0.8]"`, `q0:="[0.2, 0.1]"`.

- **Compare with non-ROS simulation:** Run the same physics and MPC without ROS to confirm numbers:
  ```bash
  python src/main.py --steps 50 --goal 0.5 0.8
  ```
  You get the same dynamics and MPC; the ROS node is just publishing that same loop to `/joint_states` for RViz.

- **Headless test:** Run `./scripts/verify_ros2_arm_viz.sh` to confirm `/joint_states` and `/robot_description` are published without opening RViz.

### Data flow (who does what)

1. **joint_state_mpc_node** (our node)  
   - Runs the **same MPC + dynamics** as `src/main.py`: at each time step it builds the QP, solves it (OSQP), steps the 2-DOF arm, and publishes the current joint angles and velocities.  
   - Publishes **`/joint_states`** (`sensor_msgs/JointState`): `name = [joint1, joint2]`, `position = [q1, q2]`, `velocity = [dq1, dq2]`.  
   - So this node is the “truth” for where the arm is.

2. **robot_state_publisher** (ROS2 standard)  
   - Subscribes to **`/joint_states`** and reads the **robot description** (URDF) from the parameter `robot_description` (set by the launch file).  
   - From the URDF it knows the kinematic tree: `base_link → joint1 → link1 → joint2 → link2`.  
   - It computes the 3D pose of every link from the joint positions and publishes **TF** (transforms between `base_link`, `link1`, `link2`).  
   - RViz (and any other node) can then know “where is link2 in the base_link frame?”.

3. **RViz**  
   - Subscribes to **TF** and optionally to **`/joint_states`**.  
   - The **RobotModel** display uses the URDF + TF (and/or joint_states) to draw the robot.  
   - **Fixed Frame** is the reference frame that stays still on the screen; we use **`base_link`** so the arm moves in front of you. If you set Fixed Frame to `map` and nothing publishes a `map` frame, you get **“No tf data”**.

### Why “No tf data” appears

- RViz needs a **Fixed Frame** that exists in the TF tree.  
- Our system only publishes frames: **base_link**, **link1**, **link2**. There is no **map** or **odom** frame.  
- If Fixed Frame is **map**, there is no TF for `map` → hence “No tf data”.  
- **Fix:** set Fixed Frame to **base_link** (our launch opens RViz with a config that already does this).

### URDF vs your dynamics

- **URDF** (`src/ros2_arm_viz/urdf/planar_arm_2dof.urdf`): describes **geometry and kinematics** for visualization (link lengths 0.5 m, joint axes, etc.). It does not run physics.  
- **Dynamics** (`src/dynamics/arm2dof.py`): used by the MPC node for **simulation** (masses, inertia, gravity, torques). The MPC node publishes the **resulting** joint angles from that simulation so the URDF in RViz matches what the controller is doing.

---

## 4. Run the visualization

### Launch with RViz (default)

```bash
./scripts/run_ros2_arm_viz.sh
```

You should see the arm in RViz moving from the default initial pose toward the goal.

### Launch without RViz (e.g. you start RViz yourself)

```bash
./scripts/run_ros2_arm_viz.sh --no-rviz
```

Then start RViz, set **Fixed Frame** to **base_link**, add a **RobotModel** display (Topic: leave default so it uses TF). The arm will appear and move as long as the launch is running.

### Change MPC parameters

```bash
# After sourcing (e.g. via run script or source_ros2_workspace.sh)
ros2 launch ros2_arm_viz display_arm_mpc.launch.py \
  steps:=300 \
  goal:="[0.5, 0.8]" \
  q0:="[0.0, 0.5]"
```

| Parameter       | Default       | Description                    |
|----------------|---------------|--------------------------------|
| `steps`        | 200           | Number of MPC steps            |
| `horizon`      | 20            | MPC horizon N                  |
| `dt`           | 0.02          | Time step (s)                  |
| `q0`           | [0.0, 0.0]    | Initial joint angles (rad)      |
| `dq0`          | [0.0, 0.0]    | Initial joint velocities       |
| `goal`         | [π/3, π/6]    | Goal joint angles (rad)        |
| `publish_rate` | 50.0          | JointState publish rate (Hz)   |

### Inspect topics

- `ros2 topic echo /joint_states` — joint names, positions, velocities.
- `ros2 run tf2_tools view_frames` (optional) — generates a PDF of the TF tree (base_link → link1 → link2).

---

## 5. File layout (reference)

| Path | Role |
|------|-----|
| `src/ros2_arm_viz/` | ROS2 package (URDF, launch, node). |
| `src/ros2_arm_viz/urdf/planar_arm_2dof.urdf` | 2-DOF arm for visualization (l1=l2=0.5 m). |
| `src/ros2_arm_viz/launch/display_arm_mpc.launch.py` | Starts robot_state_publisher, joint_state_mpc_node, and optionally RViz2. |
| `src/ros2_arm_viz/ros2_arm_viz/joint_state_mpc_node.py` | Runs MPC + dynamics, publishes `/joint_states`. |
| `src/ros2_arm_viz/rviz/planar_arm.rviz` | RViz config (Grid + RobotModel, Fixed Frame = base_link). |
| `scripts/run_ros2_arm_viz.sh` | Build (if needed), source base + workspace, launch (with or without RViz). |
| `scripts/verify_ros2_arm_viz.sh` | **Headless check**: start launch (no RViz), verify `/joint_states` and `/robot_description`; exit 0 if both received. Use in CI or to confirm the pipeline. |
| `tests/verify_ros2_arm_viz.py` | Python checker (subscriptions with correct QoS); run while launch is running, or via `verify_ros2_arm_viz.sh`. |
| `scripts/source_ros2_workspace.sh` | Source base ROS2 + repo workspace (run with bash). |
| `scripts/fix_rviz_freetype.sh` | Install freetype and set DYLD_LIBRARY_PATH for RViz2. |

---

## 6. Troubleshooting

- **“Package 'ros2_arm_viz' not found”**  
  Source the **workspace** after base ROS2:  
  `source /Users/alvin/y/envs/ros_env/setup.bash` then `source install/setup.bash` (from repo root). Use bash for sourcing (or `./scripts/run_ros2_arm_viz.sh`).

- **“robot_state_publisher not found”**  
  Base ROS2 is not sourced. Source the conda env’s setup first:  
  `source /Users/alvin/y/envs/ros_env/setup.bash`.

- **“No tf data” / “Fixed Frame does not exist”**  
  Set Fixed Frame to **base_link** and ensure the launch is running (robot_state_publisher + joint_state_mpc_node). Don’t use `map` unless you add something that publishes it.

- **RViz aborts (freetype)**  
  Install freetype in the env and set `DYLD_LIBRARY_PATH` (see §1). The run script sets it automatically.

- **“No module named 'src.dynamics'” / “Project root not found”**  
  Set `NEUROMORPHIC_ARM_PROJECT_ROOT` to the full path of the repo (the folder containing `src/main.py`). The run script sets it automatically.

- **“No module named 'casadi'” (or osqp, numpy)**  
  The MPC node runs in your ROS2 Python env and needs the project’s dependencies. In the same env as ROS2 (e.g. `ros_env`), run:  
  `pip install -r requirements.txt`

- **Zsh: “not found: .../local_setup.bash”**  
  Under zsh, `source install/setup.bash` can resolve the path wrong. Use `bash scripts/source_ros2_workspace.sh` or `./scripts/run_ros2_arm_viz.sh` (which uses bash).

- **RobotModel enabled but 3D view is empty (only grid)**  
  Set **Description Topic** to **`/robot_description`**: in the left panel expand **RobotModel**, find **Description Topic**, type `/robot_description` and press Enter. The default config now sets this; if it’s still empty, set it by hand. Ensure the launch is running (so the topic is published).

- **Camera doesn’t move / “cannot control the camera”**  
  Click **inside the 3D view** (the grid area) so it has focus, then: **left-drag** = rotate, **middle-drag** or **scroll** = zoom, **right-drag** = pan. If the view is empty, try **Reset** (bottom left) or **Focus Camera** (toolbar) to recenter.

- **RViz2 exits with code -11 (SIGSEGV)**  
  On some macOS setups the RViz2 process can segfault. Run **without** embedding RViz and open it yourself so the arm still works:
  ```bash
  ./scripts/run_ros2_arm_viz.sh --no-rviz
  ```
  In a **second terminal** (with `ros_env` active and the same `DYLD_LIBRARY_PATH` fix):
  ```bash
  micromamba activate ros_env
  export DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$DYLD_LIBRARY_PATH"
  rviz2
  ```
  In RViz: set **Fixed Frame** to `base_link`, click **Add** → **By display type** → **RobotModel**. You should see the arm moving.
