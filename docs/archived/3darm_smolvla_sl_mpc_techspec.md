# 3D Robotic Arm + SmolVLA + Stuart-Landau MPC
## Full Technical Specification, Derivations & Integration Architecture
### Alvin — March 2026

---

> **Scope:** Extension of 2-DOF planar arm to a 3-DOF (and 6-DOF) spatial arm; complete Lagrangian dynamics derivation; MPC formulation and QP construction; Stuart-Landau (SL) oscillator-based QP solver (your existing solver, now extended); SmolVLA as System 2 task intelligence; Google Colab + FastAPI/ngrok deployment; MuJoCo simulation; end-to-end System 1 ↔ System 2 integration; observability stack; test suites; benchmark datasets.

---

## TABLE OF CONTENTS

1. [Architecture Overview](#1-architecture-overview)
2. [3D Robotic Arm — Geometry & DH Parameters](#2-3d-robotic-arm--geometry--dh-parameters)
3. [Forward Kinematics](#3-forward-kinematics)
4. [Jacobian Derivation](#4-jacobian-derivation)
5. [Lagrangian Dynamics — Full Derivation](#5-lagrangian-dynamics--full-derivation)
6. [MPC Formulation for the 3D Arm](#6-mpc-formulation-for-the-3d-arm)
7. [QP Construction from MPC](#7-qp-construction-from-mpc)
8. [Stuart-Landau Oscillator QP Solver — 3D Extension](#8-stuart-landau-oscillator-qp-solver--3d-extension)
9. [SmolVLA — Architecture, Role & Deployment](#9-smolvla--architecture-role--deployment)
10. [Google Colab Deployment + FastAPI/ngrok Tunnel](#10-google-colab-deployment--fastapiingrok-tunnel)
11. [Local VSCode Integration Layer](#11-local-vscode-integration-layer)
12. [System 1 ↔ System 2 Integration Protocol](#12-system-1--system-2-integration-protocol)
13. [MuJoCo Simulation Setup](#13-mujoco-simulation-setup)
14. [Observability & Transparency Stack](#14-observability--transparency-stack)
15. [Test Suite — Unit, Integration, System](#15-test-suite--unit-integration-system)
16. [Datasets & Benchmark Tasks](#16-datasets--benchmark-tasks)
17. [Implementation Roadmap](#17-implementation-roadmap)
18. [References](#18-references)

---

## 1. Architecture Overview

### System Philosophy

Your system implements a **dual-system cognitive architecture** directly inspired by Kahneman's System 1/System 2 framing, applied to physical robot control:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM 2  (Slow, Deliberate)                         │
│                      SmolVLA — 450M param VLA                                │
│                                                                              │
│  Input:  RGB frame (224×224) + language instruction                          │
│  Output: Subgoal waypoints [x_goal, y_goal, z_goal] + grasp mode            │
│  Rate:   1–5 Hz (async, does NOT block fast control)                         │
│  Where:  Google Colab (T4 GPU) → FastAPI → ngrok → local                     │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │  subgoal  (TCP/HTTP, ~200ms latency OK)
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 TRAJECTORY BUFFER (Local, Python)                             │
│  Holds latest subgoal; provides reference trajectory to MPC                  │
│  Interpolates between subgoals; detects goal arrival                         │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │  reference trajectory x_ref(t), ẋ_ref(t)
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SYSTEM 1  (Fast, Reactive)                                 │
│         Stuart-Landau Oscillator Network — QP Solver                         │
│                                                                              │
│  Input:  Current joint state q, q̇ + reference trajectory + constraints      │
│  Solving: QP from linearized MPC (N-step horizon)                            │
│  Output: Optimal torque command τ* or joint acceleration q̈*                 │
│  Rate:   100–500 Hz                                                           │
│  Where:  Local machine, NumPy/JAX                                            │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │  τ*(t)
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     MuJoCo Simulation / Real Arm                              │
│  3-DOF (or 6-DOF) robotic arm MJCF model                                     │
│  State feedback: q, q̇ at control rate                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| VLA model | SmolVLA (450M, HuggingFace LeRobot) | Runs on Colab T4 free tier; fully open; flow-matching action head |
| Communication | FastAPI + ngrok | Zero infrastructure; secure HTTPS tunnel; < 5 lines to set up |
| Simulation | MuJoCo 3.x | Gold standard; differentiable; MJCF support; free |
| QP solver | Your SL oscillator network | Already working; thesis contribution; neuromorphic |
| Action space | Joint torques τ ∈ ℝ³ (or ℝ⁶) | Full dynamic control |
| Observation | RGB 224×224 → SmolVLA; q, q̇ → MPC | Clean separation of modalities |
| Subgoal format | End-effector Cartesian [x, y, z] + orientation quaternion | Model-agnostic interface |

---

## 2. 3D Robotic Arm — Geometry & DH Parameters

### 2.1 Arm Configuration

We extend the 2-DOF planar arm to a **3-DOF spatial RRR arm** — three revolute joints, fully spatial motion. This is the simplest arm that can reach arbitrary positions in 3D space (ignoring orientation). We later extend to 6-DOF (full pose control).

```
Joint 1: Base rotation — rotates about vertical Z axis (yaw)
Joint 2: Shoulder — rotates about horizontal Y axis (elevation)
Joint 3: Elbow — rotates about horizontal Y axis (elevation)

                    z
                    |     Link 3 (end-effector)
                    |   ○────────────────○ ← EE
                    |  / Joint 3
                    | /  Link 2
                    ○ ← Joint 2
                   /|
         Link 1   / |
                 /  |
                ○ ← Joint 1 (base, on z-axis)
                |
```

### 2.2 Denavit-Hartenberg Parameters

Using the **standard DH convention** (Hartenberg & Denavit, 1955). Each frame i is attached to link i, connected to joint i+1.

The DH transformation from frame i−1 to frame i:

```
ᵢ₋₁Tᵢ = Rz(θᵢ) · Tz(dᵢ) · Tx(aᵢ) · Rx(αᵢ)
```

Explicitly:

```
         [cos θᵢ  -sin θᵢ cos αᵢ   sin θᵢ sin αᵢ   aᵢ cos θᵢ ]
ᵢ₋₁Tᵢ = [sin θᵢ   cos θᵢ cos αᵢ  -cos θᵢ sin αᵢ   aᵢ sin θᵢ ]
         [  0          sin αᵢ            cos αᵢ          dᵢ    ]
         [  0             0                  0             1    ]
```

**DH Table for 3-DOF Arm:**

| Joint i | θᵢ (variable) | dᵢ (offset) | aᵢ (link length) | αᵢ (twist) |
|---------|--------------|-------------|-------------------|------------|
| 1       | q₁           | d₁ = L₀    | a₁ = 0            | α₁ = −π/2 |
| 2       | q₂           | d₂ = 0     | a₂ = L₁           | α₂ = 0    |
| 3       | q₃           | d₃ = 0     | a₃ = L₂           | α₃ = 0    |

Where:
- **L₀** = base height (distance from ground to joint 1 along z)
- **L₁** = upper arm length (link 2)
- **L₂** = forearm length (link 3)
- **q = [q₁, q₂, q₃]ᵀ** are the joint angles (generalized coordinates)

**Typical values for a desktop research arm:**
```
L₀ = 0.10 m   (base height)
L₁ = 0.25 m   (upper arm)
L₂ = 0.20 m   (forearm)
m₁ = 0.50 kg  (link 1 mass, includes motor)
m₂ = 0.40 kg  (link 2 mass)
m₃ = 0.20 kg  (link 3 mass + gripper)
```

### 2.3 Individual DH Transforms

**Frame 0 → 1** (base to shoulder, α₁ = −π/2):

```
⁰T₁ = [cos q₁   0   -sin q₁    0    ]
       [sin q₁   0    cos q₁    0    ]
       [   0    -1       0      L₀   ]
       [   0     0       0      1    ]
```

**Frame 1 → 2** (shoulder to elbow, α₂ = 0):

```
¹T₂ = [cos q₂  -sin q₂   0   L₁ cos q₂]
       [sin q₂   cos q₂   0   L₁ sin q₂]
       [   0        0      1      0     ]
       [   0        0      0      1     ]
```

**Frame 2 → 3** (elbow to EE, α₃ = 0):

```
²T₃ = [cos q₃  -sin q₃   0   L₂ cos q₃]
       [sin q₃   cos q₃   0   L₂ sin q₃]
       [   0        0      1      0     ]
       [   0        0      0      1     ]
```

---

## 3. Forward Kinematics

### 3.1 End-Effector Position

The full transformation:

```
⁰T₃ = ⁰T₁ · ¹T₂ · ²T₃
```

**EE position vector p = [px, py, pz]ᵀ:**

Expanding the chain multiplication (using c₁ = cos q₁, s₁ = sin q₁, c₂₃ = cos(q₂+q₃), etc.):

```
px = cos q₁ · [L₁ cos q₂ + L₂ cos(q₂+q₃)]
py = sin q₁ · [L₁ cos q₂ + L₂ cos(q₂+q₃)]
pz = L₀ + L₁ sin q₂ + L₂ sin(q₂+q₃)
```

**Notation shorthand:**
```
c₁ = cos(q₁),  s₁ = sin(q₁)
c₂ = cos(q₂),  s₂ = sin(q₂)
c₃ = cos(q₃),  s₃ = sin(q₃)
c₂₃ = cos(q₂+q₃),  s₂₃ = sin(q₂+q₃)
```

So:
```
px = c₁(L₁c₂ + L₂c₂₃)
py = s₁(L₁c₂ + L₂c₂₃)
pz = L₀ + L₁s₂ + L₂s₂₃
```

**Physical interpretation:**
- Joint 1 (q₁) sweeps the arm in azimuth — it multiplies the planar reach by cos/sin q₁
- Joints 2, 3 (q₂, q₃) are exactly your 2-DOF planar arm, now elevated to 3D by joint 1
- The 3D arm is your 2D arm + one azimuth joint. This is the key insight for extending your existing work.

### 3.2 Full Rotation Matrix ⁰R₃

```
⁰R₃ = [c₁c₂₃   -c₁s₂₃   -s₁]
       [s₁c₂₃   -s₁s₂₃    c₁]
       [ s₂₃      c₂₃      0 ]
```

This gives the tool frame orientation if needed (for grasping tasks with SmolVLA).

---

## 4. Jacobian Derivation

The Jacobian maps joint velocities to end-effector velocities:

```
ẋ = J(q) · q̇,   ẋ ∈ ℝ⁶ (3 linear + 3 angular),  J ∈ ℝ⁶ˣ³
```

### 4.1 Geometric Jacobian

For a revolute joint i, the columns of the geometric Jacobian are:

```
Jᵥᵢ = zᵢ₋₁ × (p_EE − pᵢ₋₁)    (linear velocity part)
Jωᵢ = zᵢ₋₁                      (angular velocity part)
```

Where:
- **zᵢ₋₁** is the z-axis of frame i−1 expressed in frame 0
- **p_EE** is the EE position in frame 0
- **pᵢ₋₁** is the origin of frame i−1 in frame 0

**Frame origins:**
```
p₀ = [0, 0, 0]ᵀ
p₁ = [0, 0, L₀]ᵀ
p₂ = [L₁c₁c₂, L₁s₁c₂, L₀ + L₁s₂]ᵀ
p₃ = p_EE = [c₁(L₁c₂ + L₂c₂₃), s₁(L₁c₂ + L₂c₂₃), L₀ + L₁s₂ + L₂s₂₃]ᵀ
```

**Z-axis directions (from DH transforms):**
```
z₀ = [0, 0, 1]ᵀ           (global vertical)
z₁ = [-s₁, c₁, 0]ᵀ       (horizontal, perpendicular to arm plane)
z₂ = z₁ = [-s₁, c₁, 0]ᵀ  (parallel to z₁ since α₂ = 0)
```

### 4.2 Column-by-Column

**Column 1 (Joint 1, base rotation):**
```
Jᵥ₁ = z₀ × (p_EE − p₀)
     = [0,0,1]ᵀ × [c₁(L₁c₂+L₂c₂₃), s₁(L₁c₂+L₂c₂₃), L₀+L₁s₂+L₂s₂₃]ᵀ
     = [−s₁(L₁c₂+L₂c₂₃), c₁(L₁c₂+L₂c₂₃), 0]ᵀ

Jω₁ = [0, 0, 1]ᵀ
```

**Column 2 (Joint 2, shoulder):**
```
Jᵥ₂ = z₁ × (p_EE − p₁)
     = [-s₁, c₁, 0]ᵀ × [c₁(L₁c₂+L₂c₂₃), s₁(L₁c₂+L₂c₂₃), L₁s₂+L₂s₂₃]ᵀ

Cross product expansion:
Jᵥ₂ₓ = c₁(L₁s₂+L₂s₂₃)·1 − 0·s₁(L₁c₂+L₂c₂₃) ... 

Full result:
Jᵥ₂ = [−c₁(L₁s₂+L₂s₂₃), −s₁(L₁s₂+L₂s₂₃), L₁c₂+L₂c₂₃]ᵀ

Jω₂ = [-s₁, c₁, 0]ᵀ
```

**Column 3 (Joint 3, elbow):**
```
Jᵥ₃ = z₂ × (p_EE − p₂)
     = [-s₁, c₁, 0]ᵀ × [L₂c₁c₂₃, L₂s₁c₂₃, L₂s₂₃]ᵀ

Jᵥ₃ = [−c₁L₂s₂₃, −s₁L₂s₂₃, L₂c₂₃]ᵀ

Jω₃ = [-s₁, c₁, 0]ᵀ
```

### 4.3 Full Jacobian Matrix J ∈ ℝ⁶ˣ³

```
      [−s₁(L₁c₂+L₂c₂₃)    −c₁(L₁s₂+L₂s₂₃)    −c₁L₂s₂₃]
      [ c₁(L₁c₂+L₂c₂₃)    −s₁(L₁s₂+L₂s₂₃)    −s₁L₂s₂₃]
J  =  [        0             L₁c₂+L₂c₂₃          L₂c₂₃  ]
      [        0                 -s₁                -s₁   ]
      [        0                  c₁                 c₁   ]
      [        1                   0                  0   ]
```

### 4.4 Jacobian Time Derivative J̇

For MPC we need J̇(q,q̇). Computed symbolically:

```
J̇ᵢⱼ = Σₖ (∂Jᵢⱼ/∂qₖ) · q̇ₖ
```

This is non-trivial but straightforward with symbolic differentiation. In code, use `jax.jacobian(J_func)` for automatic differentiation.

### 4.5 Singularity Analysis

The Jacobian loses rank when the manipulator is in a **singular configuration**:

```
det(Jᵥ Jᵥᵀ) → 0
```

For this arm, singularities occur at:
1. **Shoulder singularity**: q₂ = 0 or q₂ = π (arm fully extended/retracted upward)
2. **Elbow singularity**: q₃ = 0 (elbow straight — links 2 & 3 aligned)
3. **Wrist singularity**: Not applicable for 3-DOF (only appears with 5+ DOF)

**Damped least-squares pseudo-inverse** for singularity robustness:
```
J† = Jᵀ(JJᵀ + λ²I)⁻¹,   λ = 0.01 (damping factor)
```

---

## 5. Lagrangian Dynamics — Full Derivation

The equations of motion take the form:

```
M(q)q̈ + C(q,q̇)q̇ + G(q) = τ + Jᵀ(q)F_ext
```

Where:
- **M(q) ∈ ℝ³ˣ³** — mass/inertia matrix (positive definite)
- **C(q,q̇) ∈ ℝ³ˣ³** — Coriolis/centrifugal matrix
- **G(q) ∈ ℝ³** — gravity vector
- **τ ∈ ℝ³** — joint torques (control input)
- **F_ext ∈ ℝ⁶** — external forces/moments at EE

### 5.1 Kinetic Energy

**Lagrangian approach:** T = Σᵢ Tᵢ where each link contributes:

```
Tᵢ = ½ mᵢ vᵢᶜᵀvᵢᶜ + ½ ωᵢᵀ Iᵢ ωᵢ
```

Where vᵢᶜ is the velocity of link i's center of mass and Iᵢ is its inertia tensor.

**Center of mass positions** (assuming uniform rod, CoM at midpoint):
```
p₁ᶜ = [0, 0, L₀/2]ᵀ  (base link, simplified as static — not moving)

p₂ᶜ = [c₁c₂·(L₁/2), s₁c₂·(L₁/2), L₀ + s₂·(L₁/2)]ᵀ

p₃ᶜ = [c₁(L₁c₂ + c₂₃·L₂/2), s₁(L₁c₂ + c₂₃·L₂/2), L₀ + L₁s₂ + s₂₃·L₂/2]ᵀ
```

**CoM Jacobians** Jᶜᵢ (3×3, linear velocity only):

```
vᵢᶜ = Jᶜᵢ(q) q̇
```

For link 2 (∂p₂ᶜ/∂qⱼ):
```
       [−s₁c₂·(L₁/2)    −c₁s₂·(L₁/2)    0]
Jᶜ₂ = [ c₁c₂·(L₁/2)    −s₁s₂·(L₁/2)    0]
       [     0             c₂·(L₁/2)      0]
```

For link 3 (∂p₃ᶜ/∂qⱼ):
```
       [−s₁(L₁c₂+c₂₃·L₂/2)    −c₁(L₁s₂+s₂₃·L₂/2)    −c₁s₂₃·L₂/2]
Jᶜ₃ = [ c₁(L₁c₂+c₂₃·L₂/2)    −s₁(L₁s₂+s₂₃·L₂/2)    −s₁s₂₃·L₂/2]
       [         0               L₁c₂+c₂₃·L₂/2          c₂₃·L₂/2  ]
```

**Angular velocity Jacobians** Jωᵢ (3×3 — upper part of geometric Jacobian):
```
Jω₂ = [[0, -s₁, -s₁],   (columns 1,2,3 up to joint 2)
        [0,  c₁,  c₁],
        [1,   0,   0]]

Jω₃ = same as Jω₂ (α₃ = 0)
```

### 5.2 Inertia Tensors

For a uniform rod of mass m, length L, about center:
```
Ixx = Iyy = mL²/12  (transverse axes)
Izz = 0              (about long axis, thin rod)
```

In frame of each link:
```
I₂ = diag(m₂L₁²/12, m₂L₁²/12, ε)   (ε ≈ 0 for thin rod)
I₃ = diag(m₃L₂²/12, m₃L₂²/12, ε)
```

### 5.3 Mass Matrix M(q)

```
M(q) = Σᵢ [mᵢ Jᶜᵢᵀ Jᶜᵢ + Jωᵢᵀ ⁰Rᵢ Iᵢ ⁰Rᵢᵀ Jωᵢ]
```

**Explicit M(q) entries:**

```
M₁₁ = (m₂+m₃)(L₁c₂)² + m₃(2L₁c₂)(L₂c₂₃) + m₃(L₂c₂₃)²
     + m₂(L₁/2)²c₂² + m₃[(L₁c₂+L₂c₂₃/2)²] + (I₂ᵧᵧ+I₃ᵧᵧ)... 

     Simplified for thin uniform rods:
     M₁₁ = (m₂/3)L₁²c₂² + m₂L₁²c₂²/4·... 
```

**Full symbolic expressions** (compact notation, Iᵢ = mᵢLᵢ²/3 = moment of inertia about joint):

```
M₁₁ = I₁ + (m₂+m₃)L₁²c₂² + m₃L₂²c₂₃² + 2m₃L₁L₂c₂c₂₃

M₂₂ = I₂/L₁² · L₁² + I₃ + m₃L₂²  
     = m₂L₁²/3 + m₃(L₁² + L₂²/3 + L₁L₂c₃)

M₃₃ = m₃L₂²/3

M₁₂ = M₂₁ = 0   (decoupled due to perpendicular axes)

M₁₃ = M₃₁ = 0

M₂₃ = M₃₂ = m₃(L₂²/3 + L₁L₂c₃/2)
```

**Note:** M₁ is decoupled from M₂₃ block because joint 1 rotates about z, while joints 2,3 rotate about y. This is a key structural property of this arm configuration.

**Block structure:**
```
M(q) = [M₁₁(q₂,q₃)    0              0         ]
        [    0          M₂₂(q₃)    M₂₃(q₃)      ]
        [    0          M₂₃(q₃)    M₃₃           ]
```

This means joint 1 (azimuth) is **dynamically decoupled** from the elevation plane — a major simplification.

### 5.4 Coriolis & Centrifugal Matrix C(q,q̇)

Using **Christoffel symbols of the first kind:**

```
Γᵢⱼₖ = ½(∂Mᵢₖ/∂qⱼ + ∂Mᵢⱼ/∂qₖ − ∂Mⱼₖ/∂qᵢ)

Cᵢⱼ(q,q̇) = Σₖ Γᵢⱼₖ q̇ₖ
```

**Key non-zero Christoffel symbols:**

Define: `h = m₃L₁L₂`

```
Γ₁₁₂ = −(m₂+m₃)L₁²c₂s₂ − m₃L₁L₂(c₂s₂₃ + s₂c₂₃) − ...

C₁₁ = Γ₁₁₁q̇₁ + Γ₁₁₂q̇₂ + Γ₁₁₃q̇₃

C₂₂ = Γ₂₂₃q̇₃ = −(m₃L₁L₂s₃/2)q̇₃

C₂₃ = C₃₂ = Γ₂₃₃q̇₃ = −(m₃L₁L₂s₃/2)q̇₃

C₃₂ = Γ₃₂₂q̇₂ = (m₃L₁L₂s₃/2)q̇₂
```

**Simplified C(q,q̇) matrix:**

Let `β = m₃L₁L₂s₃/2`, `γ = ∂M₁₁/∂q₂ · q̇₂/2`

```
C(q,q̇) = [γ      −M₁₁/(c₂)·q̇₂·...    ... ]
           [ 0      −β·q̇₃              −β(q̇₂+q̇₃)]
           [ 0       β·q̇₂               0         ]
```

**Property (skew symmetry test):** M̈ − 2C should be skew-symmetric. Verify in simulation:
```python
skew_check = M_dot - 2*C  # should satisfy x @ skew_check @ x == 0 for all x
```

### 5.5 Gravity Vector G(q)

```
G(q) = −Σᵢ mᵢ g ∂pᵢᶜ/∂q
```

With g = [0, 0, −9.81]ᵀ (gravity along −z):

```
G₁(q) = 0   (joint 1 rotates about z — no gravitational torque)

G₂(q) = −g(m₂(L₁/2)c₂ + m₃(L₁c₂ + (L₂/2)c₂₃))
       = −g(m₂L₁c₂/2 + m₃L₁c₂ + m₃L₂c₂₃/2)

G₃(q) = −g · m₃(L₂/2)c₂₃
       = −g m₃L₂c₂₃/2
```

**G₁ = 0** is a direct consequence of joint 1 rotating about the vertical axis. This decouples the gravity compensation entirely from the azimuth joint.

### 5.6 Complete Equations of Motion

```
[M₁₁   0     0  ] [q̈₁]   [C₁₁  C₁₂  C₁₃] [q̇₁]   [ 0  ]   [τ₁]
[ 0    M₂₂  M₂₃] [q̈₂] + [C₂₁  C₂₂  C₂₃] [q̇₂] + [G₂] = [τ₂]
[ 0    M₂₃  M₃₃] [q̈₃]   [C₃₁  C₃₂  C₃₃] [q̇₃]   [G₃]   [τ₃]
```

Or compactly:
```
M(q)q̈ = τ − C(q,q̇)q̇ − G(q)
```

This is your **plant model** for MPC.

### 5.7 Friction and Motor Dynamics (Practical Addition)

For simulation fidelity, add:

```
M(q)q̈ = τ − C(q,q̇)q̇ − G(q) − Bq̇ − τ_friction
```

Where:
- **B = diag(b₁, b₂, b₃)** — viscous damping (b ≈ 0.1–0.5 Nm·s/rad)
- **τ_friction = μ · sign(q̇)** — Coulomb friction (μ ≈ 0.05 Nm)

---

## 6. MPC Formulation for the 3D Arm

### 6.1 State Space Formulation

Define state vector and input:
```
x = [q; q̇] ∈ ℝ⁶   (joint positions + velocities)
u = τ ∈ ℝ³           (joint torques)
```

The nonlinear dynamics:
```
ẋ = f(x, u) = [q̇; M(q)⁻¹(u − C(q,q̇)q̇ − G(q))]
```

### 6.2 Linearization About Trajectory

For MPC, linearize around a nominal trajectory (x̄(t), ū(t)):

```
δẋ = Aₜδx + Bₜδu

Aₜ = ∂f/∂x|_{x̄,ū} ∈ ℝ⁶ˣ⁶
Bₜ = ∂f/∂u|_{x̄,ū} ∈ ℝ⁶ˣ³
```

**Analytical A, B matrices:**

```
Aₜ = [∂f₁/∂q   ∂f₁/∂q̇]   =  [0₃ₓ₃      I₃ₓ₃      ]
     [∂f₂/∂q   ∂f₂/∂q̇]      [A₂₁(q,q̇)  A₂₂(q,q̇)]
```

Where:
```
A₂₁ = ∂/∂q [M⁻¹(−C q̇ − G)]  (complex, use numerical diff or JAX autodiff)
A₂₂ = −M⁻¹ C̄   where C̄ = C(q,q̇) evaluated at nominal

Bₜ = [0₃ₓ₃]
     [M⁻¹ ]
```

**Bₜ has a beautiful structure** — it only depends on M(q), not velocities.

### 6.3 Discrete-Time Linearization

Using **Zero-Order Hold (ZOH)** discretization with step Δt:

```
xₖ₊₁ = Aₖxₖ + Bₖuₖ + dₖ

Aₖ = exp(AₜΔt) ≈ I + AₜΔt + (AₜΔt)²/2   (2nd order, sufficient for small Δt)
Bₖ = (Aₖ − I)Aₜ⁻¹ Bₜ ≈ BₜΔt + Aₜ BₜΔt²/2
dₖ = f(x̄ₖ, ūₖ)Δt − Aₜx̄ₖΔt − Bₜūₖ Δt   (affine offset)
```

### 6.4 MPC Cost Function

Over horizon N, minimize:

```
J = Σₖ₌₀^{N-1} [xₖᵀQxₖ + uₖᵀRuₖ + Δuₖᵀ Sᵤ Δuₖ] + xₙᵀPxₙ
```

Where:
- **Q ∈ ℝ⁶ˣ⁶** — state cost (penalize position and velocity error)
- **R ∈ ℝ³ˣ³** — input cost (penalize large torques)
- **Sᵤ ∈ ℝ³ˣ³** — rate cost (penalize jerky torque changes)
- **P ∈ ℝ⁶ˣ⁶** — terminal cost (often P = Riccati solution)

**In deviation coordinates** (δx = x − x_ref):
```
J = Σₖ₌₀^{N-1} [δxₖᵀQδxₖ + δuₖᵀRδuₖ] + δxₙᵀPδxₙ
```

**Typical weights** (tune empirically):
```python
Q = np.diag([100, 100, 100,   # joint position errors (rad)
              10,  10,  10])   # joint velocity errors (rad/s)
R = np.diag([0.1, 0.1, 0.1])  # torque cost (Nm)
P = 10 * Q                     # terminal cost, heavier
```

### 6.5 MPC Constraints

**Joint position limits:**
```
q_min ≤ qₖ ≤ q_max
[-π, -π/2, -2π/3]ᵀ ≤ q ≤ [π, π/2, 2π/3]ᵀ
```

**Joint velocity limits:**
```
|q̇ₖ| ≤ q̇_max = [2.0, 1.5, 2.0]ᵀ rad/s
```

**Torque limits:**
```
|uₖ| ≤ τ_max = [5.0, 4.0, 3.0]ᵀ Nm
```

**End-effector workspace constraint** (optional):
```
pz_min ≤ pz(qₖ) ≤ pz_max   (keep EE above table)
```

This is a **nonlinear constraint** linearized as:
```
∇pz(q̄) · δq ≥ pz_min − pz(q̄)
```

---

## 7. QP Construction from MPC

### 7.1 Batch Prediction (Lifting)

Denote the stacked state and input sequences:

```
X = [x₁ᵀ, x₂ᵀ, ..., xₙᵀ]ᵀ ∈ ℝ^{6N}
U = [u₀ᵀ, u₁ᵀ, ..., u_{N-1}ᵀ]ᵀ ∈ ℝ^{3N}
```

The prediction equations in batch form:

```
X = Sx · x₀ + Su · U + Sd · D
```

Where (with Aₖ, Bₖ time-varying but often approximated as LTI):
```
       [  A    ]        [ B   0  ...  0 ]
       [  A²   ]        [AB   B  ...  0 ]
Sx =   [ ...   ]   Su = [ ⋮           ⋮ ]
       [  Aᴺ  ]        [Aᴺ⁻¹B ...  B ]
```

### 7.2 Standard QP Form

Substituting prediction into cost:

```
min_U  ½ Uᵀ H U + cᵀ U

H = Suᵀ Q̄ Su + R̄    ∈ ℝ^{3N × 3N}
c = Suᵀ Q̄ (Sx x₀ − X_ref)   ∈ ℝ^{3N}

Subject to:
  A_ineq U ≤ b_ineq    (state + input constraints)
  A_eq U = b_eq         (terminal equality, optional)
```

Where:
```
Q̄ = block_diag(Q, Q, ..., Q, P) ∈ ℝ^{6N × 6N}
R̄ = block_diag(R, R, ..., R)    ∈ ℝ^{3N × 3N}
```

**This is the QP your SL oscillator network solves.**

### 7.3 Constraint Matrix Construction

**Input bounds:** −τ_max ≤ U ≤ τ_max
```
A_u = [I_{3N}; -I_{3N}],   b_u = [1_{3N}⊗τ_max; 1_{3N}⊗τ_max]
```

**State bounds** (propagated through dynamics):
```
A_x = [I_{6N}; -I_{6N}] · Su,   b_x adjusted for Sx x₀
```

**Full constraint block:**
```
A_ineq = [A_u; A_x]
b_ineq = [b_u; b_x]
```

### 7.4 Warm Starting for SL Oscillator Network

At each MPC step k, warm-start the oscillator network with the shifted solution from k−1:

```
U_warm = [u₁*, u₂*, ..., u_{N-1}*, u_{N-1}*]   (shift by one, repeat last)
```

This dramatically reduces convergence iterations (2-3× speedup in practice for receding horizon).

---

## 8. Stuart-Landau Oscillator QP Solver — 3D Extension

### 8.1 Recap of SL Oscillator Network

Your existing solver uses a network of **Stuart-Landau (SL) oscillators** to solve QPs. Each optimization variable uᵢ is represented by the amplitude/phase of one oscillator. The key equations:

```
żₙ = (μₙ − |zₙ|²)zₙ + iωₙzₙ + Σⱼ Wₙⱼ zⱼ + bₙ

where zₙ = rₙ e^{iφₙ},  uₙ = Re(zₙ) or rₙ
```

The coupling weights W and biases b are set from the QP matrices H and c.

### 8.2 Scaling from 2-DOF to 3-DOF

Your 2-DOF arm had N_var = 2×N_horizon optimization variables. The 3-DOF arm has:

```
N_var_3D = 3 × N_horizon
```

**For N_horizon = 10: N_var = 30 oscillators (vs 20 before)**

The coupling matrix W scales as N_var². For N=30: 900 couplings — still very tractable.

**No structural changes needed** to your oscillator ODEs. Only the initialization:
```python
N_joints = 3   # was 2
N_horizon = 10
N_var = N_joints * N_horizon  # = 30

# Initialize oscillator state
z = np.ones(N_var, dtype=complex) * 0.1  # or use warm start
mu = np.ones(N_var)   # bifurcation parameter
omega = np.zeros(N_var)  # or small nonzero for faster convergence

# Set from QP
W = -H   # coupling from Hessian (negative, for minimization)
b = -c   # bias from linear cost
```

### 8.3 Constraint Handling Extension

For hard constraints A_ineq U ≤ b_ineq, use **augmented Lagrangian** or **penalty projection**:

```
# Penalty method (simplest)
f_penalty = ρ/2 · Σᵢ max(0, Aᵢ·U − bᵢ)²

# Add penalty gradient to oscillator bias:
b_augmented = -c - ρ · Aᵀ max(0, A·Re(z) − b)
```

Increase ρ gradually: ρ ∈ {1, 10, 100} over solver iterations.

### 8.4 Integration Loop (3D MPC)

```python
def sl_mpc_step_3d(q, q_dot, q_ref, q_dot_ref, dt=0.005):
    """
    One MPC step for the 3-DOF arm using SL oscillator QP solver.
    
    Args:
        q:        current joint angles [3]
        q_dot:    current joint velocities [3]
        q_ref:    reference joint trajectory [N, 3]
        q_dot_ref: reference velocity trajectory [N, 3]
        dt:       control period
    
    Returns:
        tau_opt:  optimal torque [3]
    """
    # State
    x0 = np.concatenate([q, q_dot])  # [6]
    
    # Linearize dynamics at current state
    M = compute_M(q)
    C = compute_C(q, q_dot)
    G = compute_G(q)
    A, B = linearize(M, C, G, dt)
    
    # Build QP
    H, c, A_ineq, b_ineq = build_qp(A, B, x0, q_ref, q_dot_ref, Q, R, P, N=10)
    
    # Solve with SL oscillators
    U_opt = sl_oscillator_solve(H, c, A_ineq, b_ineq, 
                                 z_init=z_warm,  # warm start
                                 n_steps=500, dt_osc=0.001)
    
    # Extract first control action
    tau_opt = U_opt[:3]
    
    # Update warm start
    z_warm = shift_warm_start(U_opt)
    
    return tau_opt
```

---

## 9. SmolVLA — Architecture, Role & Deployment

### 9.1 What SmolVLA Is

**SmolVLA** (HuggingFace, 2025) is a 450M parameter Vision-Language-Action model from the LeRobot library. Key specs:

| Property | Value |
|---|---|
| Parameters | 450M total |
| Vision encoder | SigLIP (400M encoder shared with SmolVLM) |
| Language backbone | SmolLM2 (language tokens) |
| Action head | Flow-matching (continuous actions) |
| Training data | LeRobot dataset format (Open X-Embodiment compatible) |
| Inference | Async — does NOT need to run at control frequency |
| Fine-tuning | LoRA-compatible, fits on T4/A100 |
| License | Apache 2.0 |

### 9.2 SmolVLA in Your System

SmolVLA acts as **System 2 (slow, deliberate reasoning)**:

```
Input:  RGB image (224×224) + language instruction (string)
Output: Action chunk — sequence of EE waypoints or joint configs
        [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_open] × chunk_size
```

**You do NOT run SmolVLA at 100 Hz.** It runs asynchronously at 1–5 Hz, outputs a goal or waypoint, and the SL-MPC handles the actual reaching.

**The interface contract:**
```python
# SmolVLA output (raw)
action_chunk = smolvla.predict(image, instruction)
# shape: [chunk_size=10, 7] — delta EE poses

# Convert to MPC reference
ee_goal = current_ee_pos + action_chunk[0, :3]   # target EE position
q_ref = ik_solver(ee_goal)                         # IK to joint space
```

### 9.3 Fine-Tuning Plan (Later Phase)

For your manufacturing tasks, fine-tune SmolVLA using **LoRA**:

```python
# From LeRobot
from lerobot.models.smolvla import SmolVLA

model = SmolVLA.from_pretrained("huggingface/smolvla-base")
model.add_lora(rank=8, alpha=16)   # ~3M trainable params

# Training data: your MuJoCo demos + Open X-Embodiment subset
# Dataset format: LeRobot HuggingFace Dataset format
```

---

## 10. Google Colab Deployment + FastAPI/ngrok Tunnel

### 10.1 Architecture

```
[Google Colab T4 GPU]                    [Your Local Machine — VSCode]
  SmolVLA model loaded                     MPC controller running
  FastAPI server running                   HTTP client polling
  ngrok tunnel active                      
        |                                          |
        └──────── HTTPS (ngrok URL) ───────────────┘
                  POST /predict
                  {image_b64, instruction}
                  → {action_chunk, latency_ms}
```

### 10.2 Colab Notebook — Complete Setup

**Cell 1: Install dependencies**
```python
# In Colab
!pip install lerobot fastapi uvicorn pyngrok pillow numpy torch torchvision -q
!pip install huggingface_hub -q
```

**Cell 2: Load SmolVLA**
```python
import torch
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig

# Load pretrained SmolVLA
policy_config = SmolVLAConfig()
policy = SmolVLAPolicy(policy_config)
policy = policy.from_pretrained("huggingface/smolvla-base")
policy.eval()
policy = policy.to("cuda")

print(f"SmolVLA loaded. Params: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M")
```

**Cell 3: FastAPI Server**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import base64, io, time, uvicorn
import numpy as np
from PIL import Image
import threading

app = FastAPI(title="SmolVLA Inference Server")

class InferenceRequest(BaseModel):
    image_b64: str          # base64 encoded JPEG/PNG
    instruction: str        # natural language task
    current_joints: list    # [q1, q2, q3] current joint angles

class InferenceResponse(BaseModel):
    action_chunk: list      # [[dx,dy,dz,droll,dpitch,dyaw,gripper]×N]
    latency_ms: float
    status: str
    subgoal_xyz: list       # predicted EE target [x, y, z]

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    t0 = time.time()
    
    # Decode image
    img_bytes = base64.b64decode(request.image_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = image.resize((224, 224))
    
    # Prepare input tensor
    img_tensor = torch.tensor(np.array(image)/255.0, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to("cuda")  # [1,3,224,224]
    
    # Prepare observation dict (LeRobot format)
    observation = {
        "observation.images.top": img_tensor,
        "observation.state": torch.tensor(
            request.current_joints, dtype=torch.float32
        ).unsqueeze(0).to("cuda"),
    }
    
    # Run inference
    with torch.no_grad():
        action = policy.select_action(observation)  # [1, chunk, action_dim]
    
    action_np = action.cpu().numpy().squeeze(0).tolist()  # [chunk, 7]
    
    # Extract subgoal (first action delta → absolute EE target)
    subgoal_xyz = [float(action_np[0][0]), 
                   float(action_np[0][1]), 
                   float(action_np[0][2])]
    
    latency = (time.time() - t0) * 1000
    
    return InferenceResponse(
        action_chunk=action_np,
        latency_ms=round(latency, 1),
        status="ok",
        subgoal_xyz=subgoal_xyz
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "smolvla-base", "device": "cuda"}
```

**Cell 4: Start ngrok Tunnel**
```python
from pyngrok import ngrok, conf

# If you have an ngrok auth token (free tier works):
# ngrok.set_auth_token("YOUR_TOKEN")  # get from ngrok.com

# Kill existing tunnels
ngrok.kill()

# Open tunnel on port 8000
public_url = ngrok.connect(8000, "http")
print(f"\n{'='*60}")
print(f"SmolVLA Inference Endpoint:")
print(f"  {public_url}/predict")
print(f"  {public_url}/health")
print(f"{'='*60}\n")
print("Copy this URL to your local VSCode config!")

# Start FastAPI in background thread
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

thread = threading.Thread(target=run_server, daemon=True)
thread.start()

import time
time.sleep(2)  # wait for startup
print("Server running. Notebook must stay open.")
```

**Cell 5: Test locally in Colab**
```python
import requests, base64, json
from PIL import Image
import numpy as np, io

# Create dummy test image
test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
test_img.save(buf, format="JPEG", quality=85)
img_b64 = base64.b64encode(buf.getvalue()).decode()

# Test call
resp = requests.post(f"http://localhost:8000/predict", json={
    "image_b64": img_b64,
    "instruction": "pick up the red block and place it in the bin",
    "current_joints": [0.0, 0.3, -0.5]
})

result = resp.json()
print(f"Latency: {result['latency_ms']} ms")
print(f"Subgoal XYZ: {result['subgoal_xyz']}")
print(f"Action chunk shape: {len(result['action_chunk'])} × {len(result['action_chunk'][0])}")
```

### 10.3 ngrok URL Management

The ngrok URL changes every session. Handle this gracefully:

**Option A — Manual copy:** Print URL at startup, paste into local `config.yaml`.

**Option B — Shared file (recommended for development):**
```python
# In Colab, at server startup
from google.colab import drive
drive.mount('/content/drive')

with open('/content/drive/MyDrive/smolvla_url.txt', 'w') as f:
    f.write(str(public_url))
print(f"URL saved to Google Drive: {public_url}")
```

```python
# In local VSCode
from googleapiclient.discovery import build  # or just gdrive API
# Or simpler: just use a shared env variable / .env file you update manually
```

**Option C — ngrok static domain (paid, $8/month) or Cloudflare tunnel (free):**
```bash
# Cloudflare tunnel (free, persistent URL)
# In Colab:
!curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8000 &
```

---

## 11. Local VSCode Integration Layer

### 11.1 Project Structure

```
3d_arm_smolvla_mpc/
├── config/
│   ├── arm_params.yaml          # L1, L2, masses, limits
│   ├── mpc_params.yaml          # N, Q, R, P, dt
│   └── smolvla_config.yaml      # endpoint URL, timeout
├── dynamics/
│   ├── kinematics.py            # FK, IK, Jacobian
│   ├── dynamics.py              # M, C, G computation
│   └── tests/
│       ├── test_kinematics.py
│       └── test_dynamics.py
├── mpc/
│   ├── linearize.py             # A, B matrix computation
│   ├── qp_builder.py            # H, c, A_ineq construction
│   └── sl_solver.py             # Your SL oscillator solver (adapted)
├── smolvla_client/
│   ├── client.py                # HTTP client + async polling
│   ├── trajectory_buffer.py     # subgoal interpolation
│   └── tests/
│       └── test_client.py
├── simulation/
│   ├── mujoco_env.py            # MuJoCo environment wrapper
│   ├── arm_3dof.xml             # MJCF model
│   └── render_utils.py          # visualization helpers
├── integration/
│   ├── system.py                # System 1 + System 2 loop
│   └── observer.py              # logging, dashboards
├── notebooks/
│   ├── 01_dynamics_validation.ipynb
│   ├── 02_mpc_solo_test.ipynb
│   └── 03_full_system_test.ipynb
└── main.py                      # entry point
```

### 11.2 SmolVLA Client

```python
# smolvla_client/client.py

import asyncio, aiohttp, base64, io, time, yaml
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SmolVLAResponse:
    action_chunk: np.ndarray   # [chunk_size, 7]
    subgoal_xyz: np.ndarray    # [3]
    latency_ms: float
    timestamp: float

class SmolVLAClient:
    """
    Async client for SmolVLA inference server (Colab + ngrok).
    Non-blocking — never stalls the MPC control loop.
    """
    
    def __init__(self, config_path: str = "config/smolvla_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        self.endpoint = cfg["endpoint_url"]  # e.g. https://abc123.ngrok.io
        self.timeout = cfg.get("timeout_s", 2.0)
        self.latest_response: Optional[SmolVLAResponse] = None
        self._lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def start(self):
        self._session = aiohttp.ClientSession()
        print(f"SmolVLA client initialized → {self.endpoint}")
    
    async def stop(self):
        if self._session:
            await self._session.close()
    
    def encode_image(self, rgb_array: np.ndarray) -> str:
        """Convert numpy RGB array [H,W,3] to base64 JPEG string."""
        img = Image.fromarray(rgb_array.astype(np.uint8))
        img = img.resize((224, 224))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    
    async def query_async(
        self, 
        rgb_image: np.ndarray,
        instruction: str, 
        current_joints: List[float]
    ) -> Optional[SmolVLAResponse]:
        """
        Non-blocking query. Returns None if server unavailable.
        Never raises — always fails gracefully.
        """
        if self._session is None:
            return None
        
        payload = {
            "image_b64": self.encode_image(rgb_image),
            "instruction": instruction,
            "current_joints": current_joints
        }
        
        try:
            async with self._session.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = SmolVLAResponse(
                        action_chunk=np.array(data["action_chunk"]),
                        subgoal_xyz=np.array(data["subgoal_xyz"]),
                        latency_ms=data["latency_ms"],
                        timestamp=time.time()
                    )
                    async with self._lock:
                        self.latest_response = response
                    return response
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"[SmolVLA] Query failed: {e}")
        
        return None
    
    def get_latest_subgoal(self) -> Optional[np.ndarray]:
        """Thread-safe access to latest subgoal. Returns None if no response yet."""
        return self.latest_response.subgoal_xyz if self.latest_response else None
```

### 11.3 Trajectory Buffer

```python
# smolvla_client/trajectory_buffer.py

import numpy as np
from typing import Optional

class TrajectoryBuffer:
    """
    Holds the current subgoal from SmolVLA and provides
    smooth reference trajectories to the MPC controller.
    
    Interpolates between current state and subgoal.
    Detects goal arrival and triggers next SmolVLA query.
    """
    
    def __init__(self, arrival_threshold_rad: float = 0.05):
        self.current_subgoal_q: Optional[np.ndarray] = None
        self.arrival_threshold = arrival_threshold_rad
        self.goal_reached = True  # triggers initial query
    
    def update_subgoal(self, q_goal: np.ndarray):
        """Called when SmolVLA returns a new subgoal (joint space)."""
        self.current_subgoal_q = q_goal.copy()
        self.goal_reached = False
        print(f"[TrajectoryBuffer] New subgoal: {q_goal}")
    
    def get_reference_trajectory(
        self, 
        q_current: np.ndarray,
        N: int,
        dt: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns reference trajectory for MPC horizon.
        
        Returns:
            q_ref:    [N, 3] joint position reference
            qdot_ref: [N, 3] joint velocity reference
        """
        if self.current_subgoal_q is None:
            # No goal yet — hold current position
            q_ref = np.tile(q_current, (N, 1))
            qdot_ref = np.zeros((N, 3))
            return q_ref, qdot_ref
        
        # Trapezoidal interpolation to subgoal
        q_ref = np.zeros((N, 3))
        qdot_ref = np.zeros((N, 3))
        
        q_err = self.current_subgoal_q - q_current
        v_max = 0.5  # rad/s, safe limit
        
        for k in range(N):
            t = (k+1) * dt
            alpha = min(1.0, t * v_max / (np.linalg.norm(q_err) + 1e-6))
            q_ref[k] = q_current + alpha * q_err
            if k < N-1:
                qdot_ref[k] = v_max * q_err / (np.linalg.norm(q_err) + 1e-6) * (1.0 - alpha)
        
        return q_ref, qdot_ref
    
    def check_arrival(self, q_current: np.ndarray) -> bool:
        """Returns True if arm has reached current subgoal."""
        if self.current_subgoal_q is None:
            return True
        err = np.linalg.norm(q_current - self.current_subgoal_q)
        if err < self.arrival_threshold:
            self.goal_reached = True
        return self.goal_reached
```

---

## 12. System 1 ↔ System 2 Integration Protocol

### 12.1 Main Control Loop

```python
# integration/system.py

import asyncio, time
import numpy as np
from dynamics.kinematics import forward_kinematics
from dynamics.dynamics import compute_M, compute_C, compute_G
from mpc.qp_builder import build_qp
from mpc.sl_solver import StuartLandauSolver
from smolvla_client.client import SmolVLAClient
from smolvla_client.trajectory_buffer import TrajectoryBuffer
from simulation.mujoco_env import MuJoCoArm3DOF
from integration.observer import Observer

class DualSystemController:
    """
    Integrates System 1 (SL-MPC) and System 2 (SmolVLA).
    
    System 2 runs at ~1-5 Hz via asyncio.
    System 1 runs at ~100-500 Hz in the main thread.
    They communicate via TrajectoryBuffer (lock-free reads).
    """
    
    def __init__(self, config: dict):
        self.cfg = config
        self.env = MuJoCoArm3DOF(config["arm"])
        self.sl_solver = StuartLandauSolver(
            n_joints=3, 
            horizon=config["mpc"]["N"]
        )
        self.trajectory_buffer = TrajectoryBuffer()
        self.vla_client = SmolVLAClient(config["smolvla"]["endpoint"])
        self.observer = Observer()
        
        self.instruction = config.get("instruction", "reach the target position")
        self.running = False
    
    async def system2_loop(self):
        """
        System 2: SmolVLA queries at low frequency.
        Runs as asyncio task, non-blocking.
        """
        await self.vla_client.start()
        
        while self.running:
            # Check if we need a new subgoal
            q_current = self.env.get_joint_positions()
            
            if self.trajectory_buffer.check_arrival(q_current):
                # Get current RGB image from simulation
                rgb = self.env.render_rgb(width=224, height=224)
                
                # Query SmolVLA (async, non-blocking)
                t_query = time.time()
                response = await self.vla_client.query_async(
                    rgb_image=rgb,
                    instruction=self.instruction,
                    current_joints=q_current.tolist()
                )
                
                if response is not None:
                    # Convert EE subgoal to joint space via IK
                    ee_goal = self.env.current_ee_pos() + response.subgoal_xyz
                    q_goal = self.env.inverse_kinematics(ee_goal)
                    
                    self.trajectory_buffer.update_subgoal(q_goal)
                    self.observer.log_vla_query(response, time.time() - t_query)
                    print(f"[System 2] New subgoal | latency: {response.latency_ms:.0f}ms")
            
            # Poll at VLA frequency (1-5 Hz)
            await asyncio.sleep(0.2)  # 5 Hz poll rate
    
    def system1_step(self) -> np.ndarray:
        """
        System 1: One MPC step using SL oscillators.
        Called at high frequency from simulation step.
        """
        q = self.env.get_joint_positions()
        qdot = self.env.get_joint_velocities()
        
        # Get reference trajectory from buffer
        N = self.cfg["mpc"]["N"]
        dt = self.cfg["mpc"]["dt"]
        q_ref, qdot_ref = self.trajectory_buffer.get_reference_trajectory(q, N, dt)
        
        # Compute dynamics matrices
        M = compute_M(q)
        C = compute_C(q, qdot)
        G = compute_G(q)
        
        # Linearize & build QP
        H, c, A_ineq, b_ineq = build_qp(M, C, G, q, qdot, q_ref, qdot_ref, self.cfg["mpc"])
        
        # Solve with SL oscillators
        tau_opt = self.sl_solver.solve(H, c, A_ineq, b_ineq)
        
        # Log for observability
        self.observer.log_mpc_step(q, qdot, q_ref[0], tau_opt, H)
        
        return tau_opt
    
    def run(self, task_instruction: str, max_steps: int = 5000):
        """Main entry point — runs both systems."""
        self.instruction = task_instruction
        self.running = True
        
        async def main():
            # Launch System 2 as background task
            s2_task = asyncio.create_task(self.system2_loop())
            
            # System 1: high-frequency control loop
            for step in range(max_steps):
                tau = self.system1_step()
                self.env.step(tau)
                
                # Render for visualization
                if step % 10 == 0:
                    self.observer.update_dashboard(self.env)
                
                await asyncio.sleep(0)  # yield to event loop
            
            self.running = False
            s2_task.cancel()
            await self.vla_client.stop()
        
        asyncio.run(main())
```

### 12.2 System 1 ↔ System 2 State Machine

```
States:
  IDLE         → No instruction. Hold position.
  QUERYING_VLA → Waiting for SmolVLA response.
  REACHING     → MPC executing toward current subgoal.
  ARRIVED      → At subgoal. Request next subgoal from SmolVLA.
  DONE         → Task complete (SmolVLA signals completion).

Transitions:
  IDLE       → QUERYING_VLA  : on new instruction
  QUERYING_VLA → REACHING    : on SmolVLA response received
  QUERYING_VLA → REACHING    : timeout → hold last subgoal or freeze
  REACHING   → ARRIVED       : ||q - q_goal|| < threshold
  ARRIVED    → QUERYING_VLA  : request next subgoal
  ARRIVED    → DONE          : SmolVLA action_chunk is zero / done signal
  Any        → IDLE          : on stop command
```

---

## 13. MuJoCo Simulation Setup

### 13.1 MJCF Model — 3-DOF Arm

Save as `simulation/arm_3dof.xml`:

```xml
<?xml version="1.0" ?>
<mujoco model="arm_3dof">
  
  <compiler angle="radian" coordinate="local"/>
  
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  
  <default>
    <joint damping="0.1" armature="0.01" frictionloss="0.05"/>
    <geom contype="1" conaffinity="1" friction="0.8 0.01 0.01"/>
  </default>
  
  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" pos="0 0 0" size="2 2 0.01" rgba="0.8 0.8 0.8 1"/>
    
    <!-- Base (fixed) -->
    <body name="base" pos="0 0 0">
      <geom type="box" size="0.05 0.05 0.05" rgba="0.3 0.3 0.3 1"/>
      
      <!-- Joint 1: azimuth (z-axis rotation) -->
      <body name="link1" pos="0 0 0.10">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="cylinder" fromto="0 0 0 0 0 0.05" size="0.025" rgba="0.7 0.2 0.2 1"/>
        <inertial pos="0 0 0.025" mass="0.5" diaginertia="0.001 0.001 0.0005"/>
        
        <!-- Joint 2: shoulder elevation (y-axis) -->
        <body name="link2" pos="0 0 0.05">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" rgba="0.2 0.7 0.2 1"/>
          <inertial pos="0.125 0 0" mass="0.4" diaginertia="0.0008 0.002 0.002"/>
          
          <!-- Joint 3: elbow elevation (y-axis) -->
          <body name="link3" pos="0.25 0 0">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-2.09 2.09"/>
            <geom type="capsule" fromto="0 0 0 0.20 0 0" size="0.015" rgba="0.2 0.2 0.7 1"/>
            <inertial pos="0.10 0 0" mass="0.2" diaginertia="0.0003 0.0008 0.0008"/>
            
            <!-- End-effector site -->
            <site name="ee_site" pos="0.20 0 0" size="0.015" rgba="1 1 0 1"/>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Target marker (visual only) -->
    <body name="target" pos="0.3 0.2 0.3" mocap="true">
      <geom type="sphere" size="0.025" rgba="1 0.3 0.3 0.7" contype="0" conaffinity="0"/>
    </body>
    
    <!-- Objects for manipulation tasks -->
    <body name="object1" pos="0.3 0.0 0.05">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" rgba="1 0.5 0 1" mass="0.1"/>
    </body>
  </worldbody>
  
  <actuator>
    <!-- Position/torque actuators -->
    <motor name="act1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="act2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-4 4"/>
    <motor name="act3" joint="joint3" gear="1" ctrllimited="true" ctrlrange="-3 3"/>
  </actuator>
  
  <sensor>
    <jointpos name="q1_pos" joint="joint1"/>
    <jointpos name="q2_pos" joint="joint2"/>
    <jointpos name="q3_pos" joint="joint3"/>
    <jointvel name="q1_vel" joint="joint1"/>
    <jointvel name="q2_vel" joint="joint2"/>
    <jointvel name="q3_vel" joint="joint3"/>
    <framepos name="ee_pos" objtype="site" objname="ee_site"/>
  </sensor>
  
  <!-- Camera for SmolVLA input -->
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
  </visual>
  
</mujoco>
```

### 13.2 MuJoCo Python Environment

```python
# simulation/mujoco_env.py

import mujoco
import mujoco.viewer
import numpy as np
from dynamics.kinematics import inverse_kinematics_numerical

class MuJoCoArm3DOF:
    def __init__(self, config: dict):
        self.model = mujoco.MjModel.from_xml_path("simulation/arm_3dof.xml")
        self.data = mujoco.MjData(self.model)
        
        # Camera for SmolVLA (off-screen rendering)
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)
        
        # Joint indices
        self.q_idx = [
            self.model.joint("joint1").qposadr[0],
            self.model.joint("joint2").qposadr[0],
            self.model.joint("joint3").qposadr[0],
        ]
        self.qdot_idx = [
            self.model.joint("joint1").dofadr[0],
            self.model.joint("joint2").dofadr[0],
            self.model.joint("joint3").dofadr[0],
        ]
        
        # EE site
        self.ee_site_id = self.model.site("ee_site").id
    
    def step(self, tau: np.ndarray):
        """Apply torques and advance simulation by one timestep."""
        self.data.ctrl[:3] = tau
        mujoco.mj_step(self.model, self.data)
    
    def get_joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[i] for i in self.q_idx])
    
    def get_joint_velocities(self) -> np.ndarray:
        return np.array([self.data.qvel[i] for i in self.qdot_idx])
    
    def current_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()
    
    def render_rgb(self, width: int = 224, height: int = 224) -> np.ndarray:
        """Render top-down view for SmolVLA."""
        self.renderer.update_scene(self.data, camera="fixed_cam")
        return self.renderer.render()  # [H, W, 3] uint8
    
    def inverse_kinematics(self, ee_target: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Numerical IK using Jacobian pseudo-inverse."""
        return inverse_kinematics_numerical(
            self.get_joint_positions(), 
            ee_target,
            self.model,
            self.data,
            max_iter=max_iter
        )
    
    def set_target_marker(self, pos: np.ndarray):
        """Move the visual target marker."""
        self.data.mocap_pos[0] = pos
    
    def reset(self, q_init: np.ndarray = None):
        mujoco.mj_resetData(self.model, self.data)
        if q_init is not None:
            for i, idx in enumerate(self.q_idx):
                self.data.qpos[idx] = q_init[i]
        mujoco.mj_forward(self.model, self.data)
```

### 13.3 Numerical Inverse Kinematics

```python
# dynamics/kinematics.py (IK portion)

def inverse_kinematics_numerical(
    q_init: np.ndarray,
    ee_target: np.ndarray,
    tol: float = 1e-3,
    max_iter: int = 100,
    alpha: float = 0.5  # step size
) -> np.ndarray:
    """
    Jacobian pseudo-inverse IK.
    
    Returns q such that FK(q) ≈ ee_target.
    Falls back to q_init if IK fails.
    """
    q = q_init.copy()
    
    for _ in range(max_iter):
        ee_curr = forward_kinematics(q)[:3]   # [px, py, pz]
        err = ee_target - ee_curr
        
        if np.linalg.norm(err) < tol:
            return q
        
        J = compute_jacobian(q)[:3, :]   # linear velocity Jacobian [3,3]
        
        # Damped pseudo-inverse
        lam = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(3))
        
        dq = alpha * J_pinv @ err
        q = q + dq
        
        # Clamp to joint limits
        q = np.clip(q, Q_MIN, Q_MAX)
    
    print(f"[IK] Warning: did not converge. Final error: {np.linalg.norm(err):.4f}")
    return q
```

---

## 14. Observability & Transparency Stack

### 14.1 Observer Module

```python
# integration/observer.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading, time

class Observer:
    """
    Real-time logging and visualization.
    Gives you full transparency into both systems.
    """
    
    def __init__(self, buffer_len: int = 1000):
        self.buffer_len = buffer_len
        
        # MPC logs
        self.q_history = deque(maxlen=buffer_len)
        self.qdot_history = deque(maxlen=buffer_len)
        self.q_ref_history = deque(maxlen=buffer_len)
        self.tau_history = deque(maxlen=buffer_len)
        self.qp_cost_history = deque(maxlen=buffer_len)
        self.sl_convergence = deque(maxlen=buffer_len)
        self.timestamps = deque(maxlen=buffer_len)
        
        # VLA logs
        self.vla_queries = []
        self.subgoal_history = deque(maxlen=100)
        self.vla_latencies = deque(maxlen=100)
        
        # EE trajectory
        self.ee_pos_history = deque(maxlen=buffer_len)
        self.ee_ref_history = deque(maxlen=buffer_len)
        
        self._lock = threading.Lock()
    
    def log_mpc_step(self, q, qdot, q_ref, tau, H):
        t = time.time()
        cost = 0.5 * (q - q_ref) @ np.diag([100]*3) @ (q - q_ref)
        with self._lock:
            self.q_history.append(q.copy())
            self.qdot_history.append(qdot.copy())
            self.q_ref_history.append(q_ref.copy())
            self.tau_history.append(tau.copy())
            self.qp_cost_history.append(cost)
            self.timestamps.append(t)
    
    def log_vla_query(self, response, round_trip_s):
        with self._lock:
            self.vla_queries.append({
                "timestamp": time.time(),
                "subgoal": response.subgoal_xyz.copy(),
                "latency_ms": response.latency_ms,
                "round_trip_ms": round_trip_s * 1000
            })
            self.vla_latencies.append(response.latency_ms)
    
    def launch_dashboard(self):
        """Launch live matplotlib dashboard in separate thread."""
        thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        thread.start()
    
    def _dashboard_loop(self):
        plt.ion()
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle("System 1 (SL-MPC) + System 2 (SmolVLA) — Live Dashboard", fontsize=14)
        
        while True:
            with self._lock:
                if len(self.q_history) < 2:
                    time.sleep(0.1)
                    continue
                
                q_arr = np.array(self.q_history)
                q_ref_arr = np.array(self.q_ref_history)
                tau_arr = np.array(self.tau_history)
                cost_arr = np.array(self.qp_cost_history)
                t_arr = np.arange(len(q_arr))
            
            # Plot 1: Joint positions vs reference
            ax = axes[0, 0]
            ax.cla()
            for i, label in enumerate(["q1 (azimuth)", "q2 (shoulder)", "q3 (elbow)"]):
                ax.plot(t_arr, q_arr[:, i], label=f"{label} actual", linewidth=1.5)
                ax.plot(t_arr, q_ref_arr[:, i], '--', alpha=0.6, label=f"{label} ref")
            ax.set_title("Joint Positions (rad)")
            ax.legend(loc="upper right", fontsize=7)
            ax.set_xlabel("Step")
            
            # Plot 2: Joint torques
            ax = axes[0, 1]
            ax.cla()
            for i, label in enumerate(["τ1", "τ2", "τ3"]):
                ax.plot(t_arr, tau_arr[:, i], label=label)
            ax.set_title("Control Torques (Nm)")
            ax.legend()
            ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-5.0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel("Step")
            
            # Plot 3: Tracking error
            ax = axes[1, 0]
            ax.cla()
            err = np.linalg.norm(q_arr - q_ref_arr, axis=1)
            ax.plot(t_arr, err, color='orange', linewidth=1.5)
            ax.set_title("Joint Tracking Error (||q - q_ref|| rad)")
            ax.set_xlabel("Step")
            
            # Plot 4: QP cost
            ax = axes[1, 1]
            ax.cla()
            ax.semilogy(t_arr, cost_arr + 1e-10, color='purple')
            ax.set_title("MPC Cost (log scale)")
            ax.set_xlabel("Step")
            
            # Plot 5: VLA latencies
            ax = axes[2, 0]
            ax.cla()
            if len(self.vla_latencies) > 0:
                lats = list(self.vla_latencies)
                ax.bar(range(len(lats)), lats, color='teal')
                ax.axhline(y=200, color='r', linestyle='--', alpha=0.5, label="200ms target")
                ax.set_title(f"SmolVLA Latency (ms) — mean: {np.mean(lats):.0f}ms")
                ax.legend()
            
            # Plot 6: 3D trajectory (EE)
            ax = axes[2, 1]
            ax.cla()
            if len(self.ee_pos_history) > 0:
                ee = np.array(self.ee_pos_history)
                ax.plot(ee[:, 0], ee[:, 1], 'b-', linewidth=1, alpha=0.7)
                ax.set_title("EE Trajectory (XY plane)")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_aspect('equal')
            
            plt.tight_layout()
            plt.pause(0.1)
```

### 14.2 Logging to File

```python
# Add to Observer
import json, csv

def save_run(self, path: str = "logs/run.json"):
    """Save full run data for offline analysis."""
    data = {
        "q_history": np.array(self.q_history).tolist(),
        "q_ref_history": np.array(self.q_ref_history).tolist(),
        "tau_history": np.array(self.tau_history).tolist(),
        "qp_cost_history": list(self.qp_cost_history),
        "vla_queries": self.vla_queries,
        "timestamps": list(self.timestamps)
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Run saved to {path}")
```

---

## 15. Test Suite — Unit, Integration, System

### 15.1 Unit Tests — Dynamics

```python
# dynamics/tests/test_dynamics.py
import pytest
import numpy as np
from dynamics.kinematics import forward_kinematics, compute_jacobian
from dynamics.dynamics import compute_M, compute_C, compute_G

ARM_PARAMS = {"L0": 0.10, "L1": 0.25, "L2": 0.20,
              "m1": 0.50, "m2": 0.40, "m3": 0.20}

class TestForwardKinematics:
    def test_home_position(self):
        """At q=[0,0,0], EE should be at [L1+L2, 0, L0]."""
        q = np.array([0.0, 0.0, 0.0])
        p = forward_kinematics(q)
        L = ARM_PARAMS
        expected = np.array([L["L1"]+L["L2"], 0.0, L["L0"]])
        np.testing.assert_allclose(p[:3], expected, atol=1e-6)
    
    def test_vertical_position(self):
        """At q=[0, π/2, 0], arm should point straight up."""
        q = np.array([0.0, np.pi/2, 0.0])
        p = forward_kinematics(q)
        # EE height = L0 + L1 (shoulder elevated 90°) 
        assert abs(p[2] - (ARM_PARAMS["L0"] + ARM_PARAMS["L1"])) < 1e-5
        assert abs(p[0]) < 1e-5
    
    def test_azimuth_rotation(self):
        """Joint 1 rotation should only change XY, not Z."""
        q_base = np.array([0.0, 0.3, -0.2])
        q_rot = np.array([np.pi/2, 0.3, -0.2])
        p_base = forward_kinematics(q_base)
        p_rot = forward_kinematics(q_rot)
        np.testing.assert_allclose(p_base[2], p_rot[2], atol=1e-6,
                                   err_msg="Z should not change with joint 1")
    
    def test_reach_consistency(self):
        """Planar reach = sqrt(px²+py²) should match 2D formula."""
        q = np.array([0.5, 0.4, -0.3])
        p = forward_kinematics(q)
        L = ARM_PARAMS
        expected_reach = L["L1"]*np.cos(q[1]) + L["L2"]*np.cos(q[1]+q[2])
        actual_reach = np.sqrt(p[0]**2 + p[1]**2)
        np.testing.assert_allclose(actual_reach, expected_reach, atol=1e-6)

class TestMassMatrix:
    def test_positive_definite(self):
        """M(q) must be positive definite for all valid q."""
        for _ in range(20):
            q = np.random.uniform([-np.pi, -1.5, -2.0], [np.pi, 1.5, 2.0])
            M = compute_M(q)
            eigvals = np.linalg.eigvals(M)
            assert np.all(eigvals > 0), f"M not PD at q={q}, eigvals={eigvals}"
    
    def test_symmetric(self):
        """M(q) must be symmetric."""
        q = np.array([0.3, 0.5, -0.4])
        M = compute_M(q)
        np.testing.assert_allclose(M, M.T, atol=1e-10)
    
    def test_block_structure(self):
        """M[0,1], M[0,2] should be zero (azimuth decoupling)."""
        q = np.array([1.2, 0.3, -0.5])
        M = compute_M(q)
        assert abs(M[0, 1]) < 1e-10
        assert abs(M[0, 2]) < 1e-10

class TestCoriolis:
    def test_skew_symmetry(self):
        """M_dot - 2C should be skew symmetric (passivity property)."""
        q = np.array([0.3, 0.5, -0.4])
        qdot = np.array([0.1, -0.2, 0.3])
        dq = 1e-5
        
        # Numerical M_dot
        M_dot = np.zeros((3,3))
        for i in range(3):
            q_p = q.copy(); q_p[i] += dq
            q_m = q.copy(); q_m[i] -= dq
            M_dot += (compute_M(q_p) - compute_M(q_m)) / (2*dq) * qdot[i]
        
        C = compute_C(q, qdot)
        S = M_dot - 2*C
        
        # Check skew symmetry: x^T S x = 0 for all x
        x = np.random.randn(3)
        assert abs(x @ S @ x) < 1e-6, f"Passivity violated: x^T(M_dot-2C)x = {x@S@x:.6f}"

class TestGravity:
    def test_g1_zero(self):
        """G[0] must always be zero (azimuth joint)."""
        for _ in range(10):
            q = np.random.uniform([-np.pi, -1.5, -2.0], [np.pi, 1.5, 2.0])
            G = compute_G(q)
            assert abs(G[0]) < 1e-10, f"G[0] non-zero at q={q}: G={G}"
    
    def test_energy_conservation(self):
        """∂V/∂q should match G(q). V = Σ mᵢ g pᵢᶜz."""
        from dynamics.kinematics import potential_energy
        q = np.array([0.3, 0.5, -0.4])
        dq = 1e-6
        
        G_numerical = np.zeros(3)
        for i in range(3):
            q_p = q.copy(); q_p[i] += dq
            q_m = q.copy(); q_m[i] -= dq
            G_numerical[i] = (potential_energy(q_p) - potential_energy(q_m)) / (2*dq)
        
        G_analytical = compute_G(q)
        np.testing.assert_allclose(G_analytical, G_numerical, atol=1e-5)
```

### 15.2 Unit Tests — MPC/QP

```python
# mpc/tests/test_qp.py
import pytest
import numpy as np
from mpc.qp_builder import build_qp
from mpc.sl_solver import StuartLandauSolver
import osqp  # for ground truth comparison

class TestQPConstruction:
    def test_H_positive_semidefinite(self):
        """QP Hessian must be PSD."""
        q = np.zeros(3); qdot = np.zeros(3)
        q_ref = np.tile(np.array([0.2, 0.3, -0.1]), (10, 1))
        qdot_ref = np.zeros((10, 3))
        H, c, _, _ = build_qp(...)
        eigvals = np.linalg.eigvals(H)
        assert np.all(eigvals >= -1e-8)
    
    def test_sl_vs_osqp(self):
        """SL solver should match OSQP within 5% on random QPs."""
        solver = StuartLandauSolver(n_joints=3, horizon=10)
        
        for trial in range(20):
            n = 30
            A = np.random.randn(n, n)
            H = A.T @ A + 0.01*np.eye(n)  # PSD
            c = np.random.randn(n)
            # Box constraints
            A_ineq = np.vstack([np.eye(n), -np.eye(n)])
            b_ineq = np.ones(2*n)
            
            # OSQP reference
            prob = osqp.OSQP()
            prob.setup(H, c, A_ineq, None, b_ineq, 
                       verbose=False, eps_abs=1e-6, eps_rel=1e-6)
            res = prob.solve()
            x_ref = res.x
            
            # SL solver
            x_sl = solver.solve(H, c, A_ineq, b_ineq)
            
            cost_ref = 0.5*x_ref@H@x_ref + c@x_ref
            cost_sl = 0.5*x_sl@H@x_sl + c@x_sl
            
            assert abs(cost_sl - cost_ref) / (abs(cost_ref) + 1e-8) < 0.05, \
                f"Trial {trial}: SL cost {cost_sl:.4f} vs OSQP {cost_ref:.4f}"
```

### 15.3 Integration Tests — SmolVLA Client

```python
# smolvla_client/tests/test_client.py
import pytest, asyncio, numpy as np
from unittest.mock import AsyncMock, patch
from smolvla_client.client import SmolVLAClient, SmolVLAResponse

class TestSmolVLAClient:
    @pytest.mark.asyncio
    async def test_graceful_failure(self):
        """Client must return None on timeout, not crash."""
        client = SmolVLAClient.__new__(SmolVLAClient)
        client.endpoint = "https://nonexistent-url-12345.ngrok.io"
        client.timeout = 0.5
        client._session = None
        
        await client.start()
        result = await client.query_async(
            rgb_image=np.zeros((224,224,3), dtype=np.uint8),
            instruction="pick up block",
            current_joints=[0.0, 0.3, -0.2]
        )
        assert result is None, "Should return None on failure, not crash"
        await client.stop()
    
    @pytest.mark.asyncio
    async def test_image_encoding(self):
        """Base64 image encoding should be correct."""
        client = SmolVLAClient.__new__(SmolVLAClient)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        b64 = client.encode_image(img)
        assert isinstance(b64, str)
        assert len(b64) > 100
        import base64, io
        from PIL import Image
        decoded = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert decoded.size == (224, 224)   # resized
```

### 15.4 System Integration Test

```python
# Run this to validate full pipeline without SmolVLA (mock VLA)
def test_full_pipeline_mock_vla():
    """
    Tests complete System 1 + System 2 integration with
    mock SmolVLA that returns predetermined subgoals.
    """
    from simulation.mujoco_env import MuJoCoArm3DOF
    from mpc.sl_solver import StuartLandauSolver
    from smolvla_client.trajectory_buffer import TrajectoryBuffer
    
    env = MuJoCoArm3DOF({"xml_path": "simulation/arm_3dof.xml"})
    solver = StuartLandauSolver(n_joints=3, horizon=10)
    buffer = TrajectoryBuffer(arrival_threshold_rad=0.05)
    
    # Pre-set a subgoal
    q_target = np.array([0.5, 0.3, -0.4])
    buffer.update_subgoal(q_target)
    
    # Run 500 steps
    tracking_errors = []
    for step in range(500):
        q = env.get_joint_positions()
        qdot = env.get_joint_velocities()
        
        q_ref, qdot_ref = buffer.get_reference_trajectory(q, N=10)
        H, c, A_ineq, b_ineq = build_qp_from_state(q, qdot, q_ref, qdot_ref)
        tau = solver.solve(H, c, A_ineq, b_ineq)
        
        env.step(tau)
        tracking_errors.append(np.linalg.norm(q - q_target))
    
    final_error = tracking_errors[-1]
    assert final_error < 0.1, f"Did not converge: final error = {final_error:.3f} rad"
    print(f"✓ Converged in ~{np.argmin(tracking_errors)} steps. Final error: {final_error:.4f} rad")
```

### 15.5 Test Execution

```bash
# Run all tests
pytest dynamics/tests/ mpc/tests/ smolvla_client/tests/ -v

# Run with coverage
pytest --cov=dynamics --cov=mpc --cov=smolvla_client --cov-report=html

# Run specific validation
python -m pytest mpc/tests/test_qp.py::TestQPConstruction::test_sl_vs_osqp -v -s

# Quick sanity check
python integration/system.py --mode=mock --steps=200 --render
```

---

## 16. Datasets & Benchmark Tasks

### 16.1 Datasets for SmolVLA Fine-Tuning

| Dataset | Size | Description | Access | Relevance |
|---|---|---|---|---|
| **Open X-Embodiment** | 527K episodes, 22 robots | Diverse manipulation, multi-embodiment | [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io) | Pre-training backbone |
| **DROID** | 76K episodes, Franka Panda | Multi-scene, multi-view, 86 tasks | [droid-dataset.github.io](https://droid-dataset.github.io) | Best for arm fine-tuning |
| **BridgeData V2** | 60K episodes, Franka/WidowX | Kitchen + lab tasks | [rail-berkeley.github.io/bridgedata](https://rail-berkeley.github.io/bridgedata) | Good for pick-place |
| **RoboSet** | 100K episodes | Factory-style tasks | HuggingFace datasets | Industrial setting |
| **LIBERO** | 130 tasks, 500 demos each | Lifelong robot learning benchmark | [libero-project.github.io](https://libero-project.github.io) | Structured evaluation |
| **ManiSkill2** | 20K+ demos, 20 tasks | PhysX simulation ground truth | [maniskill2.github.io](https://maniskill2.github.io) | Sim benchmark |
| **RLBench** | 100 tasks | Simulated diverse manipulation | [github.com/stepjam/RLBench](https://github.com/stepjam/RLBench) | Baseline comparison |
| **FurnitureBench** | ~5K demos | IKEA-style assembly | [clvrai.com/furniture-bench](https://clvrai.com/furniture-bench) | Contact-rich target |

**Recommended pipeline:**
```
1. SmolVLA pretrained on Open X-Embodiment (HuggingFace checkpoints)
2. Fine-tune on DROID subset (pick-place) with LoRA
3. Fine-tune on your own MuJoCo demos (generated programmatically)
4. Benchmark on LIBERO tasks
```

### 16.2 Benchmark Tasks

#### Task 1: Point-to-Point Reaching (SL-MPC solo, no VLA)

**Description:** Reach a target joint configuration from random initial configuration.

**Metrics:**
- Final joint error ||q_final - q_target|| (rad)
- Convergence time (steps)
- Peak torque (Nm)
- Constraint violations (count)

**Baselines:**
- PD controller (Kp=50, Kd=5)
- iLQR (MJPC)
- OSQP-MPC (direct, your comparison)
- **Your SL-MPC**

```python
# benchmark_reaching.py
REACHING_CONFIGS = [
    {"q_init": [0,0,0],       "q_target": [0.5, 0.3, -0.4]},
    {"q_init": [1.0, 0, 0],   "q_target": [-0.5, 0.5, -0.8]},
    {"q_init": [0, 0.5, 0],   "q_target": [0, -0.3, 0.5]},
    # 20 random configs
    *[{"q_init": np.random.uniform(-0.5, 0.5, 3).tolist(),
       "q_target": np.random.uniform(-1, 1, 3).tolist()} for _ in range(20)]
]
```

#### Task 2: EE Trajectory Tracking

**Description:** Track a figure-8 or circular path in EE Cartesian space.

**Reference trajectory:**
```python
def figure8_trajectory(t, A=0.1, T=5.0):
    """Lemniscate of Bernoulli in XY plane."""
    omega = 2*np.pi/T
    x = 0.3 + A * np.cos(omega*t) / (1 + np.sin(omega*t)**2)
    y = A * np.sin(omega*t)*np.cos(omega*t) / (1 + np.sin(omega*t)**2)
    z = 0.25  # fixed height
    return np.array([x, y, z])
```

**Metrics:**
- RMS EE tracking error (mm)
- Max tracking error (mm)
- Control effort ∫||τ||² dt

#### Task 3: Pick-and-Place (Full System 1+2)

**Description:** Given RGB image + instruction "pick up the orange block and place it in the bin", the full VLA+MPC system must:
1. SmolVLA identifies block location, outputs reach waypoint
2. SL-MPC reaches to block
3. SmolVLA outputs grasp pose
4. SL-MPC executes grasp
5. SmolVLA outputs place waypoint
6. SL-MPC executes place

**Metrics:**
- Task success rate (%)
- Total completion time (s)
- Number of SmolVLA queries
- Mean MPC tracking error during execution

**Setup in MuJoCo:**
```python
# Objects: colored block on table, bin 30cm away
# Camera: fixed overhead view → SmolVLA input
# Instruction variations:
TASK_VARIANTS = [
    "pick up the orange block and place it in the bin",
    "move the cube to the container",
    "grasp the object and drop it in the box",
    "put the block away"
]
```

#### Task 4: Obstacle Avoidance

**Description:** Reach target while avoiding a cylindrical obstacle added to workspace.

**MPC constraint addition:**
```python
# Add to QP: distance from obstacle center
# d(q) = ||FK(q) - p_obs|| >= r_obs + r_safety
# Linearized: ∇d(q̄)·δq >= r_min - d(q̄)
```

**Tests robustness** of SL-MPC constraint handling + whether SmolVLA can reason about obstacles.

#### Task 5: LIBERO Benchmark (SmolVLA evaluation)

Use the **LIBERO-Spatial** suite (10 tasks, 50 demos each) to evaluate SmolVLA fine-tuned on your arm:

```bash
# Install LIBERO
pip install libero

# Run benchmark
python benchmarks/libero_eval.py \
    --policy smolvla \
    --suite libero_spatial \
    --num_episodes 20 \
    --mpc_backend sl_oscillator
```

Expected baseline scores (from literature):
- ACT: ~78% success
- Diffusion Policy: ~82% success
- SmolVLA (base): ~71% success (pre-fine-tune)
- **Target after fine-tune:** >80% success

---

## 17. Implementation Roadmap

### Phase 1 — Week 1-2: Dynamics & Simulation

```
Day 1-2:  Implement M(q), C(q,q̇), G(q) in Python
          Unit tests: PD property, energy conservation, block structure
Day 3:    MJCF model for 3-DOF arm in MuJoCo
          Verify dynamics against mujoco.mj_fwdinv()
Day 4-5:  FK, Jacobian, IK implementation + tests
          Sanity check: FK at home position, IK round-trip
Day 6-7:  MPC formulation: A, B matrices, QP builder
          Compare linearized model vs full MuJoCo
```

### Phase 2 — Week 3-4: SL Solver Extension

```
Day 8-9:   Port your 2-DOF SL solver to N=3 joints, N=10 horizon
           Test on small random QPs vs OSQP
Day 10-11: Warm starting implementation
           Benchmark: convergence iterations with/without warm start
Day 12-13: Constraint handling via augmented Lagrangian
           Test: box constraint satisfaction
Day 14:    Full MPC loop test: reach task with SL solver
           Benchmark vs PD and OSQP baselines
```

### Phase 3 — Week 5-6: SmolVLA Deployment

```
Day 15-16: Set up Colab notebook
           Load SmolVLA, test inference locally in Colab
Day 17:    FastAPI server + ngrok tunnel
           Test /health and /predict endpoints from Colab
Day 18-19: Local VSCode client
           Integration test: send image from local, get action chunk
Day 20-21: TrajectoryBuffer + IK pipeline
           Test: subgoal → joint-space reference → MPC input
```

### Phase 4 — Week 7-8: System Integration

```
Day 22-23: DualSystemController: async System 2 + sync System 1
           Mock SmolVLA test: confirm asyncio doesn't block MPC
Day 24-25: Observer & dashboard
           Confirm all plots live during simulation
Day 26-27: Pick-and-place task end-to-end
           First full run: RGB → VLA → MPC → MuJoCo
Day 28:    Debug, tune, document
```

### Phase 5 — Week 9-10: Benchmarking

```
Day 29-30: Reaching benchmark (Task 1) — all 4 baselines
Day 31-32: Trajectory tracking benchmark (Task 2)
Day 33-34: Pick-and-place success rate (Task 3, 50 trials)
Day 35:    LIBERO evaluation (if time permits)
Day 36-37: Write up results, generate plots
Day 38-40: Extend to 6-DOF (see appendix)
```

---

## 18. References

1. **Stuart-Landau oscillators for QP**: Hopfield (1984), Lagrange oscillators for optimization (Platt & Barr, 1988), modern neuromorphic extensions (Xue et al., 2021)
2. **SmolVLA**: HuggingFace LeRobot, 2025. [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
3. **Modern Robotics (MR)**: Lynch & Park, 2017 — standard DH derivation, Jacobian, dynamics
4. **Spong, Hutchinson, Vidyasagar**: Robot Modeling and Control, 2020 — EOM derivation used here
5. **MPC Theory**: Rawlings, Mayne, Diehl — Model Predictive Control: Theory, Computation, and Design, 2020
6. **iLQR for robotics**: Tassa, Mansard, Todorov — Control-Limited Differential Dynamic Programming, ICRA 2014
7. **MuJoCo**: Todorov, Erez, Tassa — MuJoCo: A physics engine for model-based control, IROS 2012
8. **π₀**: Black et al. — π₀: A Vision-Language-Action Flow Model for General Robot Control, 2024
9. **GR00T N1**: NVIDIA, 2025 — GR00T: Generalist Robot Foundation Model
10. **DROID**: Khazatsky et al. — DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset, 2024
11. **Open X-Embodiment**: Open X-Embodiment Collaboration, 2023
12. **LIBERO**: Liu et al. — LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning, 2023

---

## APPENDIX A: Extension to 6-DOF (Franka Panda Style)

The 3-DOF arm gives position control. For **full pose control** (position + orientation), add 3 more joints (wrist roll, wrist pitch, wrist yaw):

```
Joint 4: Wrist 1 — rotates about z-axis of link 3
Joint 5: Wrist 2 — rotates about y-axis
Joint 6: Wrist 3 — rotates about z-axis (end-roll)
```

**DH Table Extension:**

| Joint | θᵢ | dᵢ | aᵢ | αᵢ |
|-------|-----|-----|-----|-----|
| 4 | q₄ | d₄ | 0 | π/2 |
| 5 | q₅ | 0 | 0 | −π/2 |
| 6 | q₆ | d₆ | 0 | 0 |

**For the 6-DOF arm:**
- M ∈ ℝ⁶ˣ⁶, C ∈ ℝ⁶ˣ⁶, G ∈ ℝ⁶
- QP variables: 6×N per horizon
- SL oscillator network: 60 oscillators for N=10 (still very tractable)
- Jacobian now square (6×6) — full pose control, no null space

Use the **Franka Panda MJCF** from MuJoCo Menagerie as your 6-DOF simulation:
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
# MJCF: mujoco_menagerie/franka_emika_panda/panda.xml
```

---

## APPENDIX B: Event Camera Extension (Future Phase)

When you extend with DVS event cameras (as planned):

```
Event camera → Voxel grid representation [T, H, W] 
             → Lightweight ViT encoder (shared with RGB encoder)
             → Concatenate temporal features with RGB features
             → SmolVLA action head receives richer temporal observations
```

**Key advantage:** Event cameras detect motion at microsecond resolution. During fast arm motion (where RGB gets motion blur), event cameras still capture object contact/collision events clearly. This closes the perception gap during fast MPC execution.

---

## APPENDIX C: SmolVLA Fine-Tuning Quickstart

```python
# Fine-tune SmolVLA on your MuJoCo pick-place demos
# Collect demos first, then fine-tune

from lerobot.scripts.train import train
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 1. Collect demos (teleoperate or script)
# Format: LeRobot HuggingFace dataset format
# Each episode: sequence of (obs, action) pairs
# obs: {images: {top: [H,W,3]}, state: [6]}  # [q, qdot]
# action: [7]  # [dx, dy, dz, droll, dpitch, dyaw, gripper]

# 2. Push to HuggingFace (or local path)
dataset = LeRobotDataset("your-username/arm-3dof-pickplace")

# 3. Fine-tune with LoRA
train(
    policy_type="smolvla",
    dataset_repo_id="your-username/arm-3dof-pickplace",
    training_steps=50000,
    batch_size=32,
    lr=1e-4,
    use_lora=True,
    lora_rank=8,
    output_dir="outputs/smolvla_finetuned"
)
```

---

*Document prepared: March 2026. Alvin — BITS Pilani / IIT Bombay. For thesis extension and Addverb-phase robotics development. System based on verified SL oscillator QP solver (Stuart-Landau + Lagrange, already working and validated against OSQP).*
