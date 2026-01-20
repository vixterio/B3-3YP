# B3-3YP
(Dual-Hormone MPC for Blood Glucose Regulation)

This repository contains our 3YP group's implementation of the *Dual-Hormone MPC Algorithm* model described in  
**Dias et al., “A Dual-Hormone Closed-Loop Artificial Pancreas for Type 1 Diabetes”**.  The MPC strategy is used to regulate blood glucose in Type 1 Diabeters Mellitus (T1DM) using the nonlinear Sorensen physiological model as the plant and a linearised Sorensen model for prediction and control.

---

## Repository Structure
```text
B3-3YP/
│
├── CL_simulation_LY.py        # Closed-loop simulation
├── MPC_controller_LY.py       # Constrained linear MPC implementation
├── SORENSEN_MODEL_TPB.py      # Nonlinear Sorensen physiological model
├── LINEARISED_MODEL_TPB.py    # Linearised Sorensen model for MPC
│
├── README.md                 # This file
```
---

## High-level Architecture
The closed-loop system follows a standard MPC architecture
- *Plant*: Full nonlinear Sorensen glucose-insulin-glucagon model
- *Controller*: Linear MPC with constraints and supervisory switching
- *Feedback*: Peripheral glucose measurement (G_PI)
- *Disturbances*: Unannounced meals applied only to the nonlinear plant

---

## Key Features

### Physiological Plant
- Full nonlinear Sorensen model with glucose, insulin, and glucagon dynamics  
- Endogenous glucose production and uptake mechanisms preserved  
- Continuous-time ODE integration using `solve_ivp`  

### Model Predictive Control
- Linear MPC based on linearised Sorensen dynamics  
- Prediction horizon: **25 steps**  
- Control horizon: **15 steps**  
- Sampling time: **5 minutes**  
- Output constraints: **80–120 mg/dL**  
- Input constraints and rate limits enforced  

### Dual-Hormone Supervisory Logic
The controller uses a mode-based supervisory structure:

| Glucose Region | Mode      | Hormone Action              |
|----------------|-----------|-----------------------------|
| G < 80 mg/dL   | GLUCAGON  | Basal glucagon infusion     |
| 80 ≤ G ≤ 120   | BASAL     | Minimal / no insulin        |
| G > 120 mg/dL  | BOLUS     | Insulin infusion            |

Additional rules:
- Insulin and glucagon are **never** infused simultaneously  
- Glucagon acts as a **safety hormone**, not a continuous control input  

--

## File Descriptions

### `CL_simulation_LY.py`
Main simulation script.

**Responsibilities:**
- Runs the closed-loop simulation  
- Applies unannounced meal disturbances  
- Handles supervisory mode switching 
- Generates glucose, insulin, and glucagon plots  

**Run the simulation:**
```bash
python CL_simulation_LY.py
```

### `MPC_controller_LY.py`
Constrained linear MPC implementation.

**Features:**
- Discrete-time linear MPC
- Input and rate constraints
- Solved using CVXPY
- MPC operates in deviation coordinates around a physiological equilibrium.  

### `SORENSEN_MODEL_TPB.py`
Nonlinear Sorensen physiological model.

<!-- **Features:**
Includes:
- Glucose compartments
- Insulin compartments
- Glucagon dynamics
- Endogenous production and uptake
- No endogenous insulin production (T1DM assumption)
- Exogenous insulin and glucagon enter only through ODE source terms, not via state injection. -->


### `LINEARISED_MODEL_TPB.py`
Linearised Sorensen model used for control.

<!-- **Features:**
- Continuous-time A, B, C matrices
- Equilibrium state x_star
- Equilibrium input u_star
- Sampling time
- This model is used only for prediction, not for plant simulation. -->
 


--

## How to Run Requirements
- Python ≥ 3.9
- NumPy
- SciPy
- Matplotlib
- CVXPY
- OSQP

**Install dependencies:**
``` bash
pip install numpy scipy matplotlib cvxpy osqp
```
*Run simulation:*
```bash
python CL_simulation_LY.py
```


<!-- # ## Team Responsibilities

# ### **Victor**  
# **Status:** Completed  
# - Implemented **all physiological constants and volume/flow parameters**, including  
#   organ volumes, blood flow rates, insulin distribution volumes, glucagon parameters,  
#   metabolic constants, and baseline values.  
# - Created the **initial conditions** and variable definitions needed for the model.  
# - Provided the parameter dictionary used by the ODE solver.  

# ### **Lin**  
# **Status:** In progress  
# - Implemented the **Insulin subsystem ODEs** from the Dias/Sorensen model.  
# - Implemented the **Glucagon subsystem ODE** including GH-dependence.  
# - Designed the `insulin_glucagon_odes()` function with clean layout and clear documentation.  
# - Added support for:
#   - time-varying inputs (U1, U2),
#   - external glucose signal GH(t),
#   - parameter injection for future tuning.
# - Created the **state vector assembly** (IB, IH, IG, IL, IK, IPV, IPI, Γ).  
# - Ensured the output returns a **state derivative vector** suitable for plotting and coupling.  

# ### **Tommy**
# **Status:** In progress
# - Created **WBS** for the whole project
# - implemented the non-linear **Sorensen model**
# - checked that the output confirms what we expect for a fasting state with no insulin and glucagon exogeneous injections.
# - created a new sorensen model file, implemented the initial conditions for a T1D and checked the output is correct
# - need to add the non-linear ODEs to the file before I linearise it
# - linearise the system using Jacobians

# ---

# ## How to Run the ODE Subsystem and Plots -->