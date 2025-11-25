# B3-3YP
(Insulin and Glucagon Pump Controller Simulation)

This repository contains our 3YP group's implementation of the *Dual-Hormone MPC Algorithm* model described in  
**Dias et al., “A Dual-Hormone Closed-Loop Artificial Pancreas for Type 1 Diabetes”**.  

---

## Team Responsibilities

### **Victor**  
**Status:** Completed  
- Implemented **all physiological constants and volume/flow parameters**, including  
  organ volumes, blood flow rates, insulin distribution volumes, glucagon parameters,  
  metabolic constants, and baseline values.  
- Created the **initial conditions** and variable definitions needed for the model.  
- Provided the parameter dictionary used by the ODE solver.  

### **Lin**  
**Status:** In progress  
- Implemented the **Insulin subsystem ODEs** from the Dias/Sorensen model.  
- Implemented the **Glucagon subsystem ODE** including GH-dependence.  
- Designed the `insulin_glucagon_odes()` function with clean layout and clear documentation.  
- Added support for:
  - time-varying inputs (U1, U2),
  - external glucose signal GH(t),
  - parameter injection for future tuning.
- Created the **state vector assembly** (IB, IH, IG, IL, IK, IPV, IPI, Γ).  
- Ensured the output returns a **state derivative vector** suitable for plotting and coupling.  

###

---

## How to Run the ODE Subsystem and Plots