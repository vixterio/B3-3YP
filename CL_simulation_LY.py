"""
CL_simulation_LY.py

Closed-loop simulation of dual-hormone MPC controlling the
nonlinear Sorensen glucose–insulin–glucagon model.

This script uses the state-space formulation:

    x_{k+1} = A x_k + B u_k
    y_k     = C x_k + e_k

where:
- x is the nonlinear Sorensen state
- u = [u1, u2] are insulin and glucagon infusion rates
- y is the peripheral glucose measurement (G_PI)
- e_k represents unmeasured disturbances (e.g. meals)
    - this is not modelled explicitly, propagated or fed into the MPC
- The MPC uses a linearised model (A,B,C) around the nominal point
- Feedback occurs via G_PI measurement only

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import SORENSEN_MODEL_TPB as sorensen


# Basal hormone levels
BASAL_GLUCAGON = 0.34   # mg/min
BASAL_INSULIN = 0.0

# Pump-like basal insulin as periodic micro-bolus pulses
ENABLE_BASAL_PULSES = True
BASAL_PERIOD_MIN = 200.0      # matches the figure's ~200 min spacing
BASAL_PULSE_WIDTH_MIN = 10.0  # short pulse duration
BASAL_PULSE_RATE = 10.0       # mU/min during the pulse (tune slightly if needed)


# Disable Sorensen internal controller
def zero_controller(y):
    return 0.0, 0.0
sorensen.temporary_controller = zero_controller

from MPC_controller_LY import (
    load_continuous_linear_model,
    zoh_discretise,
    MPCController,
    TAU_S_MIN,
    R_SETPOINT,
    NP,
    NC,
    Y_MIN,
    Y_MAX,
    THRESH_HIGH,
    THRESH_LOW
)

# Import nonlinear Sorensen plant for states
from SORENSEN_MODEL_TPB import (
    sorensen_odes,
    initial_conditions,
    STATE_ORDER
)

# Meal absorption model
K_ABS = 1.0 / 40.0  # 1/min (time constant ~40 min)

# Simulate nonlinear Sorensen for one MPC step
def sorensen_step(x_aug, u, dt, meal_mg=0.0, n_substeps=50):
    """
    x_aug includes the 19 Sorensen states plus one extra state at the end:
      D_meal [mg] = glucose in a gut reservoir that appears into plasma with 1st-order kinetics.

    meal_mg is the bolus amount added to D_meal at the start of this step.
    """

    uI_mU, uG_mg = u

    # Conver to Sorensen internal units
    uI = uI_mU * 1e3
    uG = uG_mg * 1e9

    # Add meal bolus to resevoir (mg)
    x0 = x_aug.copy()
    x0[-1] += float(meal_mg)  # add meal bolus into reservoir (mg)

    # Apply meal disturbance

    def wrapped_odes(t, y):

        ys = y[:19] # Soresen states
        Dm = y[19] # meal resevoir state

        # PLant ODEs in Sorensen coordinates
        dydt_s = sorensen_odes(t, ys, uI, uG)

         # meal reservoir -> plasma appearance
        dDm = -K_ABS * Dm
        Ra = K_ABS * Dm  # mg/min

        # add appearance to plasma glucose derivative
        dydt_s[0] += Ra / sorensen.Vg_PV

        return np.concatenate([dydt_s, [dDm]])

    t_eval = np.linspace(0.0, dt, n_substeps)

    sol = solve_ivp(
        wrapped_odes,
        (0.0, dt),
        x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )

    # Return final state AND glucose trajectory
    return sol.y[:, -1], sol.y[7, :]

# Closed-loop simulation
def run_closed_loop(sim_duration_min=2000):

    # Load linear MPC model
    model = load_continuous_linear_model()
    A, B, C = model["A"], model["B"], model["C"]
    x_star, u_star = model["x_star"], model["u_star"]

    y_star = float((C @ x_star)[0])

    # Convert TPB linear model input units -> MPC paper units
    B = B.copy()
    B[:, 0] *= 1e3   # µU/min basis -> mU/min basis
    B[:, 1] *= 1e9   # pg/min basis -> mg/min basis

    # Convert TPB equilibrium inputs to paper units too
    u_star = np.array(u_star, dtype=float).copy()
    u_star[0] /= 1e3  # µU/min -> mU/min
    u_star[1] /= 1e9  # pg/min -> mg/min

    # Discretise
    A_d, B_d = zoh_discretise(A, B, TAU_S_MIN)

    # MPC
    mpc = MPCController(A_d, B_d, C, tau_s_min=TAU_S_MIN, Np=NP, Nc=NC)

    # Initial nonlinear plant state
    x_true_19 = np.array([initial_conditions[k] for k in STATE_ORDER], dtype=float)
    x_true = np.concatenate([x_true_19, [0.0]])  # D_meal = 0 mg

    # Deviation coordinates for MPC
    x_dev = x_true[:19] - x_star
    u_prev_dev = np.zeros(2)

    sim_steps = int(sim_duration_min / TAU_S_MIN)
    t_glucose = []
    glucose = []    

    t_mpc = []
    insulin = []
    glucagon_inf = []

    ## --- Disturbance schedule (Fig. 13 vs Fig. 14 style) ---
    # Case 1: one disturbance
    # Case 2: one disturbance + another at 1600 min
    DISTURBANCE_CASE = 1
    MEAL_GRAMS = 50.0  # paper refers to meal modelling in grams

    if DISTURBANCE_CASE == 1:
        meal_events = [250.0]
    else:
        meal_events = [250.0, 1600.0]

    for k in range(sim_steps):

        current_time = k * TAU_S_MIN

        # Meal disturbance (unannounced)
        meal_mg = 0.0
        for t_evt in meal_events:
            # trigger exactly on the sample that hits the event time
            if abs(current_time - t_evt) < 1e-12:
                meal_mg = MEAL_GRAMS * 1000.0  # g -> mg
                break

        # Measurement
        G_PI = x_true[7]
        
        t_mpc.append(current_time)
        t_glucose.append(current_time) 
        glucose.append(G_PI)

        # MPC state correction (paper assumption)
        # enforce that the deviation state's measured output matches the plant measurement
        y_dev_meas = G_PI - y_star
        y_dev_model = float((C @ x_dev)[0])

        x_dev[7] += (y_dev_meas - y_dev_model)   # C selects state 7 (G_PI) in TPB model


        # Supervisory switching MPC
        if G_PI > THRESH_HIGH:
            mode = "BOLUS"
        elif G_PI < THRESH_LOW:
            mode = "GLUCAGON"
        else:
            mode = "BASAL"

        # MPC
        u_dev, _ = mpc.compute_control(
            xk=x_dev,
            uk_prev=u_prev_dev,
            r_dev=R_SETPOINT - y_star,
            y_min_dev=Y_MIN - y_star,
            y_max_dev=Y_MAX - y_star,
            mode=mode,
            lambda_u=None
        )

        if mode in ["BOLUS", "BASAL"]:
            u_star_mode = np.array([u_star[0], 0.0], dtype=float)  # no glucagon baseline
        elif mode == "GLUCAGON":
            u_star_mode = np.array([0.0, BASAL_GLUCAGON], dtype=float)  # no insulin baseline


        u = u_dev + u_star_mode
    
        # Add pump basal pulses on top of MPC command (independent of controller logic)
        if ENABLE_BASAL_PULSES:
            phase = (current_time % BASAL_PERIOD_MIN)
            if phase < BASAL_PULSE_WIDTH_MIN:
                u[0] += BASAL_PULSE_RATE


        # Store absolute values for plotting
        insulin.append(u[0])
        glucagon_inf.append(u[1])


        # Nonlinear plant update
        x_true, G_trace = sorensen_step(x_true, u, TAU_S_MIN, meal_mg=meal_mg, n_substeps=20)

        # Append internal glucose trajectory for smooth plot
        for i, g in enumerate(G_trace[1:], start=1):
            t_glucose.append(current_time + i * (TAU_S_MIN / (len(G_trace) - 1)))
            glucose.append(g)

        # Linear model update (for MPC)
        x_dev = A_d @ x_dev + B_d @ u_dev
        u_prev_dev = u_dev.copy()

    return np.array(t_glucose), np.array(glucose), np.array(t_mpc),np.array(insulin), np.array(glucagon_inf)



if __name__ == "__main__":

    t_g, G, t_mpc, I, Gg = run_closed_loop()

    plt.figure(figsize=(12, 9))

    # Glucose (continuous)
    plt.subplot(3, 1, 1)
    plt.plot(t_g, G, label="Glucose (G_PI)")
    plt.axhline(90, linestyle="--", color="k", label="Setpoint 90")
    plt.ylabel("Glucose (mg/dL)")
    plt.grid(True)
    plt.legend()

    # Insulin (MPC, ZOH)
    plt.subplot(3, 1, 2)
    plt.step(t_mpc, I, where="post", label="Insulin infusion")
    plt.ylabel("Insulin (mU/min)")
    plt.grid(True)
    plt.legend()

    # Glucagon (MPC, ZOH)
    plt.subplot(3, 1, 3)
    plt.step(t_mpc, Gg, where="post", label="Glucagon infusion")
    plt.ylabel("Glucagon (mg/min)")
    plt.xlabel("Time (min)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

