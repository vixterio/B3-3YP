"""
CL_simulation_LY.py

Closed-loop simulation of dual-hormone MPC controlling the
nonlinear Sorensen glucose–insulin–glucagon model.

This script uses the state-space formulation:

    x_{k+1} = A x_k + B u_k
    y_k     = C x_k + e_k

where:
- x is the linearised Sorensen state
- u = [u1, u2] are insulin and glucagon infusion rates
- y is the peripheral glucose measurement (G_PI)
- e_k represents unmeasured disturbances (e.g. meals)
    - this is not modelled explicitly, propagated or fed into the MPC

• The MPC uses a LINEARISED Sorensen model (A,B,C)
• The PLANT is the NONLINEAR Sorensen model
• Meals/disturbances act ONLY on the nonlinear plant
• Disturbances are unmeasured and NOT included in MPC predictions
• Feedback occurs via peripheral glucose measurement (G_PI)

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import SORENSEN_MODEL_TPB as sorensen

# Physical scaling factors
INSULIN_SCALE = 1e5     # maps MPC deviation -> mU/min
GLUCAGON_SCALE = 0.5    # maps MPC deviation -> mg/min
# Basal hormone levels
BASAL_GLUCAGON = 0.34   # mg/min
INSULIN_ON_THRESHOLD = 1e-3  # mU/min (numerical zero)


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

# Nonlinear Sorensen plant
from SORENSEN_MODEL_TPB import (
    sorensen_odes,
    initial_conditions,
    STATE_ORDER
)

# Helper: simulate nonlinear Sorensen for one MPC step
def sorensen_step(x, u, dt, meal=0.0, n_substeps=20):

    uI, uG_mg = u
    uG = uG_mg * 1e6  # convert mg/min → pg/min

    x0 = x.copy()

    # Apply meal to G_PI (state index 7)
    x0[7] += meal

    t_eval = np.linspace(0.0, dt, n_substeps)

    sol = solve_ivp(
        lambda t, y: sorensen_odes(t, y, uI, uG),
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

    A_d, B_d = zoh_discretise(A, B, TAU_S_MIN)

    # MPC
    mpc = MPCController(A_d, B_d, C, tau_s_min=TAU_S_MIN, Np=NP, Nc=NC)

    # Initial nonlinear plant state
    x_true = np.array([initial_conditions[k] for k in STATE_ORDER], dtype=float)

    # Deviation coordinates for MPC
    x_dev = np.zeros(A_d.shape[0])
    u_prev_dev = np.zeros(2)

    sim_steps = int(sim_duration_min / TAU_S_MIN)
    t_glucose = []
    glucose = []    

    t_mpc = []
    insulin = []
    glucagon_inf = []

    # Random unannounced meal
    rng = np.random.default_rng(7)

    for k in range(sim_steps):

        # Meal disturbance
        meal = 0.0
        current_time = k * TAU_S_MIN
        if current_time >= 250:
            if rng.random() < 0.002:
                meal = rng.uniform(30, 50)  # mg/min
            else:
                meal = 0.0

        # Measurement
        G_PI = x_true[7]
        
        t_mpc.append(k * TAU_S_MIN)
        t_glucose.append(k * TAU_S_MIN) 
        glucose.append(G_PI)

        # --- MPC state correction (paper assumption) ---
        y_dev_meas = G_PI - R_SETPOINT

        y_model = (C @ x_dev)[0]
        e = (G_PI - R_SETPOINT) - y_model
        x_dev = x_dev + 0.05 * C.T.flatten() * e


        # Supervisory switching
        if G_PI > THRESH_HIGH:
            mode = "BOLUS"
            lambda_mode = np.diag([1e-5, 1e-4])
        elif G_PI < THRESH_LOW:
            mode = "GLUCAGON"
            lambda_mode = np.diag([1e-4, 1e-5])
        else:
            mode = "BASAL"
            lambda_mode = np.diag([1e-3, 1e-3])

        # MPC
        u_dev, _ = mpc.compute_control(
            xk=x_dev,
            uk_prev=u_prev_dev,
            r_dev=0.0,
            y_min_dev=Y_MIN - R_SETPOINT,
            y_max_dev=Y_MAX - R_SETPOINT,
            mode=mode,
            lambda_u=lambda_mode
        )

        # Insulin: MPC-controlled, scaled to physical units
        uI = INSULIN_SCALE * u_dev[0]   # mU/min

        # Glucagon: basal unless insulin is active
        if uI > INSULIN_ON_THRESHOLD:
            uG = 0.0
        else:
            uG = BASAL_GLUCAGON          # mg/min

        # Combined control input (NO deviation glucagon)
        u = np.array([uI, uG]) + u_star

        # Store absolute values for plotting
        insulin.append(uI)
        glucagon_inf.append(uG)


        # Nonlinear plant update
        x_true, G_trace = sorensen_step(x_true, u, TAU_S_MIN, meal=meal, n_substeps=20)

        dt_sub = TAU_S_MIN / (len(G_trace)-1)
        # Append internal glucose trajectory for smooth plot
        for i, g in enumerate(G_trace[1:]):
            t_glucose.append(k * TAU_S_MIN + (i+1) * (TAU_S_MIN / len(G_trace)))
            glucose.append(g)

        # Linear model update (for MPC)
        x_dev = A_d @ x_dev + B_d @ u_dev
        u_prev_dev = u_dev.copy()

    return np.array(t_glucose), np.array(glucose), np.array(t_mpc),np.array(insulin), np.array(glucagon_inf)



if __name__ == "__main__":

    t_g, G, t_mpc, I, Gg = run_closed_loop()

    plt.figure(figsize=(12, 9))

    # ---- Glucose (continuous) ----
    plt.subplot(3, 1, 1)
    plt.plot(t_g, G, label="Glucose (G_PI)")
    plt.axhline(90, linestyle="--", color="k", label="Setpoint 90")
    plt.ylabel("Glucose (mg/dL)")
    plt.grid(True)
    plt.legend()

    # ---- Insulin (MPC, ZOH) ----
    plt.subplot(3, 1, 2)
    plt.step(t_mpc, I, where="post", label="Insulin infusion")
    plt.ylabel("Insulin (mU/min)")
    plt.grid(True)
    plt.legend()

    # ---- Glucagon (MPC, ZOH) ----
    plt.subplot(3, 1, 3)
    plt.step(t_mpc, Gg, where="post", label="Glucagon infusion")
    plt.ylabel("Glucagon (mg/min)")
    plt.xlabel("Time (min)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

