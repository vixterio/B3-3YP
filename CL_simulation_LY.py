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
def sorensen_step(x, u, dt, meal=0.0):

    # Apply meal disturbance to glucose compartment (G_PI index = 7)
    x0 = x.copy()
    x0[7] += meal

    uI, uG = u
    
    # Inject MPC control into state
    x0[8] += uI * dt # I_PV index = 8
    x0[18] += uG * dt # Gamma index = 18

    sol = solve_ivp(
        sorensen_odes,
        (0.0, dt),
        x0,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )

    return sol.y[:, -1]

    """
    Advance nonlinear Sorensen model by dt minutes.
    Meal enters as glucose disturbance (mg/min).
    """

    def rhs(t, y):
        uI, uG = u
        # Meal added to glucose appearance (G_PI)
        y = y.copy()
        y[7] += meal
        return sorensen_odes(t, y, uI, uG)

    sol = solve_ivp(
        rhs,
        (0, dt),
        x,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )

    return sol.y[:, -1]


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
    t = np.arange(sim_steps) * TAU_S_MIN

    glucose = np.zeros(sim_steps)
    insulin = np.zeros(sim_steps)
    glucagon = np.zeros(sim_steps)

    # Random unannounced meal
    rng = np.random.default_rng(7)

    for k in range(sim_steps):

        # Meal disturbance
        meal = 0.0
        if rng.random() < 0.02:
            meal = rng.uniform(20, 60)  # mg/min

        # Measurement
        G_PI = x_true[7]
        glucose[k] = G_PI

        # --- MPC state correction (paper assumption) ---
        y_dev_meas = G_PI - R_SETPOINT
        x_dev = (C.T / (C @ C.T))[:, 0] * y_dev_meas


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

        u = u_dev + u_star
        insulin[k], glucagon[k] = u

        # Nonlinear plant update
        x_true = sorensen_step(x_true, u, TAU_S_MIN, meal=meal)

        # Linear model update (for MPC)
        x_dev = A_d @ x_dev + B_d @ u_dev
        u_prev_dev = u_dev.copy()

    return t, glucose, insulin, glucagon



if __name__ == "__main__":

    t, G, I, Gg = run_closed_loop()

    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(t, G)
    plt.axhline(90, linestyle="--", color="k")
    plt.ylabel("Glucose (mg/dL)")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.step(t, I, where="post")
    plt.ylabel("Insulin (mU/min)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.step(t, Gg, where="post")
    plt.ylabel("Glucagon (mg/min)")
    plt.xlabel("Time (min)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
