"""
CL_test_simulation.py

Closed-loop simulation of a dual-hormone Model Predictive Controller
designed to reproduce the results presented of the
referenced dual-hormone MPC paper.

This script implements the control architecture and assumptions used
in the original publication, prioritising qualitative agreement with
the reported results rather than physiological realism.

System formulation:

    Linearised prediction model used by MPC:
        x_{k+1} = A x_k + B u_k
        y_k     = C x_k + e_k

    Nonlinear plant:
        Sorensen glucose–insulin–glucagon model

where:
- x is the nonlinear Sorensen state
- u = [u1, u2] are insulin and glucagon infusion rates
- y is the peripheral glucose measurement (G_PI)
- e_k represents an abstract, unmeasured disturbance acting on glucose

Key assumptions in this simulation:
- The Sorensen model is linearised around a single nominal operating
  point corresponding to basal glucose regulation (~90 mg/dL).
- Meal effects are represented as abstract disturbances or output
  bias terms, rather than explicit physiological absorption dynamics.
- The disturbance is structured such that it can be compensated by
  the linear MPC within its prediction horizon.
- Feedback to the MPC is provided only via peripheral glucose
  measurements.
- Supervisory logic enforces mutually exclusive insulin and glucagon
  actuation, consistent with the paper’s flowchart.

Purpose of this script:
- To reproduce the qualitative behaviour shown in Figure 13 of the
  paper, including bounded glucose excursions and hormone infusion
  patterns.
- To serve as a reference implementation of the published controller
  under its stated assumptions.
- To provide a baseline against which more realistic simulations and
  controller extensions can be compared.

This script is intended as a faithful reproduction of the paper’s
theoretical setup and does not aim to model real postprandial
glucose dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import SORENSEN_MODEL_TPB as sorensen

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

from SORENSEN_MODEL_TPB import (
    sorensen_odes,
    initial_conditions,
    STATE_ORDER
)


# Nonlinear Sorensen plant step 
def sorensen_step(x, u, dt, glucose_disturbance=0.0, n_substeps=25):

    uI_mU, uG_mg = u
    uI = uI_mU * 1e3
    uG = uG_mg * 1e9

    def wrapped_odes(t, y):
        dydt = sorensen_odes(t, y, uI, uG)
        # Inject abstract disturbance directly into plasma glucose
        dydt[0] += glucose_disturbance / dt
        return dydt

    t_eval = np.linspace(0.0, dt, n_substeps)
    sol = solve_ivp(
        wrapped_odes,
        (0.0, dt),
        x,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )

    return sol.y[:, -1], sol.y[7, :]


def run_closed_loop(sim_duration_min=2000):

    # Load linear model
    model = load_continuous_linear_model()
    A, B, C = model["A"], model["B"], model["C"]
    x_star, u_star = model["x_star"], model["u_star"]

    y_star = float((C @ x_star)[0])

    # Discretise
    A_d, B_d = zoh_discretise(A, B, TAU_S_MIN)

    # MPC
    mpc = MPCController(A_d, B_d, C, tau_s_min=TAU_S_MIN, Np=NP, Nc=NC)

    # Initial nonlinear state
    x_true = np.array([initial_conditions[k] for k in STATE_ORDER], dtype=float)

    # MPC deviation state
    x_dev = x_true - x_star
    u_prev_dev = np.zeros(2)

    # Offset-free disturbance estimate (initialised once)
    d_hat = 0.0

    # Disturbance observer gain (tune 0.1–0.3)
    alpha_d = 0.2

    sim_steps = int(sim_duration_min / TAU_S_MIN)

    t_glucose, glucose = [], []
    t_mpc, insulin, glucagon = [], [], []

    for k in range(sim_steps):

        current_time = k * TAU_S_MIN

        
        # Abstract disturbance 
        glucose_disturbance = 0.0
        if 250.0 <= current_time <= 350.0:
            glucose_disturbance = 40.0  # mg/dL equivalent disturbance

        # Measurement
        G_PI = x_true[7]

        t_mpc.append(current_time)
        t_glucose.append(current_time)
        glucose.append(G_PI)

        # Supervisory logic
        if G_PI > THRESH_HIGH:
            mode = "BOLUS"
        elif G_PI < THRESH_LOW:
            mode = "GLUCAGON"
        else:
            mode = "BASAL"

        # Output bias estimation
        y_dev_meas  = G_PI - y_star
        y_dev_model = float((C @ x_dev)[0])

        
        innovation = y_dev_meas - (y_dev_model + d_hat)

        # Disturbacne estimation 
        d_hat = d_hat + alpha_d * innovation

        # Augemented state passed to MPC
        x_aug = np.concatenate([x_dev, [d_hat]])

        # Ensure disabled actuator has zero previous input
        u_prev_for_mpc = u_prev_dev.copy()
        if mode in ["BOLUS", "BASAL"]:
            u_prev_for_mpc[1] = 0.0
        elif mode == "GLUCAGON":
            u_prev_for_mpc[0] = 0.0

        # MPC solve
        u_dev, info = mpc.compute_control(
            xk=x_aug,
            uk_prev=u_prev_for_mpc,
            r_dev=R_SETPOINT - y_star,
            y_min_dev=Y_MIN - y_star,
            y_max_dev=Y_MAX - y_star,
            mode=mode
        )

        # Apply equilibrium baseline
        u = u_dev + u_star

        insulin.append(u[0])
        glucagon.append(u[1])

        # Nonlinear plant step
        x_true, G_trace = sorensen_step(
            x_true, u, TAU_S_MIN,
            glucose_disturbance=glucose_disturbance
        )

        # Smooth glucose plot
        for i, g in enumerate(G_trace[1:], start=1):
            t_glucose.append(current_time + i * (TAU_S_MIN / (len(G_trace) - 1)))
            glucose.append(g)

        # Linear predictor update
        u_applied_dev = u - u_star
        x_dev = A_d @ x_dev + B_d @ u_applied_dev
        u_prev_dev = u_applied_dev.copy()

    return (
        np.array(t_glucose),
        np.array(glucose),
        np.array(t_mpc),
        np.array(insulin),
        np.array(glucagon)
    )


if __name__ == "__main__":

    t_g, G, t_mpc, I, Gg = run_closed_loop()

    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(t_g, G, label="Glucose (G_PI)")
    plt.axhline(90, linestyle="--", color="k", label="Setpoint")
    plt.ylabel("Glucose (mg/dL)")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.step(t_mpc, I, where="post", label="Insulin")
    plt.ylabel("Insulin (mU/min)")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.step(t_mpc, 1e3* Gg, where="post", label="Glucagon")
    plt.ylabel("Glucagon (mg/min)")
    plt.xlabel("Time (min)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
