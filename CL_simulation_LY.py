"""
CL_simulation_LY.py

Closed-loop simulation of a dual-hormone MPC controller applied to the
nonlinear Sorensen glucose–insulin–glucagon model under physiologically
realistic disturbances.

This script evaluates the performance and limitations of a fixed linear
Model Predictive Controller augmented with an output disturbance model when subjected to unannounced meals modelled
using explicit glucose absorption dynamics.

System formulation:

    Nonlinear plant (Sorensen model):
        ẋ = f(x, u, d)

    Linear prediction model used by MPC:
        x_{k+1} = A x_k + B u_k
        d_{k+1} = d_k
        y_k     = C x_k + d_k

where:
- x represents the nonlinear Sorensen physiological state
- u = [u1, u2] are insulin and glucagon infusion rates
- y is the peripheral glucose measurement (G_PI)
- d represents an unmeasured, persistent output disturbance accounting for
  modelling errors, unannounced meals, and physiological mismatch

Key characteristics of this simulation:
- Meals are modelled explicitly using a gut glucose reservoir with
  first-order absorption dynamics, producing delayed and nonlinear
  glucose appearance in the nonlinear plant.
- The MPC relies on a single linearisation of the Sorensen model
  around a nominal basal operating point (approximately 90 mg/dL).
- Unannounced disturbances are not modelled explicitly in the MPC
  prediction model, but are compensated using a disturbance-augmented
  (offset-free) formulation with an online disturbance estimate.
- Feedback to the controller is provided solely via peripheral glucose
  measurements (G_PI).
- Supervisory logic switches between insulin-only, basal, and glucagon-only
  control modes based on glucose thresholds.



"""

# DISTURBANCE SELECTION
DISTURBANCE_CASE = 1
# 1: abstract disturbance. Direct perturbation of glucose that does not obey physiology.
#    idealised, instantaneous meal, easy for a controller to counteract.
# 2: realistic physiological meal (50g)
#    delayed, persistent, nonlinear
#    glucose enters the body gradually through the gut with delays and nonlinear absorption

DISTURBANCE_CONFIG = {
    1: {
        "mode": "abstract",
        "abstract_disturbance": {
            "t_start": 250.0,
            "t_end": 350.0,
            "amplitude": 50.0,   # mg/dL equivalent
        },
        "meal_grams": None,
        "meal_events": [],
        "alpha_d": 0.08, # manually tuned output disturbance gain 
    },
    2: {
        "mode": "realistic",
        "meal_grams": 50.0,
        "meal_events": [250.0],
        "alpha_d": 0.25
    },
}


import math
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
BASAL_PULSE_RATE = 1.0      # mU/min during the pulse (tune slightly if needed)


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
def sorensen_step(x_aug, u, dt, DISTURBANCE_MODE, meal_mg=0.0, glucose_disturbance=0.0, n_substeps=50):
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
        if DISTURBANCE_MODE == "realistic":
            ys = y[:19] # Soresen states
            Dm = y[19] # meal resevoir state
        else:
            ys = y

        # PLant ODEs in Sorensen coordinates
        dydt_s = sorensen_odes(t, ys, uI, uG)

        if DISTURBANCE_MODE == "realistic":
            dDm = -K_ABS * Dm
            Ra = K_ABS * Dm
            dydt_s[0] += Ra / sorensen.Vg_PV
            return np.concatenate([dydt_s, [dDm]])
        else:
            dydt_s[0] += glucose_disturbance / dt
            return dydt_s


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

    # Discretise
    A_d, B_d = zoh_discretise(A, B, TAU_S_MIN)

    print("||B_d||:", np.linalg.norm(B_d))
    print("B_d (first 4 rows):\n", B_d[:4,:])
    print("Np, Nc:", NP, NC)


    # MPC
    mpc = MPCController(A_d, B_d, C, tau_s_min=TAU_S_MIN, Np=NP, Nc=NC)

    # Initial nonlinear plant state
    x_true_19 = np.array([initial_conditions[k] for k in STATE_ORDER], dtype=float)

    cfg = DISTURBANCE_CONFIG[DISTURBANCE_CASE]
    DISTURBANCE_MODE = cfg["mode"]

    if DISTURBANCE_MODE == "realistic":
        x_true = np.concatenate([x_true_19, [0.0]])  # D_meal = 0 mg
    else:
        x_true = x_true_19.copy()

    # Deviation coordinates for MPC
    x_dev = x_true[:len(x_star)] - x_star
    u_prev_dev = np.zeros(2)

    # Offset-free disturbance estimate (initialised once)
    d_hat = 0.0

    # Disturbance observer gain (MANUALLY TUNED, included in configuration)
    alpha_d = cfg["alpha_d"]

    sim_steps = int(sim_duration_min / TAU_S_MIN)
    t_glucose = []
    glucose = []    

    t_mpc = []
    insulin = []
    glucagon_inf = []

    ## --- Disturbance schedule ---
    ## --- Disturbance schedule ---
    for k in range(sim_steps):

        current_time = k * TAU_S_MIN

        meal_mg = 0.0
        glucose_disturbance = 0.0

        if DISTURBANCE_MODE == "realistic":
            for t_evt in cfg.get("meal_events", []):
                if abs(current_time - t_evt) < 1e-12:
                    meal_mg = cfg["meal_grams"] * 1000.0  # g -> mg
                    break

        elif DISTURBANCE_MODE == "abstract":
            dcfg = cfg["abstract_disturbance"]
            if dcfg["t_start"] <= current_time <= dcfg["t_end"]:
                glucose_disturbance = dcfg["amplitude"]

        # Measurement
        G_PI = x_true[7]
        
        t_mpc.append(current_time)
        t_glucose.append(current_time) 
        glucose.append(G_PI)

        # Supervisory switching MPC
        if G_PI > THRESH_HIGH:
            mode = "BOLUS"
        elif G_PI < THRESH_LOW:
            mode = "GLUCAGON"
        else:
            mode = "BASAL"

        # MPC state correction
        y_dev_meas  = G_PI - y_star
        y_dev_model = float((C @ x_dev)[0])

        innovation = y_dev_meas - (y_dev_model + d_hat)

        # Disturbance estimation 
        d_hat = d_hat + alpha_d * innovation

        # Augemented state passed to MPC
        x_aug = np.concatenate([x_dev, [d_hat]])

        u_prev_for_mpc = u_prev_dev.copy()
    
        if mode in ["BOLUS", "BASAL"]:
            u_prev_for_mpc[1] = 0.0  # glucagon is disabled in these modes
        elif mode == "GLUCAGON":
            u_prev_for_mpc[0] = 0.0  # insulin is disabled in this mode


        # MPC
        u_dev, info = mpc.compute_control(
            xk=x_aug,
            uk_prev=u_prev_for_mpc,
            r_dev=R_SETPOINT - y_star,
            y_min_dev=Y_MIN - y_star,
            y_max_dev=Y_MAX - y_star,
            mode=mode,
            lambda_u=None
        )

        if abs(current_time - 250.0) < 50:   # a small window around meal
            print(current_time, mode, "G_PI", G_PI, "u_dev", u_dev, "status", info.get("qp_status"))


        if mode in ["BOLUS", "BASAL"]:
            u_star_mode = np.array([u_star[0], 0.0], dtype=float)  # no glucagon baseline
        elif mode == "GLUCAGON":
            u_star_mode = np.array([0.0, BASAL_GLUCAGON], dtype=float)  # no insulin baseline

        u_star_mode = np.array([u_star[0], u_star[1]], dtype=float)  # insulin baseline is zero, glucagon baseline is constant

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
        x_true, G_trace = sorensen_step(
            x_true,
            u,
            TAU_S_MIN,
            DISTURBANCE_MODE=DISTURBANCE_MODE,
            meal_mg=meal_mg,
            glucose_disturbance=glucose_disturbance,
            n_substeps=20)

        # Append internal glucose trajectory for smooth plot
        for i, g in enumerate(G_trace[1:], start=1):
            t_glucose.append(current_time + i * (TAU_S_MIN / (len(G_trace) - 1)))
            glucose.append(g)

        # Linear model update (for MPC)
        u_applied_dev = u - u_star  # deviation from mode-specific baseline
        x_dev = A_d @ x_dev + B_d @ u_applied_dev

        # Store previous deviation input for next MPC iteration
        u_prev_dev = u_applied_dev.copy()

    

    # --- evaluate performance (exclude first 200 minutes) --- Tommy
    
    from PERFORMANCE_METRICS_TPB import evaluate_glucose_trace, compute_metrics

    # --- prepare arrays collected during the loop (your existing names) ---
    glucose_array = np.asarray(glucose, dtype=float)    # full trace (for plotting / return)
    t_g_array     = np.asarray(t_glucose, dtype=float)  # time vector collected in sim

    
        # --- infer sampling period robustly from the time vector ---
    if t_g_array.size >= 2:
        dt_vals = np.diff(t_g_array)
        median_dt = float(np.median(dt_vals))

        # Robust unit test:
        # - If the trace runs to large times (e.g. > 500) we assume t_g_array is already in minutes.
        # - Otherwise fall back to the previous heuristic (if median_dt very large, treat as minutes).
        if np.nanmax(t_g_array) > 500.0:
            dt_minutes = median_dt
            time_unit = "minutes (from t_g_array max > 500)"
        else:
            # If median_dt is large ( > 10 ) treat it as minutes; otherwise treat it as seconds and convert.
            if median_dt > 10.0:
                dt_minutes = median_dt
                time_unit = "minutes (median_dt > 10)"
            else:
                dt_minutes = median_dt / 60.0
                time_unit = "seconds (converted to minutes)"
    else:
        # fallback: try to use TAU_S_MIN if defined, else default to 1.0 minute
        try:
            dt_minutes = float(TAU_S_MIN)
            time_unit = "fallback TAU_S_MIN (minutes)"
        except NameError:
            dt_minutes = 1.0
            time_unit = "fallback default (1.0 minute)"

    # safety: do not let dt_minutes be zero or NaN
    if not np.isfinite(dt_minutes) or dt_minutes <= 0:
        dt_minutes = float(TAU_S_MIN) if 'TAU_S_MIN' in globals() else 1.0
        time_unit += " [corrected fallback]"

    print(f"DEBUG: inferred sampling interval = {dt_minutes:.6f} minutes  ({time_unit})")
    print(f"DEBUG: total samples = {glucose_array.size} -> total duration = {glucose_array.size * dt_minutes:.1f} minutes")

    # --- compute n_skip to remove first 200 minutes ---
    n_skip = int(math.floor(200.0 / dt_minutes))
    # clamp n_skip so we always keep at least 2 samples for evaluation
    n_skip = max(0, min(n_skip, max(0, glucose_array.size - 2)))

    print(f"Ignoring first {n_skip} samples (~{n_skip * dt_minutes:.1f} minutes) for evaluation")

    glucose_trimmed = glucose_array[n_skip:]
    # double-check durations
    print(f"Trimmed samples: {glucose_trimmed.size}, Trimmed duration (min): {glucose_trimmed.size * dt_minutes:.1f}")



    # --- call evaluator with the correct dt_minutes ---
    res = evaluate_glucose_trace(glucose_trimmed, dt_minutes=dt_minutes)

    print("Final composite J (trimmed):", res['J'])
    
    # return numeric arrays (safer for later scaling / plotting)
    return t_g_array, glucose_array, np.asarray(t_mpc, dtype=float), np.asarray(insulin, dtype=float), np.asarray(glucagon_inf, dtype=float)



if __name__ == "__main__":

    t_g, G, t_mpc, I, Gg = run_closed_loop()

    plt.figure(figsize=(12, 9))

    # # Glucose (continuous)
    # plt.subplot(3, 1, 1)
    # plt.plot(t_g, G, label="Glucose (G_PI)")
    # plt.axhline(90, linestyle="--", color="k", label="Setpoint 90")
    # plt.axhline(70, linestyle=":", color="r", label="Hypo threshold 70")
    # plt.ylabel("Glucose (mg/dL)")
    # plt.grid(True)
    # plt.legend()

    # Glucose (continuous)
    plt.subplot(3, 1, 1)
    cutoff = 200  # minutes
    # Masks
    mask_initial = t_g <= cutoff
    mask_rest = t_g > cutoff
    # First 200 min → dashed
    plt.plot(t_g[mask_initial],
            G[mask_initial],
            linestyle='--',
            color = 'blue',
            label="Glucose (first 200 min)")
    # After 200 min → solid
    plt.plot(t_g[mask_rest],
            G[mask_rest],
            linestyle='-',
            color = 'blue',
            label="Glucose (G_PI)")
    plt.axhline(90, linestyle=":", color="k", label="Setpoint 90")
    plt.axhline(70, linestyle=":", color="r", label="Hypo threshold 70")
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
    #plt.step(t_mpc, 1e3*Gg, where="post", label="Glucagon infusion")
    # convert to numpy and scale to desired units for plotting
    plt.step(t_mpc, 1e3 * np.asarray(Gg, dtype=float), where="post", label="Glucagon infusion (x1e3)")
    plt.ylabel("Glucagon (mg/min)")
    plt.xlabel("Time (min)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

