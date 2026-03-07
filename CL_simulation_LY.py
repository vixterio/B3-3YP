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

This script can be run with 3 different disturbance configurations (set by DISTURBANCE_CASE):
1) Multi-meal day with per-meal absorption (physiological, Sorensen + gut reservoir).
2) Exercise disturbance implemented ONLY in this simulation file:
   - glucose sink term (extra uptake) + effective insulin scaling (increased sensitivity).
3) Limit case: large slow meal + exercise + temporary insulin delivery cap (fault/saturation).
4) Single meal test to observe steady state response 
"""

# DISTURBANCE SELECTION
DISTURBANCE_CASE = 3
# 1: realistic multi-meal day (Sorensen + gut reservoir with per-meal Kabs)
# 2: exercise disturbance (implemented in simulation file only)
# 3: near-breaking compound: huge slow meal + exercise + insulin cap fault
# 4: single meal test to observe steady state

# Disturbance configurations and specifications for each case
# Case 1 (multi-meal day):
#  - breakfast 45g fast absorption (tau ~ 30 min)
#  - lunch     80g medium absorption (tau ~ 55 min)
#  - snack     20g fast absorption (tau ~ 25 min)
#  - dinner    70g slow absorption (tau ~ 80 min; high fat)

# Case 2 (exercise):
#  - moderate exercise 45 min (e.g. brisk walk / bike) starting at t=600 min
#  - extra uptake sink ~ 40 mg/min ramping up/down
#  - increased insulin sensitivity:
#       +30% during exercise,
#       then decays with ~120 min time constant (post-exercise effect)

# Case 3 (limit case):
#  - huge dinner 140g with very slow absorption (tau ~ 95 min)
#  - exercise 60 min starting 60 min after dinner
#  - insulin delivery cap fault: insulin max reduced to 0.6 mU/min for 90 min
#    (temporary occlusion / max-delivery limitation)

# Case 4 (single meal): 
#  - single 60g meal with moderate absorption (tau ~ 45 min) at t=250 min


DISTURBANCE_CONFIG = {
    1: {
        "mode": "realistic_meals",
        "meal_schedule": [
            {"t": 250.0,  "grams": 45.0,  "tau_abs_min": 30.0},  # breakfast
            {"t": 700.0,  "grams": 80.0,  "tau_abs_min": 55.0},  # lunch
            {"t": 1050.0, "grams": 20.0,  "tau_abs_min": 25.0},  # snack
            {"t": 1350.0, "grams": 70.0,  "tau_abs_min": 80.0},  # dinner (slow)
        ],
        "exercise_schedule": [],
        "fault_schedule": [],
        "alpha_d": 0.20,
    },

    2: {
        "mode": "exercise_only",
        "meal_schedule": [
            {"t":450.0, "grams":40.0, "tau_abs_min":55.0},  # single moderate meal to fuel exercise (40g, moderate absorption)
        ],
        "exercise_schedule": [
            {
                "t_start": 600.0,
                "duration_min": 45.0,
                # glucose sink parameters (mg/min)
                "sink_mg_per_min_peak": 40.0,
                "ramp_min": 8.0,
                # insulin sensitivity multiplier
                "sens_during": 0.30,         # +30% during exercise
                "sens_post": 0.20,           # +20% immediately after, decays
                "sens_post_tau_min": 120.0,  # decay time constant
            }
        ],
        "fault_schedule": [],
        "alpha_d": 0.18,
    },

    3: {
        "mode": "limit_compound",
        "meal_schedule": [
            {"t": 600.0, "grams": 140.0, "tau_abs_min": 95.0},  # huge slow meal
        ],
        "exercise_schedule": [
            {
                "t_start": 660.0,            # 60 min after meal starts
                "duration_min": 60.0,
                "sink_mg_per_min_peak": 65.0,
                "ramp_min": 10.0,
                "sens_during": 0.40,
                "sens_post": 0.30,
                "sens_post_tau_min": 180.0,
            }
        ],
        "fault_schedule": [
            {
                "t_start": 610.0,
                "t_end": 700.0,
                "insulin_u1_max_mU_per_min": 0.6,  # hard cap on insulin during fault window
            }
        ],
        "alpha_d": 0.22,
    },

    4: {
    "mode": "realistic_meals",
    "meal_schedule": [
        {"t": 250.0, "grams": 60.0, "tau_abs_min": 45.0},  # single 60g meal, moderate absorption
    ],
    "exercise_schedule": [],
    "fault_schedule": [],
    "alpha_d": 0.18,
    },
}


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import SORENSEN_MODEL_TPB as sorensen


# Basal hormone levels
BASAL_INSULIN = 0.0

# Basal insulin as periodic micro-bolus pulses
ENABLE_BASAL_PULSES = True
BASAL_PERIOD_MIN = 200.0   
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

# Simulation exercise helpers
def trapezoid_profile(t, t_start, t_end, ramp): # for exercise disturbance ramps
    """
    0 -> 1 -> 0 trapezoid on [t_start, t_end], with linear ramps of length 'ramp'.
    """
    if t <= t_start or t >= t_end:
        return 0.0
    if ramp <= 1e-9:
        return 1.0
    if t < t_start + ramp:
        return (t - t_start) / ramp
    if t > t_end - ramp:
        return (t_end - t) / ramp
    return 1.0

def exercise_effects(current_time, cfg): # computes the exercise disturbance effects (glucose sink and sensitivity increase) at the current time based on the exercise schedule in the configuration
    """
    Returns:
      sink_mg_per_min (>=0): extra glucose uptake sink (mg/min)
      sens_scale (>=0): fractional insulin sensitivity increase (e.g., 0.3 means +30%)
    """
    sink = 0.0
    sens = 0.0

    for ex in cfg.get("exercise_schedule", []):
        t0 = float(ex["t_start"])
        dur = float(ex["duration_min"])
        t1 = t0 + dur

        ramp = float(ex.get("ramp_min", 8.0))
        p = trapezoid_profile(current_time, t0, t1, ramp)

        # during exercise
        sink += p * float(ex["sink_mg_per_min_peak"])
        sens += p * float(ex["sens_during"])

        # post-exercise sensitivity tail (exponential decay)
        if current_time > t1:
            sens_post0 = float(ex.get("sens_post", 0.0))
            tau = float(ex.get("sens_post_tau_min", 120.0))
            if tau > 1e-9:
                sens += sens_post0 * math.exp(-(current_time - t1) / tau)

    # clip to sane values
    sink = max(0.0, sink)
    sens = max(0.0, sens)
    return sink, sens

def insulin_fault_cap(current_time, cfg): # checks if there is an active insulin delivery cap fault at the current time based on the fault schedule in the configuration
    """
    Returns insulin cap (mU/min) or None if no cap active.
    """
    for f in cfg.get("fault_schedule", []):
        if float(f["t_start"]) <= current_time <= float(f["t_end"]):
            return float(f["insulin_u1_max_mU_per_min"])
    return None

# # Meal absorption model
# K_ABS = 1.0 / 40.0  # 1/min (time constant ~40 min)

# Simulate nonlinear Sorensen for one MPC step
def sorensen_step(
    x_aug,
    u_abs,
    dt,
    mode,
    meal_mg=0.0,
    k_abs=1.0/40.0,
    exercise_sink_mg_per_min=0.0,
    insulin_sens_scale=0.0,
    n_substeps=50
):
    """
    x_aug includes the 19 Sorensen states plus one extra state at the end:
      D_meal [mg] = glucose in a gut reservoir that appears into plasma with 1st-order kinetics.

    meal_mg is the bolus amount added to D_meal at the start of this step.

    u_abs: [uI_mU_per_min, uG_mg_per_min] absolute pump commands

    exercise implemented here
      - extra sink (mg/min) removed from glucose (approx)
      - effective insulin scaling (uI multiplied by (1 + insulin_sens_scale))
    """
    
    uI_mU, uG_mg = float(u_abs[0]), float(u_abs[1])

    # effective insulin scaling to mimic increased sensitivity
    uI_mU_eff = uI_mU * (1.0 + float(insulin_sens_scale))

    # Conver to Sorensen internal units
    uI = uI_mU_eff * 1e3
    uG = uG_mg * 1e9

    # Add meal bolus to resevoir (mg)
    x0 = x_aug.copy()

    # add meal bolus to reservoir
    if mode in ["realistic_meals", "limit_compound"]:
        x0[-1] += float(meal_mg)

    def wrapped_odes(t, y):
        if mode in ["realistic_meals", "limit_compound"]:
            ys = y[:19]
            Dm = y[19]
        else:
            ys = y

        dydt_s = sorensen_odes(t, ys, uI, uG)

        # physiological meal appearance into plasma (G_PV index 0)
        if mode in ["realistic_meals", "limit_compound"]:
            k = float(k_abs)
            dDm = -k * Dm
            Ra = k * Dm  # mg/min
            dydt_s[0] += Ra / sorensen.Vg_PV
        else:
            dDm = None
    
        # exercise extra uptake sink (approx: subtract from plasma and interstitial)
        # NOTE: this is a modeling approximation since we do not edit Sorensen fluxes.
        sink = float(exercise_sink_mg_per_min)
        if sink > 0.0:
            # split sink between plasma and peripheral interstitial (simple, stable)
            dydt_s[0] -= 0.5 * sink / sorensen.Vg_PV
            dydt_s[7] -= 0.5 * sink / sorensen.Vg_PI

        if mode in ["realistic_meals", "limit_compound"]:
            return np.concatenate([dydt_s, [dDm]])
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

    # Return final state AND interstitial glucose trajectory (G_PI index 7)
    if mode in ["realistic_meals", "limit_compound"]:
        return sol.y[:, -1], sol.y[7, :]
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

    # print("||B_d||:", np.linalg.norm(B_d))
    # print("B_d (first 4 rows):\n", B_d[:4,:])
    # print("Np, Nc:", NP, NC)

    # MPC
    mpc = MPCController(A_d, B_d, C, tau_s_min=TAU_S_MIN, Np=NP, Nc=NC)

    # Initial nonlinear plant state
    x_true_19 = np.array([initial_conditions[k] for k in STATE_ORDER], dtype=float)

    cfg = DISTURBANCE_CONFIG[DISTURBANCE_CASE]
    plant_mode = cfg["mode"]

    # add meal reservoir state only for meal modes
    if plant_mode in ["realistic_meals", "limit_compound"]:
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
    t_glucose, glucose = [], []

    t_mpc, insulin, glucagon_inf = [], [], []

    # for meal Kabs scheduling
    k_abs_current = 1.0 / 40.0

    # Disturbance schedule
    for k in range(sim_steps):
        current_time = k * TAU_S_MIN

        # -------------------------
        # determine meal bolus and k_abs at this step
        # -------------------------
        meal_mg = 0.0
        if plant_mode in ["realistic_meals", "limit_compound"]:
            for evt in cfg.get("meal_schedule", []):
                if abs(current_time - float(evt["t"])) < 1e-12:
                    meal_mg = float(evt["grams"]) * 1000.0
                    k_abs_current = 1.0 / float(evt["tau_abs_min"])
                    break

        # determine exercise effects (simulation-only)
        exercise_sink, sens_scale = exercise_effects(current_time, cfg)

        # Measurement
        G_PI = x_true[7]
        
        t_mpc.append(current_time)
        t_glucose.append(current_time) 
        glucose.append(G_PI)

        # Supervisory switching MPC
        if G_PI > THRESH_HIGH:
            mpc_mode = "BOLUS"
        elif G_PI < THRESH_LOW:
            mpc_mode = "GLUCAGON"
        else:
            mpc_mode = "BASAL"

        # MPC state correction
        y_dev_meas  = G_PI - y_star
        y_dev_model = float((C @ x_dev)[0])

        innovation = y_dev_meas - (y_dev_model + d_hat)

        # Disturbance estimation 
        d_hat = d_hat + alpha_d * innovation

        # Augemented state passed to MPC
        x_aug = np.concatenate([x_dev, [d_hat]])

        u_prev_for_mpc = u_prev_dev.copy()
    
        # adding mode disabling for deviation input to MPC
        if mpc_mode in ["BOLUS", "BASAL"]:
            u_prev_for_mpc[1] = 0.0  # glucagon is disabled in these modes
        elif mpc_mode == "GLUCAGON":
            u_prev_for_mpc[0] = 0.0  # insulin is disabled in this mode


        # MPC
        u_dev, info = mpc.compute_control(
            xk=x_aug,
            uk_prev=u_prev_for_mpc,
            r_dev=R_SETPOINT - y_star,
            y_min_dev=Y_MIN - y_star,
            y_max_dev=Y_MAX - y_star,
            mode=mpc_mode,
            lambda_u=None
        )

        u_abs = u_dev + u_star

        if mpc_mode in ["BOLUS", "BASAL"]:
            u_abs[1] = 0.0  # no glucagon baseline
        elif mpc_mode == "GLUCAGON":
            u_abs[0] = 0.0  # no insulin baseline

        # Hard nonnegativity (physical pump constraint)
        u_abs[0] = max(0.0, u_abs[0])
        u_abs[1] = max(0.0, u_abs[1])
    
        # Add pump basal pulses on top of MPC command (independent of controller logic)
        if ENABLE_BASAL_PULSES:
            phase = (current_time % BASAL_PERIOD_MIN)
            if phase < BASAL_PULSE_WIDTH_MIN:
                u_abs[0] += BASAL_PULSE_RATE

        # case 3 fault: cap insulin delivery
        cap = insulin_fault_cap(current_time, cfg)
        if cap is not None:
            u_abs[0] = min(u_abs[0], cap)

        # Store absolute values for plotting
        insulin.append(u_abs[0])
        glucagon_inf.append(u_abs[1])


        # Nonlinear plant update
        x_true, G_trace = sorensen_step(
            x_true,
            u_abs,
            TAU_S_MIN,
            mode=plant_mode,
            meal_mg=meal_mg,
            exercise_sink_mg_per_min=exercise_sink,
            insulin_sens_scale=sens_scale,
            k_abs=k_abs_current,
            n_substeps=20)

        # Append internal glucose trajectory for smooth plot
        for i, g in enumerate(G_trace[1:], start=1):
            t_glucose.append(current_time + i * (TAU_S_MIN / (len(G_trace) - 1)))
            glucose.append(g)

        # Linear model update (for MPC)
        u_applied_dev = u_abs - u_star  # deviation from mode-specific baseline
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

    # return (
    #     np.asarray(t_glucose, dtype=float),
    #     np.asarray(glucose, dtype=float),
    #     np.asarray(t_mpc, dtype=float),
    #     np.asarray(insulin, dtype=float),
    #     np.asarray(glucagon_inf, dtype=float),
    # )



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
    plt.step(t_mpc, 1e3*Gg, where="post", label="Glucagon infusion")
    plt.ylabel("Glucagon (mg/min)")
    plt.xlabel("Time (min)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

