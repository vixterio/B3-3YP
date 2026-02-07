""" MPC_controller_LY.py

    Constrained linear MPC for the Sorensen linearised model.

    REQUIREMENT: 
    - The repository module LINEARISED_MODEL_TPB.py exports continuous-time linearisation:
        A : (n x n) continuous-time A matrix
        B : (n x m) continuous-time B matrix
        C : (1 x n) output matrix measuring glucose (row vector or 1D)
        tau_s : sampling time in MINUTES (scalar)
        x_star : equilibrium state (n,)
        u_star : equilibrium input (m,)
        
    The script will:
    - Discretise (A,B) with a ZOH using tau_s (minutes -> seconds conversion).
    - Build and solve a constrained MPC QP (cvxpy, OSQP).

    Specs enforced:
    - Output: peripheral glucose y(k) = C x(k) (assumed by provided C)
    - Manipulated vars: insulin (u1) and glucagon (u2)
    - Disturbance: random unannounced meal
    - Setpoint: 90 mg/dL
    - Sampling time: 5 min (we discretise using 5 min)
    - Prediction horizon Np = 25
    - Control horizon Nc = 15
    - Output constraints: 80 <= y <= 120 mg/dL
    - Insulin constraints: 0 <= u1 <= 80 (mU/min)
    - Glucagon constraints: 0 <= u2 <= 0.5 (mg/min)
    - Rate limits: |Δu1| <= 16.7, |Δu2| <= 0.1 per sample

    Flowchart logic (supervisory switching):
    - If BG > 120: "Bolus insulin" mode -> insulin-only MPC (glucagon forced to 0), aggressive tuning
    - If 80 <= BG <= 120: "Basal insulin" mode -> insulin-only MPC (glucagon forced to 0), insulin capped to a basal max
    - If BG < 80: "Glucagon infusion" mode -> glucagon-only MPC (insulin forced to 0)
"""

import sys
import importlib
import numpy as np
from scipy import signal
import cvxpy as cp


# Paper constants / specs
TAU_S_MIN = 5.0           # sampling time (min)
R_SETPOINT = 90.0         # mg/dL
NP = 25                   # prediction horizon
NC = 15                   # control horizon
Y_MIN = 80.0              # mg/dL
Y_MAX = 120.0             # mg/dL

# Input constraints
U1_MIN, U1_MAX = 0.0, 80.0     # insulin (mU/min)
U2_MIN, U2_MAX = 0.0, 0.5      # glucagon (mg/min)

# Rate limits (per sample)
DU1_MAX = 16.7
DU2_MAX = 0.1

# Flowchart thresholds (mg/dL)
THRESH_HIGH = 120.0
THRESH_LOW = 80.0

# Basal insulin cap
BASAL_U1_MAX = 10.0  # mU/min (tune if needed)

# Soft-constraint penalty for output bounds (prevents infeasible QP -> zero control)
SOFT_Y_PENALTY = 1e3



def load_continuous_linear_model():
    modname = "LINEARISED_MODEL_TPB"
    try: # fallback if not found
        mod = importlib.import_module(modname)
    except Exception as e:
        raise RuntimeError(f"Unable to import module '{modname}'. Make sure '{modname}.py' is available. Error: {e}") 

    A = getattr(mod, "A", None)
    B = getattr(mod, "B", None)
    C = getattr(mod, "C", None)
    tau_s = getattr(mod, "tau_s", None)

    # fallback if continuous model not found
    missing = []
    if A is None: missing.append("A")
    if B is None: missing.append("B")
    if C is None: missing.append("C")
    if tau_s is None: missing.append("tau_s")
    if missing:
        raise RuntimeError(f"Missing in {modname}: {missing}")

    # ensuring matrices are arrays, data types are as expected
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape((1, -1))
    tau_s = float(tau_s)
    x_star = getattr(mod, "x_star", None)
    u_star = getattr(mod, "u_star", None)

    if u_star is not None:
        # If you use u_star anywhere (baseline), convert it too:
        u_star = np.array(u_star, dtype=float)
        u_star[0] /= 1e3   # µU/min -> mU/min
        u_star[1] /= 1e9   # pg/min -> mg/min
    else: 
        u_star = np.zeros((2,), dtype=float)

    return {"A": A, "B": B, "C": C, "tau_s": tau_s, "x_star": x_star, "u_star": u_star, "source": modname}


def zoh_discretise(A, B, tau_s_min):
    """Discretise (A,B) with ZOH. tau_s_min in minutes -> convert to seconds.
        Also includes a unit conversion.
    """
    dt = float(tau_s_min) #* 60.0  # convert minutes to seconds for discretisation
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    B = B.copy()
    B[:, 0] *= 1e3
    B[:,1] *= 1e9

    C_dummy = np.zeros((1, A.shape[0])) # ZOH needs C,D but we only want A,B. Discretisation of A,B is independent of C,D.
    D_dummy = np.zeros((1, B.shape[1]))
    out = signal.cont2discrete((A, B, C_dummy, D_dummy), dt, method="zoh")
    A_d, B_d = np.asarray(out[0]), np.asarray(out[1])

    # Diagnostic print (temporary)
    print("[zoh_discretise] dt (min):", dt)
    print("[zoh_discretise] ||B_cont (scaled)||:", np.linalg.norm(B))
    print("[zoh_discretise] ||B_d||:", np.linalg.norm(B_d))
    print("[zoh_discretise] B_d[7,:] (G_PI row):", B_d[7, :])

    return A_d, B_d


class MPCController:
    """
    Standard linear MPC on an LTI model with input constraints, rate constraints, and
    SOFT output constraints (slack) to avoid infeasibility.

    Note: We enforce the flowchart by passing a "mode" that constrains which actuator(s)
    are allowed (insulin-only / glucagon-only) and by using basal cap in the basal zone.
    """
    # store in structure,
    def __init__(self, A_d, B_d, C, tau_s_min,
                 Np=NP, Nc=NC,
                 lambda_u=None):
        self.A = np.asarray(A_d, dtype=float)
        self.B = np.asarray(B_d, dtype=float)
        self.C = np.asarray(C, dtype=float).reshape((1, -1))
        self.tau_s = float(tau_s_min)
        self.Np = int(Np)           # normalise
        self.Nc = int(min(Nc, Np))  # enforce controller design constraint Nc <= Np
        self.n = self.A.shape[0]            # no of states
        self.m = self.B.shape[1]            # no of inputs
        if self.m != 2:
            raise ValueError("This controller expects m=2 (insulin, glucagon).")
        
        # damping factor in cost function
        if lambda_u is None:
            # arbitrary default
            lambda_u = np.diag([1e-4, 1e-4])
        self.lambda_u = np.asarray(lambda_u, dtype=float)
        if self.lambda_u.ndim == 1:
            self.lambda_u = np.diag(self.lambda_u)

        self._build_prediction_matrices()

    def _build_prediction_matrices(self):
        # Phi maps xk -> stacked future outputs (free response, zero future input)
        Phi_rows = []
        A_pow = np.eye(self.n)
        for _ in range(1, self.Np + 1):
            A_pow = A_pow @ self.A
            Phi_rows.append(self.C @ A_pow)
        self.Phi = np.vstack(Phi_rows)  # (Np x n), gives observability matrix ie CA, CA^2, ...

        # Build convolution matrix Gamma for piecewise-constant u over horizon
        H_list = []
        A_pow = np.eye(self.n)
        for _ in range(1, self.Np + 1):
            H_list.append((self.C @ A_pow @ self.B).reshape((1, self.m)))
            A_pow = A_pow @ self.A          # gives CB, CAB, CA^2B, ...

        Gamma_full = np.zeros((self.Np, self.m * self.Np))

        for row in range(self.Np):
            for col in range(row + 1):
                Hi = H_list[row - col]
                Gamma_full[row, col*self.m:(col+1)*self.m] = Hi     # creates discrete-time convolution of inputs with the system, Markov parameter
        self.Gamma_full = Gamma_full
        self.Gamma = self.Gamma_full[:, :self.m * self.Nc] # applys control horizon

    def compute_control(self, xk, uk_prev, r_dev,
                        y_min_dev, y_max_dev,
                        mode: str,
                        lambda_u=None,
                        y_bias_dev=0.0,
                        solver=cp.OSQP, verbose=False):
        
        # Shortcuts
        m = self.m
        n = self.n
        Nc = self.Nc
        Np = self.Np

        # Dimension checking inputs
        xk = np.asarray(xk).reshape((n,))
        uk_prev = np.asarray(uk_prev).reshape((m,))

        # Passing lambda_u
        if lambda_u is None:
            lambda_u = self.lambda_u
        else:
            lambda_u = np.asarray(lambda_u, dtype=float)
            if lambda_u.ndim == 1:
                lambda_u = np.diag(lambda_u)

        # Reference vector (deviation)
        r_vec = np.ones((Np, 1)) * float(r_dev)

        Y0 = (self.Phi @ xk).reshape((Np, 1)) # free response (no future input)
        bias_vec = np.ones((Np, 1)) * float(y_bias_dev) # bias term to correct model mismatch (if needed)
        deltaU = cp.Variable((m * Nc, 1)) # decision variable: stacked input increments 


        # Cumulative sum operator to convert deltaU -> u sequence
        T = np.zeros((m * Nc, m * Nc))
        for i in range(Nc):
            for j in range(i + 1):
                T[i*m:(i+1)*m, j*m:(j+1)*m] = np.eye(m)

        uk_prev_repeat = np.tile(uk_prev, Nc).reshape((-1, 1))
        u_seq = uk_prev_repeat + T @ deltaU

        Gamma_T = self.Gamma @ T
        Gamma_u0 = self.Gamma @ uk_prev_repeat
        Y_pred = Y0 + bias_vec + Gamma_u0 + Gamma_T @ deltaU

        # Soft output constraint slack
        # Allow violation of output constraints via slack variables (no infeasibility)
        s = cp.Variable((Np, 1), nonneg=True)

        # Objective
        Lambda_block = np.kron(np.eye(Nc), lambda_u)
        obj = (
            cp.sum_squares(Y_pred - r_vec)
            + cp.quad_form(deltaU, Lambda_block)
            + SOFT_Y_PENALTY * cp.sum(s)
        )

        constraints = []

        # Rate bounds (Δu) for actuator limits
        dumin = np.array([-DU1_MAX, -DU2_MAX], dtype=float)
        dumax = np.array([ DU1_MAX,  DU2_MAX], dtype=float)

        # If an actuator is OFF, don't constrain its rate (otherwise mode switches can be infeasible)
        if mode in ["BOLUS", "BASAL"]:
            dumin[1] = -1e6
            dumax[1] =  1e6
        elif mode == "GLUCAGON":
            dumin[0] = -1e6
            dumax[0] =  1e6

        for i in range(Nc):
            du_i = deltaU[i*m:(i+1)*m, 0]
            constraints += [du_i >= dumin, du_i <= dumax]

        # Input bounds (u)
        umin = np.array([U1_MIN, U2_MIN], dtype=float)
        umax = np.array([U1_MAX, U2_MAX], dtype=float)

        # Flowchart-mode enforcement (hard constraints)
        # - BOLUS: insulin allowed up to 80, glucagon forced 0
        # - BASAL: insulin allowed but capped to BASAL_U1_MAX, glucagon forced 0
        # - GLUCAGON: glucagon allowed up to 0.5, insulin forced 0
        if mode == "BOLUS":
            umax_mode = np.array([U1_MAX, 0.0], dtype=float)
            umin_mode = np.array([U1_MIN, 0.0], dtype=float)
        elif mode == "BASAL":
            umax_mode = np.array([BASAL_U1_MAX, 0.0], dtype=float)
            umin_mode = np.array([U1_MIN, 0.0], dtype=float)
        elif mode == "GLUCAGON":
            umax_mode = np.array([0.0, U2_MAX], dtype=float)
            umin_mode = np.array([0.0, U2_MIN], dtype=float)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for i in range(Nc):
            constraints += [
                u_seq[i*m + 0, 0] >= umin_mode[0],
                u_seq[i*m + 0, 0] <= umax_mode[0],
                u_seq[i*m + 1, 0] >= umin_mode[1],
                u_seq[i*m + 1, 0] <= umax_mode[1],
            ]

        # Soft output bounds
        constraints += [
            Y_pred >= y_min_dev - s,
            Y_pred <= y_max_dev + s
        ]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        #prob.solve(solver=solver, verbose=verbose)
        # Giving OSQP more iterations and less tolerances for convergence
        prob.solve(
            solver=cp.OSQP,
            verbose=False,
            max_iter=20000,
            eps_abs=1e-4,
            eps_rel=1e-4,
            polish=True,
            warm_start=True
        )

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return uk_prev.copy(), {"qp_status": prob.status}

        deltaU_opt = np.array(deltaU.value).reshape(-1)


        du0 = deltaU_opt[:m]
        u0 = uk_prev + du0 # receding horizon: apply first input only

        info = {"qp_status": prob.status} 

        return u0, info

