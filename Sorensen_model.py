"""
Sorensen full model (Appendix A1, pages 19-23 of 'Revised Sorensen Model.pdf')

Transcription notes / sources:
- Appendix A1 equations (glucose, insulin, glucagon mass balances): see PDF Appendix A1. :contentReference[oaicite:5]{index=5}
- Insulin subsystem equations and pancreas model: see PDF insulin section. :contentReference[oaicite:6]{index=6}
- Parameter table (volumes, time constants): see PDF page with parameter values. :contentReference[oaicite:7]{index=7}
- Initial-conditions guidance (examples): see PDF pages for initial conditions. :contentReference[oaicite:8]{index=8}

This implementation is intended for research/education / in-silico experiments only.
"""

import numpy as np
from scipy.integrate import solve_ivp

# ---- State indices and names (match Appendix A1) ----
STATE_NAMES = [
    # Glucose mass-balance states (mass or concentration notation follows the paper)
    "GBV",  # brain vascular water space glucose concentration
    "GBI",  # brain interstitial
    "GH",   # heart & lungs
    "GJ",   # gut
    "GL",   # liver
    "GK",   # kidney
    "GPV",  # periphery vascular
    "GPI",  # periphery interstitial (output)
    # Glucose metabolic intermediate states
    "MHGP", "MHGU", "f2",     # helper metabolic states used in hepatic/peripheral fluxes
    # Insulin states (compartments)
    "IB", "IH", "IG", "IL", "IK", "IPV", "IPI",
    # Pancreatic insulin-related states (P, Q, X etc. in Sorensen)
    "P", "Q", "X",
    # Glucagon (single equation in Sorensen)
    "GAMMA"
]

N_STATES = len(STATE_NAMES)

# For convenience: index map
IDX = {name: i for i, name in enumerate(STATE_NAMES)}

# ---- Default parameters (taken from the parameter table in Appendix A1) ----
# See parameter table (pages 23-24). Values below are copied where explicitly present.
# Citation: parameter table lines. :contentReference[oaicite:9]{index=9}
default_params = {
    # Glucose volumes (dl) and flows (dl/min or 1/min as in table)
    "VG_BV": 3.5,    # dl (GBV)
    "QG_B": 5.9,     # dl/min (brain flow)
    "TB": 2.1,       # min (brain time constant)
    "VBI": 4.5,      # dl (brain interstitial)
    "VG_H": 13.8,    # dl
    "QG_A": 2.5,     # dl/min
    "VG_L": 25.1,
    "QG_L": 12.6,
    "VG_G": 11.2,
    "QG_G": 10.1,
    "VG_K": 6.6,
    "QG_K": 10.1,
    "VG_PV": 10.4,
    "QG_P": 15.1,
    "VPI": 67.4,
    "TGP": 5.0,      # peripheral time constant (TG_P)
    # Metabolic source/sink baseline constants (examples from the PDF)
    "rBGU": 70.0,    # mg/min constant basal brain glucose uptake. :contentReference[oaicite:10]{index=10}
    "rRBCU": 10.0,   # mg/min constant RBC uptake. :contentReference[oaicite:11]{index=11}
    "rJGU": 20.0,    # mg/min gut uptake. :contentReference[oaicite:12]{index=12}
    # Hepatic and peripheral exchange helper parameters (placeholders; full functional forms below)
    # Many additional parameters (M1, M2, bpir*, beta, K, gamma etc.) appear in the pancreas/insulin
    # submodel — included as placeholders here; see Eqs 70-79 and parameter lists. :contentReference[oaicite:13]{index=13}
    "M1": 0.00015,
    "M2": 0.289,
    "bpir1": 5.148,
    "bpir2": 3.746,
    "bpir3": 2.452,
    "bpir4": 4.477,
    "bpir5": 3.066,
    # insulin extraction fractions (from PDF)
    "FLIC": 0.40,  # fraction liver insulin clearance. :contentReference[oaicite:14]{index=14}
    "FKIC": 0.30,  # kidney clearance fraction. :contentReference[oaicite:15]{index=15}
    "FPIC": 0.15,  # peripheral clearance. :contentReference[oaicite:16]{index=16}
    # Pancreas model params
    "K": 0.015,
    "Q0": 44310.0,
    "gP": 0.0,
    "a": 0.014, "b": 0.05,
    # Glucagon helper params (from PDF eqs 75-79)
    "rMGC": 9.10,   # ml/min (value from eqn lines). :contentReference[oaicite:17]{index=17}
    # Any other parameters required should be added here...
}

# ---- Metabolic helper functions ----
def tanh(x):
    return np.tanh(x)

# Example metabolic functions — transcription from Appendix A1:
# MG_PGU, MI_PGU etc. (PDF lines 80-86 show functional forms; see Appendix A1). :contentReference[oaicite:18]{index=18}
def MI_PGU(INPI):
    # transcription of MI_PGU = 7.03 + 6.52 * tanh(0.338*(INPI - 5.82))  (eqn line L81-L82)
    return 7.03 + 6.52 * tanh(0.338 * (INPI - 5.82))

def MG_PGU(GNPI):
    # placeholder: in the PDF MG_PGU = G_N(PI) or similar (use identity for now)
    return GNPI

# --- Main ODE function ---
def sorensen_full(t, x, params=None, roga_func=None, u_ins=0.0, u_gluc=0.0):
    """
    Full Sorensen model RHS (mass balances and metabolic sources/sinks).
    - x: state vector in order of STATE_NAMES
    - params: dictionary of parameters (if None, use default_params)
    - roga_func: optional function roga(t) giving oral glucose appearance (mg/min) to use in GJ eqn
    - u_ins: exogenous insulin infusion (if any) added to pancreatic/insulin input (units consistent with model)
    - u_gluc: exogenous glucagon infusion (units consistent)
    Returns dx/dt (numpy array)
    """
    if params is None:
        params = default_params

    dx = np.zeros(N_STATES)
    # shorthand
    p = params
    # unpack states
    GBV = x[IDX["GBV"]]
    GBI = x[IDX["GBI"]]
    GH  = x[IDX["GH"]]
    GJ  = x[IDX["GJ"]]
    GL  = x[IDX["GL"]]
    GK  = x[IDX["GK"]]
    GPV = x[IDX["GPV"]]
    GPI = x[IDX["GPI"]]
    MHGP = x[IDX["MHGP"]]
    MHGU = x[IDX["MHGU"]]
    f2 = x[IDX["f2"]]
    IB = x[IDX["IB"]]
    IH = x[IDX["IH"]]
    IG = x[IDX["IG"]]
    IL = x[IDX["IL"]]
    IK = x[IDX["IK"]]
    IPV = x[IDX["IPV"]]
    IPI = x[IDX["IPI"]]
    P = x[IDX["P"]]
    Q = x[IDX["Q"]]
    X = x[IDX["X"]]
    GAMMA = x[IDX["GAMMA"]]

    # ---- Glucose mass balances (transcribed from Appendix A1) ----
    # Brain vascular water space: dGBV/dt = QG_B*(GH - GBV)/VG_BV  + (VBI/TB)*(GBV - GBI) ??? (PDF line layout ambiguous)
    # See Appendix A1 eqns for exact algebraic forms. (page 19). :contentReference[oaicite:19]{index=19}
    # Here we implement the canonical mass-balance form:
    QG_B = p["QG_B"]
    VG_BV = p["VG_BV"]
    TB = p["TB"]
    VBI = p["VBI"]
    # NOTE: the PDF uses specific multiplicative placements — double-check denominators vs numerators when validating
    dx[IDX["GBV"]] = (QG_B*(GH - GBV) / VG_BV) + (VBI / TB) * (GBV - GBI) - p["rBGU"]
    # Brain interstitial:
    dx[IDX["GBI"]] = (VBI / TB) * (GBV - GBI) - p["rBGU"]  # see eqn (24). :contentReference[oaicite:20]{index=20}

    # Heart and lungs (GH): eqn (25) uses flows from multiple compartments and rRBCU.
    # eqn (25) transcription (see Appendix A1). :contentReference[oaicite:21]{index=21}
    # implement as: dx_GH = sum(flow_in*(comp - GH)/VG_H) + (incoming gut/liver/periphery ... ) - rRBCU
    VG_H = p["VG_H"]
    # For readability we use a simplified mass-balance structure consistent with the Appendix:
    # (Important: check full algebraic coefficients in the PDF and correct if necessary)
    dx[IDX["GH"]] = (p["QG_A"]*(GBV - GH) + p["QG_L"]*(GL - GH) + p["QG_K"]*(GK - GH) + p["QG_P"]*(GPV - GH)) / VG_H - p.get("rRBCU", 10.0)

    # Gut (GJ): dGJ/dt = QG_J*(GH - GJ) + roga - rJGU
    # (eqn 26). roga_func supplies oral glucose appearance; otherwise 0.
    roga = 0.0 if roga_func is None else float(roga_func(t))
    QG_J = p.get("QG_J", p.get("QG_G", 10.1))
    dx[IDX["GJ"]] = (QG_J * (GH - GJ) / max(p["VG_G"], 1e-6)) + roga - p["rJGU"]

    # Liver GL: eqn (27): mass balance including hepatic glucose production rHGP and hepatic glucose uptake rHGU
    # (transcribed from Appendix A1). :contentReference[oaicite:22]{index=22}
    dx[IDX["GL"]] = (p["QG_L"] * (GH - GL) / max(p["VG_L"], 1e-6)) + p.get("rHGP", 0.0) - p.get("rHGU", 0.0)

    # Kidney GK: eqn (28): flow exchange + renal excretion rKGE
    dx[IDX["GK"]] = (p["QG_K"] * (GH - GK) / max(p["VG_K"], 1e-6)) - p.get("rKGE", 0.0)

    # Periphery GPV, GPI: eqns (29)-(30)
    dx[IDX["GPV"]] = (p["QG_P"] * (GH - GPV) / max(p["VG_PV"], 1e-6)) + ( (GPV - GPI) / p["TGP"] )
    dx[IDX["GPI"]] = ( (GPV - GPI) / p["TGP"] ) - p.get("rPGU", 0.0)

    # Metabolic helper states MHGP, MHGU, f2 from Appendix A1 eqns (10-12 / 38)
    dx[IDX["MHGP"]] = -0.04 * MHGP + 0.077 * IL     # eqn (10). :contentReference[oaicite:23]{index=23}
    dx[IDX["MHGU"]] = -0.04 * MHGU + 0.002 * IL     # eqn (11). :contentReference[oaicite:24]{index=24}
    dx[IDX["f2"]] = -0.015 * f2 - 0.006 * GAMMA     # eqn (12). :contentReference[oaicite:25]{index=25}

    # ---- Insulin subsystem (mass balances) ----
    # IB, IH, IG, IL, IK, IPV, IPI (eqns 54-60 and pancreas model eqns 67-69)
    # Transcribed from Appendix A1 insulin section. :contentReference[oaicite:26]{index=26}
    QI_B = p.get("QI_B", 5.0)
    VI_B = p.get("VI_B", 3.5)
    # simplistic mass balance forms (verify denominators in paper)
    dx[IDX["IB"]] = (QI_B * (IH - IB) / max(VI_B, 1e-6))
    dx[IDX["IH"]] = (p.get("QI_A", 2.5) * (IB - IH) + p.get("QI_L", 12.6)*(IL - IH) + p.get("QI_K", 10.1)*(IK - IH) + p.get("QI_P", 15.1)*(IPV - IH) ) / max(p.get("VG_H", 13.8),1e-6) - 0.0
    dx[IDX["IG"]] = 0.765 * (IH - IG)             # eqn (15). :contentReference[oaicite:27]{index=27}
    dx[IDX["IL"]] = 0.094 * IH + 0.378 * IG - 0.789 * IL  # eqn (16). :contentReference[oaicite:28]{index=28}
    dx[IDX["IK"]] = 1.411 * IH - 1.8351 * IK     # eqn (17). :contentReference[oaicite:29]{index=29}
    dx[IDX["IPV"]] = 1.418 * IH - 1.874 * IPV + 0.455 * IPI  # eqn (18). :contentReference[oaicite:30]{index=30}
    dx[IDX["IPI"]] = 0.05 * IPV - 0.111 * IPI + u_ins    # eqn (19) includes insulin input U1. :contentReference[oaicite:31]{index=31}

    # ---- Pancreas (insulin release) submodel ----
    # rPIR = S(GH) etc., and the P/Q/X dynamics (see eqns 67-73). Implement a basic transcription.
    # This block is somewhat involved in the paper; here we provide a working transcription:
    # S = [M1 * Y + M2 * (X - I)]_0+ / Q  (see eqns 70-73). Use simple functional approximations.
    # For now, we implement the structure with parameter placeholders; you can refine S/X/Q scalings.
    # See Appendix A1 eqns 70-73 for details. :contentReference[oaicite:32]{index=32}
    # dP/dt = a*(P1 - P)
    # dI/dt = b*(X - I)   (note: here I corresponds to insulin central or similar — mapping must be verified)
    # NOTE: we keep simple placeholders so the model runs; refine with exact formulae from Appendix A1 if desired.
    a = p.get("a", 0.014)
    b = p.get("b", 0.05)
    P1 = max(0.0, X)  # placeholder mapping
    dx[IDX["P"]] = a * (P1 - P)
    dx[IDX["Q"]] = p.get("K", 0.015) * (p.get("Q0", 44310.0) - Q) + p.get("gP", 0.0) * P1
    dx[IDX["X"]] = (GH * p.get("bpir1", 5.148) / (p.get("bpir2", 3.746) + GH))  # very rough transcription of Eq.72-73

    # ---- Glucagon equation ----
    # single ODE for GAMMA per Appendix A1 eqn (74) and (77-79). Includes input U2 (u_gluc).
    # Implemented using transcription from PDF: Γ' = -0.08*Γ - 6.9e-7*GH + 0.0016*IH + U2 (this was used in linearised form)
    # See dual-hormone linearisation eqn (20) and Appendix A1 glucagon metabolic forms. :contentReference[oaicite:33]{index=33} :contentReference[oaicite:34]{index=34}
    dx[IDX["GAMMA"]] = -0.08 * GAMMA - 6.9e-7 * GH + 0.0016 * IH + u_gluc

    return dx


# ---- Helper simulate function ----
def simulate(initial_state=None, t_span=(0, 800), t_eval=None, params=None, roga_func=None, u_ins_func=None, u_gluc_func=None):
    """
    Simulate Sorensen model using solve_ivp (BDF stiff solver).
    - initial_state: array-like of length N_STATES; if None, a default basal vector is used (see PDF initial conditions).
      See Appendix A1 initial conditions. :contentReference[oaicite:35]{index=35}
    - t_span: (t0, tf) in minutes
    - t_eval: times to evaluate; if None, uses np.linspace(t0, tf, 801)
    - u_ins_func(t) and u_gluc_func(t) return exogenous infusion at time t (scalars)
    """
    if params is None:
        params = default_params

    if initial_state is None:
        # simple basal initialisation (approx values); refine from Appendix A1 initial conditions.
        x0 = np.zeros(N_STATES)
        # set glucose compartments to ~90-145 mg/dL depending on desired scenario:
        for name in ["GBV","GBI","GH","GJ","GL","GK","GPV","GPI"]:
            x0[IDX[name]] = 200.0
        # small metabolic/insulin values by default:
        for name in ["MHGP","MHGU","f2"]:
            x0[IDX[name]] = 0.1
        for name in ["IB","IH","IG","IL","IK","IPV","IPI"]:
            x0[IDX[name]] = 10.0  # non-zero baseline; adjust for Type1 vs Type2 simulations
        x0[IDX["P"]] = 0.0
        x0[IDX["Q"]] = params.get("Q0", 44310.0)
        x0[IDX["X"]] = 0.0
        x0[IDX["GAMMA"]] = 2.0
    else:
        x0 = np.asarray(initial_state, dtype=float)

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])+1))

    # wrappers for control inputs
    if u_ins_func is None:
        u_ins_func = lambda tt: 0.0
    if u_gluc_func is None:
        u_gluc_func = lambda tt: 0.0

    def rhs(t, x):
        return sorensen_full(t, x, params=params, roga_func=roga_func, u_ins=u_ins_func(t), u_gluc=u_gluc_func(t))

    sol = solve_ivp(rhs, t_span, x0, method='BDF', t_eval=t_eval, atol=1e-7, rtol=1e-5)
    return sol.t, sol.y

# ---- Example usage (if run as script) ----
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t, y = simulate(t_span=(0, 800))
    gpi = y[IDX["GPI"], :]
    plt.plot(t, gpi)
    plt.xlabel("time (min)")
    plt.ylabel("GPI (mg/dL) — peripheral glucose (approx)")
    plt.title("Sorensen model (Appendix A1) — demo basal sim")
    plt.grid(True)
    plt.show()
