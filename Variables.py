import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from typing import Callable, Sequence

# Default Parameters

#anything after the underscore is subscript from the pubmed document
#Glucose Values
#Volumes are in dl
V_BV=3.5
V_BI=4.5
V_H=13.8
V_L=25.1
V_G=11.2
V_K=6.6
V_PV=10.4
V_PI=67.4
#Flow rates are in dl/min
Q_B=5.9
Q_B2=43.7
Q_A=2.5
Q_L=12.6
Q_G=10.1
Q_K=10.1
Q_P=15.1
#Time is in minutes
T_B=2.1
T_P=5
#insulin values indicated by the I
#Volume is in l
VI_B=0.26
VI_H=0.99
VI_G=0.94
VI_L=1.14
VI_K=0.51
VI_PV=0.74
VI_PI=6.74
#Flow rate in l/min
QI_B=0.45
QI_H=3.12
QI_A=0.18
QI_K=0.72
QI_P=1.05
QI_G=0.72
QI_L=0.90
#Random constants
TI_P=20 #minutes
beta_pir1=3.27
beta_pir2=132 #mg/dl
beta_pir3=5.93
beta_pir4=3.02
beta_pir5=1.11
M1=0.00747 #min^-1
M2=0.0958 #min^-1
Q0=6.33 #U
alpha=0.0482 #min^-1
beta=0.931 #min^-1
K=0.575 #U/min
#Glucagon Values
Vgamma=11310 #ml
#gamma_B=input ["glucagon concentration"]

## Insulin and glucagon ODEs (Eqs. 13-20 inclusive)

def insulin_glucagon_odes(
    t: float,
    y: Sequence[float],
    u1_func: Callable[[float], float],
    u2_func: Callable[[float], float],
    gh_func: Callable[[float], float],
) -> np.ndarray:
    """
    Compute derivatives for insulin (7 states) + glucagon (1 state).

    State vector y ordering:
      y[0] I_B   (blood)
      y[1] I_H   (heart/lungs)
      y[2] I_G   (gut)
      y[3] I_L   (liver)
      y[4] I_K   (kidney)
      y[5] I_PV  (peripheral vascular)
      y[6] I_PI  (peripheral interstitial / infusion)
      y[7] gamma (glucagon)

    Inputs:
      u1_func(t) -> U_1 (insulin infusion, mU/min)
      u2_func(t) -> U_2 (glucagon infusion)
      gh_func(t) -> GH (glucose in heart compartment, mg/dL)
    """
    # Unpack state variables
    IB, IH, IG, IL, IK, IPV, IPI, gamma = y

    # Evaluate inputs at time t
    U1 = u1_func(t)
    U2 = u2_func(t)
    GH = gh_func(t)

    # Compute Insulin derivatives
        # All numerical constants lifted from the paper directly
    dIB_dt = 1.73*(IH - IB)
    dIH_dt = Q0*0.454*IB + 0.909*IL + 0.727*IK + 1.061*IPV - 3.151*IH
    dIG_dt = 0.765*(IH - IG)
    dIL_dt = 0.094*IH + 0.378*IG - 0.789*IL
    dIK_dt = 1.411*IH - 1.8351*IK
    dIPV_dt = 1.418*IH - 1.874*IPV + 0.455*IPI
    dIPI_dt = 0.05*IPV - 0.111*IPI + U1

    # Compute Glucagon derivative
    dgamma_dt = -0.08*gamma - 0.00000069*GH + 0.0016*IH + U2

    return np.array([dIB_dt, dIH_dt, dIG_dt, dIL_dt, dIK_dt, dIPV_dt, dIPI_dt, dgamma_dt])

#Source and Sinks-Glucose in mg/min
GN_PI = 0  
## once josh writes glucode ode someone needs to turn these into another function that calculates these rates as instantaneous at each time step -lin
r_RBCU=10
r_BGU=70
r_JGU=20
rB_PGU=35
MG_PGU=GN_PI
r_PGU=MI_PGU*MG_PGU*rB_PGU
rB_HGP=155
r_HGP=MI_HGP*Mgamma_HGP*MG_HGP*rB_HGP
tao_1=25 #mins
tao_gamma=65 #mins
rB_HGU=20
r_HGU=MI_HGU*MG_HGU*rB_HGU
#Source and sinks- INsulin
F_LIC=0.4
F_KIC=0.3
F_PIC=0.15
#Source and sinks- Glucagon
r_MgammaC=9.1
# r_PgammaR=r_MgammaC*gamma ##why do we define r_PgammaR twice is it typo or like a recursive thing :'( -lin
# r_PgammaR=MG_PgammaR*MI_PgammaR*rB_PgammaR
MI_HGP=1
MI_HGU=1
f2=0
#for Insulin and Glucose concentration it depends person to person but Ill use the average values
GB_PV= 40 #micro U/ml so adjust based on other values
IB_PV= 5.5 #mmol/L so adjust
#check units throughout the code to make sure they are consistent

## Running a demo simulation
if __name__ == "__main__":
    # Minimal initial conditions 
    IB0 = 5.5   # mU/ml (random initial IB to start the ode solving from) — @Tommy can you figure out what this is meant to be -lin
    y0 = np.array([IB0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # @Tommy here as well change initial values if needed -lin

    # time horizon (paper often uses long sims; use 2000 min for full reproduction)
    t_span = (0.0, 200.0)     # demo uses 200; change to (0.0, 2000.0) for full runs
    t_eval = np.linspace(t_span[0], t_span[1], 401)

    # inputs: no infusion in demo
    u1 = lambda t: 0.0  # insulin infusion (mU/min)
    u2 = lambda t: 0.0  # glucagon infusion (mg/min)

    # GH: placeholder constant. @Tommy Replace GH(t) as needed
    gh = lambda t: 140.0  # mg/dL placeholder

    sol = solve_ivp(lambda tt, yy: insulin_glucagon_odes(tt, yy, u1, u2, gh),
                    t_span=t_span, y0=y0, t_eval=t_eval, method="RK45",
                    atol=1e-6, rtol=1e-6)

    if not sol.success:
        raise RuntimeError("Integration failed: " + str(sol.message))
