import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from typing import Callable, Sequence

#DEFAULT PARAMETERS FROM 'REVISED SORENSEN MODEL' PAPER

#Glucose values in dL
Vg_BV=3.5
Vg_BI=4.5
Vg_H=13.8
Vg_L=25.1
Vg_G=11.2
Vg_K=6.6
Vg_PV=10.4
Vg_PI=67.4
#Glucose flow rates are in dl/min
Qg_B=5.9
Qg_B2=43.7
Qg_A=2.5
Qg_L=12.6
Qg_G=10.1
Qg_K=10.1
Qg_P=15.1
#Time is in minutes
Tg_B=2.1
Tg_P=5

#Insulin values indicated by the I
#Volume is in L
Vi_B=0.26
Vi_H=0.99
Vi_G=0.94
Vi_L=1.14
Vi_K=0.51
Vi_PV=0.74
Vi_PI=6.74
#Flow rate in L/min
Qi_B=0.45
Qi_H=3.12
Qi_A=0.18
Qi_K=0.72
Qi_P=1.05
Qi_G=0.72
Qi_L=0.90
#Random constants
Ti_P=20 #minutes
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


#USER SUPPLIED BASAL INPUT VALUES
G_input = 90.0 #glucose concentration
I_input = 0 #insulin concentration
gamma_input = 50.0 #glucagon concentration

#METABOLIC SOURCES AND SINKS
F_LIC = 0.40
F_KIC = 0.30
F_PIC = 0.15

#BASELINE FLUX CONSTANTS
r_BGU = 70.0 #brain glucose uptake baseline mg/min eqn 31
r_RBCU = 10.0 #RBC uptake baseline mg/min eqn 32
r_JGU = 20.0 #cardiac (gut/intestinal) uptake mg/min eqn 33
r_PGU_B = 35.0 #baseline peripheral glucose uptake mg/min eqn 35
r_HGP_B = 155.0 #baseline hepatic glucose production mg/min eqn 155
r_HGU_B = 20.0 #baseline hepatic glucose uptake mg/min eqn 49


#INITIAL CONDITIONS
#Glucose mass balance
G_PV_B = G_input   #eqn 80

# For initial steady-state we use the baseline flux values r^B_* from the paper:
r_PGU = r_PGU_B    
r_BGU = r_BGU      
r_CGU = r_JGU      
#r_PGU = MI_PGU(I_PI_B) * MG_PGU(G_PI_B) * r_PGU_B
#r_HGU = MI_HGU(I_H_B) * MG_HGU(G_L_B) * r_HGU_B
#r_HGP = MI_HGP(I_H_B) * MF_HGP(G_H_B) * MG_HGP(G_L_B) * r_HGP_B


G_H_B  = G_PV_B + r_PGU / Qg_P #eqn 81
G_K_B  = G_H_B #eqn 82
G_BV_B = G_H_B - r_BGU / Qg_B #eqn 83
G_G_B  = G_H_B - r_CGU / Qg_G #eqn 84
r_B_HGP = r_HGP_B
r_B_HGU = r_HGU_B
#the paper has a typographical error as it uses 1/G_B_L instead of 1/Vg_L in eqn 85
G_L_B = (1.0 / Vg_L) * (Qg_A * G_H_B + Qg_G * G_G_B + r_B_HGP - r_B_HGU) #eqn 85
G_BI_B = G_BV_B - (r_BGU * Tg_B) / Vg_BI #eqn 86
G_PI_B = G_PV_B - (r_PGU * Tg_P) / Vg_PI #eqn 87

#Insulin mass balance
I_PV_B = I_input #eqn 91
I_H_B = I_PV_B / (1.0 - F_PIC) #eqn 92
I_K_B = I_H_B * (1.0 - F_KIC) #eqn 93
I_B_B = I_H_B #eqn 94
I_G_B = I_H_B #eqn 95
I_PI_B = I_PV_B - (Qi_P * Ti_P / Vi_PI) * (I_H_B - I_PV_B) #eqn 96
I_L_B = (1.0 / Qi_L) * (Qi_H * I_H_B + Qi_B * I_B_B + Qi_K * I_K_B + Qi_P * I_PI_B) #eqn 97
r_PIR_B = 0 #No pancreatic insulin release in T1D eqn 98
#Model Pancreas
X_B = 0 #eqn 99
Y_B = 0 #eqn 101
P_inf = 0 #eqn 100
I_pancreas_B = 0 #eqn 103

# Eq.104: Q_pancreas_B = (H * Q0 + gamma * P_inf) / (H + M1 * Y_B)
# The paper uses H, gamma in pancreas equation; in earlier code Q0 used as baseline secretion constant.
# In the version I used earlier I set Q_pancreas_B = (Q0 + gamma_input * P_inf) / (1 + M1 * Y_B)
# to keep consistent with earlier script. If I have specific H and gamma scalars, replace accordingly.
Q_pancreas_B = 0 #eqn 104

#Glucagon mass balance
gamma_B = gamma_input #eqn 105



#PRINTING RESULTS
print("\n=== Glucose Initial Conditions (paper baseline fluxes) ===")
print(f"G_PV_B = {G_PV_B:.4f} mg/dL")
print(f"G_H   = {G_H_B:.4f} mg/dL")
print(f"G_K   = {G_K_B:.4f} mg/dL")
print(f"G_BV  = {G_BV_B:.4f} mg/dL")
print(f"G_G   = {G_G_B:.4f} mg/dL")
print(f"G_L   = {G_L_B:.4f} mg/dL")
print(f"G_BI  = {G_BI_B:.4f} mg/dL")
print(f"G_PI  = {G_PI_B:.4f} mg/dL")

print("\n=== Insulin Initial Conditions ===")
print(f"I_PV = {I_PV_B:.4f} μU/mL")
print(f"I_H  = {I_H_B:.4f} μU/mL")
print(f"I_K  = {I_K_B:.4f} μU/mL")
print(f"I_B  = {I_B_B:.4f} μU/mL")
print(f"I_G  = {I_G_B:.4f} μU/mL")
print(f"I_PI = {I_PI_B:.4f} μU/mL")
print(f"I_L  = {I_L_B:.4f} μU/mL")
print(f"r_PIR (pancreatic release baseline) = {r_PIR_B:.4f} (units consistent with paper)")

print("\n=== Glucagon Initial Condition ===")
print(f"Gamma_B = {gamma_B:.4f} pg/mL")

print("\n=== Pancreas Steady-State ===")
print(f"X_B = {X_B:.6g}")
print(f"P_inf = {P_inf:.6g}")
print(f"Y_B = {Y_B:.6g}")
print(f"Q_pancreas_B = {Q_pancreas_B:.6g}")




#THE NON-LINEAR SORENSEN MODEL (from the 'Revised Sorensen Model' paper)
def sorensen_odes(t, y, u_insulin, u_glucagon):
    """
    Full nonlinear Sorensen model (T1D version).
    y = 19-state vector (in the order listed below)
    u_insulin  = external insulin infusion (U1)
    u_glucagon = external glucagon infusion (U2)
    """

     #unpack state vector
    (G_BV, G_BI, G_H, G_G, G_L, G_K, G_PV, G_PI, I_B, I_H, I_G, I_L, I_K, I_PV, I_PI, M_I_HGP, M_I_HGU, f2, Gamma) = y

    #normalisation of the terms
    G_BV_N = G_BV / G_BV_B
    G_BI_N = G_BI / G_BI_B
    G_H_N  = G_H  / G_H_B
    G_G_N  = G_G  / G_G_B
    G_L_N  = G_L  / G_L_B
    G_K_N  = G_K  / G_K_B
    G_PV_N = G_PV / G_PV_B
    G_PI_N = G_PI / G_PI_B

    I_B_N  = I_B  / max(I_B_B, 1e-12) #the max function makes sure it never divides by 0, if I_B_B=0 then it will divide by 1e-12
    I_H_N  = I_H  / max(I_H_B, 1e-12)
    I_G_N  = I_G  / max(I_G_B, 1e-12)
    I_L_N  = I_L  / max(I_L_B, 1e-12)
    I_PI_N = I_PI / max(I_PI_B, 1e-12)

    # nonlinear metabolic flux functions
    #PGU (peripheral uptake)
    M_I_PGU = 7.03 + 6.52 * np.tanh(0.338 * (I_PI_N - 5.82)) #eqn 36
    M_G_PGU = G_PI_N #eqn 37
    r_PGU = M_I_PGU * M_G_PGU * r_PGU_B #eqn 34

    #HGP (hepatic glucose production)
    M_I1_HGP = 1.21 - 1.14 * np.tanh(1.66 * (I_L_N - 0.89)) #eqn 42
    M_G0_HGP = 2.7 * np.tanh(0.39 * G_H_N) #eqn 44
    M_F_HGP  = M_G0_HGP - f2 #eqn 43
    M_G_HGP  = 1.42 - 1.41 * np.tanh(0.62 * (G_L_N - 0.497)) #eqn 47
    r_HGP = M_I1_HGP * M_F_HGP * M_G_HGP * r_HGP_B #eqn 38

    #HGU (hepatic glucose uptake)
    M_I1_HGU = 2.0 * np.tanh(0.55 * I_L_N) #eqn 51
    M_G_HGU  = 5.66 + 5.66 * np.tanh(2.44 * (G_L_N - 1.48)) #eqn 52
    r_HGU = M_I1_HGU * M_G_HGU * r_HGU_B #eqn 48

    #Kidney excretion eqn 53
    if G_K < 460: #mg/min
        r_KGE = 71 + 71 * np.tanh(0.011 * (G_K - 460))
    else:
        r_KGE = -330 + 0.872 * G_K

    #Static fluxes
    r_BGU  = 70.0 #eqn 31
    r_RBCU = 10.0 #eqn 32
    r_JGU  = 20.0 #eqn 33

    #Glucose ODEs (Eqs. 23–30)
    dG_BV = (Qg_B * (G_H - G_BV) - (Vg_BI/Tg_B) * (G_BV - G_BI) - 0) / Vg_BV #eqn 23
    dG_BI = ((Vg_BI/Tg_B) * (G_BV - G_BI) - r_BGU) / Vg_BI #eqn 24

    dG_H  = (Qg_B * G_BV + Qg_L * G_L + Qg_K * G_K + Qg_P * G_PV #eqn 25
             - Qg_B2 * G_H - r_RBCU) / Vg_H
    
    #paper uses J not G but doesn't define any J terms, so assuming typo
    dG_G  = (Qg_G * (G_H - G_G) - r_JGU) / Vg_G #eqn 26

    dG_L  = (Qg_A * G_H + Qg_G * G_G - Qg_L * G_L + r_HGP - r_HGU) / Vg_L #eqn 27

    dG_K  = (Qg_K * (G_H - G_K) - r_KGE) / Vg_K #eqn 28

    dG_PV = (Qg_P * (G_H - G_PV) - (Vg_PI/Tg_P)*(G_PV - G_PI)) / Vg_PV #eqn 29

    dG_PI = ((Vg_PI/Tg_P)*(G_PV - G_PI) - r_PGU) / Vg_PI #eqn 30

    #Insulin ODEs (T1D = NO PANCREAS)
    dI_B  = (Qi_B * (I_H - I_B)) / Vi_B #eqn 54

    dI_H  = (Qi_B * I_B + Qi_L * I_L + Qi_K * I_K + Qi_P * I_PV #eqn 55
             - Qi_H * I_H) / Vi_H

    dI_G  = (Qi_G * (I_H - I_G)) / Vi_G #eqn 56

    #No endogenous secretion, so r_PIR = 0
    r_LIC_val = F_LIC * (Qi_L * I_H) #eqn 61
    dI_L  = (Qi_A * I_H + Qi_G * I_G - Qi_L * I_L - r_LIC_val) / Vi_L #eqn 57

    r_KIC_val = F_KIC * (Qi_K * I_H) #eqn 63
    dI_K  = (Qi_K * (I_H - I_K) - r_KIC_val) / Vi_K #eqn 58

    dI_PV = (Qi_P * (I_H - I_PV) - (Vi_PI/Ti_P)*(I_PV - I_PI)) / Vi_PV #eqn 59
    
    #simplified version of eqn 65, used because the original equation's denominator can become
    #unstable when insulin is zero or ver small. 
    r_PIC_val = F_PIC * Qi_P * I_PI #eqn 65
    dI_PI = ((Vi_PI/Ti_P)*(I_PV - I_PI) - r_PIC_val + u_insulin) / Vi_PI #eqn 60

    #equations 67-73 are part of the pancreas (beta-cell) submodel, which is not used in T1D

    # GLUCAGON ODE (T1D has normal glucagon, as alpha-cells are functional, but production is dysregulated)
    # Production
    M_G_PFR = 2.93 - 2.10 * np.tanh(4.18 * (G_H_N - 0.61)) #eqn 78
    M_I_PFR = 1.31 - 0.61 * np.tanh(1.06 * (I_H_N - 0.47)) #eqn 79
    r_PFR = M_G_PFR * M_I_PFR * 1.0 #eqn 77, using baseline production rate of 1.0 pg/min

    # Clearance
    r_MTC = 9.10 * Gamma #eqn 75 mL/min
    dGamma = (r_PFR - r_MTC + u_glucagon) / Vgamma #eqn 74
    

    #Metabolic Dynamics (slow regulators) from the original Sorensen model, not included in the Revised Model
    #as they didn't change from the original model
    dM_I_HGP = (1/25.0) * (M_I1_HGP - M_I_HGP)
    dM_I_HGU = (1/25.0) * (M_I1_HGU - M_I_HGU)
    d_f2     = (1/65.0) * (M_G0_HGP - 0.5 - f2)

    #non-negativity clamps, preventing glucose, insulin and glucagon values from becoming negative
    if G_BV <= 0 and dG_BV < 0: dG_BV = 0
    if G_BI <= 0 and dG_BI < 0: dG_BI = 0
    if G_H  <= 0 and dG_H  < 0: dG_H  = 0
    if G_G  <= 0 and dG_G  < 0: dG_G  = 0
    if G_L  <= 0 and dG_L  < 0: dG_L  = 0
    if G_K  <= 0 and dG_K  < 0: dG_K  = 0
    if G_PV <= 0 and dG_PV < 0: dG_PV = 0
    if G_PI <= 0 and dG_PI < 0: dG_PI = 0       

    if I_B  <= 0 and dI_B  < 0: dI_B  = 0
    if I_H  <= 0 and dI_H  < 0: dI_H  = 0
    if I_G  <= 0 and dI_G  < 0: dI_G  = 0
    if I_L  <= 0 and dI_L  < 0: dI_L  = 0
    if I_K  <= 0 and dI_K  < 0: dI_K  = 0
    if I_PV <= 0 and dI_PV < 0: dI_PV = 0
    if I_PI <= 0 and dI_PI < 0: dI_PI = 0

    if Gamma < 0 and dGamma < 0:
        dGamma = 0.0


    #Return all derivatives
    return np.array([
        dG_BV, dG_BI, dG_H, dG_G, dG_L, dG_K, dG_PV, dG_PI,
        dI_B, dI_H, dI_G, dI_L, dI_K, dI_PV, dI_PI,
        dM_I_HGP, dM_I_HGU, d_f2,
        dGamma
    ])



#EXAMPLE INPUT FUNCTIONS FOR EXTERNAL INFUSIONS
def u1_insulin(t):
    #example: no external insulin infusion 0mU/min
    return 0.0

def u2_glucagon(t):
    #example: no external glucagon infusion 0pg/min
    return 0.0

#build the initial state vector (must match sorensen_odes unpack order)
initial_state = np.array([
    G_BV_B, G_BI_B, G_H_B, G_G_B, G_L_B, G_K_B, G_PV_B, G_PI_B,
    I_B_B, I_H_B, I_G_B, I_L_B, I_K_B, I_PV_B, I_PI_B,
    1.0,    # M_I_HGP baseline (unitless)
    1.0,    # M_I_HGU baseline (unitless)
    0.0,    # f2 baseline
    gamma_B # Gamma baseline
])

#time span for simualtion (0 to 300 minutes
t_final = 300.0
t_eval = np.linspace(0.0,t_final, 601) #every 0.5 minutes
#now need to solve the ODEs, solve_ivp solves an initial value problem for a system of ODEs
sol = solve_ivp(
    fun=lambda t, y: sorensen_odes(t, y, u1_insulin(t), u2_glucagon(t)),
    t_span=(0.0, t_final),
    y0=initial_state,
    t_eval=t_eval,
    method='RK45',
    atol=1e-6, #absolute tolerance
    rtol=1e-3, #relative tolerance
)

if not sol.success:
    raise RuntimeError("Integration failed: " + sol.message)

#Plotting glucose, insulin, glucagon results
plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[7], label="G_PI (periph interstitial glucose) -- index 7")
plt.xlabel("Time (min)")
plt.ylabel("Glucose (mg/dL)")
plt.title("Glucose compartments")
plt.legend()
plt.grid(True)

plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[9],  label="I_H (heart insulin) -- index 9")
plt.xlabel("Time (min)")
plt.ylabel("Insulin (μU/mL)")
plt.title("Insulin compartments")
plt.legend()
plt.grid(True)

plt.figure(figsize=(10,4))
plt.plot(sol.t, sol.y[18], label="Gamma (glucagon) -- index 18")
plt.xlabel("Time (min)")
plt.ylabel("Glucagon (pg/mL)")
plt.title("Glucagon")
plt.legend()
plt.grid(True)

plt.show()