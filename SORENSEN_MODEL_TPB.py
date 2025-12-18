import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from typing import Callable, Sequence
from pprint import pprint

#Model is based only on Revised Sorensen Model paper

#PARAMTER VALUES

#Glucose
Vg_BV = 3.5 #dL
Vg_BI = 4.5 #dL
Vg_H  = 13.8 #dL
Vg_L  = 25.1 #dL
Vg_G  = 11.2 #dL
Vg_K  = 6.6 #dL
Vg_PV = 10.4 #dL
Vg_PI = 67.4 #dL

Qg_B  = 5.9 #dL/min
Qg_A  = 2.5 #dL/min
Qg_L  = 12.6 #dL/min
Qg_G  = 10.1 #dL/min
Qg_K  = 10.1 #dL/min
Qg_P = 15.1 #dL/min

Tg_B = 2.1 #min
Tg_P = 5.0 #min

#Insulin
Vi_B  = 0.26 #L
Vi_H  = 0.99 #L
Vi_G  = 0.94 #L
Vi_L  = 1.14 #L
Vi_K  = 0.51 #L
Vi_PV = 0.74 #L
Vi_PI = 6.74 #L

Qi_B = 0.45 #L/min
Qi_H = 3.12 #L/min
Qi_G = 0.72 #L/min
Qi_L = 0.90 #L/min
Qi_K = 0.72 #L/min
Qi_P = 1.05 #L/min
Qi_A = 0.18 #L/min


#these are pancreas paramters - not used in T1D model but here for completeness
Ti_P = 20 #min
beta_pir1 = 3.27
beta_pir2 = 132 #mg/dL
beta_pir3 = 5.93 
beta_pir4 = 3.02
beta_pir5 = 1.11
M_1 = 0.00747 #1/min
M_2 = 0.0958 #1/min
Q_0 = 6.33 #U
alpha = 0.00482 #1/min
beta = 0.931 #1/min
K = 0.575 #U/min


#Glucagon
V_gamma = 11310 #mL


STATE_ORDER = [
    "G_PV",
    "G_H",
    "G_K",
    "G_BV",
    "G_G",
    "G_L",
    "G_BI",
    "G_PI",
    "I_PV",
    "I_H",
    "I_K",
    "I_B",
    "I_G",
    "I_PI",
    "I_L",
    "M_I_HGP",
    "M_I_HGU",
    "f2",
    "Gamma",
]

#18 Gamma       glucagon

n_states = 19 #number of states in the model

F_LIC = 0.40
F_KIC = 0.30
F_PIC = 0.15

#INITIAL CONDITIONS (b refers to baseline)
#bsebaseline flux constants
rB_PGU = 35.0 #mg/min peripheral glucose uptake 
rB_HGP = 155 #mg/min hepatic glucose production
rB_HGU = 20 #mg/min hepatic glucose uptake
rB_JGU = 20 #mg/min gut glucose uptake
rB_BGU = 70 #mg/min brain glucose uptake


#steady state glucose initial conditions
GB_PV = 90 #mg/dL [input glucose concentration]
GB_H = GB_PV + rB_PGU/Qg_P #mg/dL
GB_K = GB_H 
GB_BV = GB_H - rB_BGU/Qg_B #mg/dL
GB_G = GB_H - rB_JGU/Qg_G #mg/dL rB_JGU = rB_CGU
GB_L=(1/Vg_L)*(Qg_A*GB_H + Qg_G*GB_G + rB_HGP - rB_HGU)
GB_BI = GB_BV - (rB_BGU*Tg_B)/Vg_BI #mg/dL
GB_PI = GB_PV - (rB_PGU*Tg_P)/Vg_PI #mg/dL

#steady state insulin initial conditions
IB_PV = 0.5  #uU/L [input insulin concentration]
IB_H = IB_PV/(1-F_PIC) #uU/L
IB_K = IB_H*(1-F_KIC)
IB_B = IB_H
IB_G = IB_H
IB_PI = IB_PV - ((Qi_P*Ti_P)/Vi_PI)*(IB_H - IB_PV) #uU/L
IB_L = (1/Qi_L)*(Qi_H*IB_H + Qi_B*IB_B + Qi_K*IB_K + Qi_P*IB_PV) #uU/L
rB_PIR = 0 #T1D assumption - no endogenous insulin production

#glucagon
gamma_B = 50.0 #pg/mL [input plasma glucagon concentration

#normalised values
GN_H = 1.0
IN_L = 1.0 

#regulator initial conditions
M_I_HGP_initial =float(1.21 - 1.14*np.tanh(1.66*(IN_L - 0.89)))
M_I_HGU_initial = float(2*np.tanh(0.55*IN_L))
f2_initial = float(2.7*np.tanh(0.39*GN_H) - 0.5) #not explicityly stated, calculated from setting d(f2)/dt = 0

#collect everything]
initial_conditions = { # Glucose
    "G_PV": GB_PV,
    "G_H":  GB_H,
    "G_K":  GB_K,
    "G_BV": GB_BV,
    "G_G":  GB_G,
    "G_L":  GB_L,
    "G_BI": GB_BI,
    "G_PI": GB_PI,

    "I_PV": IB_PV,
    "I_H":  IB_H,
    "I_K":  IB_K,
    "I_B":  IB_B,
    "I_G":  IB_G,
    "I_PI": IB_PI,
    "I_L":  IB_L,

    "M_I_HGP": M_I_HGP_initial,
    "M_I_HGU": M_I_HGU_initial,
    "f2":      f2_initial,
    "Gamma":   gamma_B
}

#printing initial conditions
#pprint(initial_conditions)





#exogenous insulin
def u_insulin(t):
    return 0  #µU/min

#exogenous glucagon
def u_glucagon(t):
    return 0  #pg/min





#NON-LINEAR SORENSEN MODEL

#compute metabolic fluxes -> compute glucose derivatives -> compute insulin derivatives -> compute regulator and glucagon derivatives

def sorensen_odes(t, y):
    #y is 19 state vector, u_insulin and u_glucagon are exogenous insulin and glucagon infusions respectively
    #calling exogenous functions
    uI = u_insulin(t)
    uG = u_glucagon(t)
    #unpack state vector
    (G_PV, G_H, G_K, G_BV, G_G, G_L, G_BI, G_PI, I_PV, I_H, I_K, I_B, I_G, I_PI, I_L, M_I_HGP, M_I_HGU, f2, Gamma) = y

    #metabolic fluxes 
    rB_PGU = 35.0 #mg/min peripheral glucose uptake 
    rB_HGP = 155 #mg/min hepatic glucose production
    rB_HGU = 20 #mg/min hepatic glucose uptake
    rB_JGU = 20 #mg/min gut glucose uptake
    rB_BGU = 70 #mg/min brain glucose uptake
    rB_PFR = 0 #for a T1D patient

    #peripheral glucose uptake r_PGU 
    IN_PI = I_PI/max(IB_PI, 1e-12)
    GN_PI = G_PI/max(GB_PI, 1e-12)
    M_I_PGU = float(7.03 + 6.52*np.tanh(0.338*(IN_PI - 5.82))) #eq 36
    M_G_PGU = GN_PI #eq 37
    r_PGU = M_I_PGU * M_G_PGU * rB_PGU #eq 34

    #hepatic glucose production r_HGP
    GN_H = G_H/max(GB_H, 1e-12)
    GN_L = G_L/max(GB_L, 1e-12) 
    M_G0_HGP = float(2.7 * np.tanh(0.39 * GN_H)) #eq 44 immediate glucose effect
    M_F_HGP  = M_G0_HGP - f2  #eq 43 slow glucose feedback via f2    
    M_G_HGP = float(1.42 - 1.41 * np.tanh(0.62 * (GN_L - 0.497))) #eq 47
    r_HGP = M_I_HGP * M_F_HGP * M_G_HGP * rB_HGP #eq 38

    #hepatic glucose uptake r_HGU
    M_G_HGU = float(5.66 + 5.66 * np.tanh(2.44 * (GN_L - 1.48))) #eq 52
    r_HGU = M_G_HGU * M_I_HGU * rB_HGU

    #kidney glucose excretion r_KGE
    if G_K < 460.0: #mg/min
        r_KGE = float(71.0 + 71.0 * np.tanh(0.011 * (G_K - 460.0)))
    else:
        r_KGE = 330.0 + 0.872 * G_K  #eq 53

    #constant metabolic sinks
    r_BGU = rB_BGU
    r_JGU = rB_JGU
    r_RBCU = 10 #mg/min

    #glucose ODEs
    dG_BV = (1/Vg_BV) * (Qg_B * (G_H - G_BV) - (Vg_BI/Tg_B) * (G_BV - G_BI)) #eq 23
    dG_BI = (1/Vg_BI) * ((Vg_BI/Tg_B) * (G_BV - G_BI) - r_BGU) #eq 24
    #dG_H might need to have gut terms added, not sure - paper doesn't have it but chatGPT seems to think it is needed
    dG_H = (1/Vg_H) * (Qg_B*G_BV + Qg_L*G_L + Qg_K*G_K + Qg_P*G_PV - (Qg_B + Qg_L + Qg_K + Qg_P)*G_H - r_RBCU) #eq 25
    dG_G = (1/Vg_G) * (Qg_G * (G_H - G_G) - r_JGU) #eq 26 they had a typo and wrote J instead of G in the paper
    dG_L = (1/Vg_L) * (Qg_A*G_H + Qg_G*G_G - Qg_L*G_L + r_HGP - r_HGU) #eq 27
    dG_K = (1/Vg_K) * (Qg_K * (G_H - G_K) - r_KGE) #eq 28
    dG_PV = (1/Vg_PV) * (Qg_P * (G_H - G_PV) - (Vg_PI/Tg_P)*(G_PV - G_PI)) #eqn 29
    dG_PI = (1/Vg_PI) * ((Vg_PI/Tg_P)*(G_PV - G_PI) - r_PGU) #eq 30

    #insulin ODEs
    dI_B = (1/Vi_B) * (Qi_B * (I_H - I_B)) #eq 54
    dI_H = (1/Vi_H) * (Qi_B*I_B + Qi_L*I_L + Qi_K*I_K + Qi_P*I_PV - Qi_H*I_H) #eq 55
    dI_G = (1/Vi_G) * (Qi_G * (I_H - I_G)) #eq 56
    r_LIC = F_LIC * (Qi_A * I_H + Qi_G * I_G) #eq 64
    dI_L = (1/Vi_L) * (Qi_A*I_H + Qi_G*I_G - Qi_L*I_L - r_LIC) #eq 57 r_PIR = 0 as rB_PIR = 0
    r_KIC = F_KIC * (Qi_K*I_H) #eq 63
    dI_K = (1/Vi_K) * (Qi_K * (I_H - I_K) - r_KIC) #eq 58
    dI_PV = (1/Vi_PV) * (Qi_P * (I_H - I_PV) - (Vi_PI/Ti_P) * (I_PV - I_PI)) + uI/Vi_PV #eq 59
    r_PIC = F_PIC * Qi_P * I_PI #simplified version of equation 65
    dI_PI = (1/Vi_PI) * ((Vi_PI/Ti_P) * (I_PV - I_PI) - r_PIC)  #eq 60

    #regulator ODEs
    IN_L = I_L/max(IB_L,1e-12)
    MI1_HGP = float(1.21 - 1.14 * np.tanh(1.66 * (IN_L - 0.89)))
    t1 = 25 #min
    dM_I_HGP = (MI1_HGP - M_I_HGP) / t1 #eq 
    MI1_HGU = float(2.0 * np.tanh(0.55 * IN_L))
    dM_I_HGU = (MI1_HGU - M_I_HGU) / t1 #eq 50
    GN_H = G_H/max(GB_H,1e-12)
    M_G0_HGP = float(2.7 * np.tanh(0.39 * GN_H))
    t_G = 65 #min
    d_f2 = (0.5*M_G0_HGP - 0.5 - f2)/t_G #eq 45

    #glucagon ODE
    IN_H = I_H/max(IB_H,1e-12)
    M_G_PFR = float(2.93 - 2.10 * np.tanh(4.18 * (GN_H - 0.61)))
    M_I_PFR = float(1.31 - 0.61 * np.tanh(1.06 * (IN_H - 0.47)))
    r_MTC = 9.10 * Gamma  #pg/min
    #rB_PFR = 0.3 * r_MTC
    r_PFR = rB_PFR * M_I_PFR * M_G_PFR #pg/min 
    dGamma = (r_PFR - r_MTC + uG) / V_gamma #eq 74


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


    #return all the derivatives
    return np.array([dG_PV, dG_H, dG_K, dG_BV, dG_G, dG_L, dG_BI, dG_PI, dI_PV, dI_H, dI_K, dI_B, dI_G, dI_PI, dI_L, dM_I_HGP, dM_I_HGU, d_f2, dGamma])


 



#time span for simulation
t_final = 1000.0
t_eval = np.linspace(0.0,t_final, 2001) #every 0.5 minutes





#SOLVING THE ODEs
y0 = np.array([initial_conditions[name] for name in STATE_ORDER], dtype=float)
sol = solve_ivp(
    sorensen_odes,
    (0.0, t_final),
    y0,
    method="RK45",
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-8
)

#PLOTS
#G_PI 
plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[7], label="G_PI (periph interstitial glucose) -- index 7")
plt.xlabel("Time (min)")
plt.ylabel("Glucose (mg/dL)")
plt.title("Glucose compartments")
plt.legend()
plt.grid(True)

#Glucagon
plt.figure(figsize=(10,4))
plt.plot(sol.t, sol.y[18], label="Gamma (glucagon) -- index 18")
plt.xlabel("Time (min)")
plt.ylabel("Glucagon (pg/mL)")
plt.title("Glucagon")
plt.legend()
plt.grid(True)

#Insulin
plt.figure(figsize=(10,4))
plt.plot(sol.t, sol.y[8], label="I_PV -- index 8")
plt.xlabel("Time (min)")
plt.ylabel("Insulin (μU/L)")
plt.title("Insulin")
plt.legend()
plt.grid(True)
plt.show()















    
