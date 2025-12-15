import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from typing import Callable, Sequence

# Model is based only on Revised Sorensen Model paper

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


#STATE VECTOR
#0 G_BV      brain vascular glucose
#1 G_BI      brain interstitial glucose
#2 G_H      heart and lungs glucose
#3 G_G      gut glucose
#4 G_L      liver glucose
#5 G_K      kidney glucose
#6 G_PV     peripheral vascular glucose
#7 G_PI     peripheral interstitial glucose
#8 I_B      blood insulin
#9 I_H      heart and lungs insulin
#10 I_G     gut insulin
#11 I_L     liver insulin
#12 I_K     kidney insulin
#13 I_PV    peripheral vascular insulin
#14 I_PI    peripheral interstitial insulin
#15 M_I_HGP      regulator  
#16 M_I_HGU     regulator
#17 f2 slow     regulator
#18 Gamma       glucagon

n_states = 19 #number of states in the model

F_LIC = 0.40
F_KIC = 0.30
F_PIC = 0.15

#INITIAL CONDITIONS (b refers to baseline)
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
IB_PV = 10.0  #uU/mL [input insulin concentration]
IB_H = IB_PV/(1-F_PIC) #uU/mL
IB_K = IB_H*(1-F_KIC)
IB_B = IB_H
IB_G = IB_H
IB_PI = IB_PV - ((Qi_P*Ti_P)/Vi_PI)*(IB_H - IB_PV) #uU/mL
IB_L = (1/Qi_L)*(Qi_H*IB_H + Qi_B*IB_B + Qi_K*IB_K + Qi_P*IB_PV) #uU/mL
rB_PIR = 0 #T1D assumption - no endogenous insulin production

#glucagon
gamma_B = 50.0 #pg/mL [input plasma glucagon concentration

#normalised values
GN_H = 1.0
IN_L = 1.0 

#regulator initial conditions
M_I_HGP_initial = 1.21 - 1.14 *np.tanh(1.66*(IN_L - 0.89))
M_I_HGU_initial = 2*np.tanh(0.55*IN_L)
f2_initial = 2.7*np.tanh(0.39*GN_H) - 0.5 #not explicity stated, calculated by setting d(f2)/dt = 0

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

print (initial_conditions)