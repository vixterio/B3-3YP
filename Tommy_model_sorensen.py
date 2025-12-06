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
I_input = 10.0 #insulin concentration
gamma_input = 50.0 #glucagon concentration

#METABOLIC SOURCES AND SINKS
F_LIC = 0.40
F_KIC = 0.30
F_PIC = 0.15

#BASELINE FLUX CONSTANTS
r_BGU = 70.0 #brain glucose uptake baseline mg/min
r_RBCU = 10.0 #RBC uptake baseline mg/min
r_JGU = 20.0 #cardiac (gut/intestinal) uptake mg/min
r_PGU_B = 35.0 #baseline peripheral glucose uptake mg/min
r_HGP_B = 155.0 #baseline hepatic glucose production mg/min
r_HGU_B = 20.0 #baseline hepatic glucose uptake mg/min


#INITIAL CONDITIONS
#Glucose mass balance
G_PV_B = G_input   

# For initial steady-state we use the baseline flux values r^B_* from the paper:
r_PGU = r_PGU_B    
r_BGU = r_BGU      
r_CGU = r_JGU      


G_H_B  = G_PV_B + r_PGU / Qg_P
G_K_B  = G_H_B
G_BV_B = G_H_B - r_BGU / Qg_B
G_G_B  = G_H_B - r_CGU / Qg_G
r_B_HGP = r_HGP_B
r_B_HGU = r_HGU_B
G_L_B = (1.0 / Vg_L) * (Qg_A * G_H_B + Qg_G * G_G_B + r_B_HGP - r_B_HGU)
G_BI_B = G_BV_B - (r_BGU * Tg_B) / Vg_BI
G_PI_B = G_PV_B - (r_PGU * Tg_P) / Vg_PI

#Insulin mass balance
I_PV_B = I_input  
I_H_B = I_PV_B / (1.0 - F_PIC)
I_K_B = I_H_B * (1.0 - F_KIC)
I_B_B = I_H_B
I_G_B = I_H_B
I_PI_B = I_PV_B - (Qi_P * Ti_P / Vi_PI) * (I_H_B - I_PV_B)
I_L_B = (1.0 / Qi_L) * (Qi_H * I_H_B + Qi_B * I_B_B + Qi_K * I_K_B + Qi_P * I_PI_B)
r_PIR_B = (Qi_L / (1.0 - F_LIC)) * (I_L_B - Qi_B * I_B_B - Qi_G * I_G_B - Qi_H * I_H_B)
#Model Pancreas
X_B = (G_H_B ** beta_pir1) / ((beta_pir2 ** beta_pir1) + beta_pir3 * (G_H_B ** beta_pir4))
P_inf = X_B ** beta
Y_B = X_B ** beta_pir5
P_B = P_inf
I_pancreas_B = X_B

# Eq.104: Q_pancreas_B = (H * Q0 + gamma * P_inf) / (H + M1 * Y_B)
# The paper uses H, gamma in pancreas equation; in earlier code Q0 used as baseline secretion constant.
# In the version I used earlier I set Q_pancreas_B = (Q0 + gamma_input * P_inf) / (1 + M1 * Y_B)
# to keep consistent with earlier script. If I have specific H and gamma scalars, replace accordingly.
Q_pancreas_B = (Q0 + gamma_input * P_inf) / (1.0 + M1 * Y_B)

#Glucagon mass balance
gamma_B = gamma_input



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