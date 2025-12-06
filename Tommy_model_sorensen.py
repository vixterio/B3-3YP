import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from typing import Callable, Sequence

#Default Parameters from 'Revised Sorensen Model'
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
#Volume is in l
Vi_B=0.26
Vi_H=0.99
Vi_G=0.94
Vi_L=1.14
Vi_K=0.51
Vi_PV=0.74
Vi_PI=6.74
#Flow rate in l/min
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