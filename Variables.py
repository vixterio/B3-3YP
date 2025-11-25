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
gamma_B=input ["glucagon concentration"]
#Source and Sinks-Glucose in mg/min
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
r_PgammaR=r_MgammaC*gamma
r_PgammaR=MG_PgammaR*MI_PgammaR*rB_PgammaR
MI_HGP=1
MI_HGU=1
f2=0
#for Insulin and Glucose concentration it depends person to person but Ill use the average values
GB_PV= 40 #micro U/ml so adjust based on other values
IB_PV= 5.5 #mmol/L so adjust
#check units throughout the code to make sure they are consistent