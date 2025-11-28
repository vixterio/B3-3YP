import numpy as np
import matplotlib.pyplot as plt


class Patient:
    """
    Patient class for glucose-glucagon dynamics simulation.
    States are named with convention: variable_subscript where subscript comes after underscore.
    Uses Runge-Kutta 4th order method for numerical integration.
    """
    
    def __init__(self, dt=1.0):
        """
        Initialize patient with random state values and parameters.
        
        Args:
            dt: Time step size for integration (minutes)
        """
        self.dt = dt  # Step size for integration
        
        # State variables (G for glucose concentrations)
        # Initialize with random values around physiological ranges
        self.G_BV = np.random.uniform(95, 95)   # mg/dl
        self.G_BI = np.random.uniform(60, 60)   # mg/dl
        self.G_H = np.random.uniform(95, 95)    # mg/dl
        self.G_G = np.random.uniform(95, 95)    # mg/dl
        self.G_L = np.random.uniform(95, 95)    # mg/dl
        self.G_K = np.random.uniform(95, 95)    # mg/dl
        self.G_PV = np.random.uniform(95, 95)   # mg/dl
        self.G_PI = np.random.uniform(95, 95)   # mg/dl
        
        # Insulin state variables (I for insulin concentrations)
        self.I_B = 0      # μU/ml
        self.I_H = 0      # μU/ml
        self.I_G = 0      # μU/ml
        self.I_L = 0      # μU/ml
        self.I_K = 0      # μU/ml
        self.I_PV = 0     # μU/ml
        self.I_PI = 0     # μU/ml
        
        # Multiplier state variables
        self.M_HGP_i = np.random.uniform(0.8, 1.2)
        self.M_HGU_i = np.random.uniform(0.8, 1.2)
        self.f_2 = np.random.uniform(0, 0.5)
        
        # Glucagon state variable (Γ)
        self.Gamma = np.random.uniform(0.5, 1)  # mg/L
        
        # Control inputs (can be set externally)
        self.U_1 = 0  # Insulin input
        self.U_2 = 0  # Glucagon input
        
    def derivatives(self, G_BV, G_BI, G_H, G_G, G_L, G_K, G_PV, G_PI,
                    I_B, I_H, I_G, I_L, I_K, I_PV, I_PI,
                    M_HGP_i, M_HGU_i, f_2, Gamma):
        """
        Calculate derivatives for all state variables based on the ODEs.
        
        Returns:
            Tuple of all derivatives
        """
        # Glucose ODEs (Equations 2-9)
        dG_BV = 1.685*G_H - 2.297*G_BV + 0.612*G_BI
        
        dG_BI = 0.476*(G_BV - G_BI)
        
        dG_H = (0.427*G_BV + 0.913*G_L + 0.731*G_K + 
                1.094*G_PV - 3.166*G_H)
        
        dG_G = 0.901*(G_H - G_G)
        
        dG_L = (0.099*G_H + 0.402*G_G - 0.501*G_L + 
                2.755*M_HGP_i - 5.299*f_2 - 
                8.467*M_HGU_i + 4.354*Gamma)
        
        dG_K = 1.53*(G_H - G_K)
        
        dG_PV = 1.451*G_H - 2.748*G_PV + 1.296*G_PI
        
        # Equation (9): G*_PI
        dG_PI = 0.2*G_PV - 0.204*G_PI - 0.007*I_PI
        
        # Multiplier ODEs (Equations 10-12)
        # Equation (10): M*_HGP^i
        dM_HGP_i = -0.04*M_HGP_i + 0.077*I_L
        
        # Equation (11): M*_HGU^i
        dM_HGU_i = -0.04*M_HGU_i + 0.002*I_L
        
        # Equation (12): f*_2
        df_2 = -0.015*f_2 - 0.006*Gamma
        
        # Insulin ODEs (Equations 13-19)
        # Equation (13): I*_B
        dI_B = 1.73*(I_H - I_B)
        
        # Equation (14): I*_H
        dI_H = (0.454*I_B + 0.909*I_L + 0.727*I_K + 
                1.061*I_PV - 3.151*I_H)
        
        # Equation (15): I*_G
        dI_G = 0.765*(I_H - I_G)
        
        # Equation (16): I*_L
        dI_L = 0.094*I_H + 0.378*I_G - 0.789*I_L
        
        # Equation (17): I*_K
        dI_K = 1.411*I_H - 1.8351*I_K
        
        # Equation (18): I*_PV
        dI_PV = 1.418*I_H - 1.874*I_PV + 0.455*I_PI
        
        # Equation (19): I*_PI
        dI_PI = 0.05*I_PV - 0.111*I_PI + self.U_1
        
        # Glucagon ODE (Equation 20)
        # Equation (20): Γ*
        dGamma = -0.08*Gamma - 0.00000069*G_H + 0.0016*I_H + self.U_2
        
        return (dG_BV, dG_BI, dG_H, dG_G, dG_L, dG_K, dG_PV, dG_PI,
                dI_B, dI_H, dI_G, dI_L, dI_K, dI_PV, dI_PI,
                dM_HGP_i, dM_HGU_i, df_2, dGamma)
    
    def transition(self):
        """
        Evolve all states forward by one time step using 4th order Runge-Kutta method.
        Updates all state variables in place.
        """
        # Current state vector
        y0 = np.array([
            self.G_BV, self.G_BI, self.G_H, self.G_G, 
            self.G_L, self.G_K, self.G_PV, self.G_PI,
            self.I_B, self.I_H, self.I_G, self.I_L, 
            self.I_K, self.I_PV, self.I_PI,
            self.M_HGP_i, self.M_HGU_i, self.f_2, self.Gamma
        ])
        
        # RK4 integration
        k1 = np.array(self.derivatives(*y0))
        k2 = np.array(self.derivatives(*(y0 + 0.5*self.dt*k1)))
        k3 = np.array(self.derivatives(*(y0 + 0.5*self.dt*k2)))
        k4 = np.array(self.derivatives(*(y0 + self.dt*k3)))
        
        # Update state
        y_new = y0 + (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Unpack new state values
        (self.G_BV, self.G_BI, self.G_H, self.G_G, 
         self.G_L, self.G_K, self.G_PV, self.G_PI,
         self.I_B, self.I_H, self.I_G, self.I_L, 
         self.I_K, self.I_PV, self.I_PI,
         self.M_HGP_i, self.M_HGU_i, self.f_2, self.Gamma) = y_new
    
    def get_state(self):
        """
        Return current state as a dictionary.
        """
        return {
            'G_BV': self.G_BV,
            'G_BI': self.G_BI,
            'G_H': self.G_H,
            'G_G': self.G_G,
            'G_L': self.G_L,
            'G_K': self.G_K,
            'G_PV': self.G_PV,
            'G_PI': self.G_PI,
            'I_B': self.I_B,
            'I_H': self.I_H,
            'I_G': self.I_G,
            'I_L': self.I_L,
            'I_K': self.I_K,
            'I_PV': self.I_PV,
            'I_PI': self.I_PI,
            'M_HGP_i': self.M_HGP_i,
            'M_HGU_i': self.M_HGU_i,
            'f_2': self.f_2,
            'Gamma': self.Gamma
        }



if __name__ == "__main__": #only lets this run when dynamics.py is executed directly
    # Create patient with zero control inputs
    patient = Patient(dt=1.0)
    
    # Simulation parameters
    n_steps = 10
    time = np.arange(n_steps + 1)
    
    # Storage for state history
    history = {
        'G_BV': [], 'G_BI': [], 'G_H': [], 'G_G': [], 'G_L': [], 'G_K': [], 'G_PV': [], 'G_PI': [],
        'I_B': [], 'I_H': [], 'I_G': [], 'I_L': [], 'I_K': [], 'I_PV': [], 'I_PI': [],
        'M_HGP_i': [], 'M_HGU_i': [], 'f_2': [], 'Gamma': []
    }
    
    # Record initial state
    state = patient.get_state()
    for key in history.keys():
        history[key].append(state[key])
    
    # Run simulation
    print("Simulating patient dynamics with zero control inputs...")
    for i in range(n_steps):
        patient.transition()
        state = patient.get_state()
        for key in history.keys():
            history[key].append(state[key])
    
    # Convert history to numpy arrays
    for key in history.keys():
        history[key] = np.array(history[key])
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot 1: Glucose concentrations
    ax = axes[0]
    ax.plot(time, history['G_BV'], label='G_BV', alpha=0.7)
    ax.plot(time, history['G_BI'], label='G_BI', alpha=0.7)
    ax.plot(time, history['G_H'], label='G_H', alpha=0.7)
    ax.plot(time, history['G_G'], label='G_G', alpha=0.7)
    ax.plot(time, history['G_L'], label='G_L', alpha=0.7)
    ax.plot(time, history['G_K'], label='G_K', alpha=0.7)
    ax.plot(time, history['G_PV'], label='G_PV', alpha=0.7)
    ax.plot(time, history['G_PI'], label='G_PI', alpha=0.7)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Glucose Concentration (mg/dl)')
    ax.set_title('Glucose Compartments Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Insulin concentrations
    ax = axes[1]
    ax.plot(time, history['I_B'], label='I_B', alpha=0.7)
    ax.plot(time, history['I_H'], label='I_H', alpha=0.7)
    ax.plot(time, history['I_G'], label='I_G', alpha=0.7)
    ax.plot(time, history['I_L'], label='I_L', alpha=0.7)
    ax.plot(time, history['I_K'], label='I_K', alpha=0.7)
    ax.plot(time, history['I_PV'], label='I_PV', alpha=0.7)
    ax.plot(time, history['I_PI'], label='I_PI', alpha=0.7)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Insulin Concentration (μU/ml)')
    ax.set_title('Insulin Compartments Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    

    # Plot 3: Glucagon
    ax = axes[2]
    ax.plot(time, history['Gamma'], label='Γ (Glucagon)', alpha=0.7, color='purple')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Glucagon Concentration (mg/L)')
    ax.set_title('Glucagon Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nFinal state:")
    # print(patient.get_state()) #commented out to reduce output