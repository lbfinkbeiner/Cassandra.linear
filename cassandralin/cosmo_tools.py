# THIS WAS WRITTEN BY MATTEO. NONE OF IT IS FROM LUKAS.

# A library with useful cosmology routines

import numpy as np

class Cosmology():

    def __init__(self, omega_cdm = 0.268584094, omega_de = 0.681415906, omega_baryon = 0.05, 
                 hubble = 0.67, ns = 0.96, A_s = None, sigma8 = 0.82755, ReNormalizeInputSpectrum = True, 
                 expfactor = 0.33342518307993363, tau = 0.0952, w0 = -1, wa = 0, Om_EdE = False, de_model = 'lcdm'):
        
        ''' Cosmology class. It contains the cosmological parameters and some useful functions.
        
        Parameters
        -----------
        omega_cdm : float, optional, default = 0.268584094
            Cold dark matter density parameter. Non-physical i.e. this is \Omega_{cdm}
        omega_de : float, optional, default = 0.681415906
            Dark energy density parameter. Non-physical i.e. this is \Omega_{de}
        omega_baryon : float, optional, default = 0.05
            Baryon density parameter. Non-physical i.e. this is \Omega_{b}
        hubble : float, optional, default = 0.67
            Hubble parameter in units of 100 km/s/Mpc i.e. this is h
        ns : float, optional, default = 0.96
            Scalar spectral index of initial power spectrum
        A_s : float, optional, default = None
            Amplitude of initial power spectrum. If None, it is computed from sigma8 (when needed, TODO)
        sigma8 : float, optional, default = 0.82755
            Normalization of the power spectrum at z=0. Used for compatibility with other codes (TODO: use sigma12)
        ReNormalizeInputSpectrum : bool, optional, default = True
            If True, the input power spectrum is renormalized to sigma8 (TODO, just set for retro-compatibility)
        expfactor : float, optional, default = 0.33342518307993363
            Scale factor for this specific cosmology.
        tau : float, optional, default = 0.0952
            Optical depth for reionization. (TODO, just set for retro-compatibility)
        w0 : float, optional, default = -1
            Dark energy equation of state parameter, constant part.
        wa : float, optional, default = 0
            Dark energy equation of state parameter, time-dependent part (linear parametrization).
        Om_EdE : float, optional, default = False
            EdE dark energy model parameter. If False, the dark energy model is LCDM.
        de_model : str, optional, default = 'lcdm'
            Dark energy model. Can be 'lcdm', 'w0wa' or 'EdE'.

        Methods
        -----------
        # Cosmology methods #
        E_z : compute the normalized Hubble parameter at a given scale factor.
        DE_density : compute the dark energy density at a given scale factor.
        w_z : compute the dark energy equation of state parameter at a given scale factor.
        w_eff : compute the effective dark energy equation of state parameter at a given scale factor.

        # Growth methods #
        compute_growth : compute the growth factor and the growth rate at a given scale factor.
        growth_factor : compute the growth factor at a given scale factor.
        growth_rate : compute the growth rate at a given scale factor.

        '''
        
        self.pars = {}
        self.pars['Omega_cdm'] = omega_cdm
        self.pars['Omega_b'] = omega_baryon
        self.pars['Omega_m'] = omega_cdm + omega_baryon
        self.pars['Omega_DE'] = omega_de
        self.pars['Om_EdE'] = Om_EdE
        self.pars['w0'] = w0
        self.pars['wa'] = wa
        self.pars['de_model'] = de_model
        self.pars['sigma8'] = sigma8
        self.pars['h'] = hubble
        self.pars['ns'] = ns
        self.pars['As'] = A_s
        self.pars['tau'] = tau
        self.expfactor = expfactor
        # Setting the physical density parameters
        self.pars['omega_cdm'] = self.pars['Omega_cdm'] * self.pars['h']**2
        self.pars['omega_b'] = self.pars['Omega_b'] * self.pars['h']**2
        self.pars['omega_m'] = self.pars['Omega_m'] * self.pars['h']**2
        self.pars['omega_de'] = self.pars['Omega_DE'] * self.pars['h']**2
        # Setting cached values for the camb power spectrum (maybe this should be done in a different way)
        self.camb_Pk_z0 = None
        self.camb_results_z0 = None
        self.knl = None

    ####### Cosmology methods #######

    def E_z(self, a, deriv=False):
        ''' Compute the normalized Hubble parameter at a given scale factor.
            If deriv=True, it returns the derivative of E_z with respect to a.'''
        if deriv:
            return (1/2)*(-3*self.pars['Omega_m']/a**4 - 2*(1-self.pars["Omega_m"]-self.pars["Omega_DE"])/a**3 + self.DE_density(a, deriv=True))/self.E_z(a)
        else:
            return np.sqrt(self.pars["Omega_m"]/a**3 + (1-self.pars["Omega_m"]-self.pars["Omega_DE"])/a**2 + self.DE_density(a))
            
    def DE_density(self, a, deriv=False):
        ''' Compute the dark energy density at a given scale factor.
            If deriv=True, it returns the derivative of the dark energy density with respect to a.'''
        if self.pars['de_model'] == 'lcdm':
            if deriv:
                return 0
            else:
                return self.pars['Omega_DE']
        else:
            if deriv:
                return self.pars['Omega_DE'] * (-3) * a**(-3*(1+self.w_eff(a))-1) * (a*np.log(a)*self.w_eff(a, deriv=True) + 1+self.w_eff(a))
            else:
                return self.pars['Omega_DE']*a**(-3*(1+self.w_eff(a)))
            

    def w_eff(self, a, deriv=False):
        ''' Compute the effective dark energy equation of state parameter at a given scale factor.
            If deriv=True, it returns the derivative of w_eff with respect to a.
            In models with varying dark energy equation of state, this is defined as
                w_eff: Omega_DE(a) = Omega_DE * a**(-3*(1+w_eff(a)))'''
        
        # First we calculate the w0 contribution
        if self.pars['Om_EdE']:
            b = - 3*self.pars['w0']/( np.log(1/self.pars['Om_EdE']-1) + np.log(1/self.pars['Omega_m']-1) )
            w0 = self.pars['w0']/(1 - b*np.log(a)) if not deriv else self.pars['w0']*b/(a*(1-b*np.log(a))**2)
        else:
            w0 = self.pars['w0'] if not deriv else 0
        # Then the wa one
        if self.pars['wa'] == 0:
            wa = 0
        else:
            if not deriv:
                wa = np.where(a == 1, 0, self.pars['wa'] * (1+(1-a)/np.log(a)))
            else:
                wa = np.where(a == 1, -self.pars['wa']/2, self.pars['wa'] * (a-a*np.log(a)-1) / (a * np.log(a)**2))
        return w0 + wa
    
    ####### Growth methods #######
    
    def compute_growth(self, a, a0=1e-3, f0=1, solver='odeint'):
        ''' Compute the growth factor and the growth rate at a given scale factor, through solving the generic ODE.
            Works for w0, wa and EdE dark energy models, but not for a generic w(z).
            
            a : float or array
                Scale factor at which to compute the growth factor and the growth rate.
            a0 : float, optional, default = 1e-3
                Initial scale factor value for solving the ODE.
            f0 : float, optional, default = 1
                Initial growth rate value for solving the ODE. The initial D value is set to D0 = a0, assuming matter domination.
            solver : str, optional, default = 'odeint'
                Solver to use for solving the ODE. Can be 'odeint' or 'solve_ivp'.

            return : tuple
                Growth factor and growth rate at the given scale factor.

            '''
        
        from scipy.integrate import solve_ivp, odeint 

        def dy_dt(t, y):
            return [y[1], -(3/t+self.E_z(t, deriv=True)/(self.E_z(t)))*y[1]+y[0]*3*self.pars['Omega_m']/(2*t**5*(self.E_z(t))**2)]
        
        a = np.atleast_1d(a)
        y0 = [a0, f0]      #Initial conditions for D, dD/da
        tspan = (a0, 2)
        if solver == 'solve_ivp':
            sol = solve_ivp(dy_dt, tspan, y0, t_eval=a)
            D, f = sol.y
        else:
            sol = odeint(dy_dt, y0, tfirst=True, t=np.insert(a, 0, a0))
            D = sol[1:, 0]; f = sol[1:, 1]
        
        return D, f*a/D
    
    def growth_factor(self, a, a0=1e-3, solver='odeint'):
        return self.compute_growth(a, a0=a0, solver=solver)[0]
    
    def growth_rate(self, a, a0=1e-3, solver='odeint'):
        return self.compute_growth(a, a0=a0, solver=solver)[1]
    
