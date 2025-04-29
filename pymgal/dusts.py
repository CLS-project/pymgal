import numpy as np
from pymgal import utils
from numba import njit, prange




# Speed up dust attenuation with numba since charlot_fall can be slow otherwise
@njit(parallel=True)
def charlot_fall_numba(ts1, ls1, tau1, tau2, tbreak):
    ls1 = np.reshape(ls1, (ls1.size, 1))
    ts1 = np.reshape(ts1, (1, ts1.size))

    taus = np.full(ts1.size, tau1)
    m = ts1.ravel() > tbreak
    if m.any():
        taus[m] = tau2
    return np.exp(-1.0 * taus * (ls1 / 5500.0) ** -0.7)


class charlot_fall(object):
    """ callable-object implementation of the Charlot and Fall (2000) dust law """
    tau1 = 0.0
    tau2 = 0.0
    tbreak = 0.0

    def __init__(self, tau1=1.0, tau2=0.5, tbreak=0.01):
        """ dust_obj = charlot_fall(tau1=1.0, tau2=0.3, tbreak=0.01)
        Return a callable object for returning the dimming factor as a function of age
        for a Charlot and Fall (2000) dust law.  The dimming is:

        np.exp(-1*Tau(t)(lambda/5500angstroms))

        Where Tau(t) = `tau1` for t < `tbreak` (in gyrs) and `tau2` otherwise. """

        self.tau1 = tau1
        self.tau2 = tau2
        self.tbreak = tbreak


    def __call__(self, ts, ls):
        ts1 = np.copy(ts)
        ls1 = np.copy(ls) 
        dust = charlot_fall_numba(ts1, ls1, self.tau1, self.tau2, self.tbreak)        
        return dust



# Calzetti seems to run faster without numba, but here is the implementation just in case
@njit(parallel=True)
def numba_calzetti(ls, esbv, rv):
    ks = np.zeros(ls.size)
    s = ls < .63
    if s.any():
        ks[s] = 2.659 * (-2.156 + 1.509 / ls[s] - 0.198 / ls[s]**2.0 +
                         0.011 / ls[s]**3.0) + rv
    l = ~s
    if l.any(): ks[l] = 2.659 * (-1.857 + 1.040 / ls[l]) + rv

    # calculate dimming factor as a function of lambda
    factors = 10.0**(-0.4 * esbv * ks)
    return factors



class calzetti(object):
    """ callable-object implementation of the Calzetti et al. (2000) dust law """
    av = 0.0
    rv = 0.0
    ebv = 0.0
    esbv = 0.0

    def __init__(self, av=1.0, rv=4.05):
        """ dust_obj = calzetti( av=1.0, rv=4.05 )
		Return a callable object for returning the dimming factor as a function of age
		for a Calzetti et al. (2000) dust law.  The dimming is:

		 """

        self.av = av
        self.rv = rv
        self.ebv = self.av / self.rv
        self.esbv = self.ebv * 0.44

    def __call__(self, ts, ls):

        # calzetti was fit in microns...
        ls = utils.convert_length(np.asarray(ls), incoming='a', outgoing='um')
        
        ks = np.zeros(ls.size)
        s = ls < .63
        if s.any():
            ks[s] = 2.659 * (-2.156 + 1.509 / ls[s] - 0.198 / ls[s]**2.0 +
                             0.011 / ls[s]**3.0) + self.rv
        l = ~s
        if l.any(): ks[l] = 2.659 * (-1.857 + 1.040 / ls[l]) + self.rv

        # calculate dimming factor as a function of lambda
        factors = 10.0**(-0.4 * self.esbv * ks)     #numba_calzetti(ls, self.esbv, self.rv)  # for numba use

        # need to return an array of shape (nls,nts).  Therefore, repeat
        return factors.reshape((ls.size, 1)).repeat(len(ts), axis=1)



@njit(parallel=True)   # Numba drastically speeds this up,
def gas_masses_los(g_xbins, g_ybins, g_depth, s_xbins, s_ybins, s_depth, gmass, kappa=None):
    """
    Compute the number of gas particles in front of each stellar particle along the line of sight using Numba.
    
    Parameters:
    g_xbins, g_ybins, g_depth: 1D arrays of gas particle coordinates
    s_xbins, s_ybins, s_depth: 1D arrays of stellar particle coordinates
    gmass: 1D array of gas particle masses
    kappa: 1D array of gas opacities. If None, the opacity will be ignored
    
    Returns:
    m_gas: 1D array where m_gas[i] is the total mass of gas particles in front of the i-th stellar particle.
    """
    m_gas = np.zeros(len(s_xbins), dtype=np.float64)
    avg_kappas = np.zeros(len(s_xbins), dtype=np.float64) if kappa is not None else None
    
    for i in prange(len(s_xbins)):
        count = 0
        mass_sum = 0.0
        kappa_sum = 0.0
        for j in range(len(g_xbins)):
            if g_xbins[j] == s_xbins[i] and g_ybins[j] == s_ybins[i] and g_depth[j] < s_depth[i]:
                count += 1
                mass_sum += gmass[j]
                if kappa is not None:
                    kappa_sum += kappa[j]
                    
        m_gas[i] = mass_sum
        if kappa is not None:
            avg_kappas[i] = kappa_sum / count if count > 0 else 0 # compute average kappa
    
    return m_gas, avg_kappas




class los_extinction(object):
    """
    Determine the effect of exctinction line of sight (LoS) extinction for a given projection
    The dimming goes like e^-tau, where tau = int (kappa * rho) ds for an optical depth tau, an opacity kappa, a density rho, and a line of sight s
    Assuming a constant kappa and rho, this just becomes tau = kappa * m_gas / A_pixel (where m_gas in the gas mass between a stellar particle and the observer and A_pixel is the area of a pixel)
    kappa can then vary based on contributions from electron scattering (kappa_e), Kramer's law (kappa_k) for bound-free/free-free interactions, negative hydrogen ion (kappa_h), and/or molecules such as H20 or CO (kappa_m)
    For more details on how to calculate kappa, see https://www.astro.princeton.edu/~gk/A403/opac.pdf
    IMPROTANT: kappa must be in cm^2/g, meaning that m_g is in kg and s is in cm 
    
    Parameters:
    kappa      :    A custom opacity. If this is not None, all other kappas (kappa_e, k, h, m) will be ignored, and the opacity will be set to this value
    use_kappa_e:    Do you want to consider opacity from electron scattering? (Important in highly ionized environments). If False, contributions from electron scattering are ignored
    use_kappa_k:    Do you want to consider opacity from Kramer's law for bound-free/free-free interactions? (Important in partially ionized environments). If False, contributions from Kramer's law are ignored
    use_kappa_h:    Do you want to consider opacity from the negative hydrogen ion? (Important in solar-like environments). If False, contributions from H^- are ignored are ignored
    use_kappa_m:    Do you want to consider opacity from molecules? (Important in cold environments). If False, contributions from molecules are ignored

    
    """
    def __init__(self, kappa="kappa_e"):
        self.kappa = kappa
        
        

    def get_kappa(self, temps, densities, metals, xH=0.76):
        temps = np.array(temps, dtype=np.float64)  # Increase floating point precision since this can get raised to high powers

        kappa_e = 0.2 * (1 + xH)
        kappa_k = 4e25 * metals * (1+xH) * densities * temps**(-3.5)
        kappa_m = 0.1 * metals  
        kappa_h = 2.5e-31 * metals * densities**0.5 * np.exp(9 * np.log(temps))

        kappa_arr = None    # initialize

        
        if isinstance(self.kappa, (int, float)):                                      # If the user provides a constant kappa
            kappa_arr = np.full(metals.shape, self.kappa, dtype=float)
        elif self.kappa == "kappa_e":                                                 # For electron scattering
            kappa_arr = np.full(metals.shape, kappa_e, dtype=float)
        elif self.kappa == "kappa_k":                                                 # For Kramer's opacity for free-free/bound-freee/bound-bound absorption
            kappa_arr = kappa_k                 
        elif self.kappa == "kappa_m":                                                 # For molecular opacity
            kappa_arr = kappa_m
        elif self.kappa == "kappa_h":                                                 # For negative hydrogen
            self.kappa = kappa_h
        elif self.kappa == "kappa_rad":                                               # For the combined radiative opacity (e+k+m+h) as per https://www.astro.princeton.edu/~gk/A403/opac.pdf
            kappa_h, kappa_e, kappa_k = [np.where(kappa == 0, np.inf, kappa) for kappa in [kappa_h, kappa_e, kappa_k]]   # Replace zero values with np.inf to prevent division by zero in the parallel opacity combination
            kappa_arr = kappa_m + 1/(1/kappa_h + 1/(kappa_e + kappa_k))   
        elif self.kappa == "kappa_d":
            raise ValueError("Line of sight exticntion with dust opacity is currently under development. Please select an alternate opacity source for now.")

        else:
            raise ValueError("Invalid opacity option. Must be either 'kappa_e' (electron scattering), 'kappa_k' (Kramer's law), 'kappa_m' (molecules), 'kappa_rad' (combined radiation), 'kappa_d' (dust), or a float/int (constant).")

        return kappa_arr


    def get_tau(self, kappa, m_gas, area):
        """
        Determine the optical depth associated with a stellar particle based on the gas mass between it and the observer
        Parameters:
        m_gas: An array of all gas masses in the line of sight of each stellar particle. IMPORTANT: This must be in grams
        area:  The area associated associated with the line of sight, which is equal to one pixel area. IMPORTANT: This must be in cm^2
        kappa: The opacity of the medium along the line of sight, usually derived from properties of gas particles. IMPORTANT: This must be in cm^2/g

        Returns:
        The optical depth tau = integral kappa*rho ds = kappa * m_gas / area
        """
        taus = np.zeros(len(m_gas), dtype=np.float64)
        #kappa = np.full(len(m_gas), 0.02)
        taus = np.where(area > 0, kappa * (m_gas / area), 0.0)

        return taus
