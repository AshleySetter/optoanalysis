import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import least_squares, curve_fit

def steady_state_potential(xdata,HistBins=100):
    """ 
    Calculates the steady state potential.

    Parameters
    ----------
    xdata : ndarray
        Position data for a degree of freedom
    HistBins : int
        Number of bins to use for histogram
        of xdata. Number of position points
        at which the potential is calculated.

    Returns
    -------
    position : ndarray
        positions at which potential has been 
        calculated
    potential : ndarray
        value of potential at the positions above
    
    """  
    import numpy as np
    
    pops=np.histogram(xdata,HistBins)[0]
    bins=np.histogram(xdata,HistBins)[1]
    bins=bins[0:-1]
    bins=bins+np.mean(np.diff(bins))
    
    #normalise pops
    pops=pops/float(np.sum(pops))
    
    return bins,-np.log(pops)

def dynamical_potential(xdata, dt, order=3):
    """
    Computes potential from spring function

    Parameters
    ----------
    xdata : ndarray
        Position data for a degree of freedom,
        at which to calculate potential
    dt : float
        time between measurements
    order : int
        order of polynomial to fit

    Returns
    -------
    Potential : ndarray
        valued of potential at positions in
        xdata

    """
    import numpy as np
    adata = CalcAcceleration(xdata, dt)
    xdata = xdata[2:] # removes first 2 values as differentiating twice means
    # we have acceleration[n] corresponds to position[n-2]
    
    z=np.polyfit(xdata,adata,order)
    p=np.poly1d(z)
    spring_pot=np.polyint(p)
    return -spring_pot

def CalcAcceleration(xdata, dt):
    """
    Calculates the acceleration from the position
    
    Parameters
    ----------
    xdata : ndarray
        Position data
    dt : float
        time between measurements

    Returns
    -------
    acceleration : ndarray
        values of acceleration from position 
        2 to N.

    """
    acceleration = np.diff(np.diff(xdata))/dt**2
    return acceleration

import scipy.constants

def FitRadius(z, SampleFreq, Damping, HistBins=100):
    """
    Fits the dynamical potential to the Steady 
    State Potential by varying the Radius.
    
    z : ndarray
        Position data
    SampleFreq : float
        frequency at which the position data was 
        sampled
    Damping : float
        value of damping (in radians/second)
    HistBins : int
        number of values at which to evaluate 
        the steady state potential / perform
        the fitting to the dynamical potential

    Returns
    -------
    Radius : float
        Radius of the nanoparticle
    RadiusError : float
        One Standard Deviation Error in the Radius from the Fit
        (doesn't take into account possible error in damping)
    """
    dt = 1/SampleFreq
    boltzmann=scipy.constants.Boltzmann
    temp=300 # why halved??
    density=1800
    SteadyStatePotnl = list(steady_state_potential(z, HistBins=HistBins))
    yoffset=min(SteadyStatePotnl[1])
    SteadyStatePotnl[1] -= yoffset

    SpringPotnlFunc = dynamical_potential(z, dt)
    SpringPotnl = SpringPotnlFunc(z)
    kBT_Gamma = temp*boltzmann*1/Damping
    
    #FitSoln = least_squares(GetResiduals, 50, args=(SteadyStatePotnl, SpringPotnlFunc, kBT_Gamma), full_output=True)
    #print(FitSoln)
    #RADIUS = FitSoln['x'][0]

    DynamicPotentialFunc = MakeDynamicPotentialFunc(kBT_Gamma, density, SpringPotnlFunc)
    FitSoln = curve_fit(DynamicPotentialFunc, SteadyStatePotnl[0], SteadyStatePotnl[1], p0 = 50)
    print(FitSoln)
    popt, pcov = FitSoln
    perr = np.sqrt(np.diag(pcov))
    Radius, RadiusError = popt[0], perr[0]

    mass=((4/3)*np.pi*((Radius*10**-9)**3))*density
    yfit=(kBT_Gamma/mass)
    Y = yfit*SpringPotnl
    
    fig, ax = plt.subplots()
    ax.plot(SteadyStatePotnl[0], SteadyStatePotnl[1], 'bo', label="Steady State Potential")
    plt.plot(z,Y, 'r-', label="Dynamical Potential")
    ax.legend(loc='best')
    ax.set_ylabel('U ($k_{B} T $ Joules)')
    ax.set_xlabel('Distance (mV)')
    plt.tight_layout()
    plt.show()
    return Radius, RadiusError

def GetResiduals(Radius, SteadyStatePotnl, SpringPotnlFunc, kBT_Gamma):
    density=1800
    mass = ((4/3)*np.pi*((Radius*10**-9)**3))*density
    yfit=(kBT_Gamma/mass)
    ZSteadyState = SteadyStatePotnl[0]
    Y = yfit*SpringPotnlFunc(ZSteadyState)
    Residuals = SteadyStatePotnl[1] - Y
    return Residuals

def MakeDynamicPotentialFunc(kBT_Gamma, density, SpringPotnlFunc):
    """
    Creates the function that calculates the potential given
    the position (in volts) and the radius of the particle. 

    Parameters
    ----------
    kBT_Gamma : float
        Value of kB*T/Gamma
    density : float
        density of the nanoparticle
    SpringPotnlFunc : function
        Function which takes the value of position (in volts)
        and returns the spring potential
    
    Returns
    -------
    PotentialFunc : function
        function that calculates the potential given
        the position (in volts) and the radius of the 
        particle.

    """
    def PotentialFunc(xdata, Radius):
        """
        calculates the potential given the position (in volts) 
        and the radius of the particle.

        Parameters
        ----------
        xdata : ndarray
            Positon data (in volts)
        Radius : float
            Radius in units of nm

        Returns
        -------
        Potential : ndarray
            Dynamical Spring Potential at positions given by xdata
        """
        mass = ((4/3)*np.pi*((Radius*10**-9)**3))*density
        yfit=(kBT_Gamma/mass)
        Y = yfit*SpringPotnlFunc(xdata)
        return Y
    return PotentialFunc

import optoanalysis as oa 

dat = oa.load_data('testData.raw')
w0, A, G, _, _ = dat.get_fit_auto(70e3)
gamma = G.n
print(gamma)
z, t, _, _ = dat.filter_data(w0.n/(2*np.pi), 3, 20e3)
SampleFreq = dat.SampleFreq/3

R = FitRadius(z, SampleFreq, Damping=gamma, HistBins=120)

print(R)

