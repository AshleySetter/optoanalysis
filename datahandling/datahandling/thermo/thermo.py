import datahandling
import numpy as _np
import scipy as _scipy
from scipy import constants
try:
    from numba import jit as _jit
except (OSError, ModuleNotFoundError) as e:
    def _jit(func):        
        return func
    print("Numba not present on system, not importing jit")


class ThermoObject(datahandling.DataObject):
    """
    Creates an object containing some data and all it's properties
    for thermodynamics analysis.
    
    Attributes
    ----------
        SampleFreq : float
                The sample frequency used in generating the data.
        time : ndarray
                Contains the time data in seconds
        voltage : ndarray
                Contains the voltage data in Volts - with noise and clean signals
                all added together
        SampleFreq : sample frequency used to sample the data (when it was
                taken by the oscilloscope)
        freqs : ndarray
                Contains the frequencies corresponding to the PSD (Pulse Spectral
                Density)
        PSD : ndarray
                Contains the values for the PSD (Pulse Spectral Density) as calculated
                at each frequency contained in freqs        
    """
    def __init__(self, filepath, RelativeChannelNo=None):
        """
        Initialises the object by generating the data and calculating the PSD.
        """
        super(ThermoObject, self).__init__(filepath, RelativeChannelNo) # calls the init func from datahandling
        return None

   # @_jit
    def calc_hamiltonian(self, mass, omega_array):
        """
        Calculates the standard (pot+kin) Hamiltonian of your system.
        
        Parameters
        ----------
        mass : float
            The mass of the particle in kg
        omega_array : array
            array which represents omega at every point in your time trace
            and should therefore have the same length as self.position_data
        
        Requirements
        ------------
        self.position_data : array
            Already filtered for the degree of freedom of intrest and converted into meters. 

        Returns
        -------
        Hamiltonian : array
            The calculated Hamiltonian
        """
        Kappa_t= mass*omega_array**2
        self.E_pot = 0.5*Kappa_t*self.position_data**2
        self.E_kin = 0.5*mass*(_np.insert(_np.diff(self.position_data), 0, (self.position_data[1]-self.position_data[0]))*self.SampleFreq)**2
        self.Hamiltonian = self.E_pot + self.E_kin
        return self.Hamiltonian

  #  @_jit
    def calc_phase_space_density(self, mass, omega_array, temperature_array):
        """
        Calculates the partition function of your system at each point in time.
    
        Parameters
        ----------
        mass : float
            The mass of the particle in kg
        omega_array : array
            array which represents omega at every point in your time trace
            and should therefore have the same length as the Hamiltonian
        temperature_array : array
            array which represents the temperature at every point in your time trace
            and should therefore have the same length as the Hamiltonian
        
        Requirements
        ------------
        self.position_data : array
            Already filtered for the degree of freedom of intrest and converted into meters. 

        Returns:
        -------
        Phasespace-density : array
            The Partition Function at every point in time over a given trap-frequency and temperature change.
        """

        return self.calc_hamiltonian(mass, omega_array)/calc_partition_function(mass, omega_array,temperature_array)

 #   @_jit
    def extract_thermodynamic_quantities(self,temperature_array):
        """
        Calculates the thermodynamic quantities of your system at each point in time.
        Calculated Quantities: self.Q (heat),self.W (work), self.Delta_E_kin, self.Delta_E_pot
        self.Delta_E (change of Hamiltonian),
    
        Parameters
        ----------
        temperature_array : array
            array which represents the temperature at every point in your time trace
            and should therefore have the same length as the Hamiltonian
        
        Requirements
        ------------
        execute calc_hamiltonian on the DataObject first

        Returns:
        -------
        Q : array
            The heat exchanged by the particle at every point in time over a given trap-frequency and temperature change.
        W : array
            The work "done"  by the particle at every point in time over a given trap-frequency and temperature change.
        """
        beta = 1/(_scipy.constants.Boltzmann*temperature_array)
        self.Q = self.Hamiltonian*(_np.insert(_np.diff(beta),0,beta[1]-beta[0])*self.SampleFreq)
        self.W = self.Hamiltonian-self.Q
        self.Delta_E_kin = _np.diff(self.E_kin)*self.SampleFreq
        self.Delta_E_pot = _np.diff(self.E_pot)*self.SampleFreq
        self.Delta_E = _np.diff(self.Hamiltonian)*self.SampleFreq
        
        return self.Q, self.W

#@_jit
def calc_partition_function(mass, omega_array, temperature_array):
    """
    Calculates the partition function of your system at each point in time.

    Parameters
    ----------
    mass : float
        The mass of the particle in kg
    omega_array : array
        array which represents omega at every point in your time trace
        and should therefore have the same length as the Hamiltonian
    temperature_array : array
        array which represents the temperature at every point in your time trace
        and should therefore have the same length as the Hamiltonian

    Returns:
    -------
    Partition function : array
        The Partition Function at every point in time over a given trap-frequency and temperature change.
    """
    Kappa_t= mass*omega_array**2    
    return _np.sqrt(4*_np.pi**2*_scipy.constants.Boltzmann**2*temperature_array**2/(mass*Kappa_t))
        
    
@_jit
def calc_entropy(phase_space_density_array):
    """
    Calculates the entropy of your system at each point in time 
    for your given phase space density evolution in time.
    
    Parameters
    ----------
    phase_space_density_array : array
        array which represents the phase space density at every point in time
     Returns:
    -------
    entropy : array
        The entropy of the particle at every point in time via the phase space density method.  
    """
    entropy = -_scipy.constants.Boltzmann*_np.log(phase_space_density_array)
    return entropy
