import datahandling

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
