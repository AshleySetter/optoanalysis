from datahandling import DataObject
from datahandling import take_closest
import numpy as _np
import matplotlib.pyplot as _plt

class SimData(DataObject):
    """
    Creates an object containing some simulated data and all it's properties.
    
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
        SignalFreqs : ndarray
                Contains the frequencies of the signals present in the signal
        Amplitudes : ndarray
                Contains the amplitudes of the signals present in the signal
        NoiseStdDev : float
                The standard deviation of the noise present in the signal
        Noise : ndarray
                The array containing the noise signal with time
        TimeTuple : tuple
                The start and stop time of the generated data 
        TrueSignals : dict
                Dictionary containing the clean signals. The keys are the frequencies
                of the signals and the values are the ndarrays containing the signal 
                values with time.
        
    """
    def __init__(self, SampleFreq, SignalFreqs, Amplitudes, NoiseStdDev, TimeTuple, MeanFreeTime):
        """
        Initialises the object by generating the data and calculating the PSD.
        """
        self.SampleFreq = SampleFreq
        self.SignalFreqs = _np.array(SignalFreqs)
        self.Amplitudes = _np.array(Amplitudes)
        self.NoiseStdDev = NoiseStdDev
        self.TimeTuple = (TimeTuple[0], TimeTuple[1])
        self.MeanFreeTime = MeanFreeTime
        self.generate_simulated_data()
        self.get_PSD()
        return None

    def get_time_data(self):
        """ 
        Returns the time and voltage data.

        Returns
        -------
        time : ndarray
                Contains the time data in seconds
        voltage : ndarray
                Contains the voltage data in Volts - with noise and clean signals
                all added together
        """
        return self.time, self.voltage

    def generate_simulated_data(self):
        if self.MeanFreeTime == None:
            self.generate_simulated_data_no_phase_noise()
        else:
            self.generate_simulated_data_with_phase_noise()
    
    def generate_simulated_data_no_phase_noise(self):
        """
        Generates the simulated data (several sine waves with noise).
        """
        Ts = 1/self.SampleFreq
        self.time = _np.arange(self.TimeTuple[0], self.TimeTuple[1], Ts)
        self.TrueSignals = {}
        for Freq in self.SignalFreqs:
            w = 2*_np.pi*Freq
            self.TrueSignals[Freq] = _np.sin(w*self.time)
        self.Noise = _np.random.normal(0, self.NoiseStdDev, len(self.time))
        self.voltage = _np.copy(self.Noise)
        for signal in [self.TrueSignals[key] for key in self.TrueSignals]:
            self.voltage += signal
        return None

    def generate_simulated_data_with_phase_noise(self):
        """
        Generates the simulated data (several sine waves with noise)
        with phase noise.
        """
        Ts = 1/self.SampleFreq
        self.time = _np.arange(self.TimeTuple[0], self.TimeTuple[1], Ts)
        self.TrueSignals = {}
        for FreqIndex, Freq in enumerate(self.SignalFreqs):
            w = 2*_np.pi*Freq
            TrueSignal = []
            Phase = 0
            TSinceLastPhaseChange = 0
            TimeForNextPhaseChange = self.MeanFreeTime + _np.random.normal(0, self.MeanFreeTime/2.0)
            for t in self.time:
                if TSinceLastPhaseChange > TimeForNextPhaseChange:
                    Phase = _np.random.uniform(-180, 180)
                    TSinceLastPhaseChange = 0
                    TimeForNextPhaseChange = self.MeanFreeTime + _np.random.normal(0, self.MeanFreeTime/2.0)
                TrueSignal.append(self.Amplitudes[FreqIndex]*_np.sin(w*t+Phase))
                TSinceLastPhaseChange += Ts
            self.TrueSignals[Freq] = _np.array(TrueSignal)
        self.Noise = _np.random.normal(0, self.NoiseStdDev, len(self.time))
        self.voltage = _np.copy(self.Noise)
        for signal in [self.TrueSignals[key] for key in self.TrueSignals]:
            self.voltage += signal
        return None

    
    def sim_multi_plot(self, timeLimits="Default", ShowFig=True):
        """
        Plots the full signal, noise signal, and each true signal in 1 figure
        inside of subplots.
        """
        if timeLimits == "Default":
            timeLimits = [self.time[0], self.time[-1]]
        lowerIndex = self.time.tolist().index(takeClosest(self.time, timeLimits[0]))
        upperIndex = self.time.tolist().index(takeClosest(self.time, timeLimits[1]))
        fig = _plt.figure()
        NumPlots = len(self.TrueSignals) + 2
        axList = []
        ax = fig.add_subplot('{}1{}'.format(NumPlots, 0))
        ax.plot(self.time[lowerIndex:upperIndex], self.voltage[lowerIndex:upperIndex])
        ax.set_title("Total Data")
        axList.append(ax)        
        ax = fig.add_subplot('{}1{}'.format(NumPlots, 1))
        ax.plot(self.time[lowerIndex:upperIndex], self.Noise[lowerIndex:upperIndex])
        ax.set_title("Noise Data")
        axList.append(ax)
        for i, freq in enumerate(self.TrueSignals):
            ax = fig.add_subplot('{}1{}'.format(NumPlots, i+2))
            ax.plot(self.time[lowerIndex:upperIndex], self.TrueSignals[freq][lowerIndex:upperIndex])
            ax.set_title("Data for {} Hz".format(freq))
            axList.append(ax)
        if ShowFig == True:
            _plt.show()
        return fig, ax

