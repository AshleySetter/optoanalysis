from DataHandling import DataObject
from DataHandling import takeClosest
import numpy as _np
import matplotlib.pyplot as _plt

class SimData(DataObject):
    """
    Creates an object containing some simulated data and all it's properties.
    
    Attributes
    ----------
        time : ndarray
                Contains the time data in seconds
        Voltage : ndarray
                Contains the voltage data in Volts
        SampleFreq : sample frequency used to sample the data (when it was
                taken by the oscilloscope)
        freqs : ndarray
                Contains the frequencies corresponding to the PSD (Pulse Spectral
                Density)
        PSD : ndarray
                Contains the values for the PSD (Pulse Spectral Density) as calculated
                at each frequency contained in freqs    
    """
    def __init__(self, SampleFreq, Freqs, Amplitudes, NoiseStdDev, TimeTuple):
        self.SampleFreq = SampleFreq
        self.Freqs = Freqs
        self.Amplitudes = Amplitudes
        self.NoiseStdDev = NoiseStdDev
        self.TimeTuple = TimeTuple
        self.GenerateSimulatedData()
        self.getTimeData()
        self.getPSD()
        return None

    def getTimeData(self):
        return self.time, self.Voltage

    def GenerateSimulatedData(self):
        Ts = 1/self.SampleFreq
        self.time = _np.arange(self.TimeTuple[0], self.TimeTuple[1], Ts)
        self.TrueSignals = {}
        for Freq in self.Freqs:
            w = 2*_np.pi*Freq
            self.TrueSignals[Freq] = _np.sin(w*self.time)
        self.Noise = _np.random.normal(0, self.NoiseStdDev, len(self.time))
        self.Voltage = self.Noise
        for signal in [self.TrueSignals[key] for key in self.TrueSignals]:
            self.Voltage += signal
        return None

    def SimMultiPlot(self, timeLimits="Default", ShowFig=True):
        if timeLimits == "Default":
            timeLimits = [self.time[0], self.time[-1]]
        lowerIndex = self.time.tolist().index(takeClosest(self.time, timeLimits[0]))
        upperIndex = self.time.tolist().index(takeClosest(self.time, timeLimits[1]))
        fig = _plt.figure()
        NumPlots = len(self.TrueSignals) + 2
        axList = []
        ax = fig.add_subplot('{}1{}'.format(NumPlots, 0))
        ax.plot(self.time[lowerIndex:upperIndex], self.Voltage[lowerIndex:upperIndex])
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

