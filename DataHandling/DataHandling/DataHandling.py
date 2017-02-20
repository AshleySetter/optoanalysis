import DataHandling.LeCroy
import matplotlib.pyplot as _plt
import numpy as _np
import scipy.signal
from bisect import bisect_left as _bisect_left
from scipy.optimize import curve_fit as _curve_fit
import uncertainties as _uncertainties
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
import matplotlib.animation as _animation
from glob import glob
import re

def LoadData(Filepath):
    """
    Parameters
    ----------
        Filepath : string
            filepath to the file containing the data used to initialise 
            and create an instance of the DataObject class

    Returns
    -------
        Data : DataObject
            An instance of the DataObject class contaning the data
            that you requested to be loaded.
    """
    return DataObject(Filepath)
    
def MultiLoad(DirectoryPath, Channels, RunNos, RepeatNos):
    """
    This function uses ReGeX to search the direcory provided as DirectoryPath
    for files matching the specifications given in the arguments. Data
    loaded from this function will have additional properties identifying it:
    ChannelNo - The Channel Number
    RunNo - The Run Number
    RepeatNo - The Repeat Number

    Parameters
    ----------
    Channels : list
        The channel numbers you want to load in the form [Lower, Higher]
    RunNos : list
        The run nubmers you want to load in the form [Lower, Higher]
    RepeatNos : list
        The repeat numbers you want to load in the form [Lower, Higher]
    Returns
    -------
    DataList : list
        A list containing the instances of the DataObject class 
        contaning the data that you requested to be loaded.
        Data loaded from this function will have additional 
        properties identifying it:
        ChannelNo - The Channel Number
        RunNo - The Run Number
        RepeatNo - The Repeat Number
    """
    if RepeatNos[1] > 9:
        raise NotImplementedError ("Repeat numbers of with 2 or more digits have not been implemented") 
    if Channels[1] > 9:
        raise NotImplementedError ("Channel numbers of with 2 or more digits have not been implemented")
    
    if RunNos[1] < 10:
        REGEXPattern = "CH([{0}-{1}]+)_RUN0*([{2}-{3}])_REPEAT000([{4}-{5}])".format(Channels[0], Channels[1], RunNos[0], RunNos[1], RepeatNos[0], RepeatNos[1])
    if RunNos[1] > 9 and RunNos[1] < 100:
        if RunNos[0] > 9:
            lower1stDigit = int(str(RunNos[0])[0])
            lower2ndDigit = int(str(RunNos[0])[1])
        else:
            lower1stDigit = 0
            lower2ndDigit = int(str(RunNos[0]))
        higher1stDigit = str(RunNos[1])[0]
        higher2ndDigit = str(RunNos[1])[1]
        REGEXPattern = "CH([0-4]+)_RUN0*([0-9][0-9])_REPEAT000([0-5])"
        REGEXPattern = "CH([{0}-{1}]+)_RUN0*([{2}-{3}][{4}-{5}])_REPEAT000([{6}-{7}])".format(Channels[0], Channels[1], lower1stDigit, higher1stDigit, lower2ndDigit, higher2ndDigit, RepeatNos[0], RepeatNos[1])
        
    ListOfFiles = glob(DirectoryPath+'/*')
    ListOfMatchingFiles = []

    for Filepath in ListOfFiles:
        matchObj = re.search(REGEXPattern, Filepath)
        if matchObj != None:
            Data = LoadData(Filepath)
            Data.ChannelNo = matchObj.group(1)
            Data.RunNo = matchObj.group(2)
            Data.RepeatNo = matchObj.group(3)
            ListOfMatchingFiles.append(Data)
    return ListOfMatchingFiles


class DataObject():
    """
    Creates an object containing data and all it's properties. 

    Attributes
    ----------
	filepath : string
		filepath to the file containing the data used to initialise
		this particular instance of the DataObject class
	filename : string
		filename of the file containing the data used to initialise
		this particular instance of the DataObject class
	waveDescription : dictionary
		Contains various information about the data as it was collected.
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
	
	The following attributes are only assigned after getFit has been called.

	A : uncertainties.ufloat
		Fitting constant A
		A = γ**2*Γ_0*(K_b*T_0)/(π*m) 
		where:
			γ = conversionFactor
			Γ_0 = Damping factor due to environment
			π = pi
	Ftrap : uncertainties.ufloat
		Trapping frequency as determined from the fitting function
	Gamma : uncertainties.ufloat
		The damping factor Gamma = Γ = Γ_0 + δΓ
		where:
			Γ_0 = Damping factor due to environment
			δΓ = extra damping due to feedback
    """
    def __init__(self, filepath):
        """
	Parameters
	----------
	filepath : string
	    The filepath to the data file to initialise this object instance.

        Initialisation - assigns values to the following attributes:
        - filepath
        - filename
        - time
        - Voltage
        - freqs
        - PSD
        """
        self.filepath = filepath
        self.filename = filepath.split("/")[-1]
        self.getTimeData()
        self.getPSD()
        return None

    def getTimeData(self):
        """
        Gets the time and voltage data and the wave description.
        
        Returns
        -------
        time : ndarray
			array containing the value of time (in seconds) at which the
			voltage is sampled
        Voltage : ndarray 
			array containing the sampled voltages
        """
        f = open(self.filepath,'rb')
        raw = f.read()
        f.close()
        self.waveDescription, self.time, self.Voltage, _ = \
        	DataHandling.LeCroy.InterpretWaveform(raw)
        self.SampleFreq = (1/self.waveDescription["HORIZ_INTERVAL"])
        return self.time, self.Voltage
    
    def plotTimeData(self, ShowFig=True):
        """
        plot time data against voltage data.

        Parameters
        ----------
        ShowFig : bool, optional 
            If True runs plt.show() before returning figure
            if False it just returns the figure object.
            (the default is True, it shows the figure) 

        Returns
        -------
        fig : plt.figure
	    The figure object created
	ax : fig.add_subplot(111)
	    The subplot object created
        """
        fig = _plt.figure(figsize=[10, 6])
        ax = fig.add_subplot(111)        
        ax.plot(self.time, self.Voltage)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Voltage (V)")
        if ShowFig == True:
            _plt.show()
        return fig, ax
    
    def getPSD(self, NPerSegment=100000, window="hann"):
        """
        Extracts the pulse spectral density (PSD) from the data.

	Parameters
	----------
        NPerSegment : int, optional
            Length of each segment used in scipy.welch
            default =100000

        window : str or tuple or array_like, optional
            Desired window to use. See get_window for a list of windows 
            and required parameters. If window is array_like it will be 
            used directly as the window and its length will be used for 
            nperseg.
            default = "hann"

        Returns
        -------
        freqs : ndarray
        	Array containing the frequencies at which the PSD has been
        	calculated
        PSD : ndarray
        	Array containing the value of the PSD at the corresponding 
        	frequency value in V**2/Hz
        """
        self.freqs, self.PSD = scipy.signal.welch(self.Voltage, self.SampleFreq, 
                                window=window, nperseg=NPerSegment)
        return self.freqs, self.PSD
    
    def plotPSD(self, xlim, ShowFig=True):
        """
        plot the pulse spectral density.

        Parameters
        ----------
        xlim : array_like
            The x limits of the plotted PSD [LowerLimit, UpperLimit]
        ShowFig : bool, optional
            If True runs plt.show() before returning figure
            if False it just returns the figure object.
            (the default is True, it shows the figure)

        Returns
        -------
        fig : plt.figure
			The figure object created
		ax : fig.add_subplot(111)
			The subplot object created
        """
        self.getPSD()
        fig = _plt.figure(figsize=[10, 6])
        ax = fig.add_subplot(111)
        ax.semilogy(self.freqs, self.PSD, color="blue")
        ax.set_xlabel("Frequency Hz")
        ax.set_xlim(xlim)
        ax.grid(which="major")
        ax.set_ylabel("PSD ($v^2/Hz$)")
        if ShowFig == True:
            _plt.show()
        return  fig, ax

    def getFit(self, WidthOfPeakToFit, NMovAveToFit, TrapFreq, A_Initial, Gamma_Initial, Verbosity=1):
        """
        Function that fits peak to the PSD.

        Parameters
        ----------
	
	Returns
	-------
	A : uncertainties.ufloat
		Fitting constant A
		A = γ**2*Γ_0*(K_b*T_0)/(π*m) 
		where:
			γ = conversionFactor
			Γ_0 = Damping factor due to environment
			π = pi
	Ftrap : uncertainties.ufloat
		The trapping frequency in the z axis
	Gamma : uncertainties.ufloat
		The damping factor Gamma = Γ = Γ_0 + δΓ
		where:
			Γ_0 = Damping factor due to environment
			δΓ = extra damping due to feedback
        """
        Params, ParamsErr = fitPSD(self, WidthOfPeakToFit, NMovAveToFit, Verbosity, TrapFreq, A_Initial, Gamma_Initial)

        print("\n")
        print("A: {} +- {}% ".format(Params[0], ParamsErr[0]/Params[0]*100))
        print("Trap Frequency: {} +- {}% ".format(Params[1], ParamsErr[1]/Params[1]*100))
        print("Big Gamma: {} +- {}% ".format(Params[2], ParamsErr[2]/Params[2]*100))

        self.A = _uncertainties.ufloat(Params[0], ParamsErr[0])
        self.Ftrap = _uncertainties.ufloat(Params[1], ParamsErr[1])
        self.Gamma = _uncertainties.ufloat(Params[2], ParamsErr[2])

        return self.A, self.Ftrap, self.Gamma

#    def extractXYZMotion(self, [zf, xf, yf], uncertaintyInFreqs, PeakWidth, subSampleFraction):
#        """
#        Extracts the x, y and z signals (in volts) from the 
#        
#        """
#        zf, xf, yf = DataHandling.GetxyzFreqs(self, 64000, 160000, 185000, bandwidth=5000)
#        self.zVolts, self.xVolts, self.yVolts = DataHandling.getXYZData(self, zf, xf, yf, 2, zwidth=3000, xwidth=#3000, ywidth=3000)
#        return self.zVolts, self.xVolts, self.yVolts
        
    
def calcTemp(Data_ref, Data):
    T = 300*(Data.A/Data.Gamma)/(Data_ref.A/Data_ref.Gamma)
    return T

def fit_curvefit(p0, datax, datay, function, yerr=None, **kwargs):

    pfit, pcov = \
         _curve_fit(function,datax,datay,p0=p0,\
                            sigma=yerr, epsfcn=0.0001, **kwargs)
    error = [] 
    for i in range(len(pfit)):
        try:
            error.append(_np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    pfit_curvefit = pfit
    perr_curvefit = _np.array(error)
    return pfit_curvefit, perr_curvefit 

def moving_average(a, n=3) :
    ret = _np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = _bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

def PSD_Fitting(A, Omega0, gamma, omega):
    # Amp = amplitude
    # Omega0 = trapping (Angular) frequency
    # gamma = Big Gamma - damping (due to environment and feedback (if feedback is on))
    return 10*_np.log10(A/((Omega0**2-omega**2)**2 + (omega*gamma)**2))

def fitPSD(Data, bandwidth, NMovAve, verbosity, TrapFreqGuess, AGuess=0.1e10, GammaGuess=400):
    """
    Fits theory PSD to Data. Assumes highest point of PSD is the
    trapping frequency.
    
    Parameters
    ----------
    Data - data object to be fitted 
    bandwidth - bandwidth around trapping frequency peak to
                fit the theory PSD to
    NMovAve - amount of moving averages to take before the fitting
    verbosity - (defaults to 0) if set to 1 this function plots the
        PSD of the data, smoothed data and theory peak from fitting.
    
    Returns
    -------
    ParamsFit - Fitted parameters: 
        [A, TrappingFrequency, Gamma]
    ParamsFitErr - Error in fitted parameters: 
        [AErr, TrappingFrequencyErr, GammaErr]
        
    """    
    AngFreqs = 2*_np.pi*Data.freqs
    Angbandwidth = 2*_np.pi*bandwidth
    AngTrapFreqGuess = 2*_np.pi*TrapFreqGuess
    
    ClosestToAngTrapFreqGuess = takeClosest(AngFreqs, AngTrapFreqGuess)
    index_ftrap = _np.where(AngFreqs == ClosestToAngTrapFreqGuess)
    ftrap = AngFreqs[index_ftrap]
    
    f_fit_lower = takeClosest(AngFreqs, ftrap-Angbandwidth/2)
    f_fit_upper = takeClosest(AngFreqs, ftrap+Angbandwidth/2)
    
    indx_fit_lower = int(_np.where(AngFreqs==f_fit_lower)[0])
    indx_fit_upper = int(_np.where(AngFreqs==f_fit_upper)[0])

#    print(f_fit_lower, f_fit_upper)
#    print(AngFreqs[indx_fit_lower], AngFreqs[indx_fit_upper])

    index_ftrap = _np.where(Data.PSD == max(Data.PSD[indx_fit_lower:indx_fit_upper])) # find highest point in region about guess for trap frequency - use that as guess for trap frequency and recalculate region about the trap frequency

    ftrap = AngFreqs[index_ftrap]

#    print(ftrap)
    
    f_fit_lower = takeClosest(AngFreqs, ftrap-Angbandwidth/2)
    f_fit_upper = takeClosest(AngFreqs, ftrap+Angbandwidth/2)
    
    indx_fit_lower = int(_np.where(AngFreqs==f_fit_lower)[0])
    indx_fit_upper = int(_np.where(AngFreqs==f_fit_upper)[0])
    
    PSD_smoothed = moving_average(Data.PSD, NMovAve)
    freqs_smoothed = moving_average(AngFreqs, NMovAve) 
    
    logPSD_smoothed = 10*_np.log10(PSD_smoothed) 
    
    def CalcTheoryPSD_curvefit(freqs, A, TrapFreq, BigGamma):
        Theory_PSD = PSD_Fitting(A, TrapFreq, BigGamma, freqs)
        if A < 0 or TrapFreq < 0 or BigGamma < 0:
            return 1e9
        else:
            return Theory_PSD 
    
    datax = freqs_smoothed[indx_fit_lower:indx_fit_upper]
    datay = logPSD_smoothed[indx_fit_lower:indx_fit_upper]
    
    p0 = _np.array([AGuess, ftrap, GammaGuess])
    
    Params_Fit, Params_Fit_Err = fit_curvefit(p0,
                                      datax, datay, CalcTheoryPSD_curvefit)

    if verbosity == 1:
        #    print("Params Fitted:", Params_Fit, "Error in Params:", Params_Fit_Err)

        PSDTheory_fit_initial = PSD_Fitting(p0[0], p0[1], 
                                    p0[2], freqs_smoothed)

        PSDTheory_fit = PSD_Fitting(Params_Fit[0], Params_Fit[1], 
                                    Params_Fit[2], freqs_smoothed)
        
        _plt.plot(AngFreqs/(2*_np.pi), 10*_np.log10(Data.PSD), color="darkblue", label="Raw PSD Data", alpha=0.5)
        _plt.plot(freqs_smoothed/(2*_np.pi), logPSD_smoothed, color='blue', label="smoothed", linewidth=1.5)
        _plt.plot(freqs_smoothed/(2*_np.pi), PSDTheory_fit_initial, color="purple", label="initial")
        _plt.plot(freqs_smoothed/(2*_np.pi), PSDTheory_fit, color="red", label="fitted")
        _plt.xlim([(ftrap-5*Angbandwidth)/(2*_np.pi), (ftrap+5*Angbandwidth)/(2*_np.pi)])
        _plt.plot([(ftrap-Angbandwidth)/(2*_np.pi), (ftrap-Angbandwidth)/(2*_np.pi)],
                 [min(logPSD_smoothed), max(logPSD_smoothed)], '--',
                 color="grey")
        _plt.plot([(ftrap+Angbandwidth)/(2*_np.pi), (ftrap+Angbandwidth)/(2*_np.pi)],
                 [min(logPSD_smoothed), max(logPSD_smoothed)], '--',
                 color="grey")
        _plt.legend()
        _plt.show()
    return Params_Fit, Params_Fit_Err



def ExtractParameters(Pressure, A, AErr, Gamma0, Gamma0Err):
    """
    Calculates the radius, mass and conversion factor and thier uncertainties.
    For values to be correct data must have been taken with feedback off and
    at pressures of around 1mbar (this is because the equations assume
    harmonic motion and at lower pressures the uncooled particle experiences
    anharmonic motion (due to exploring furthur outside the middle of the trap).
    When cooled the value of Gamma (the damping) is a combination of the 
    enviromental damping and feedback damping and so is not the correct value
    for use in this equation (as it requires the enviromental damping). 
    Environmental damping can be predicted though as A=const*Gamma0. By 
    fitting to 1mbar data one can find the value of the const and therefore
    Gamma0 = A/const

    Parameters
    ----------
    Pressure : float
    	Pressure in mbar when the data was taken
    A : float
		Fitting constant A
		A = γ**2*Γ_0*(K_b*T_0)/(π*m) 
		where:
			γ = conversionFactor
			Γ_0 = Damping factor due to environment
			π = pi
    AErr : float
		Error in Fitting constant A
    Gamma0 : float
		The enviromental damping factor Gamma_0 = Γ_0
    Gamma0Err : float
    	The error in the enviromental damping factor Gamma_0 = Γ_0

    Returns:
    Params : list
    	[radius, mass, conversionFactor]
    ParamsError : list
    	[radiusError, massError, conversionFactorError]
    """
    PressureErr = 0.15
    Pressure = 100*Pressure # conversion to Pascals
    
    rho= 2200 # kgm^3
    dm = 0.372e-9 # m I'Hanlon, 2003
    T0 = 300 # kelvin 
    kB = 1.38e-23 # m^2 kg s^-2 K-1
    eta = 18.27e-6 # Pa s, viscosity of air 
    
    radius = (0.169*9*_np.pi*eta*dm**2)/(_np.sqrt(2)*rho*kB*T0)*(Pressure)/(Gamma0)
    err_radius = radius*_np.sqrt(((PressureErr*Pressure)/Pressure)**2+(Gamma0Err/Gamma0)**2);
    mass = rho*((4*_np.pi*radius**3)/3);
    err_mass = mass*2*err_radius/radius;
    conversionFactor = _np.sqrt(A*_np.pi*mass/(kB*T0*Gamma0));
    err_conversionFactor = conversionFactor*_np.sqrt((AErr/A)**2+(err_mass/mass)**2 + (Gamma0Err/Gamma0)**2);  
    
    return [radius, mass, conversionFactor], [err_radius, err_mass, err_conversionFactor]

def GetxyzFreqs(Data, zfreq, xfreq, yfreq, bandwidth=5000):
    """
    Determines the exact z, x and y peak frequencies from approximate
    frequencies by finding the highest peak in the PSD "close to" the 
    approximate peak frequency. By "close to" I mean within the range:
    approxFreq - bandwidth/2 to approxFreq + bandwidth/2
    
    Parameters
    ----------
    Data : DataObject
        DataObject containing the data for which you want to determine the
        z, x and y frequencies.
    zfreq : float
        An approximate frequency for the z peak
    xfreq : float
        An approximate frequency for the z peak    
    yfreq : float
        An approximate frequency for the z peak
    bandwidth : float
        The bandwidth around the approximate peak to look for the actual peak.
	
	Returns:
	trapfreqs : list
		List containing the trap frequencies in the following order (z, x, y)

    """
    trapfreqs = []
    for freq in [zfreq, xfreq, yfreq]:
        z_f_fit_lower = takeClosest(Data.freqs, freq-bandwidth/2)                                                                                                                                                                          
        z_f_fit_upper = takeClosest(Data.freqs, freq+bandwidth/2)
        z_indx_fit_lower = int(_np.where(Data.freqs==z_f_fit_lower)[0])                                                                                                                                                                           
        z_indx_fit_upper = int(_np.where(Data.freqs==z_f_fit_upper)[0]) 

        z_index_ftrap = _np.where(Data.PSD == max(Data.PSD[z_indx_fit_lower:z_indx_fit_upper])) 
        # find highest point in region about guess for trap frequency
        # use that as guess for trap frequency and recalculate region 
        # about the trap frequency
        z_ftrap = Data.freqs[z_index_ftrap]
        trapfreqs.append(z_ftrap)
    return trapfreqs

def getXYZData(Data, zf, xf, yf, FractionOfSampleFreq,
               zwidth=10000, xwidth=5000, ywidth=5000, 
               ztransition=10000, xtransition=5000, ytransition=5000,
               verbosity=True):
    """
    Given a Data object and the frequencies of the z, x and y peaks (and some
    optional parameters for the created filters) this function extracts the
    individual z, x and y signals (in volts) by creating IIR filters and filtering
    the Data.
    
    Parameters
    ----------
    Data : DataObject
        DataObject containing the data for which you want to extract the
        z, x and y signals.
    zf : float
        The frequency of the z peak in the PSD
    xf : float
        The frequency of the x peak in the PSD
    yf : float
        The frequency of the y peak in the PSD
    FractionOfSampleFreq : integer
        The fraction of the sample frequency to sub-sample the data by.
        This sometimes needs to be done because a filter with the appropriate
        frequency response may not be generated using the sample rate at which
        the data was taken. Increasing this number means the x, y and z signals
        produced by this function will be sampled at a lower rate but a higher 
        number means a higher chance that the filter produced will have a nice
        frequency response.
    zwidth : float
        The width of the pass-band of the IIR filter to be generated to 
        filter Z.
    xwidth : float
        The width of the pass-band of the IIR filter to be generated to 
        filter X.
    ywidth : float
        The width of the pass-band of the IIR filter to be generated to 
        filter Y.
    ztransition : float
        The width of the transition-band of the IIR filter to be generated to 
        filter Z.
    xtransition : float
        The width of the transition-band of the IIR filter to be generated to 
        filter X.
    ytransition : float
        The width of the transition-band of the IIR filter to be generated to 
        filter Y.
    verbosity : bool
        If True - plot unfiltered and filtered PSD for z, x and y.
        If False - don't plot anything

    Returns
    -------
    zdata : ndarray
	 	Array containing the z signal in volts with time.
   	xdata : ndarray
		Array containing the x signal in volts with time.
   	ydata : ndarray
		Array containing the y signal in volts with time.

    """
    SAMPLEFREQ = Data.SampleFreq/FractionOfSampleFreq

    bZ, aZ = IIRFilterDesign(zf, zwidth, ztransition, SAMPLEFREQ, GainStop=100)

    input_signal = Data.Voltage[0::FractionOfSampleFreq]
    zdata = scipy.signal.filtfilt(bZ, aZ, input_signal)

    bX, aX = IIRFilterDesign(xf, xwidth, xtransition, SAMPLEFREQ, GainStop=100)

    xdata = scipy.signal.filtfilt(bX, aX, input_signal)

    bY, aY = IIRFilterDesign(yf, ywidth, ytransition, SAMPLEFREQ, GainStop=100)

    ydata = scipy.signal.filtfilt(bY, aY, input_signal)
    
    if verbosity == True:
        f, PSD = scipy.signal.welch(input_signal, SAMPLEFREQ, nperseg=10000)
        f_z, PSD_z = scipy.signal.welch(zdata, SAMPLEFREQ, nperseg=10000)
        f_y, PSD_y = scipy.signal.welch(ydata, SAMPLEFREQ, nperseg=10000)
        f_x, PSD_x = scipy.signal.welch(xdata, SAMPLEFREQ, nperseg=10000)    
        _plt.plot(f, 10*_np.log10(PSD))
        _plt.plot(f_z, 10*_np.log10(PSD_z), label="z")
        _plt.plot(f_x, 10*_np.log10(PSD_x), label="x")
        _plt.plot(f_y, 10*_np.log10(PSD_y), label="y")
        _plt.legend(loc="best")
        _plt.xlim([zf-zwidth-ztransition, yf+ywidth+ytransition])
        _plt.show()
        
    return zdata, xdata, ydata

def animate(zdata, xdata, ydata, 
            conversionFactor, SampleFreq, FractionOfSampleFreq, 
            timeSteps=100, filename="particle"):
    """
    Animates the particle's motion given the z, x and y signal (in Volts)
    and the conversion factor (to convert between V and nm).
    
    Parameters
    ----------
    zdata : ndarray
        Array containing the z signal in volts with time.
    xdata : ndarray
        Array containing the x signal in volts with time.
    ydata : ndarray
        Array containing the y signal in volts with time.
    conversionFactor : float
        conversion factor (in units of Volts/Metre)
    SampleFreq : float
        The frequency at which the original data was sampled
    FractionOfSampleFreq : int
        The fraction of the sample frequency used to sub-sample the data by.
        This would have been used in the getXYZData function.
    timeSteps : int
        Number of time steps to animate 
    filename : string
        filename to create the mp4 under ({filename}.mp4)
    
    """
    timePerFrame = 0.203
    print("This will take ~ {} minutes".format(timePerFrame*timeSteps/60))

    SAMPLEFREQ = SampleFreq/FractionOfSampleFreq
    conv = conversionFactor*1e-9
    ZBoxStart = 1/conv*(_np.mean(zdata)-0.06)
    ZBoxEnd = 1/conv*(_np.mean(zdata)+0.06)
    XBoxStart = 1/conv*(_np.mean(xdata)-0.06)
    XBoxEnd = 1/conv*(_np.mean(xdata)+0.06)
    YBoxStart = 1/conv*(_np.mean(ydata)-0.06)
    YBoxEnd = 1/conv*(_np.mean(ydata)+0.06)

    FrameInterval = 1 # how many timesteps = 1 frame in animation

    a = 20
    b = 0.6*a
    myFPS = 7
    myBitrate = 1e6

    fig = _plt.figure(figsize = (a,b))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("{} us".format(1/SAMPLEFREQ*1e6*0))
    ax.set_xlabel('X (nm)')
    ax.set_xlim([XBoxStart,XBoxEnd])
    ax.set_ylabel('Y (nm)')
    ax.set_ylim([YBoxStart,YBoxEnd])
    ax.set_zlabel('Z (nm)')
    ax.set_zlim([ZBoxStart,ZBoxEnd])
    ax.view_init(20, -30)

    #ax.view_init(0, 0)

    def setup_plot():
        XArray = 1/conv*xdata[0]
        YArray = 1/conv*ydata[0]
        ZArray = 1/conv*zdata[0]
        scatter = ax.scatter(XArray, YArray, ZArray)
        return scatter,

    def animate(i):
        #print "\r {}".format(i),
        print("Frame: {}".format(i), end="\r")
        ax.clear()
        ax.view_init(20, -30)
        ax.set_title("{} us".format(1/SAMPLEFREQ*1e6*i))
        ax.set_xlabel('X (nm)')
        ax.set_xlim([XBoxStart,XBoxEnd])
        ax.set_ylabel('Y (nm)')
        ax.set_ylim([YBoxStart,YBoxEnd])
        ax.set_zlabel('Z (nm)')
        ax.set_zlim([ZBoxStart,ZBoxEnd])
        XArray = 1/conv*xdata[i]
        YArray = 1/conv*ydata[i]
        ZArray = 1/conv*zdata[i]
        scatter = ax.scatter(XArray, YArray, ZArray)
        ax.scatter([XArray], [0], [-ZBoxEnd], c='k', alpha=0.9)
        ax.scatter([-XBoxEnd], [YArray], [0], c='k', alpha=0.9)
        ax.scatter([0], [YBoxEnd], [ZArray], c='k', alpha=0.9)

        Xx, Yx, Zx, Xy, Yy, Zy, Xz, Yz, Zz = [], [], [], [], [], [], [], [], []

        for j in range(0, 30):

            Xlast = 1/conv*xdata[i-j]
            Ylast = 1/conv*ydata[i-j]
            Zlast = 1/conv*zdata[i-j]

            Alpha = 0.5-0.05*j
            if Alpha > 0:
                ax.scatter([Xlast], [0+j*10], [-ZBoxEnd], c='grey', alpha=Alpha)
                ax.scatter([-XBoxEnd], [Ylast], [0-j*10], c='grey', alpha=Alpha)
                ax.scatter([0-j*2], [YBoxEnd], [Zlast], c='grey', alpha=Alpha)

                Xx.append(Xlast)
                Yx.append(0+j*10)
                Zx.append(-ZBoxEnd)

                Xy.append(-XBoxEnd)
                Yy.append(Ylast)
                Zy.append(0-j*10)

                Xz.append(0-j*2)
                Yz.append(YBoxEnd)
                Zz.append(Zlast)

            if j < 15:
                XCur = 1/conv*xdata[i-j+1]
                YCur = 1/conv*ydata[i-j+1]
                ZCur = 1/conv*zdata[i-j+1]
                ax.plot([Xlast, XCur], [Ylast, YCur], [Zlast, ZCur], alpha=0.4)

        ax.plot_wireframe(Xx, Yx, Zx, color='grey')
        ax.plot_wireframe(Xy, Yy, Zy, color='grey')
        ax.plot_wireframe(Xz, Yz, Zz, color='grey')

        return scatter,

    anim = _animation.FuncAnimation(fig, animate, int(timeSteps/FrameInterval), init_func=setup_plot, blit=True)

    _plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    mywriter = _animation.FFMpegWriter(fps = myFPS, bitrate = myBitrate)
    anim.save('{}.mp4'.format(filename),writer=mywriter, fps = myFPS, bitrate = myBitrate)
    return None

def IIRFilterDesign(CentralFreq, bandwidth, transitionWidth, SampleFreq, GainStop=40, GainPass=0.01):
    """
    Function to calculate the coefficients of an IIR filter.
    
    Parameters
    ----------
    CentralFreq : float
        Central frequency of the IIR filter to be designed
    bandwidth : float
        The width of the passband to be created about the central frequency
    transitionWidth : float
        The width of the transition band between the pass-band and stop-band
    SampleFreq : float
        The sample frequency (rate) of the data to be filtered 
    GainStop : float
        The dB of attenuation within the stopband (i.e. outside the passband)
    GainPass : float
        The dB attenuation inside the passband (ideally close to 0 for a bandpass filter)
        
    Returns
    -------
    b : ndarray
        coefficients multiplying the current and past inputs (feedforward coefficients)
    a : ndarray
        coefficients multiplying the past outputs (feedback coefficients)
    """
    NyquistFreq = SampleFreq/2
    if (CentralFreq+bandwidth/2+transitionWidth > NyquistFreq):
        print("Need a higher Sample Frequency for this Central Freq, Bandwidth and transition Width")
        return 0, 0
    CentralFreqNormed = CentralFreq/NyquistFreq
    bandwidthNormed = bandwidth/NyquistFreq
    transitionWidthNormed = transitionWidth/NyquistFreq
    bandpass = [CentralFreqNormed-bandwidthNormed/2, CentralFreqNormed+bandwidthNormed/2]
    bandstop = [CentralFreqNormed-bandwidthNormed/2-transitionWidthNormed,
                CentralFreqNormed+bandwidthNormed/2+transitionWidthNormed]
    print(bandpass, bandstop)
    b, a = scipy.signal.iirdesign(bandpass, bandstop, GainPass, GainStop)
    return b, a


def GetFreqResponse(a, b, verbosity=1, SampleFreq=(2*_np.pi), NumOfFreqs=500, whole=False):
    """
    This function takes an array of coefficients and finds the frequency
    response of the filter using scipy.signal.freqz.
    Verbosity sets if the response should be plotted

    Parameters
    ----------
    a : array_like
        Coefficients multiplying the y values (outputs of the filter)
    b : array_like
        Coefficients multiplying the x values (inputs of the filter)
    verbosity : int
        Verbosity of function (i.e. whether to plot frequency and phase
        response or whether to just return the values.)
        Options (Default is 1):
        0 - Do not plot anything, just return values
        1 - Plot Frequency and Phase response and return values
    SampleFreq : float
        Sample frequency (in Hz) to simulate (used to convert frequency range
        to normalised frequency range)
    NumOfFreqs : int
        Number of frequencies to use to simulate the frequency and phase
        response of the filter. Default is 500.
    Whole : int (0 or 1)
        Sets whether to plot the whole response (0 to sample freq) 
        or just to plot 0 to Nyquist (SampleFreq/2):
        0 - plot 0 to Nyquist (SampleFreq/2)
        1 - plot the whole response (0 to sample freq)

    Returns
    -------
    freqList : ndarray
        Array containing the frequencies at which the gain is calculated
    GainArray : ndarray
        Array containing the gain in dB of the filter when simulated
        (20*log_10(A_out/A_in))
    PhaseDiffArray : ndarray
        Array containing the phase response of the filter - phase
        difference between the input signal and output signal at
        different frequencies
    """
    w, h = _scipy_signal.freqz(b=b, a=a, worN=NumOfFreqs, whole=whole)
    freqList = w/(_np.pi)*SampleFreq/2.0
    himag = _np.array([hi.imag for hi in h])
    GainArray = 20*_np.log10(_np.abs(h))
    PhaseDiffArray = _np.unwrap(_np.arctan2(_np.imag(h), _np.real(h)))
    if verbosity == 1:
        fig1 = _plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(freqList, GainArray, '-', label="Specified Filter")
        ax.set_title("Frequency Response")
        if SampleFreq == 2*_np.pi:
            ax.set_xlabel(("$\Omega$ - Normalized frequency "
                           "($\pi$=Nyquist Frequency)"))            
        else:
            ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_xlim([0, SampleFreq/2.0])
        fig2 = _plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(freqList, PhaseDiffArray, '-', label="Specified Filter")
        ax.set_title("Phase Response")
        if SampleFreq == 2*_np.pi:
            ax.set_xlabel(("$\Omega$ - Normalized frequency "
                           "($\pi$=Nyquist Frequency)"))            
        else:
            ax.set_xlabel("frequency (Hz)")

        ax.set_ylabel("Phase Difference")
        ax.set_xlim([0, SampleFreq/2.0])
        _plt.show()

    return freqList, GainArray, PhaseDiffArray

def MultiPlot(DataArray, xlim, LabelArray=[], ShowFig=True):
    """
    plot the pulse spectral density.

    Parameters
    ----------
    DataArray - array-like
        array of DataObject instances for which to plot the PSDs
    xlim - array-like
        2 element array specifying the lower and upper x limit for which to
        plot the Power Spectral Density
    LabelArray - array-like, optional
        array of labels for each data-set to be plotted
    ShowFig : bool, optional
       If True runs plt.show() before returning figure
       if False it just returns the figure object.
       (the default is True, it shows the figure) 

    Returns
    -------
    fig : plt.figure
        The figure object created
    ax : fig.add_subplot(111)
        The subplot object created
    """
    if LabelArray == []:
        LabelArray = ["DataSet {}".format(i) for i in _np.arange(0, len(DataArray), 1)]
    fig = _plt.figure(figsize=[10, 6])
    ax = fig.add_subplot(111)

    for i, data in enumerate(DataArray):
        ax.semilogy(data.freqs, data.PSD, alpha=0.8, label=LabelArray[i])
    ax.set_xlabel("Frequency Hz")
    ax.set_xlim(xlim)
    ax.grid(which="major")
    ax.legend(loc="best")
    ax.set_ylabel("PSD ($v^2/Hz$)")
    if ShowFig == True:
        _plt.show()
    return fig, ax
    
