import optoanalysis
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
import seaborn as _sns
_sns.reset_orig()
import pandas as _pd
import fnmatch as _fnmatch
from multiprocessing import Pool as _Pool
from multiprocessing import cpu_count as _cpu_count
from scipy.optimize import minimize as _minimize
import warnings as _warnings
from scipy.signal import hilbert as _hilbert
import matplotlib as _mpl
from scipy.io.wavfile import write as _writewav
from matplotlib import colors as _mcolors
from matplotlib import cm as _cm
import qplots as _qplots
from functools import partial as _partial
from frange import frange
from scipy.constants import Boltzmann, pi
from os.path import exists as _does_file_exist
from skimage.transform import iradon_sart as _iradon_sart
from nptdms import TdmsFile as _TdmsFile
import gc
try:
    try:
        import pycuda.autoinit
        import pycuda.gpuarray as _gpuarray
    except (OSError, ModuleNotFoundError) as e:
        print("pyCUDA not present on system, function calc_fft_with_PyCUDA and calc_ifft_with_PyCUDA will crash")
    try:
        from skcuda.fft import fft as _fft
        from skcuda.fft import ifft as _ifft
        from skcuda.fft import Plan as _Plan
    except (OSError, ModuleNotFoundError) as e:
        print("skcuda not present on system, function calc_fft_with_PyCUDA and calc_ifft_with_PyCUDA will crash")
except NameError as e:
    pass # ModuleNotFoundError not always there
        
#cpu_count = _cpu_count()
#workerPool = _Pool(cpu_count)

_mpl.rcParams['lines.markeredgewidth'] = 1 # set default markeredgewidth to 1 overriding seaborn's default value of 0
_sns.set_style("whitegrid")

def GenCmap(basecolor, ColorRange, NumOfColors, logscale=False):
    if NumOfColors > 256:
        _warnings.warn("Maximum Number of colors is 256", UserWarning)
        NumOfColors = 256
    if logscale == True:
        colors = [_sns.set_hls_values(basecolor, l=l) for l in _np.logspace(ColorRange[0], ColorRange[1], NumOfColors)]
    else:
        colors = [_sns.set_hls_values(basecolor, l=l) for l in _np.linspace(ColorRange[0], ColorRange[1], NumOfColors)]
    cmap = _sns.blend_palette(colors, as_cmap=True, n_colors=NumOfColors)
    return cmap

color = "green"
colors = [_sns.set_hls_values(color, l=l) for l in _np.logspace(-0.01, -20, 100)]
logcmap = _sns.blend_palette(colors, as_cmap=True)

properties = {
    'default_fig_size': [6.5, 4],
    'default_linear_cmap': _sns.cubehelix_palette(n_colors=1024, light=1, as_cmap=True, rot=-.4),
    'default_log_cmap': GenCmap('green', [0, -60], 256, logscale=True),
    'default_base_color': 'green',
    }

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
    time : frange
            Contains the time data as an frange object. Can get a generator 
            or array of this object.
    voltage : ndarray
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

    def __init__(self, filepath, RelativeChannelNo=None, SampleFreq=None, PointsToLoad=-1, calcPSD=True, NPerSegmentPSD=1000000, NormaliseByMonitorOutput=False):
        """
        Parameters
        ----------
        filepath : string
            The filepath to the data file to initialise this object instance.
        RelativeChannelNo : int, optional
            If loading a .bin file produced by the Saleae datalogger, used to specify
            the channel number
            If loading a .dat file produced by the labview NI5122 daq card, used to 
            specifiy the channel number if two channels where saved, if left None with 
            .dat files it will assume that the file to load only contains one channel.
            If NormaliseByMonitorOutput is True then RelativeChannelNo specifies the 
            monitor channel for loading a .dat file produced by the labview NI5122 daq card.
        SampleFreq : float, optional
            If loading a .dat file produced by the labview NI5122 daq card, used to
            manually specify the sample frequency 
        PointsToLoad : int, optional
            Number of first points to read. -1 means all points (i.e. the complete file)
            WORKS WITH NI5122 DATA SO FAR ONLY!!!
        calcPSD : bool, optional
            Whether to calculate the PSD upon loading the file, can take some time
            off the loading and reduce memory usage if frequency space info is not required
        NPerSegmentPSD : int, optional
            NPerSegment to pass to scipy.signal.welch to calculate the PSD
        NormaliseByMonitorOutput : bool, optional
            If True the particle signal trace will be divided by the monitor output, which is
            specified by the channel number set in the RelativeChannelNo parameter. 
            WORKS WITH NI5122 DATA SO FAR ONLY!!!

        Initialisation - assigns values to the following attributes:
        - filepath
        - filename
        - filedir
        - time
        - voltage
        - freqs
        - PSD
        """
        self.filepath = filepath
        self.filename = filepath.split("/")[-1]
        self.filedir = self.filepath[0:-len(self.filename)]
        self.load_time_data(RelativeChannelNo,SampleFreq,PointsToLoad,NormaliseByMonitorOutput)
        if calcPSD != False:
            self.get_PSD(NPerSegmentPSD)
        return None

    def load_time_data(self, RelativeChannelNo=None, SampleFreq=None, PointsToLoad=-1, NormaliseByMonitorOutput=False):
        """
        Loads the time and voltage data and the wave description from the associated file.

        Parameters
        ----------
        RelativeChannelNo : int, optional
             Channel number for loading saleae data files
             If loading a .dat file produced by the labview NI5122 daq card, used to 
             specifiy the channel number if two channels where saved, if left None with 
             .dat files it will assume that the file to load only contains one channel.
             If NormaliseByMonitorOutput is True then RelativeChannelNo specifies the 
             monitor channel for loading a .dat file produced by the labview NI5122 daq card.
        SampleFreq : float, optional
             Manual selection of sample frequency for loading labview NI5122 daq files
        PointsToLoad : int, optional
             Number of first points to read. -1 means all points (i.e., the complete file)
             WORKS WITH NI5122 DATA SO FAR ONLY!!!
        NormaliseByMonitorOutput : bool, optional
             If True the particle signal trace will be divided by the monitor output, which is
             specified by the channel number set in the RelativeChannelNo parameter. 
             WORKS WITH NI5122 DATA SO FAR ONLY!!!
        """
        f = open(self.filepath, 'rb')
        raw = f.read()
        f.close()
        FileExtension = self.filepath.split('.')[-1]
        if FileExtension == "raw" or FileExtension == "trc":
            with _warnings.catch_warnings(): # supress missing data warning and raise a missing
                # data warning from optoanalysis with the filepath
                _warnings.simplefilter("ignore")
                waveDescription, timeParams, self.voltage, _, missingdata = optoanalysis.LeCroy.InterpretWaveform(raw, noTimeArray=True) 
            if missingdata:
                _warnings.warn("Waveform not of expected length. File {} may be missing data.".format(self.filepath))
            self.SampleFreq = (1 / waveDescription["HORIZ_INTERVAL"])
        elif FileExtension == "bin":
            if RelativeChannelNo == None:
                raise ValueError("If loading a .bin file from the Saleae data logger you must enter a relative channel number to load")
            timeParams, self.voltage = optoanalysis.Saleae.interpret_waveform(raw, RelativeChannelNo)
            self.SampleFreq = 1/timeParams[2]
        elif FileExtension == "dat": #for importing a file written by labview using the NI5122 daq card
            if SampleFreq == None:
                raise ValueError("If loading a .dat file from the NI5122 daq card you must enter a SampleFreq")
            if RelativeChannelNo == None:
                self.voltage = _np.fromfile(self.filepath, dtype='>h',count=PointsToLoad)
            elif RelativeChannelNo != None:
                filedata = _np.fromfile(self.filepath, dtype='>h',count=PointsToLoad)
                if NormaliseByMonitorOutput == True:
                    if RelativeChannelNo == 0:
                        monitorsignal = filedata[:len(filedata):2]
                        self.voltage = filedata[1:len(filedata):2]/monitorsignal
                    elif RelativeChannelNo == 1:
                        monitorsignal = filedata[1:len(filedata):2]
                        self.voltage = filedata[:len(filedata):2]/monitorsignal
                elif NormaliseByMonitorOutput == False:
                    self.voltage = filedata[RelativeChannelNo:len(filedata):2]
            timeParams = (0,(len(self.voltage)-1)/SampleFreq,1/SampleFreq)
            self.SampleFreq = 1/timeParams[2]
        elif FileExtension == "tdms": # for importing a file written by labview form the NI7961 FPGA with the RecordDataPC VI
            if SampleFreq == None:
                raise ValueError("If loading a .tdms file saved from the FPGA you must enter a SampleFreq")
            self.SampleFreq = SampleFreq
            dt = 1/self.SampleFreq
            FIFO_SIZE = 262143 # this is the maximum size of the DMA FIFO on the NI 7961 FPGA with the NI 5781 DAC card
            tdms_file = _TdmsFile(self.filepath)
            channel = tdms_file.object('Measured_Data', 'data')
            data = channel.data[FIFO_SIZE:] # dump first 1048575 points of data
            # as this is the values that had already filled the buffer
            # from before when the record code started running
            volts_per_unit = 2/(2**14)
            self.voltage = volts_per_unit*data
            timeParams = [0, (data.shape[0]-1)*dt, dt]
        startTime, endTime, Timestep = timeParams
        self.timeStart = startTime
        self.timeEnd = endTime
        self.timeStep = Timestep
        self.time = frange(startTime, endTime+Timestep, Timestep)
        return None

    def get_time_data(self, timeStart=None, timeEnd=None):
        """
        Gets the time and voltage data.

        Parameters
        ----------
        timeStart : float, optional
            The time get data from.
            By default it uses the first time point
        timeEnd : float, optional
            The time to finish getting data from.
            By default it uses the last time point        

        Returns
        -------
        time : ndarray
                        array containing the value of time (in seconds) at which the
                        voltage is sampled
        voltage : ndarray
                        array containing the sampled voltages
        """
        if timeStart == None:
            timeStart = self.timeStart
            
        if timeEnd == None:
            timeEnd = self.timeEnd

        time = self.time.get_array()
            
        StartIndex = _np.where(time == take_closest(time, timeStart))[0][0]
        EndIndex = _np.where(time == take_closest(time, timeEnd))[0][0]

        if EndIndex == len(time) - 1:
            EndIndex = EndIndex + 1 # so that it does not remove the last element

        return time[StartIndex:EndIndex], self.voltage[StartIndex:EndIndex]
    
    def plot_time_data(self, timeStart=None, timeEnd=None, units='s', show_fig=True):
        """
        plot time data against voltage data.

        Parameters
        ----------
        timeStart : float, optional
            The time to start plotting from.
            By default it uses the first time point
        timeEnd : float, optional
            The time to finish plotting at.
            By default it uses the last time point
        units : string, optional
            units of time to plot on the x axis - defaults to s
        show_fig : bool, optional
            If True runs plt.show() before returning figure
            if False it just returns the figure object.
            (the default is True, it shows the figure)

        Returns
        -------
        fig : matplotlib.figure.Figure object
            The figure object created
        ax : matplotlib.axes.Axes object
            The subplot object created
        """
        unit_prefix = units[:-1] # removed the last char
        if timeStart == None:
            timeStart = self.timeStart
        if timeEnd == None:
            timeEnd = self.timeEnd

        time = self.time.get_array()

        StartIndex = _np.where(time == take_closest(time, timeStart))[0][0]
        EndIndex = _np.where(time == take_closest(time, timeEnd))[0][0]

        fig = _plt.figure(figsize=properties['default_fig_size'])
        ax = fig.add_subplot(111)
        ax.plot(unit_conversion(time[StartIndex:EndIndex], unit_prefix),
                self.voltage[StartIndex:EndIndex])
        ax.set_xlabel("time ({})".format(units))
        ax.set_ylabel("voltage (V)")
        ax.set_xlim([timeStart, timeEnd])
        if show_fig == True:
            _plt.show()
        return fig, ax

    def get_PSD(self, NPerSegment=1000000, window="hann", timeStart=None, timeEnd=None, override=False):
        """
        Extracts the power spectral density (PSD) from the data.

        Parameters
        ----------
        NPerSegment : int, optional
            Length of each segment used in scipy.welch
            default = 1000000

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
        if timeStart == None and timeEnd == None:
            freqs, PSD = calc_PSD(self.voltage, self.SampleFreq, NPerSegment=NPerSegment)
            self.PSD = PSD
            self.freqs = freqs
        else:
            if timeStart == None:
                timeStart = self.timeStart
            if timeEnd == None:
                timeEnd = self.timeEnd

            time = self.time.get_array()
                
            StartIndex = _np.where(time == take_closest(time, timeStart))[0][0]
            EndIndex = _np.where(time == take_closest(time, timeEnd))[0][0]

            if EndIndex == len(time) - 1:
                EndIndex = EndIndex + 1 # so that it does not remove the last element
            freqs, PSD = calc_PSD(self.voltage[StartIndex:EndIndex], self.SampleFreq, NPerSegment=NPerSegment)
            if override == True:
                self.freqs = freqs
                self.PSD = PSD

        return freqs, PSD

    def plot_PSD(self, xlim=None, units="kHz", show_fig=True, timeStart=None, timeEnd=None, *args, **kwargs):
        """
        plot the pulse spectral density.

        Parameters
        ----------
        xlim : array_like, optional
            The x limits of the plotted PSD [LowerLimit, UpperLimit]
            Default value is [0, SampleFreq/2]
        units : string, optional
            Units of frequency to plot on the x axis - defaults to kHz
        show_fig : bool, optional
            If True runs plt.show() before returning figure
            if False it just returns the figure object.
            (the default is True, it shows the figure)

        Returns
        -------
        fig : matplotlib.figure.Figure object
            The figure object created
        ax : matplotlib.axes.Axes object
            The subplot object created
        """
        #        self.get_PSD()
        if timeStart == None and timeEnd == None:
            freqs = self.freqs
            PSD = self.PSD
        else:
            freqs, PSD = self.get_PSD(timeStart=timeStart, timeEnd=timeEnd)
            
        unit_prefix = units[:-2]
        if xlim == None:
            xlim = [0, unit_conversion(self.SampleFreq/2, unit_prefix)]
        fig = _plt.figure(figsize=properties['default_fig_size'])
        ax = fig.add_subplot(111)
        ax.semilogy(unit_conversion(freqs, unit_prefix), PSD, *args, **kwargs)
        ax.set_xlabel("Frequency ({})".format(units))
        ax.set_xlim(xlim)
        ax.grid(which="major")
        ax.set_ylabel("$S_{xx}$ ($V^2/Hz$)")
        if show_fig == True:
            _plt.show()
        return fig, ax

    def calc_area_under_PSD(self, lowerFreq, upperFreq):
        """
        Sums the area under the PSD from lowerFreq to upperFreq.

        Parameters
        ----------
        lowerFreq : float
            The lower limit of frequency to sum from
        upperFreq : float
            The upper limit of frequency to sum to

        Returns
        -------
        AreaUnderPSD : float
            The area under the PSD from lowerFreq to upperFreq
        """
        Freq_startAreaPSD = take_closest(self.freqs, lowerFreq)
        index_startAreaPSD = int(_np.where(self.freqs == Freq_startAreaPSD)[0][0])
        Freq_endAreaPSD = take_closest(self.freqs, upperFreq)
        index_endAreaPSD = int(_np.where(self.freqs == Freq_endAreaPSD)[0][0])
        AreaUnderPSD = sum(self.PSD[index_startAreaPSD: index_endAreaPSD])
        return AreaUnderPSD

    def get_fit(self, TrapFreq, WidthOfPeakToFit, A_Initial=0.1e10, Gamma_Initial=400, silent=False, MakeFig=True, show_fig=True):
        """
        Function that fits to a peak to the PSD to extract the 
        frequency, A factor and Gamma (damping) factor.

        Parameters
        ----------
        TrapFreq : float
            The approximate trapping frequency to use initially
            as the centre of the peak
        WidthOfPeakToFit : float
            The width of the peak to be fitted to. This limits the
            region that the fitting function can see in order to
            stop it from fitting to the wrong peak
        A_Initial : float, optional
            The initial value of the A parameter to use in fitting
        Gamma_Initial : float, optional
            The initial value of the Gamma parameter to use in fitting
        Silent : bool, optional
            Whether to print any output when running this function
            defaults to False
        MakeFig : bool, optional
            Whether to construct and return the figure object showing
            the fitting. defaults to True
        show_fig : bool, optional
            Whether to show the figure object when it has been created.
            defaults to True

        Returns
        -------
        A : uncertainties.ufloat
            Fitting constant A
            A = γ**2*2*Γ_0*(K_b*T_0)/(π*m)
            where:
            γ = conversionFactor
            Γ_0 = Damping factor due to environment
            π = pi
        OmegaTrap : uncertainties.ufloat
            The trapping frequency in the z axis (in angular frequency)
        Gamma : uncertainties.ufloat
            The damping factor Gamma = Γ = Γ_0 + δΓ
            where:
            Γ_0 = Damping factor due to environment
            δΓ = extra damping due to feedback or other effects
        fig : matplotlib.figure.Figure object
            figure object containing the plot
        ax : matplotlib.axes.Axes object
            axes with the data plotted of the:
            - initial data
            - smoothed data
            - initial fit
            - final fit

        """
        if MakeFig == True:
            Params, ParamsErr, fig, ax = fit_PSD(
                self, WidthOfPeakToFit, TrapFreq, A_Initial, Gamma_Initial, MakeFig=MakeFig, show_fig=show_fig)
        else:
            Params, ParamsErr, _ , _ = fit_PSD(
                self, WidthOfPeakToFit, TrapFreq, A_Initial, Gamma_Initial, MakeFig=MakeFig, show_fig=show_fig)

        if silent == False:
            print("\n")
            print("A: {} +- {}% ".format(Params[0],
                                         ParamsErr[0] / Params[0] * 100))
            print(
                "Trap Frequency: {} +- {}% ".format(Params[1], ParamsErr[1] / Params[1] * 100))
            print(
                "Big Gamma: {} +- {}% ".format(Params[2], ParamsErr[2] / Params[2] * 100))

        self.A = _uncertainties.ufloat(Params[0], ParamsErr[0])
        self.OmegaTrap = _uncertainties.ufloat(Params[1], ParamsErr[1])
        self.Gamma = _uncertainties.ufloat(Params[2], ParamsErr[2])

        
        if MakeFig == True:
            return self.A, self.OmegaTrap, self.Gamma, fig, ax
        else:
            return self.A, self.OmegaTrap, self.Gamma, None, None

    def get_fit_from_peak(self, lowerLimit, upperLimit, NumPointsSmoothing=1, silent=False, MakeFig=True, show_fig=True):
        """
        Finds approximate values for the peaks central frequency, height, 
        and FWHM by looking for the heighest peak in the frequency range defined 
        by the input arguments. It then uses the central frequency as the trapping 
        frequency, peak height to approximate the A value and the FWHM to an approximate
        the Gamma (damping) value.

        Parameters
        ----------
        lowerLimit : float
            The lower frequency limit of the range in which it looks for a peak
        upperLimit : float
            The higher frequency limit of the range in which it looks for a peak
        NumPointsSmoothing : float
            The number of points of moving-average smoothing it applies before fitting the 
            peak.
        Silent : bool, optional
            Whether it prints the values fitted or is silent.
        show_fig : bool, optional
            Whether it makes and shows the figure object or not.

        Returns
        -------
        OmegaTrap : ufloat
            Trapping frequency
        A : ufloat
            A parameter
        Gamma : ufloat
            Gamma, the damping parameter
        """
        lowerIndex = _np.where(self.freqs ==
            take_closest(self.freqs, lowerLimit))[0][0]
        upperIndex = _np.where(self.freqs ==
            take_closest(self.freqs, upperLimit))[0][0]

        if lowerIndex == upperIndex:
            _warnings.warn("range is too small, returning NaN", UserWarning)
            val = _uncertainties.ufloat(_np.NaN, _np.NaN)
            return val, val, val, val, val

        MaxPSD = max(self.PSD[lowerIndex:upperIndex])

        centralIndex = _np.where(self.PSD == MaxPSD)[0][0]
        CentralFreq = self.freqs[centralIndex]

        approx_A = MaxPSD * 1e16  # 1e16 was calibrated for a number of saves to be approximately the correct conversion factor between the height of the PSD and the A factor in the fitting

        MinPSD = min(self.PSD[lowerIndex:upperIndex])

        # need to get this on log scale
        HalfMax = MinPSD + (MaxPSD - MinPSD) / 2

        try:
            LeftSideOfPeakIndex = _np.where(self.PSD ==
                                            take_closest(self.PSD[lowerIndex:centralIndex], HalfMax))[0][0]
            LeftSideOfPeak = self.freqs[LeftSideOfPeakIndex]
        except IndexError:
            _warnings.warn("range is too small, returning NaN", UserWarning)
            val = _uncertainties.ufloat(_np.NaN, _np.NaN)
            return val, val, val, val, val

        try:
            RightSideOfPeakIndex = _np.where(self.PSD ==
                                             take_closest(self.PSD[centralIndex:upperIndex], HalfMax))[0][0]
            RightSideOfPeak = self.freqs[RightSideOfPeakIndex]
        except IndexError:
            _warnings.warn("range is too small, returning NaN", UserWarning)
            val = _uncertainties.ufloat(_np.NaN, _np.NaN)
            return val, val, val, val, val

        FWHM = RightSideOfPeak - LeftSideOfPeak

        approx_Gamma = FWHM/4
        try:
            A, OmegaTrap, Gamma, fig, ax \
                = self.get_fit(CentralFreq,
                               (upperLimit-lowerLimit)/2, 
                               A_Initial=approx_A,
                               Gamma_Initial=approx_Gamma,
                               silent=silent,
                               MakeFig=MakeFig,
                               show_fig=show_fig)
        except (TypeError, ValueError) as e: 
            _warnings.warn("range is too small to fit, returning NaN", UserWarning)
            val = _uncertainties.ufloat(_np.NaN, _np.NaN)
            return val, val, val, val, val
        OmegaTrap = self.OmegaTrap
        A = self.A
        Gamma = self.Gamma

        omegaArray = 2 * pi * \
            self.freqs[LeftSideOfPeakIndex:RightSideOfPeakIndex]
        PSDArray = self.PSD[LeftSideOfPeakIndex:RightSideOfPeakIndex]

        return OmegaTrap, A, Gamma, fig, ax 

    def get_fit_auto(self, CentralFreq, MaxWidth=15000, MinWidth=500, WidthIntervals=500, MakeFig=True, show_fig=True, silent=False):
        """
        Tries a range of regions to search for peaks and runs the one with the least error
        and returns the parameters with the least errors.

        Parameters
        ----------
        CentralFreq : float
            The central frequency to use for the fittings.
        MaxWidth : float, optional
            The maximum bandwidth to use for the fitting of the peaks.
        MinWidth : float, optional
            The minimum bandwidth to use for the fitting of the peaks.
        WidthIntervals : float, optional
            The intervals to use in going between the MaxWidth and MinWidth.
        show_fig : bool, optional
            Whether to plot and show the final (best) fitting or not.

        Returns
        -------
        OmegaTrap : ufloat
            Trapping frequency
        A : ufloat
            A parameter
        Gamma : ufloat
            Gamma, the damping parameter
        fig : matplotlib.figure.Figure object
            The figure object created showing the PSD of the data 
            with the fit
        ax : matplotlib.axes.Axes object
            The axes object created showing the PSD of the data 
            with the fit

        """
        MinTotalSumSquaredError = _np.infty
        for Width in _np.arange(MaxWidth, MinWidth - WidthIntervals, -WidthIntervals):
            try:
                OmegaTrap, A, Gamma,_ , _ \
                    = self.get_fit_from_peak(
                        CentralFreq - Width / 2,
                        CentralFreq + Width / 2,
                        silent=True,
                        MakeFig=False,
                        show_fig=False)
            except RuntimeError:
                _warnings.warn("Couldn't find good fit with width {}".format(
                    Width), RuntimeWarning)
                val = _uncertainties.ufloat(_np.NaN, _np.NaN)
                OmegaTrap = val
                A = val
                Gamma = val
            TotalSumSquaredError = (
                A.std_dev / A.n)**2 + (Gamma.std_dev / Gamma.n)**2 + (OmegaTrap.std_dev / OmegaTrap.n)**2
            #print("totalError: {}".format(TotalSumSquaredError))
            if TotalSumSquaredError < MinTotalSumSquaredError:
                MinTotalSumSquaredError = TotalSumSquaredError
                BestWidth = Width
        if silent != True:
            print("found best")
        try:
            OmegaTrap, A, Gamma, fig, ax \
                = self.get_fit_from_peak(CentralFreq - BestWidth / 2,
                                         CentralFreq + BestWidth / 2,
                                         MakeFig=MakeFig,
                                         show_fig=show_fig,
                                         silent=silent)
        except UnboundLocalError:
            raise ValueError("A best width was not found, try increasing the number of widths tried by either decreasing WidthIntervals or MinWidth or increasing MaxWidth")
        OmegaTrap = self.OmegaTrap
        A = self.A
        Gamma = self.Gamma
        self.FTrap = OmegaTrap/(2*pi)
        return OmegaTrap, A, Gamma, fig, ax

    def calc_gamma_from_variance_autocorrelation_fit(self, NumberOfOscillations, GammaGuess=None, silent=False, MakeFig=True, show_fig=True):
        """
        Calculates the total damping, i.e. Gamma, by splitting the time trace
        into chunks of NumberOfOscillations oscillations and calculated the
        variance of each of these chunks. This array of varainces is then used
        for the autocorrleation. The autocorrelation is fitted with an exponential 
        relaxation function and the function returns the parameters with errors.

        Parameters
        ----------
        NumberOfOscillations : int
            The number of oscillations each chunk of the timetrace 
            used to calculate the variance should contain.
        GammaGuess : float, optional
            Inital guess for BigGamma (in radians)
        Silent : bool, optional
            Whether it prints the values fitted or is silent.
        MakeFig : bool, optional
            Whether to construct and return the figure object showing
            the fitting. defaults to True
        show_fig : bool, optional
            Whether to show the figure object when it has been created.
            defaults to True

        Returns
        -------
        Gamma : ufloat
            Big Gamma, the total damping in radians
        fig : matplotlib.figure.Figure object
            The figure object created showing the autocorrelation
            of the data with the fit
        ax : matplotlib.axes.Axes object
            The axes object created showing the autocorrelation
            of the data with the fit

        """
        try:
            SplittedArraySize = int(self.SampleFreq/self.FTrap.n) * NumberOfOscillations
        except KeyError:
            ValueError('You forgot to do the spectrum fit to specify self.FTrap exactly.')
        VoltageArraySize = len(self.voltage)
        SnippetsVariances = _np.var(self.voltage[:VoltageArraySize-_np.mod(VoltageArraySize,SplittedArraySize)].reshape(-1,SplittedArraySize),axis=1)
        autocorrelation = calc_autocorrelation(SnippetsVariances)
        time = _np.array(range(len(autocorrelation))) * SplittedArraySize / self.SampleFreq

        if GammaGuess==None:
            Gamma_Initial = (time[4]-time[0])/(autocorrelation[0]-autocorrelation[4])
        else:
            Gamma_Initial = GammaGuess
        
        if MakeFig == True:
            Params, ParamsErr, fig, ax = fit_autocorrelation(
                autocorrelation, time, Gamma_Initial, MakeFig=MakeFig, show_fig=show_fig)
        else:
            Params, ParamsErr, _ , _ = fit_autocorrelation(
                autocorrelation, time, Gamma_Initial, MakeFig=MakeFig, show_fig=show_fig)

        if silent == False:
            print("\n")
            print(
                "Big Gamma: {} +- {}% ".format(Params[0], ParamsErr[0] / Params[0] * 100))

        Gamma = _uncertainties.ufloat(Params[0], ParamsErr[0])
        
        if MakeFig == True:
            return Gamma, fig, ax
        else:
            return Gamma, None, None

    def calc_gamma_from_energy_autocorrelation_fit(self, GammaGuess=None, silent=False, MakeFig=True, show_fig=True):
        """
        Calculates the total damping, i.e. Gamma, by calculating the energy each 
        point in time. This energy array is then used for the autocorrleation. 
        The autocorrelation is fitted with an exponential relaxation function and
        the function returns the parameters with errors.

        Parameters
        ----------
        GammaGuess : float, optional
            Inital guess for BigGamma (in radians)
        silent : bool, optional
            Whether it prints the values fitted or is silent.
        MakeFig : bool, optional
            Whether to construct and return the figure object showing
            the fitting. defaults to True
        show_fig : bool, optional
            Whether to show the figure object when it has been created.
            defaults to True

        Returns
        -------
        Gamma : ufloat
            Big Gamma, the total damping in radians
        fig : matplotlib.figure.Figure object
            The figure object created showing the autocorrelation
            of the data with the fit
        ax : matplotlib.axes.Axes object
            The axes object created showing the autocorrelation
            of the data with the fit

        """
        autocorrelation = calc_autocorrelation(self.voltage[:-1]**2*self.OmegaTrap.n**2+(_np.diff(self.voltage)*self.SampleFreq)**2)
        time = self.time.get_array()[:len(autocorrelation)]

        if GammaGuess==None:
            Gamma_Initial = (time[4]-time[0])/(autocorrelation[0]-autocorrelation[4])
        else:
            Gamma_Initial = GammaGuess
        
        if MakeFig == True:
            Params, ParamsErr, fig, ax = fit_autocorrelation(
                autocorrelation, time, Gamma_Initial, MakeFig=MakeFig, show_fig=show_fig)
        else:
            Params, ParamsErr, _ , _ = fit_autocorrelation(
                autocorrelation, time, Gamma_Initial, MakeFig=MakeFig, show_fig=show_fig)

        if silent == False:
            print("\n")
            print(
                "Big Gamma: {} +- {}% ".format(Params[0], ParamsErr[0] / Params[0] * 100))

        Gamma = _uncertainties.ufloat(Params[0], ParamsErr[0])
        
        if MakeFig == True:
            return Gamma, fig, ax
        else:
            return Gamma, None, None

    def calc_gamma_from_position_autocorrelation_fit(self, GammaGuess=None, FreqTrapGuess=None, silent=False, MakeFig=True, show_fig=True):
        """
        Calculates the total damping, i.e. Gamma, by calculating the autocorrleation 
        of the position-time trace. The autocorrelation is fitted with an exponential 
        relaxation function derived in Tongcang Li's 2013 thesis (DOI: 10.1007/978-1-4614-6031-2)
        and the function (equation 4.20 in the thesis) returns the parameters with errors.

        Parameters
        ----------
        GammaGuess : float, optional
            Inital guess for BigGamma (in radians)
        FreqTrapGuess : float, optional
            Inital guess for the trapping Frequency in Hz
        silent : bool, optional
            Whether it prints the values fitted or is silent.
        MakeFig : bool, optional
            Whether to construct and return the figure object showing
            the fitting. defaults to True
        show_fig : bool, optional
            Whether to show the figure object when it has been created.
            defaults to True

        Returns
        -------
        Gamma : ufloat
            Big Gamma, the total damping in radians
        OmegaTrap : ufloat
            Trapping frequency in radians
        fig : matplotlib.figure.Figure object
            The figure object created showing the autocorrelation
            of the data with the fit
        ax : matplotlib.axes.Axes object
            The axes object created showing the autocorrelation
            of the data with the fit

        """
        autocorrelation = calc_autocorrelation(self.voltage)
        time = self.time.get_array()[:len(autocorrelation)]

        if GammaGuess==None:
            Gamma_Initial = (autocorrelation[0]-autocorrelation[int(self.SampleFreq/self.FTrap.n)])/(time[int(self.SampleFreq/self.FTrap.n)]-time[0])*2*_np.pi
        else:
            Gamma_Initial = GammaGuess

        if FreqTrapGuess==None:
            FreqTrap_Initial = self.FTrap.n
        else:
            FreqTrap_Initial = FreqTrapGuess
            
        if MakeFig == True:
            Params, ParamsErr, fig, ax = fit_autocorrelation(
                autocorrelation, time, Gamma_Initial, FreqTrap_Initial, method='position', MakeFig=MakeFig, show_fig=show_fig)
        else:
            Params, ParamsErr, _ , _ = fit_autocorrelation(
                autocorrelation, time, Gamma_Initial, FreqTrap_Initial, method='position', MakeFig=MakeFig, show_fig=show_fig)

        if silent == False:
            print("\n")
            print(
                "Big Gamma: {} +- {}% ".format(Params[0], ParamsErr[0] / Params[0] * 100))
            print(
                "Trap Frequency: {} +- {}% ".format(Params[1], ParamsErr[1] / Params[1] * 100))
            
        Gamma = _uncertainties.ufloat(Params[0], ParamsErr[0])
        OmegaTrap = _uncertainties.ufloat(Params[1], ParamsErr[1])
        
        if MakeFig == True:
            return Gamma, OmegaTrap, fig, ax
        else:
            return Gamma, OmegaTrap, None, None
    
    def extract_parameters(self, P_mbar, P_Error, method="chang"):
        """
        Extracts the Radius, mass and Conversion factor for a particle.

        Parameters
        ----------
        P_mbar : float 
            The pressure in mbar when the data was taken.
        P_Error : float
            The error in the pressure value (as a decimal e.g. 15% = 0.15)
        
        Returns
        -------
        Radius : uncertainties.ufloat
            The radius of the particle in m
        Mass : uncertainties.ufloat
            The mass of the particle in kg
        ConvFactor : uncertainties.ufloat
            The conversion factor between volts/m

        """

        [R, M, ConvFactor], [RErr, MErr, ConvFactorErr] = \
            extract_parameters(P_mbar, P_Error,
                               self.A.n, self.A.std_dev,
                               self.Gamma.n, self.Gamma.std_dev,
                               method = method)
        self.Radius = _uncertainties.ufloat(R, RErr)
        self.Mass = _uncertainties.ufloat(M, MErr)
        self.ConvFactor = _uncertainties.ufloat(ConvFactor, ConvFactorErr)

        return self.Radius, self.Mass, self.ConvFactor

    def extract_ZXY_motion(self, ApproxZXYFreqs, uncertaintyInFreqs, ZXYPeakWidths, subSampleFraction=1, NPerSegmentPSD=1000000, MakeFig=True, show_fig=True):
        """
        Extracts the x, y and z signals (in volts) from the voltage signal. Does this by finding the highest peaks in the signal about the approximate frequencies, using the uncertaintyinfreqs parameter as the width it searches. It then uses the ZXYPeakWidths to construct bandpass IIR filters for each frequency and filtering them. If too high a sample frequency has been used to collect the data scipy may not be able to construct a filter good enough, in this case increasing the subSampleFraction may be nessesary.
        
        Parameters
        ----------
        ApproxZXYFreqs : array_like
            A sequency containing 3 elements, the approximate 
            z, x and y frequency respectively.
        uncertaintyInFreqs : float
            The uncertainty in the z, x and y frequency respectively.
        ZXYPeakWidths : array_like
            A sequency containing 3 elements, the widths of the
            z, x and y frequency peaks respectively.
        subSampleFraction : int, optional
            How much to sub-sample the data by before filtering,
            effectively reducing the sample frequency by this 
            fraction.
        NPerSegmentPSD : int, optional
            NPerSegment to pass to scipy.signal.welch to calculate the PSD
        show_fig : bool, optional
            Whether to show the figures produced of the PSD of
            the original signal along with the filtered x, y and z.

        Returns
        -------
        self.zVolts : ndarray
            The z signal in volts extracted by bandpass IIR filtering
        self.xVolts : ndarray
            The x signal in volts extracted by bandpass IIR filtering
        self.yVolts : ndarray
            The y signal in volts extracted by bandpass IIR filtering
        time : ndarray
            The array of times corresponding to the above 3 arrays
        fig : matplotlib.figure.Figure object
            figure object containing a plot of the PSD of the original 
            signal with the z, x and y filtered signals
        ax : matplotlib.axes.Axes object
            axes object corresponding to the above figure
        
        """
        [zf, xf, yf] = ApproxZXYFreqs
        zf, xf, yf = get_ZXY_freqs(
            self, zf, xf, yf, bandwidth=uncertaintyInFreqs)
        [zwidth, xwidth, ywidth] = ZXYPeakWidths
        self.zVolts, self.xVolts, self.yVolts, time, fig, ax = get_ZXY_data(
            self, zf, xf, yf, subSampleFraction, zwidth, xwidth, ywidth, MakeFig=MakeFig, show_fig=show_fig, NPerSegmentPSD=NPerSegmentPSD)
        return self.zVolts, self.xVolts, self.yVolts, time, fig, ax

    def filter_data(self, freq, FractionOfSampleFreq=1, PeakWidth=10000,
                    filterImplementation="filtfilt",
                    timeStart=None, timeEnd=None,
                    NPerSegmentPSD=1000000,
                    PyCUDA=False, MakeFig=True, show_fig=True):
        """
        filter out data about a central frequency with some bandwidth using an IIR filter.
    
        Parameters
        ----------
        freq : float
            The frequency of the peak of interest in the PSD
        FractionOfSampleFreq : integer, optional
            The fraction of the sample frequency to sub-sample the data by.
            This sometimes needs to be done because a filter with the appropriate
            frequency response may not be generated using the sample rate at which
            the data was taken. Increasing this number means the x, y and z signals
            produced by this function will be sampled at a lower rate but a higher
            number means a higher chance that the filter produced will have a nice
            frequency response.
        PeakWidth : float, optional
            The width of the pass-band of the IIR filter to be generated to
            filter the peak. Defaults to 10KHz
        filterImplementation : string, optional
            filtfilt or lfilter - use scipy.filtfilt or lfilter
            ifft - uses built in IFFT_filter
            default: filtfilt
        timeStart : float, optional
            Starting time for filtering. Defaults to start of time data.
        timeEnd : float, optional
            Ending time for filtering. Defaults to end of time data.
        NPerSegmentPSD : int, optional
            NPerSegment to pass to scipy.signal.welch to calculate the PSD
        PyCUDA : bool, optional
            Only important for the 'ifft'-method
            If True, uses PyCUDA to accelerate the FFT and IFFT
            via using your NVIDIA-GPU
            If False, performs FFT and IFFT with conventional
            scipy.fftpack
        MakeFig : bool, optional
            If True - generate figure showing filtered and unfiltered PSD
            Defaults to True.
        show_fig : bool, optional
            If True - plot unfiltered and filtered PSD
            Defaults to True.
    
        Returns
        -------
        timedata : ndarray
            Array containing the time data
        FiletedData : ndarray
            Array containing the filtered signal in volts with time.
        fig : matplotlib.figure.Figure object
            The figure object created showing the PSD of the filtered 
            and unfiltered signal
        ax : matplotlib.axes.Axes object
            The axes object created showing the PSD of the filtered 
            and unfiltered signal
        """
        if timeStart == None:
            timeStart = self.timeStart
        if timeEnd == None:
            timeEnd = self.timeEnd

        time = self.time.get_array()

        StartIndex = _np.where(time == take_closest(time, timeStart))[0][0]
        EndIndex = _np.where(time == take_closest(time, timeEnd))[0][0]

        
        input_signal = self.voltage[StartIndex: EndIndex][0::FractionOfSampleFreq]
        SAMPLEFREQ = self.SampleFreq / FractionOfSampleFreq
        if filterImplementation == "filtfilt" or filterImplementation == "lfilter":
            if filterImplementation == "filtfilt":
                ApplyFilter = scipy.signal.filtfilt
            elif filterImplementation == "lfilter":
                ApplyFilter = scipy.signal.lfilter
                
    
            b, a = make_butterworth_bandpass_b_a(freq, PeakWidth, SAMPLEFREQ)
            print("filtering data")
            filteredData = ApplyFilter(b, a, input_signal)
    
            if(_np.isnan(filteredData).any()):
                raise ValueError(
                    "Value Error: FractionOfSampleFreq must be higher, a sufficiently small sample frequency should be used to produce a working IIR filter.")
        elif filterImplementation == "ifft":
            filteredData = IFFT_filter(input_signal, SAMPLEFREQ, freq-PeakWidth/2, freq+PeakWidth/2, PyCUDA = PyCUDA)
        else:
            raise ValueError("filterImplementation must be one of [filtfilt, lfilter, ifft] you entered: {}".format(filterImplementation))
    
        if MakeFig == True:
            f, PSD = scipy.signal.welch(
                input_signal, SAMPLEFREQ, nperseg=NPerSegmentPSD)
            f_filtdata, PSD_filtdata = scipy.signal.welch(filteredData, SAMPLEFREQ, nperseg=NPerSegmentPSD)
            fig, ax = _plt.subplots(figsize=properties["default_fig_size"])
            ax.plot(f, PSD)
            ax.plot(f_filtdata, PSD_filtdata, label="filtered data")
            ax.legend(loc="best")
            ax.semilogy()
            ax.set_xlim([freq - PeakWidth, freq + PeakWidth])
        else:
            fig = None
            ax = None
        if show_fig == True:
            _plt.show()
        timedata = time[StartIndex: EndIndex][0::FractionOfSampleFreq]
        return timedata, filteredData, fig, ax

    def plot_phase_space_sns(self, freq, ConvFactor, PeakWidth=10000, FractionOfSampleFreq=1, kind="hex", timeStart=None, timeEnd =None, PointsOfPadding=500, units="nm", logscale=False, cmap=None, marginalColor=None, gridsize=200, show_fig=True, ShowPSD=False, alpha=0.5, *args, **kwargs):
        """
        Plots the phase space of a peak in the PSD.
        
        Parameters
        ----------
        freq : float
            The frequenecy of the peak (Trapping frequency of the dimension of interest)
        ConvFactor : float (or ufloat)
            The conversion factor between Volts and Meters
        PeakWidth : float, optional
            The width of the peak. Defaults to 10KHz
        FractionOfSampleFreq : int, optional
            The fraction of the sample freq to use to filter the data.
            Defaults to 1.
        kind : string, optional
            kind of plot to draw - pass to jointplot from seaborne
        timeStart : float, optional
            Starting time for data from which to calculate the phase space.
            Defaults to start of time data.
        timeEnd : float, optional
            Ending time for data from which to calculate the phase space.
            Defaults to start of time data.
        PointsOfPadding : float, optional
            How many points of the data at the beginning and end to disregard for plotting
            the phase space, to remove filtering artifacts. Defaults to 500.
        units : string, optional
            Units of position to plot on the axis - defaults to nm
        cmap : matplotlib.colors.ListedColormap, optional
            cmap to use for plotting the jointplot
        marginalColor : string, optional
            color to use for marginal plots
        gridsize : int, optional
            size of the grid to use with kind="hex"
        show_fig : bool, optional
            Whether to show the figure before exiting the function
            Defaults to True.
        ShowPSD : bool, optional
            Where to show the PSD of the unfiltered and the 
            filtered signal used to make the phase space
            plot. Defaults to False.

        Returns
        -------
        fig : matplotlib.figure.Figure object
            figure object containing the phase space plot
        JP : seaborn.jointplot object
            joint plot object containing the phase space plot
        """
        if cmap == None:
            if logscale == True:
                cmap = properties['default_log_cmap']
            else:
                cmap = properties['default_linear_cmap']
        
        unit_prefix = units[:-1]

        _, PosArray, VelArray = self.calc_phase_space(freq, ConvFactor, PeakWidth=PeakWidth, FractionOfSampleFreq=FractionOfSampleFreq, timeStart=timeStart, timeEnd=timeEnd, PointsOfPadding=PointsOfPadding, ShowPSD=ShowPSD)

        _plt.close('all')
        
        PosArray = unit_conversion(PosArray, unit_prefix) # converts m to units required (nm by default)
        VelArray = unit_conversion(VelArray, unit_prefix) # converts m/s to units required (nm/s by default)
        
        VarPos = _np.var(PosArray)
        VarVel = _np.var(VelArray)
        MaxPos = _np.max(PosArray)
        MaxVel = _np.max(VelArray)
        if MaxPos > MaxVel / (2 * pi * freq):
            _plotlimit = MaxPos * 1.1
        else:
            _plotlimit = MaxVel / (2 * pi * freq) * 1.1

        print("Plotting Phase Space")

        if marginalColor == None:
            try:
                marginalColor = tuple((cmap.colors[len(cmap.colors)/2][:-1]))
            except AttributeError:
                try:
                    marginalColor = cmap(2)
                except:
                    marginalColor = properties['default_base_color']

        if kind == "hex":    # gridsize can only be passed if kind="hex"
            JP1 = _sns.jointplot(_pd.Series(PosArray[1:], name="$z$ ({}) \n filepath=%s".format(units) % (self.filepath)),
                                 _pd.Series(VelArray / (2 * pi * freq), name="$v_z$/$\omega$ ({})".format(units)),
                                 stat_func=None,
                                 xlim=[-_plotlimit, _plotlimit],
                                 ylim=[-_plotlimit, _plotlimit],
                                 size=max(properties['default_fig_size']),
                                 kind=kind,
                                 marginal_kws={'hist_kws': {'log': logscale},},
                                 cmap=cmap,
                                 color=marginalColor,
                                 gridsize=gridsize,
                                 alpha=alpha,
                                 *args,
                                 **kwargs,
            )
        else:
            JP1 = _sns.jointplot(_pd.Series(PosArray[1:], name="$z$ ({}) \n filepath=%s".format(units) % (self.filepath)),
                                     _pd.Series(VelArray / (2 * pi * freq), name="$v_z$/$\omega$ ({})".format(units)),
                                 stat_func=None,
                                 xlim=[-_plotlimit, _plotlimit],
                                 ylim=[-_plotlimit, _plotlimit],
                                 size=max(properties['default_fig_size']),
                                 kind=kind,
                                 marginal_kws={'hist_kws': {'log': logscale},},
                                 cmap=cmap,
                                 color=marginalColor,
                                 alpha=alpha,                                
                                 *args,
                                 **kwargs,
            )

        fig = JP1.fig
        
        if show_fig == True:
            print("Showing Phase Space")
            _plt.show()
            
        return fig, JP1
 
    def plot_phase_space(self, freq, ConvFactor, PeakWidth=10000, FractionOfSampleFreq=1, timeStart=None, timeEnd =None, PointsOfPadding=500, units="nm", show_fig=True, ShowPSD=False, xlabel='', ylabel='', *args, **kwargs):
        unit_prefix = units[:-1]

        xlabel = xlabel + "({})".format(units)
        ylabel = ylabel + "({})".format(units)
            
        _, PosArray, VelArray = self.calc_phase_space(freq, ConvFactor, PeakWidth=PeakWidth, FractionOfSampleFreq=FractionOfSampleFreq, timeStart=timeStart, timeEnd=timeEnd, PointsOfPadding=PointsOfPadding, ShowPSD=ShowPSD)

        PosArray = unit_conversion(PosArray, unit_prefix) # converts m to units required (nm by default)
        VelArray = unit_conversion(VelArray, unit_prefix) # converts m/s to units required (nm/s by default)

        VelArray = VelArray/(2*pi*freq) # converst nm/s to nm/radian
        PosArray = PosArray[1:]

        fig, axscatter, axhistx, axhisty, cb = _qplots.joint_plot(PosArray, VelArray, *args, **kwargs)
        axscatter.set_xlabel(xlabel)
        axscatter.set_ylabel(ylabel)

        if show_fig == True:
            _plt.show()
        return fig, axscatter, axhistx, axhisty, cb
    
    def calc_phase_space(self, freq, ConvFactor, PeakWidth=10000, FractionOfSampleFreq=1, timeStart=None, timeEnd =None, PointsOfPadding=500, ShowPSD=False):
        """
        Calculates the position and velocity (in m) for use in plotting the phase space distribution.

        Parameters
        ----------
        freq : float
            The frequenecy of the peak (Trapping frequency of the dimension of interest)
        ConvFactor : float (or ufloat)
            The conversion factor between Volts and Meters
        PeakWidth : float, optional
            The width of the peak. Defaults to 10KHz
        FractionOfSampleFreq : int, optional
            The fraction of the sample freq to use to filter the data.
            Defaults to 1.
        timeStart : float, optional
            Starting time for data from which to calculate the phase space.
            Defaults to start of time data.
        timeEnd : float, optional
            Ending time for data from which to calculate the phase space.
            Defaults to start of time data.
        PointsOfPadding : float, optional
            How many points of the data at the beginning and end to disregard for plotting
            the phase space, to remove filtering artifacts. Defaults to 500
        ShowPSD : bool, optional
            Where to show the PSD of the unfiltered and the filtered signal used 
            to make the phase space plot. Defaults to False.
        *args, **kwargs : optional
            args and kwargs passed to qplots.joint_plot


        Returns
        -------
        time : ndarray
            time corresponding to position and velocity
        PosArray : ndarray
            Array of position of the particle in time
        VelArray : ndarray
            Array of velocity of the particle in time
        """
        _, Pos, fig, ax = self.filter_data(
            freq, FractionOfSampleFreq, PeakWidth, MakeFig=ShowPSD, show_fig=ShowPSD, timeStart=timeStart, timeEnd=timeEnd)
        time = self.time.get_array()
        if timeStart != None:
            StartIndex = _np.where(time == take_closest(time, timeStart))[0][0]
        else:
            StartIndex = 0
        if timeEnd != None:
            EndIndex = _np.where(time == take_closest(time, timeEnd))[0][0]
        else:
            EndIndex = -1
            
        Pos = Pos[PointsOfPadding : -PointsOfPadding+1]
        time = time[StartIndex:EndIndex][::FractionOfSampleFreq][PointsOfPadding : -PointsOfPadding+1]
        
        if type(ConvFactor) == _uncertainties.core.Variable:
            conv = ConvFactor.n
        else:
            conv = ConvFactor
        PosArray = Pos / conv # converts V to m
        VelArray = _np.diff(PosArray) * (self.SampleFreq / FractionOfSampleFreq) # calcs velocity (in m/s) by differtiating position
        return time, PosArray, VelArray
        
        
class ORGTableData():
    """
    Class for reading in general data from org-mode tables.


    The table must be formatted as in the example below:

    ```
    | RunNo | ColumnName1 | ColumnName2 |
    |-------+-------------+-------------|
    |   3   |     14      |     15e3    |
    ```

    In this case the run number would be 3 and the ColumnName2-value would
    be 15e3 (15000.0).

    """
    def __init__(self, filename):
        """
        Opens the org-mode table file, reads the file in as a string,
        and runs parse_orgtable in order to read the pressure.
        """
        with open(filename, 'r') as file:
            fileContents = file.readlines()
        self.ORGTableData = parse_orgtable(fileContents)

    def get_value(self, ColumnName, RunNo):
        """
        Retreives the value of the collumn named ColumnName associated 
        with a particular run number.

        Parameters
        ----------
        ColumnName : string
            The name of the desired org-mode table's collumn

        RunNo : int
            The run number for which to retreive the pressure value
        
        Returns
        -------
        Value : float
            The value for the column's name and associated run number
        """
        Value = float(self.ORGTableData[self.ORGTableData.RunNo == '{}'.format(
            RunNo)][ColumnName])
        
        return Value 

    
def load_data(Filepath, ObjectType='data', RelativeChannelNo=None, SampleFreq=None, PointsToLoad=-1, calcPSD=True, NPerSegmentPSD=1000000, NormaliseByMonitorOutput=False, silent=False):
    """
    Parameters
    ----------
    Filepath : string
        filepath to the file containing the data used to initialise
        and create an instance of the DataObject class
    ObjectType : string, optional
        type to load the data as, takes the value 'default' if not specified.
        Options are:
        'data' : optoanalysis.DataObject
        'thermo' : optoanalysis.thermo.ThermoObject
    RelativeChannelNo : int, optional
        If loading a .bin file produced by the Saneae datalogger, used to specify
        the channel number
        If loading a .dat file produced by the labview NI5122 daq card, used to 
        specifiy the channel number if two channels where saved, if left None with 
        .dat files it will assume that the file to load only contains one channel.
        If NormaliseByMonitorOutput is True then specifies the monitor channel for
        loading a .dat file produced by the labview NI5122 daq card.
    SampleFreq : float, optional
        If loading a .dat file produced by the labview NI5122 daq card, used to
        manually specify the sample frequency
    PointsToLoad : int, optional
        Number of first points to read. -1 means all points (i.e., the complete file)
        WORKS WITH NI5122 DATA SO FAR ONLY!!!
    calcPSD : bool, optional
        Whether to calculate the PSD upon loading the file, can take some time
        off the loading and reduce memory usage if frequency space info is not required
    NPerSegmentPSD : int, optional
        NPerSegment to pass to scipy.signal.welch to calculate the PSD
    NormaliseByMonitorOutput : bool, optional
        If True the particle signal trace will be divided by the monitor output, which is
        specified by the channel number set in the RelativeChannelNo parameter. 
        WORKS WITH NI5122 DATA SO FAR ONLY!!!

    Returns
    -------
    Data : DataObject
        An instance of the DataObject class contaning the data
        that you requested to be loaded.

    """
    if silent != True:
        print("Loading data from {}".format(Filepath))
    ObjectTypeDict = {
        'data' : DataObject,
        'thermo' : optoanalysis.thermo.ThermoObject,
        }
    try:
        Object = ObjectTypeDict[ObjectType]
    except KeyError:
        raise ValueError("You entered {}, this is not a valid object type".format(ObjectType))
    data = Object(Filepath, RelativeChannelNo, SampleFreq, PointsToLoad, calcPSD, NPerSegmentPSD, NormaliseByMonitorOutput)
    try:
        channel_number, run_number, repeat_number = [int(val) for val in re.findall('\d+', data.filename)]
        data.channel_number = channel_number
        data.run_number = run_number
        data.repeat_number = repeat_number
        if _does_file_exist(data.filepath.replace(data.filename, '') + "pressure.log"):
            print("pressure.log file exists")
            for line in open(data.filepath.replace(data.filename, '') + "pressure.log", 'r'):
                run_number, repeat_number, pressure = line.split(',')[1:]
                run_number = int(run_number)
                repeat_number = int(repeat_number)
                pressure = float(pressure)
                if (run_number == data.run_number) and (repeat_number == data.repeat_number):
                    data.pmbar = pressure    
    except ValueError:
        pass
    try:
        if _does_file_exist(glob(data.filepath.replace(data.filename, '*' + data.filename[20:-4] + ' - header.dat'))[0]):
            print("header file exists")
            with open(glob(data.filepath.replace(data.filename, '*' + data.filepath[20:-4] + ' - header.dat'))[0], encoding='ISO-8859-1') as f:
                lines = f.readlines()
            data.pmbar = (float(lines[68][-9:-1])+float(lines[69][-9:-1]))/2
    except (ValueError, IndexError):
        pass
    return data

def search_data_std(Channel, RunNos, RepeatNos, directoryPath='.'):
    """
    Lets you find multiple datasets at once assuming they have a 
    filename which contains a pattern of the form:
    CH<ChannelNo>_RUN00...<RunNo>_REPEAT00...<RepeatNo>    

    Parameters
    ----------
    Channel : int
        The channel you want to load
    RunNos : sequence
        Sequence of run numbers you want to load
    RepeatNos : sequence
        Sequence of repeat numbers you want to load
    directoryPath : string, optional
        The path to the directory housing the data
        The default is the current directory

    Returns
    -------
    Data_filepaths : list
        A list containing the filepaths to the matching files
    """
    files = glob('{}/*'.format(directoryPath))
    files_CorrectChannel = []
    for file_ in files:
        if 'CH{}'.format(Channel) in file_:
            files_CorrectChannel.append(file_)
    print(files_CorrectChannel)
    files_CorrectRunNo = []
    for RunNo in RunNos:
        files_match = _fnmatch.filter(
            files_CorrectChannel, '*RUN*0{}_*'.format(RunNo))
        for file_ in files_match:
            files_CorrectRunNo.append(file_)
    files_CorrectRepeatNo = []
    for RepeatNo in RepeatNos:
        files_match = _fnmatch.filter(
            files_CorrectRunNo, '*REPEAT*0{}.*'.format(RepeatNo))
        for file_ in files_match:
            files_CorrectRepeatNo.append(file_)
    return files_CorrectRepeatNo

def multi_load_data(Channel, RunNos, RepeatNos, directoryPath='.', calcPSD=True, NPerSegmentPSD=1000000):
    """
    Lets you load multiple datasets at once assuming they have a 
    filename which contains a pattern of the form:
    CH<ChannelNo>_RUN00...<RunNo>_REPEAT00...<RepeatNo>    

    Parameters
    ----------
    Channel : int
        The channel you want to load
    RunNos : sequence
        Sequence of run numbers you want to load
    RepeatNos : sequence
        Sequence of repeat numbers you want to load
    directoryPath : string, optional
        The path to the directory housing the data
        The default is the current directory

    Returns
    -------
    Data : list
        A list containing the DataObjects that were loaded. 
    """
    matching_files = search_data_std(Channel=Channel, RunNos=RunNos, RepeatNos=RepeatNos, directoryPath=directoryPath)
    #data = []
    #for filepath in matching_files_:
    #    data.append(load_data(filepath, calcPSD=calcPSD, NPerSegmentPSD=NPerSegmentPSD))

    cpu_count = _cpu_count()
    workerPool = _Pool(cpu_count)
    load_data_partial = _partial(load_data, calcPSD=calcPSD, NPerSegmentPSD=NPerSegmentPSD)
    data = workerPool.map(load_data_partial, matching_files)
    workerPool.close()
    workerPool.terminate()
    workerPool.join()
       
    #with _Pool(cpu_count) as workerPool:
        #load_data_partial = _partial(load_data, calcPSD=calcPSD, NPerSegmentPSD=NPerSegmentPSD)
        #data = workerPool.map(load_data_partial, files_CorrectRepeatNo)
    return data

def multi_load_data_custom(Channel, TraceTitle, RunNos, directoryPath='.', calcPSD=True, NPerSegmentPSD=1000000):
    """
    Lets you load multiple datasets named with the LeCroy's custom naming scheme at once.

    Parameters
    ----------
    Channel : int
        The channel you want to load
    TraceTitle : string
        The custom trace title of the files. 
    RunNos : sequence
        Sequence of run numbers you want to load
    RepeatNos : sequence
        Sequence of repeat numbers you want to load
    directoryPath : string, optional
        The path to the directory housing the data
        The default is the current directory

    Returns
    -------
    Data : list
        A list containing the DataObjects that were loaded. 
    """
#    files = glob('{}/*'.format(directoryPath))
#    files_CorrectChannel = []
#    for file_ in files:
#        if 'C{}'.format(Channel) in file_:
#           files_CorrectChannel.append(file_)
#    files_CorrectRunNo = []
#    for RunNo in RunNos:
#        files_match = _fnmatch.filter(
#            files_CorrectChannel, '*C{}'.format(Channel)+TraceTitle+str(RunNo).zfill(5)+'.*')
#        for file_ in files_match:
#            files_CorrectRunNo.append(file_)
    matching_files = search_data_custom(Channel, TraceTitle, RunNos, directoryPath)
    cpu_count = _cpu_count()
    workerPool = _Pool(cpu_count)
    # for filepath in files_CorrectRepeatNo:
    #    print(filepath)
    #    data.append(load_data(filepath))
    load_data_partial = _partial(load_data, calcPSD=calcPSD, NPerSegmentPSD=NPerSegmentPSD)
    data = workerPool.map(load_data_partial, matching_files)
    workerPool.close()
    workerPool.terminate()
    workerPool.join()
    return data

def search_data_custom(Channel, TraceTitle, RunNos, directoryPath='.'):
    """
    Lets you create a list with full file paths of the files
    named with the LeCroy's custom naming scheme.

    Parameters
    ----------
    Channel : int
        The channel you want to load
    TraceTitle : string
        The custom trace title of the files. 
    RunNos : sequence
        Sequence of run numbers you want to load
    RepeatNos : sequence
        Sequence of repeat numbers you want to load
    directoryPath : string, optional
        The path to the directory housing the data
        The default is the current directory

    Returns
    -------
    Paths : list
        A list containing the full file paths of the files you were looking for. 
    """
    files = glob('{}/*'.format(directoryPath))
    files_CorrectChannel = []    
    for file_ in files:
        if 'C{}'.format(Channel) in file_:
            files_CorrectChannel.append(file_)
    print(files_CorrectChannel)
    files_CorrectRunNo = []
    for RunNo in RunNos:
        files_match = _fnmatch.filter(
            files_CorrectChannel, '*C{}'.format(Channel)+TraceTitle+str(RunNo).zfill(5)+'.*')
        for file_ in files_match:
            files_CorrectRunNo.append(file_)
    print(files_CorrectRunNo)
    paths = files_CorrectRunNo
    return paths


def calc_temp(Data_ref, Data):
    """
    Calculates the temperature of a data set relative to a reference.
    The reference is assumed to be at 300K.

    Parameters
    ----------
    Data_ref : DataObject
        Reference data set, assumed to be 300K
    Data : DataObject
        Data object to have the temperature calculated for

    Returns
    -------
    T : uncertainties.ufloat
        The temperature of the data set
    """
    T = 300 * ((Data.A * Data_ref.Gamma) / (Data_ref.A * Data.Gamma))
    Data.T = T
    return T

def calc_gamma_components(Data_ref, Data):
    """
    Calculates the components of Gamma (Gamma0 and delta_Gamma), 
    assuming that the Data_ref is uncooled data (ideally at 3mbar
    for best fitting). It uses the fact that A_prime=A/Gamma0 should 
    be constant for a particular particle under changes in pressure
    and therefore uses the reference save to calculate A_prime (assuming
    the Gamma value found for the uncooled data is actually equal to Gamma0
    since only collisions should be causing the damping. Therefore for
    the cooled data Gamma0 should equal A/A_prime and therefore we
    can extract Gamma0 and delta_Gamma.

    A_prime = ConvFactor**2 * (2*k_B*T0/(pi*m))

    Parameters
    ----------
    Data_ref : DataObject
        Reference data set, assumed to be 300K
    Data : DataObject
        Data object to have the temperature calculated for

    Returns
    -------
    Gamma0 : uncertainties.ufloat
        Damping due to the environment
    delta_Gamma : uncertainties.ufloat
        Damping due to other effects (e.g. feedback cooling)
    
    """
    A_prime = Data_ref.A/Data_ref.Gamma
    Gamma0 = Data.A/A_prime
    delta_Gamma = Data.Gamma - Gamma0
    return Gamma0, delta_Gamma

def fit_curvefit(p0, datax, datay, function, **kwargs):
    """
    Fits the data to a function using scipy.optimise.curve_fit

    Parameters
    ----------
    p0 : array_like
        initial parameters to use for fitting
    datax : array_like
        x data to use for fitting
    datay : array_like
        y data to use for fitting
    function : function
        funcion to be fit to the data
    kwargs 
        keyword arguments to be passed to scipy.optimise.curve_fit

    Returns
    -------
    pfit_curvefit : array
        Optimal values for the parameters so that the sum of
        the squared residuals of ydata is minimized
    perr_curvefit : array
        One standard deviation errors in the optimal values for
        the parameters
    """
    pfit, pcov = \
        _curve_fit(function, datax, datay, p0=p0,
                   epsfcn=0.0001, **kwargs)
    error = []
    for i in range(len(pfit)):
        try:
            error.append(_np.absolute(pcov[i][i])**0.5)
        except:
            error.append(_np.NaN)
    pfit_curvefit = pfit
    perr_curvefit = _np.array(error)
    return pfit_curvefit, perr_curvefit


def moving_average(array, n=3):
    """
    Calculates the moving average of an array.

    Parameters
    ----------
    array : array
        The array to have the moving average taken of
    n : int
        The number of points of moving average to take
    
    Returns
    -------
    MovingAverageArray : array
        The n-point moving average of the input array
    """
    ret = _np.cumsum(array, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.

    Parameters
    ----------
    myList : array
        The list in which to find the closest value to myNumber
    myNumber : float
        The number to find the closest to in MyList

    Returns
    -------
    closestValue : float
        The number closest to myNumber in myList
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

def _energy_autocorrelation_fitting_eqn(t, Gamma):
    """
    The value of the fitting equation:
    exp(-t*Gamma)
    to be fit to the autocorrelation-exponential decay
    Actual correct equation would be: 
    exp(-t*Gamma) * (4*Omega**2-Gamma**2 * cos(2*sqrt(Gamma**2 /4 - Omega**2)*t))
    taken from DOI: 10.1103/PhysRevE.94.062151 but since
    the additional term is negligible when the quality factor Q>1.

    Parameters
    ----------
    t : float
        time 
    Gamma : float
        Big Gamma (in radians), i.e. damping 

    Returns
    -------
    Value : float
        The value of the fitting equation
    """
    return _np.exp(-t*Gamma)

def _position_autocorrelation_fitting_eqn(t, Gamma, AngTrapFreq):
    """
    The value of the fitting equation:
    exp(-t*Gamma/2) * (cos(t* sqrt(Omega**2 - Gamma**2 /4)) + Gamma* sin(t* sqrt(Omega**2-Gamma**2 /4))/(2* sqrt(Omega**2 - Gamma**2 /4)))
    [eqn 4.20 taken from DOI: DOI: 10.1007/978-1-4614-6031-2]
    to be fit to the autocorrelation-exponential decay

    Parameters
    ----------
    t : float
        time 
    Gamma : float
        Big Gamma (in radians), i.e. damping 
    AngTrapFreq : float
        Angular Trapping Frequency in Radians

    Returns
    -------
    Value : float
        The value of the fitting equation
    """
    return _np.exp(-t*Gamma/2)* ( _np.cos(t* _np.sqrt(AngTrapFreq**2-Gamma**2/4)) + Gamma* _np.sin(t* _np.sqrt(AngTrapFreq**2-Gamma**2/4))/(2* _np.sqrt(AngTrapFreq**2-Gamma**2/4)) )


def fit_autocorrelation(autocorrelation, time, GammaGuess, TrapFreqGuess=None, method='energy', MakeFig=True, show_fig=True):
    """
    Fits exponential relaxation theory to data.

    Parameters
    ----------
    autocorrelation : array
        array containing autocorrelation to be fitted
    time : array
        array containing the time of each point the autocorrelation
        was evaluated
    GammaGuess : float
        The approximate Big Gamma (in radians) to use initially
    TrapFreqGuess : float
        The approximate trapping frequency to use initially in Hz.
    method : string, optional
        To choose which autocorrelation fit is needed.
        'position' : equation 4.20 from Tongcang Li's 2013 thesis 
                     (DOI: 10.1007/978-1-4614-6031-2)
        'energy'   : proper exponential energy correlation decay
                     (DOI: 10.1103/PhysRevE.94.062151)
    MakeFig : bool, optional
        Whether to construct and return the figure object showing
        the fitting. defaults to True
    show_fig : bool, optional
        Whether to show the figure object when it has been created.
        defaults to True

    Returns
    -------
    ParamsFit - Fitted parameters:
        'variance'-method : [Gamma]
        'position'-method : [Gamma, AngularTrappingFrequency]
    ParamsFitErr - Error in fitted parameters:
        'varaince'-method : [GammaErr]
        'position'-method : [GammaErr, AngularTrappingFrequencyErr]
    fig : matplotlib.figure.Figure object
        figure object containing the plot
    ax : matplotlib.axes.Axes object
        axes with the data plotted of the:
            - initial data
            - final fit
    """
    datax = time
    datay = autocorrelation

    method = method.lower()
    if method == 'energy':
        p0 = _np.array([GammaGuess])

        Params_Fit, Params_Fit_Err = fit_curvefit(p0,
                                                  datax,
                                                  datay,
                                                  _energy_autocorrelation_fitting_eqn)
        autocorrelation_fit = _energy_autocorrelation_fitting_eqn(_np.arange(0,datax[-1],1e-7),
                                                                  Params_Fit[0])
    elif method == 'position':
        AngTrapFreqGuess = 2 * _np.pi * TrapFreqGuess
        p0 = _np.array([GammaGuess, AngTrapFreqGuess])
        Params_Fit, Params_Fit_Err = fit_curvefit(p0,
                                                  datax,
                                                  datay,
                                                  _position_autocorrelation_fitting_eqn)
        autocorrelation_fit = _position_autocorrelation_fitting_eqn(_np.arange(0,datax[-1],1e-7),
                                                                    Params_Fit[0],
                                                                    Params_Fit[1])
        
    if MakeFig == True:
        fig = _plt.figure(figsize=properties["default_fig_size"])
        ax = fig.add_subplot(111)
        ax.plot(datax*1e6, datay,
                '.', color="darkblue", label="Autocorrelation Data", alpha=0.5)
        ax.plot(_np.arange(0,datax[-1],1e-7)*1e6, autocorrelation_fit,
                color="red", label="fit")
        ax.set_xlim([0,
                     30e6/Params_Fit[0]/(2*_np.pi)])
        legend = ax.legend(loc="best", frameon = 1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')
        ax.set_xlabel("time (us)")
        ax.set_ylabel(r"$\left | \frac{\langle x(t)x(t+\tau) \rangle}{\langle x(t)x(t) \rangle} \right |$")
        if show_fig == True:
            _plt.show()
        return Params_Fit, Params_Fit_Err, fig, ax
    else:
        return Params_Fit, Params_Fit_Err, None, None
    
def PSD_fitting_eqn(A, OmegaTrap, Gamma, omega):
    """
    The value of the fitting equation:
    A / ((OmegaTrap**2 - omega**2)**2 + (omega * Gamma)**2)
    to be fit to the PSD

    Parameters
    ----------
    A : float
        Fitting constant A
        A = γ**2*Γ_0*(2*K_b*T_0)/(π*m)
        where:
            γ = conversionFactor
            Γ_0 = Damping factor due to environment
            π = pi
    OmegaTrap : float
        The trapping frequency in the axis of interest 
        (in angular frequency)
    Gamma : float
        The damping factor Gamma = Γ = Γ_0 + δΓ
        where:
            Γ_0 = Damping factor due to environment
            δΓ = extra damping due to feedback or other effects
    omega : float
        The angular frequency to calculate the value of the 
        fitting equation at 

    Returns
    -------
    Value : float
        The value of the fitting equation
    """
    return A / ((OmegaTrap**2 - omega**2)**2 + omega**2 * (Gamma)**2)

def PSD_fitting_eqn_with_background(A, OmegaTrap, Gamma, FlatBackground, omega):
    """
    The value of the fitting equation:
    A / ((OmegaTrap**2 - omega**2)**2 + (omega * Gamma)**2) + FlatBackground
    to be fit to the PSD

    Parameters
    ----------
    A : float
        Fitting constant A
        A = γ**2*Γ_0*(2*K_b*T_0)/(π*m)
        where:
            γ = conversionFactor
            Γ_0 = Damping factor due to environment
            π = pi
    OmegaTrap : float
        The trapping frequency in the axis of interest 
        (in angular frequency)
    Gamma : float
        The damping factor Gamma = Γ = Γ_0 + δΓ
        where:
            Γ_0 = Damping factor due to environment
            δΓ = extra damping due to feedback or other effects
    FlatBackground : float
        Adds a constant offset to the peak to account for a flat 
        noise background
    omega : float
        The angular frequency to calculate the value of the 
        fitting equation at 

    Returns
    -------
    Value : float
        The value of the fitting equation
    """
    return A / ((OmegaTrap**2 - omega**2)**2 + omega**2 * (Gamma)**2) + FlatBackground

def fit_PSD(Data, bandwidth, TrapFreqGuess, AGuess=0.1e10, GammaGuess=400, FlatBackground=None, MakeFig=True, show_fig=True):
    """
    Fits theory PSD to Data. Assumes highest point of PSD is the
    trapping frequency.

    Parameters
    ----------
    Data : DataObject
        data object to be fitted
    bandwidth : float
         bandwidth around trapping frequency peak to
         fit the theory PSD to
    TrapFreqGuess : float
        The approximate trapping frequency to use initially
        as the centre of the peak
    AGuess : float, optional
        The initial value of the A parameter to use in fitting
    GammaGuess : float, optional
        The initial value of the Gamma parameter to use in fitting
    FlatBackground : float, optional
        If given a number the fitting function assumes a flat 
        background to get more exact Area, which does not factor in
        noise. defaults to None, which fits a model with no flat 
        background contribution, basically no offset
    MakeFig : bool, optional
        Whether to construct and return the figure object showing
        the fitting. defaults to True
    show_fig : bool, optional
        Whether to show the figure object when it has been created.
        defaults to True

    Returns
    -------
    ParamsFit - Fitted parameters:
        [A, TrappingFrequency, Gamma, FlatBackground(optional)]
    ParamsFitErr - Error in fitted parameters:
        [AErr, TrappingFrequencyErr, GammaErr, FlatBackgroundErr(optional)]
    fig : matplotlib.figure.Figure object
        figure object containing the plot
    ax : matplotlib.axes.Axes object
        axes with the data plotted of the:
            - initial data
            - initial fit
            - final fit
    """
    AngFreqs = 2 * pi * Data.freqs
    Angbandwidth = 2 * pi * bandwidth
    AngTrapFreqGuess = 2 * pi * TrapFreqGuess

    ClosestToAngTrapFreqGuess = take_closest(AngFreqs, AngTrapFreqGuess)
    index_OmegaTrap = _np.where(AngFreqs == ClosestToAngTrapFreqGuess)[0][0]
    OmegaTrap = AngFreqs[index_OmegaTrap]

    f_fit_lower = take_closest(AngFreqs, OmegaTrap - Angbandwidth / 2)
    f_fit_upper = take_closest(AngFreqs, OmegaTrap + Angbandwidth / 2)

    indx_fit_lower = int(_np.where(AngFreqs == f_fit_lower)[0][0])
    indx_fit_upper = int(_np.where(AngFreqs == f_fit_upper)[0][0])

    if indx_fit_lower == indx_fit_upper:
        raise ValueError("Bandwidth argument must be higher, region is too thin.")
    
#    print(f_fit_lower, f_fit_upper)
#    print(AngFreqs[indx_fit_lower], AngFreqs[indx_fit_upper])

    # find highest point in region about guess for trap frequency - use that
    # as guess for trap frequency and recalculate region about the trap
    # frequency
    index_OmegaTrap = _np.where(Data.PSD == max(
        Data.PSD[indx_fit_lower:indx_fit_upper]))[0][0]

    OmegaTrap = AngFreqs[index_OmegaTrap]

#    print(OmegaTrap)

    f_fit_lower = take_closest(AngFreqs, OmegaTrap - Angbandwidth / 2)
    f_fit_upper = take_closest(AngFreqs, OmegaTrap + Angbandwidth / 2)

    indx_fit_lower = int(_np.where(AngFreqs == f_fit_lower)[0][0])
    indx_fit_upper = int(_np.where(AngFreqs == f_fit_upper)[0][0])

    logPSD = 10 * _np.log10(Data.PSD) # putting PSD in dB

    def calc_theory_PSD_curve_fit(freqs, A, TrapFreq, BigGamma, FlatBackground=None):
        if FlatBackground == None:
            Theory_PSD = 10 * \
                _np.log10(PSD_fitting_eqn(A, TrapFreq, BigGamma, freqs)) # PSD in dB
        else:
            Theory_PSD = 10* \
                _np.log10(PSD_fitting_eqn_with_background(A, TrapFreq, BigGamma, FlatBackground, freqs)) # PSD in dB
        if A < 0 or TrapFreq < 0 or BigGamma < 0:
            return 1e9
        else:
            return Theory_PSD

    datax = AngFreqs[indx_fit_lower:indx_fit_upper]
    datay = logPSD[indx_fit_lower:indx_fit_upper]

    if FlatBackground == None:
        p0 = _np.array([AGuess, OmegaTrap, GammaGuess])

        Params_Fit, Params_Fit_Err = fit_curvefit(p0,
                                                  datax,
                                                  datay,
                                                  calc_theory_PSD_curve_fit)
    else:
        p0 = _np.array([AGuess, OmegaTrap, GammaGuess, FlatBackground])

        Params_Fit, Params_Fit_Err = fit_curvefit(p0,
                                                  datax,
                                                  datay,
                                                  calc_theory_PSD_curve_fit)

    if MakeFig == True:
        fig = _plt.figure(figsize=properties["default_fig_size"])
        ax = fig.add_subplot(111)

        if FlatBackground==None:
            PSDTheory_fit_initial = 10 * _np.log10(
                PSD_fitting_eqn(p0[0], p0[1],
                                p0[2], AngFreqs))

            PSDTheory_fit = 10 * _np.log10(
                PSD_fitting_eqn(Params_Fit[0],
                                Params_Fit[1],
                                Params_Fit[2],
                                AngFreqs))
        else:
            PSDTheory_fit_initial = 10 * _np.log10(
                PSD_fitting_eqn_with_background(p0[0], p0[1],
                                                p0[2], p0[3], AngFreqs))

            PSDTheory_fit = 10 * _np.log10(
                PSD_fitting_eqn_with_background(Params_Fit[0],
                                                Params_Fit[1],
                                                Params_Fit[2],
                                                Params_Fit[3],
                                                AngFreqs))

        ax.plot(AngFreqs / (2 * pi), Data.PSD,
                color="darkblue", label="Raw PSD Data", alpha=0.5)
        ax.plot(AngFreqs / (2 * pi), 10**(PSDTheory_fit_initial / 10),
                '--', alpha=0.7, color="purple", label="initial vals")
        ax.plot(AngFreqs / (2 * pi), 10**(PSDTheory_fit / 10),
                color="red", label="fitted vals")
        ax.set_xlim([(OmegaTrap - 5 * Angbandwidth) / (2 * pi),
                     (OmegaTrap + 5 * Angbandwidth) / (2 * pi)])
        ax.plot([(OmegaTrap - Angbandwidth) / (2 * pi), (OmegaTrap - Angbandwidth) / (2 * pi)],
                [min(10**(logPSD / 10)),
                 max(10**(logPSD / 10))], '--',
                color="grey")
        ax.plot([(OmegaTrap + Angbandwidth) / (2 * pi), (OmegaTrap + Angbandwidth) / (2 * pi)],
                [min(10**(logPSD / 10)),
                 max(10**(logPSD / 10))], '--',
                color="grey")
        ax.semilogy()
        legend = ax.legend(loc="best", frameon = 1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("$S_{xx}$ ($V^2/Hz$)")
        if show_fig == True:
            _plt.show()
        return Params_Fit, Params_Fit_Err, fig, ax
    else:
        return Params_Fit, Params_Fit_Err, None, None


def extract_parameters(Pressure, PressureErr, A, AErr, Gamma0, Gamma0Err, method="chang"):
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
    PressureErr : float
        Error in the Pressure as a decimal (e.g. 15% error is 0.15) 
    A : float
        Fitting constant A
        A = γ**2*2*Γ_0*(K_b*T_0)/(π*m)
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
        The extracted parameters
    ParamsError : list
        [radiusError, massError, conversionFactorError]
        The error in the extracted parameters
    """
    Pressure = 100 * Pressure  # conversion to Pascals

    rho = 1800 # as quoted by Microspheres and Nanospheres  # kgm^3
    dm = 0.372e-9  # m O'Hanlon, 2003
    T0 = 300  # kelvin
    kB = Boltzmann  # m^2 kg s^-2 K-1
    eta = 18.27e-6  # Pa s, viscosity of air

    method = method.lower()
    if method == "rashid":
        radius = (0.619 * 9 * pi * eta * dm**2) / \
                 (_np.sqrt(2) * rho * kB * T0) * (Pressure/Gamma0)
        
    m_air = 4.81e-26 # molecular mass of air is 28.97 g/mol and Avogadro's Number 6.0221409^23
    if method == "chang":
        vbar = (8*kB*T0/(pi*m_air))**0.5
        radius = 16/(rho*pi*vbar)*(Pressure/Gamma0)/4 # CORRECTION FACTOR OF 4 APPLIED!!!!
    # see section 4.1.1 of Muddassar Rashid's 2016 Thesis for
    # derivation of this
    # see also page 132 of Jan Giesler's Thesis
    err_radius = radius * \
        _np.sqrt(((PressureErr * Pressure) / Pressure)
                 ** 2 + (Gamma0Err / Gamma0)**2)
    mass = rho * ((4 * pi * radius**3) / 3)
    err_mass = mass * 2 * err_radius / radius
    conversionFactor = _np.sqrt(A * mass / (4 * kB * T0 * Gamma0))
    err_conversionFactor = conversionFactor * \
        _np.sqrt((AErr / A)**2 + (err_mass / mass)
                 ** 2 + (Gamma0Err / Gamma0)**2)
    return [radius, mass, conversionFactor], [err_radius, err_mass, err_conversionFactor]


def get_ZXY_freqs(Data, zfreq, xfreq, yfreq, bandwidth=5000):
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
    bandwidth : float, optional
        The bandwidth around the approximate peak to look for the actual peak. The default value is 5000

    Returns
    -------
    trapfreqs : list
        List containing the trap frequencies in the following order (z, x, y)
    """
    trapfreqs = []
    for freq in [zfreq, xfreq, yfreq]:
        z_f_fit_lower = take_closest(Data.freqs, freq - bandwidth / 2)
        z_f_fit_upper = take_closest(Data.freqs, freq + bandwidth / 2)
        z_indx_fit_lower = int(_np.where(Data.freqs == z_f_fit_lower)[0][0])
        z_indx_fit_upper = int(_np.where(Data.freqs == z_f_fit_upper)[0][0])

        z_index_OmegaTrap = _np.where(Data.PSD == max(
            Data.PSD[z_indx_fit_lower:z_indx_fit_upper]))[0][0]
        # find highest point in region about guess for trap frequency
        # use that as guess for trap frequency and recalculate region
        # about the trap frequency
        z_OmegaTrap = Data.freqs[z_index_OmegaTrap]
        trapfreqs.append(z_OmegaTrap)
    return trapfreqs

def get_ZXY_data(Data, zf, xf, yf, FractionOfSampleFreq=1,
                 zwidth=10000, xwidth=5000, ywidth=5000,
                 filterImplementation="filtfilt",
                 timeStart=None, timeEnd=None,
                 NPerSegmentPSD=1000000,
                 MakeFig=True, show_fig=True):
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
    FractionOfSampleFreq : integer, optional
        The fraction of the sample frequency to sub-sample the data by.
        This sometimes needs to be done because a filter with the appropriate
        frequency response may not be generated using the sample rate at which
        the data was taken. Increasing this number means the x, y and z signals
        produced by this function will be sampled at a lower rate but a higher
        number means a higher chance that the filter produced will have a nice
        frequency response.
    zwidth : float, optional
        The width of the pass-band of the IIR filter to be generated to
        filter Z.
    xwidth : float, optional
        The width of the pass-band of the IIR filter to be generated to
        filter X.
    ywidth : float, optional
        The width of the pass-band of the IIR filter to be generated to
        filter Y.
    filterImplementation : string, optional
        filtfilt or lfilter - use scipy.filtfilt or lfilter
        default: filtfilt
    timeStart : float, optional
        Starting time for filtering
    timeEnd : float, optional
        Ending time for filtering
    show_fig : bool, optional
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
    timedata : ndarray
        Array containing the time data to go with the z, x, and y signal.
    """
    if timeStart == None:
        timeStart = Data.timeStart
    if timeEnd == None:
        timeEnd = Data.timeEnd

    time = Data.time.get_array()

    StartIndex = _np.where(time == take_closest(time, timeStart))[0][0]
    EndIndex = _np.where(time == take_closest(time, timeEnd))[0][0]

    SAMPLEFREQ = Data.SampleFreq / FractionOfSampleFreq

    if filterImplementation == "filtfilt":
        ApplyFilter = scipy.signal.filtfilt
    elif filterImplementation == "lfilter":
        ApplyFilter = scipy.signal.lfilter
    else:
        raise ValueError("filterImplementation must be one of [filtfilt, lfilter] you entered: {}".format(
            filterImplementation))

    input_signal = Data.voltage[StartIndex: EndIndex][0::FractionOfSampleFreq]

    bZ, aZ = make_butterworth_bandpass_b_a(zf, zwidth, SAMPLEFREQ)
    print("filtering Z")
    zdata = ApplyFilter(bZ, aZ, input_signal)

    if(_np.isnan(zdata).any()):
        raise ValueError(
            "Value Error: FractionOfSampleFreq must be higher, a sufficiently small sample frequency should be used to produce a working IIR filter.")

    bX, aX = make_butterworth_bandpass_b_a(xf, xwidth, SAMPLEFREQ)
    print("filtering X")
    xdata = ApplyFilter(bX, aX, input_signal)

    if(_np.isnan(xdata).any()):
        raise ValueError(
            "Value Error: FractionOfSampleFreq must be higher, a sufficiently small sample frequency should be used to produce a working IIR filter.")

    bY, aY = make_butterworth_bandpass_b_a(yf, ywidth, SAMPLEFREQ)
    print("filtering Y")
    ydata = ApplyFilter(bY, aY, input_signal)

    if(_np.isnan(ydata).any()):
        raise ValueError(
            "Value Error: FractionOfSampleFreq must be higher, a sufficiently small sample frequency should be used to produce a working IIR filter.")

    if MakeFig == True:        
        f, PSD = scipy.signal.welch(
            input_signal, SAMPLEFREQ, nperseg=NPerSegmentPSD)
        f_z, PSD_z = scipy.signal.welch(zdata, SAMPLEFREQ, nperseg=NPerSegmentPSD)
        f_y, PSD_y = scipy.signal.welch(ydata, SAMPLEFREQ, nperseg=NPerSegmentPSD)
        f_x, PSD_x = scipy.signal.welch(xdata, SAMPLEFREQ, nperseg=NPerSegmentPSD)
        fig, ax = _plt.subplots(figsize=properties["default_fig_size"])
        ax.plot(f, PSD)
        ax.plot(f_z, PSD_z, label="z")
        ax.plot(f_x, PSD_x, label="x")
        ax.plot(f_y, PSD_y, label="y")
        ax.legend(loc="best")
        ax.semilogy()
        ax.set_xlim([zf - zwidth, yf + ywidth])
    else:
        fig = None
        ax = None
    if show_fig == True:
        _plt.show()
    timedata = time[StartIndex: EndIndex][0::FractionOfSampleFreq]
    return zdata, xdata, ydata, timedata, fig, ax


def get_ZXY_data_IFFT(Data, zf, xf, yf,
                      zwidth=10000, xwidth=5000, ywidth=5000,
                      timeStart=None, timeEnd=None,
                      show_fig=True):
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
    zwidth : float, optional
        The width of the pass-band of the IIR filter to be generated to
        filter Z.
    xwidth : float, optional
        The width of the pass-band of the IIR filter to be generated to
        filter X.
    ywidth : float, optional
        The width of the pass-band of the IIR filter to be generated to
        filter Y.
    timeStart : float, optional
        Starting time for filtering
    timeEnd : float, optional
        Ending time for filtering
    show_fig : bool, optional
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
    timedata : ndarray
        Array containing the time data to go with the z, x, and y signal.
    """
    if timeStart == None:
        timeStart = Data.timeStart
    if timeEnd == None:
        timeEnd = Data.timeEnd

    time = Data.time.get_array()

    StartIndex = _np.where(time == take_closest(time, timeStart))[0][0]
    EndIndex = _np.where(time == take_closest(time, timeEnd))[0][0]

    SAMPLEFREQ = Data.SampleFreq

    input_signal = Data.voltage[StartIndex: EndIndex]

    zdata = IFFT_filter(input_signal, SAMPLEFREQ, zf -
                        zwidth / 2, zf + zwidth / 2)

    xdata = IFFT_filter(input_signal, SAMPLEFREQ, xf -
                        xwidth / 2, xf + xwidth / 2)

    ydata = IFFT_filter(input_signal, SAMPLEFREQ, yf -
                        ywidth / 2, yf + ywidth / 2)

    if show_fig == True:
        NPerSegment = len(Data.time)
        if NPerSegment > 1e7:
            NPerSegment = int(1e7)
        f, PSD = scipy.signal.welch(
            input_signal, SAMPLEFREQ, nperseg=NPerSegment)
        f_z, PSD_z = scipy.signal.welch(zdata, SAMPLEFREQ, nperseg=NPerSegment)
        f_y, PSD_y = scipy.signal.welch(ydata, SAMPLEFREQ, nperseg=NPerSegment)
        f_x, PSD_x = scipy.signal.welch(xdata, SAMPLEFREQ, nperseg=NPerSegment)
        _plt.plot(f, PSD)
        _plt.plot(f_z, PSD_z, label="z")
        _plt.plot(f_x, PSD_x, label="x")
        _plt.plot(f_y, PSD_y, label="y")
        _plt.legend(loc="best")
        _plt.xlim([zf - zwidth, yf + ywidth])
        _plt.xlabel('Frequency (Hz)')
        _plt.ylabel(r'$S_{xx}$ ($V^2/Hz$)')
        _plt.semilogy()
        _plt.title("filepath = %s" % (Data.filepath))
        _plt.show()

    timedata = time[StartIndex: EndIndex]
    return zdata, xdata, ydata, timedata


def animate(zdata, xdata, ydata,
            conversionFactorArray, timedata,
            BoxSize,
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
    conversionFactorArray : ndarray
        Array of 3 values of conversion factors for z, x and y (in units of Volts/Metre)
    timedata : ndarray
        Array containing the time data in seconds.
    BoxSize : float
        The size of the box in which to animate the particle - in nm
    timeSteps : int, optional
        Number of time steps to animate
    filename : string, optional
        filename to create the mp4 under (<filename>.mp4)

    """
    timePerFrame = 0.203
    print("This will take ~ {} minutes".format(timePerFrame * timeSteps / 60))

    convZ = conversionFactorArray[0] * 1e-9
    convX = conversionFactorArray[1] * 1e-9
    convY = conversionFactorArray[2] * 1e-9
    
    ZBoxStart = -BoxSize  # 1/conv*(_np.mean(zdata)-0.06)
    ZBoxEnd = BoxSize  # 1/conv*(_np.mean(zdata)+0.06)
    XBoxStart = -BoxSize  # 1/conv*(_np.mean(xdata)-0.06)
    XBoxEnd = BoxSize  # 1/conv*(_np.mean(xdata)+0.06)
    YBoxStart = -BoxSize  # 1/conv*(_np.mean(ydata)-0.06)
    YBoxEnd = BoxSize  # 1/conv*(_np.mean(ydata)+0.06)

    FrameInterval = 1  # how many timesteps = 1 frame in animation

    a = 20
    b = 0.6 * a
    myFPS = 7
    myBitrate = 1000000

    fig = _plt.figure(figsize=(a, b))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("{} us".format(timedata[0] * 1000000))
    ax.set_xlabel('X (nm)')
    ax.set_xlim([XBoxStart, XBoxEnd])
    ax.set_ylabel('Y (nm)')
    ax.set_ylim([YBoxStart, YBoxEnd])
    ax.set_zlabel('Z (nm)')
    ax.set_zlim([ZBoxStart, ZBoxEnd])
    ax.view_init(20, -30)

    #ax.view_init(0, 0)

    def setup_plot():
        XArray = 1 / convX * xdata[0]
        YArray = 1 / convY * ydata[0]
        ZArray = 1 / convZ * zdata[0]
        scatter = ax.scatter(XArray, YArray, ZArray)
        return scatter,

    def animate(i):
        # print "\r {}".format(i),
        print("Frame: {}".format(i), end="\r")
        ax.clear()
        ax.view_init(20, -30)
        ax.set_title("{} us".format(int(timedata[i] * 1000000)))
        ax.set_xlabel('X (nm)')
        ax.set_xlim([XBoxStart, XBoxEnd])
        ax.set_ylabel('Y (nm)')
        ax.set_ylim([YBoxStart, YBoxEnd])
        ax.set_zlabel('Z (nm)')
        ax.set_zlim([ZBoxStart, ZBoxEnd])
        XArray = 1 / convX * xdata[i]
        YArray = 1 / convY * ydata[i]
        ZArray = 1 / convZ * zdata[i]
        scatter = ax.scatter(XArray, YArray, ZArray)
        ax.scatter([XArray], [0], [-ZBoxEnd], c='k', alpha=0.9)
        ax.scatter([-XBoxEnd], [YArray], [0], c='k', alpha=0.9)
        ax.scatter([0], [YBoxEnd], [ZArray], c='k', alpha=0.9)

        Xx, Yx, Zx, Xy, Yy, Zy, Xz, Yz, Zz = [], [], [], [], [], [], [], [], []

        for j in range(0, 30):

            Xlast = 1 / convX * xdata[i - j]
            Ylast = 1 / convY * ydata[i - j]
            Zlast = 1 / convZ * zdata[i - j]

            Alpha = 0.5 - 0.05 * j
            if Alpha > 0:
                ax.scatter([Xlast], [0 + j * 10],
                           [-ZBoxEnd], c='grey', alpha=Alpha)
                ax.scatter([-XBoxEnd], [Ylast], [0 - j * 10],
                           c='grey', alpha=Alpha)
                ax.scatter([0 - j * 2], [YBoxEnd],
                           [Zlast], c='grey', alpha=Alpha)

                Xx.append(Xlast)
                Yx.append(0 + j * 10)
                Zx.append(-ZBoxEnd)

                Xy.append(-XBoxEnd)
                Yy.append(Ylast)
                Zy.append(0 - j * 10)

                Xz.append(0 - j * 2)
                Yz.append(YBoxEnd)
                Zz.append(Zlast)

            if j < 15:
                XCur = 1 / convX * xdata[i - j + 1]
                YCur = 1 / convY * ydata[i - j + 1]
                ZCur = 1 / convZ * zdata[i - j + 1]
                ax.plot([Xlast, XCur], [Ylast, YCur], [Zlast, ZCur], alpha=0.4)

        ax.plot_wireframe(Xx, Yx, Zx, color='grey')
        ax.plot_wireframe(Xy, Yy, Zy, color='grey')
        ax.plot_wireframe(Xz, Yz, Zz, color='grey')

        return scatter,

    anim = _animation.FuncAnimation(fig, animate, int(
        timeSteps / FrameInterval), init_func=setup_plot, blit=True)

    _plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    mywriter = _animation.FFMpegWriter(fps=myFPS, bitrate=myBitrate)
    # , fps = myFPS, bitrate = myBitrate)
    anim.save('{}.mp4'.format(filename), writer=mywriter)
    return None

def animate_2Dscatter(x, y, NumAnimatedPoints=50, NTrailPoints=20, 
    xlabel="", ylabel="",
    xlims=None, ylims=None, filename="testAnim.mp4", 
    bitrate=1e5, dpi=5e2, fps=30, figsize = [6, 6]):
    """
    Animates x and y - where x and y are 1d arrays of x and y 
    positions and it plots x[i:i+NTrailPoints] and y[i:i+NTrailPoints]
    against each other and iterates through i. 

    """
    fig, ax = _plt.subplots(figsize = figsize)

    alphas = _np.linspace(0.1, 1, NTrailPoints)
    rgba_colors = _np.zeros((NTrailPoints,4))
    # for red the first column needs to be one
    rgba_colors[:,0] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas


    scatter = ax.scatter(x[0:NTrailPoints], y[0:NTrailPoints], color=rgba_colors)

    if xlims == None:
        xlims = (min(x), max(x))
    if ylims == None:
        ylims = (min(y), max(y))

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def animate(i, scatter):
        scatter.axes.clear() # clear old scatter object
        scatter = ax.scatter(x[i:i+NTrailPoints], y[i:i+NTrailPoints], color=rgba_colors, animated=True) 
        # create new scatter with updated data
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return scatter,


    ani = _animation.FuncAnimation(fig, animate, _np.arange(1, NumAnimatedPoints),
                                  interval=25, blit=True, fargs=[scatter])
    ani.save(filename, bitrate=bitrate, dpi=dpi, fps=fps)
    return None

def animate_2Dscatter_slices(x, y, NumAnimatedPoints=50, 
    xlabel="", ylabel="",
    xlims=None, ylims=None, filename="testAnim.mp4", 
    bitrate=1e5, dpi=5e2, fps=30, figsize = [6, 6]):
    """
    Animates x and y - where x and y are both 2d arrays of x and y 
    positions and it plots x[i] against y[i] and iterates through i. 

    """
    fig, ax = _plt.subplots(figsize = figsize)

    scatter = ax.scatter(x[0], y[0])

    if xlims == None:
        xlims = (_np.min(x), _np.max(x))
    if ylims == None:
        ylims = (_np.min(y), _np.max(y))

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def animate(i, scatter):
        scatter.axes.clear() # clear old scatter object
        scatter = ax.scatter(x[i], y[i], animated=True)
        # create new scatter with updated data
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return scatter,


    ani = _animation.FuncAnimation(fig, animate, _np.arange(1, NumAnimatedPoints),
                                  interval=25, blit=True, fargs=[scatter])
    ani.save(filename, bitrate=bitrate, dpi=dpi, fps=fps)
    return None


def IFFT_filter(Signal, SampleFreq, lowerFreq, upperFreq, PyCUDA = False):
    """
    Filters data using fft -> zeroing out fft bins -> ifft

    Parameters
    ----------
    Signal : ndarray
        Signal to be filtered
    SampleFreq : float
        Sample frequency of signal
    lowerFreq : float
        Lower frequency of bandpass to allow through filter
    upperFreq : float
       Upper frequency of bandpass to allow through filter
    PyCUDA : bool, optional
       If True, uses PyCUDA to accelerate the FFT and IFFT
       via using your NVIDIA-GPU
       If False, performs FFT and IFFT with conventional
       scipy.fftpack

    Returns
    -------
    FilteredData : ndarray
        Array containing the filtered data
    """
    if PyCUDA==True:
        Signalfft=calc_fft_with_PyCUDA(Signal)
    else:
        print("starting fft")
        Signalfft = scipy.fftpack.fft(Signal)
    print("starting freq calc")
    freqs = _np.fft.fftfreq(len(Signal)) * SampleFreq
    print("starting bin zeroing")
    Signalfft[_np.where(freqs < lowerFreq)] = 0
    Signalfft[_np.where(freqs > upperFreq)] = 0
    if PyCUDA==True:
        FilteredSignal = 2 * calc_ifft_with_PyCUDA(Signalfft)
    else:
        print("starting ifft")
        FilteredSignal = 2 * scipy.fftpack.ifft(Signalfft)
    print("done")
    return _np.real(FilteredSignal)

def calc_fft_with_PyCUDA(Signal):
    """
    Calculates the FFT of the passed signal by using
    the scikit-cuda libary which relies on PyCUDA

    Parameters
    ----------
    Signal : ndarray
        Signal to be transformed into Fourier space

    Returns
    -------
    Signalfft : ndarray
        Array containing the signal's FFT
    """
    print("starting fft")
    Signal = Signal.astype(_np.float32)
    Signal_gpu = _gpuarray.to_gpu(Signal)
    Signalfft_gpu = _gpuarray.empty(len(Signal)//2+1,_np.complex64)
    plan = _Plan(Signal.shape,_np.float32,_np.complex64)
    _fft(Signal_gpu, Signalfft_gpu, plan)
    Signalfft = Signalfft_gpu.get() #only 2N+1 long
    Signalfft = _np.hstack((Signalfft,_np.conj(_np.flipud(Signalfft[1:len(Signal)//2]))))
    print("fft done")
    return Signalfft

def calc_ifft_with_PyCUDA(Signalfft):
    """
    Calculates the inverse-FFT of the passed FFT-signal by 
    using the scikit-cuda libary which relies on PyCUDA

    Parameters
    ----------
    Signalfft : ndarray
        FFT-Signal to be transformed into Real space

    Returns
    -------
    Signal : ndarray
        Array containing the ifft signal
    """
    print("starting ifft")
    Signalfft = Signalfft.astype(_np.complex64)
    Signalfft_gpu = _gpuarray.to_gpu(Signalfft[0:len(Signalfft)//2+1])
    Signal_gpu = _gpuarray.empty(len(Signalfft),_np.float32)
    plan = _Plan(len(Signalfft),_np.complex64,_np.float32)
    _ifft(Signalfft_gpu, Signal_gpu, plan)
    Signal = Signal_gpu.get()/(2*len(Signalfft)) #normalising as CUDA IFFT is un-normalised
    print("ifft done")
    return Signal

def butterworth_filter(Signal, SampleFreq, lowerFreq, upperFreq):
    """
    Filters data using by constructing a 5th order butterworth
    IIR filter and using scipy.signal.filtfilt, which does
    phase correction after implementing the filter (as IIR 
    filter apply a phase change)

    Parameters
    ----------
    Signal : ndarray
        Signal to be filtered
    SampleFreq : float
        Sample frequency of signal
    lowerFreq : float
        Lower frequency of bandpass to allow through filter
    upperFreq : float
       Upper frequency of bandpass to allow through filter

    Returns
    -------
    FilteredData : ndarray
        Array containing the filtered data
    """
    b, a = make_butterworth_b_a(lowerFreq, upperFreq, SampleFreq)
    FilteredSignal = scipy.signal.filtfilt(b, a, Signal)
    return _np.real(FilteredSignal)


def make_butterworth_b_a(lowcut, highcut, SampleFreq, order=5, btype='band'):
    """
    Generates the b and a coefficients for a butterworth IIR filter.

    Parameters
    ----------
    lowcut : float
        frequency of lower bandpass limit
    highcut : float
        frequency of higher bandpass limit
    SampleFreq : float
        Sample frequency of filter
    order : int, optional
        order of IIR filter. Is 5 by default
    btype : string, optional
        type of filter to make e.g. (band, low, high)

    Returns
    -------
    b : ndarray
        coefficients multiplying the current and past inputs (feedforward coefficients)
    a : ndarray
        coefficients multiplying the past outputs (feedback coefficients)
    """
    nyq = 0.5 * SampleFreq
    low = lowcut / nyq
    high = highcut / nyq
    if btype.lower() == 'band':
        b, a = scipy.signal.butter(order, [low, high], btype = btype)
    elif btype.lower() == 'low':
        b, a = scipy.signal.butter(order, low, btype = btype)
    elif btype.lower() == 'high':
        b, a = scipy.signal.butter(order, high, btype = btype)
    else:
        raise ValueError('Filter type unknown')
    return b, a
def make_butterworth_bandpass_b_a(CenterFreq, bandwidth, SampleFreq, order=5, btype='band'):
    """
    Generates the b and a coefficients for a butterworth bandpass IIR filter.

    Parameters
    ----------
    CenterFreq : float
        central frequency of bandpass
    bandwidth : float
        width of the bandpass from centre to edge
    SampleFreq : float
        Sample frequency of filter
    order : int, optional
        order of IIR filter. Is 5 by default
    btype : string, optional
        type of filter to make e.g. (band, low, high)

    Returns
    -------
    b : ndarray
        coefficients multiplying the current and past inputs (feedforward coefficients)
    a : ndarray
        coefficients multiplying the past outputs (feedback coefficients)
    """    
    lowcut = CenterFreq-bandwidth/2
    highcut = CenterFreq+bandwidth/2
    b, a = make_butterworth_b_a(lowcut, highcut, SampleFreq, order, btype)
    return b, a


def IIR_filter_design(CentralFreq, bandwidth, transitionWidth, SampleFreq, GainStop=40, GainPass=0.01):
    """
    Function to calculate the coefficients of an IIR filter, 
    IMPORTANT NOTE: make_butterworth_bandpass_b_a and make_butterworth_b_a
    can produce IIR filters with higher sample rates and are prefereable
    due to this.

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
    GainStop : float, optional
        The dB of attenuation within the stopband (i.e. outside the passband)
    GainPass : float, optional
        The dB attenuation inside the passband (ideally close to 0 for a bandpass filter)

    Returns
    -------
    b : ndarray
        coefficients multiplying the current and past inputs (feedforward coefficients)
    a : ndarray
        coefficients multiplying the past outputs (feedback coefficients)
    """
    NyquistFreq = SampleFreq / 2
    if (CentralFreq + bandwidth / 2 + transitionWidth > NyquistFreq):
        raise ValueError(
            "Need a higher Sample Frequency for this Central Freq, Bandwidth and transition Width")

    CentralFreqNormed = CentralFreq / NyquistFreq
    bandwidthNormed = bandwidth / NyquistFreq
    transitionWidthNormed = transitionWidth / NyquistFreq
    bandpass = [CentralFreqNormed - bandwidthNormed /
                2, CentralFreqNormed + bandwidthNormed / 2]
    bandstop = [CentralFreqNormed - bandwidthNormed / 2 - transitionWidthNormed,
                CentralFreqNormed + bandwidthNormed / 2 + transitionWidthNormed]
    print(bandpass, bandstop)
    b, a = scipy.signal.iirdesign(bandpass, bandstop, GainPass, GainStop)
    return b, a

def get_freq_response(a, b, show_fig=True, SampleFreq=(2 * pi), NumOfFreqs=500, whole=False):
    """
    This function takes an array of coefficients and finds the frequency
    response of the filter using scipy.signal.freqz.
    show_fig sets if the response should be plotted

    Parameters
    ----------
    b : array_like
        Coefficients multiplying the x values (inputs of the filter)

    a : array_like
        Coefficients multiplying the y values (outputs of the filter)
    show_fig : bool, optional
        Verbosity of function (i.e. whether to plot frequency and phase
        response or whether to just return the values.)
        Options (Default is 1):
        False - Do not plot anything, just return values
        True - Plot Frequency and Phase response and return values
    SampleFreq : float, optional
        Sample frequency (in Hz) to simulate (used to convert frequency range
        to normalised frequency range)
    NumOfFreqs : int, optional
        Number of frequencies to use to simulate the frequency and phase
        response of the filter. Default is 500.
    Whole : bool, optional
        Sets whether to plot the whole response (0 to sample freq)
        or just to plot 0 to Nyquist (SampleFreq/2): 
        False - (default) plot 0 to Nyquist (SampleFreq/2)
        True - plot the whole response (0 to sample freq)

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
    w, h = scipy.signal.freqz(b=b, a=a, worN=NumOfFreqs, whole=whole)
    freqList = w / (pi) * SampleFreq / 2.0
    himag = _np.array([hi.imag for hi in h])
    GainArray = 20 * _np.log10(_np.abs(h))
    PhaseDiffArray = _np.unwrap(_np.arctan2(_np.imag(h), _np.real(h)))

    fig1 = _plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(freqList, GainArray, '-', label="Specified Filter")
    ax1.set_title("Frequency Response")
    if SampleFreq == 2 * pi:
        ax1.set_xlabel(("$\Omega$ - Normalized frequency "
                       "($\pi$=Nyquist Frequency)"))
    else:
        ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel("Gain (dB)")
    ax1.set_xlim([0, SampleFreq / 2.0])
    if show_fig == True:
        _plt.show()
    fig2 = _plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(freqList, PhaseDiffArray, '-', label="Specified Filter")
    ax2.set_title("Phase Response")
    if SampleFreq == 2 * pi:
        ax2.set_xlabel(("$\Omega$ - Normalized frequency "
                       "($\pi$=Nyquist Frequency)"))
    else:
        ax2.set_xlabel("frequency (Hz)")

    ax2.set_ylabel("Phase Difference")
    ax2.set_xlim([0, SampleFreq / 2.0])
    if show_fig == True:
        _plt.show()
    return freqList, GainArray, PhaseDiffArray, fig1, ax1, fig2, ax2


def multi_plot_PSD(DataArray, xlim=[0, 500], units="kHz", LabelArray=[], ColorArray=[], alphaArray=[], show_fig=True):
    """
    plot the pulse spectral density for multiple data sets on the same
    axes.

    Parameters
    ----------
    DataArray : array-like
        array of DataObject instances for which to plot the PSDs
    xlim : array-like, optional
        2 element array specifying the lower and upper x limit for which to
        plot the Power Spectral Density
    units : string
        units to use for the x axis
    LabelArray : array-like, optional
        array of labels for each data-set to be plotted
    ColorArray : array-like, optional
        array of colors for each data-set to be plotted
    show_fig : bool, optional
       If True runs plt.show() before returning figure
       if False it just returns the figure object.
       (the default is True, it shows the figure)

    Returns
    -------
    fig : matplotlib.figure.Figure object
        The figure object created
    ax : matplotlib.axes.Axes object
        The axes object created
    """
    unit_prefix = units[:-2] # removed the last 2 chars
    if LabelArray == []:
        LabelArray = ["DataSet {}".format(i)
                      for i in _np.arange(0, len(DataArray), 1)]
    if ColorArray == []:
        ColorArray = _np.empty(len(DataArray))
        ColorArray = list(ColorArray)
        for i, ele in enumerate(ColorArray):
            ColorArray[i] = None    

    if alphaArray == []:
        alphaArray = _np.empty(len(DataArray))
        alphaArray = list(alphaArray)
        for i, ele in enumerate(alphaArray):
            alphaArray[i] = None    

            
    fig = _plt.figure(figsize=properties['default_fig_size'])
    ax = fig.add_subplot(111)

    for i, data in enumerate(DataArray):
        ax.semilogy(unit_conversion(data.freqs, unit_prefix), data.PSD, label=LabelArray[i], color=ColorArray[i], alpha=alphaArray[i])
            
    ax.set_xlabel("Frequency ({})".format(units))
    ax.set_xlim(xlim)
    ax.grid(which="major")
    legend = ax.legend(loc="best", frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('white')
    ax.set_ylabel("PSD ($v^2/Hz$)")

    _plt.title('filedir=%s' % (DataArray[0].filedir))

    if show_fig == True:
        _plt.show()
    return fig, ax


def multi_plot_time(DataArray, SubSampleN=1, units='s', xlim=None, ylim=None, LabelArray=[], show_fig=True):
    """
    plot the time trace for multiple data sets on the same axes.

    Parameters
    ----------
    DataArray : array-like
        array of DataObject instances for which to plot the PSDs
    SubSampleN : int, optional
        Number of intervals between points to remove (to sub-sample data so
        that you effectively have lower sample rate to make plotting easier
        and quicker.
    xlim : array-like, optional
        2 element array specifying the lower and upper x limit for which to
        plot the time signal
    LabelArray : array-like, optional
        array of labels for each data-set to be plotted
    show_fig : bool, optional
       If True runs plt.show() before returning figure
       if False it just returns the figure object.
       (the default is True, it shows the figure) 

    Returns
    -------
    fig : matplotlib.figure.Figure object
        The figure object created
    ax : matplotlib.axes.Axes object
        The axes object created
    """
    unit_prefix = units[:-1] # removed the last char
    if LabelArray == []:
        LabelArray = ["DataSet {}".format(i)
                      for i in _np.arange(0, len(DataArray), 1)]
    fig = _plt.figure(figsize=properties['default_fig_size'])
    ax = fig.add_subplot(111)
    
    for i, data in enumerate(DataArray):
        ax.plot(unit_conversion(data.time.get_array()[::SubSampleN], unit_prefix), data.voltage[::SubSampleN],
                alpha=0.8, label=LabelArray[i])
    ax.set_xlabel("time (s)")
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    ax.grid(which="major")
    legend = ax.legend(loc="best", frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('white')

    ax.set_ylabel("voltage (V)")
    if show_fig == True:
        _plt.show()
    return fig, ax


def multi_subplots_time(DataArray, SubSampleN=1, units='s', xlim=None, ylim=None, LabelArray=[], show_fig=True):
    """
    plot the time trace on multiple axes

    Parameters
    ----------
    DataArray : array-like
        array of DataObject instances for which to plot the PSDs
    SubSampleN : int, optional
        Number of intervals between points to remove (to sub-sample data so
        that you effectively have lower sample rate to make plotting easier
        and quicker.
    xlim : array-like, optional
        2 element array specifying the lower and upper x limit for which to
        plot the time signal
    LabelArray : array-like, optional
        array of labels for each data-set to be plotted
    show_fig : bool, optional
       If True runs plt.show() before returning figure
       if False it just returns the figure object.
       (the default is True, it shows the figure) 

    Returns
    -------
    fig : matplotlib.figure.Figure object
        The figure object created
    axs : list of matplotlib.axes.Axes objects
        The list of axes object created
    """
    unit_prefix = units[:-1] # removed the last char
    NumDataSets = len(DataArray)

    if LabelArray == []:
        LabelArray = ["DataSet {}".format(i)
                      for i in _np.arange(0, len(DataArray), 1)]

    fig, axs = _plt.subplots(NumDataSets, 1)

    for i, data in enumerate(DataArray):
        axs[i].plot(unit_conversion(data.time.get_array()[::SubSampleN], unit_prefix), data.voltage[::SubSampleN],
                    alpha=0.8, label=LabelArray[i])
        axs[i].set_xlabel("time ({})".format(units))
        axs[i].grid(which="major")
        axs[i].legend(loc="best")
        axs[i].set_ylabel("voltage (V)")
        if xlim != None:
            axs[i].set_xlim(xlim)
        if ylim != None:
            axs[i].set_ylim(ylim)
    if show_fig == True:
        _plt.show()
    return fig, axs

def arrange_plots_on_one_canvas(FigureAxTupleArray, title='', SubtitleArray = [], show_fig=True):
    """
    Arranges plots, given in an array of tuples consisting of fig and axs, 
    onto a subplot-figure consisting of 2 horizontal times the lenght of the
    passed (fig,axs)-array divided by 2 vertical subplots 

    Parameters
    ----------
    FigureAxTupleArray : array-like
        array of Tuples(fig, axs) outputted from the other plotting funtions 
        inside optoanalysis
    title : string, optional
        string for the global title of the overall combined figure 
    SubtitleArray : array-like, optional
        array of titles for each figure-set to be plotted, i.e. subplots 
    show_fig : bool, optional
       If True runs plt.show() before returning figure
       if False it just returns the figure object.
       (the default is True, it shows the figure) 

    Returns
    -------
    fig : matplotlib.figure.Figure object
        The figure object created
    axs : list of matplotlib.axes.Axes objects
        The list of axes object created
    """
    if SubtitleArray == []:
        SubtitleArray = ["Plot {}".format(i)
                      for i in _np.arange(0, len(FigureAxTupleArray), 1)]
    SingleFigSize = FigureAxTupleArray[0][0].get_size_inches()
    combinedFig=_plt.figure(figsize=(2*SingleFigSize[0],_np.ceil(len(FigureAxTupleArray)/2)*SingleFigSize[1]))
    for index in range(len(FigureAxTupleArray)):
        individualPlot = FigureAxTupleArray[index]
        individualPlot[0].set_size_inches((2*SingleFigSize[0],_np.ceil(len(FigureAxTupleArray)/2)*SingleFigSize[1]))
        ax = individualPlot[1]
        ax.set_title(SubtitleArray[index])
        ax.remove()
        ax.figure = combinedFig
        ax.change_geometry(int(_np.ceil(len(FigureAxTupleArray)/2)),2,1+index)
        combinedFig.axes.append(ax)
        combinedFig.add_axes(ax)
        #_plt.close(individualPlot[0])
    combinedFig.subplots_adjust(hspace=.4)
    combinedFig.suptitle(title)
    if show_fig == True:
        _plt.show()
    return combinedFig

def calc_PSD(Signal, SampleFreq, NPerSegment=1000000, window="hann"):
    """
    Extracts the pulse spectral density (PSD) from the data.

    Parameters
    ----------
    Signal : array-like
        Array containing the signal to have the PSD calculated for
    SampleFreq : float
        Sample frequency of the signal array
    NPerSegment : int, optional
        Length of each segment used in scipy.welch
        default = 1000000
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
    freqs, PSD = scipy.signal.welch(Signal, SampleFreq,
                                    window=window, nperseg=NPerSegment)
    PSD = PSD[freqs.argsort()]
    freqs.sort()
    return freqs, PSD

def calc_autocorrelation(Signal, FFT=False, PyCUDA=False):
    """
    Calculates the autocorrelation from a given Signal via using
    

    Parameters
    ----------
    Signal : array-like
        Array containing the signal to have the autocorrelation calculated for
    FFT : optional, bool
        Uses FFT to accelerate autocorrelation calculation, but assumes certain
        certain periodicity on the signal to autocorrelate. Zero-padding is added
        to account for this periodicity assumption.
    PyCUDA : bool, optional
       If True, uses PyCUDA to accelerate the FFT and IFFT
       via using your NVIDIA-GPU
       If False, performs FFT and IFFT with conventional
       scipy.fftpack

    Returns
    -------
    Autocorrelation : ndarray
            Array containing the value of the autocorrelation evaluated
            at the corresponding amount of shifted array-index.
    """
    if FFT==True:
        Signal_padded = scipy.fftpack.ifftshift((Signal-_np.average(Signal))/_np.std(Signal))
        n, = Signal_padded.shape
        Signal_padded = _np.r_[Signal_padded[:n//2], _np.zeros_like(Signal_padded), Signal_padded[n//2:]]
        if PyCUDA==True:
            f = calc_fft_with_PyCUDA(Signal_padded)
        else:
            f = scipy.fftpack.fft(Signal_padded)
        p = _np.absolute(f)**2
        if PyCUDA==True:
            autocorr = calc_ifft_with_PyCUDA(p)
        else:
            autocorr = scipy.fftpack.ifft(p)
        return _np.real(autocorr)[:n//2]/(_np.arange(n//2)[::-1]+n//2)
    else:
        Signal = Signal - _np.mean(Signal)
        autocorr = scipy.signal.correlate(Signal, Signal, mode='full')
        return autocorr[autocorr.size//2:]/autocorr[autocorr.size//2]

def _GetRealImagArray(Array):
    """
    Returns the real and imaginary components of each element in an array and returns them in 2 resulting arrays.

    Parameters
    ----------
    Array : ndarray
        Input array

    Returns
    -------
    RealArray : ndarray
        The real components of the input array
    ImagArray : ndarray
        The imaginary components of the input array
    """
    ImagArray = _np.array([num.imag for num in Array])
    RealArray = _np.array([num.real for num in Array])
    return RealArray, ImagArray


def _GetComplexConjugateArray(Array):
    """
    Calculates the complex conjugate of each element in an array and returns the resulting array.

    Parameters
    ----------
    Array : ndarray
        Input array

    Returns
    -------
    ConjArray : ndarray
        The complex conjugate of the input array.
    """
    ConjArray = _np.array([num.conj() for num in Array])
    return ConjArray


def fm_discriminator(Signal):
    """
    Calculates the digital FM discriminator from a real-valued time signal.

    Parameters
    ----------
    Signal : array-like
        A real-valued time signal

    Returns
    -------
    fmDiscriminator : array-like
        The digital FM discriminator of the argument signal
    """
    S_analytic = _hilbert(Signal)
    S_analytic_star = _GetComplexConjugateArray(S_analytic)
    S_analytic_hat = S_analytic[1:] * S_analytic_star[:-1]
    R, I = _GetRealImagArray(S_analytic_hat)
    fmDiscriminator = _np.arctan2(I, R)
    return fmDiscriminator


def _approx_equal(a, b, tol):
    """
    Returns if b is approximately equal to be a within a certain percentage tolerance.

    Parameters
    ----------
    a : float
        first value
    b : float
        second value
    tol : float
        tolerance in percentage
    """
    return abs(a - b) / a * 100 < tol


def _is_this_a_collision(ArgList):
    """
    Detects if a particular point is during collision after effect (i.e. a phase shift) or not.

    Parameters
    ----------
    ArgList : array_like
        Contains the following elements:
            value : float
                value of the FM discriminator
            mean_fmd : float
                the mean value of the FM discriminator
            tolerance : float
                The tolerance in percentage that it must be away from the mean value for it
                to be counted as a collision event.

    Returns
    -------
    is_this_a_collision : bool
        True if this is a collision event, false if not.
    """
    value, mean_fmd, tolerance = ArgList
    if not _approx_equal(mean_fmd, value, tolerance):
        return True
    else:
        return False


def find_collisions(Signal, tolerance=50):
    """
    Finds collision events in the signal from the shift in phase of the signal.

    Parameters
    ----------
    Signal : array_like
        Array containing the values of the signal of interest containing a single frequency.
    tolerance : float
        Percentage tolerance, if the value of the FM Discriminator varies from the mean by this
        percentage it is counted as being during a collision event (or the aftermath of an event).

    Returns
    -------
    Collisions : ndarray
        Array of booleans, true if during a collision event, false otherwise.
    """
    fmd = fm_discriminator(Signal)
    mean_fmd = _np.mean(fmd)

    Collisions = [_is_this_a_collision(
        [value, mean_fmd, tolerance]) for value in fmd]

    return Collisions


def count_collisions(Collisions):
    """
    Counts the number of unique collisions and gets the collision index.

    Parameters
    ----------
    Collisions : array_like
        Array of booleans, containing true if during a collision event, false otherwise.

    Returns
    -------
    CollisionCount : int
        Number of unique collisions
    CollisionIndicies : list
        Indicies of collision occurance
    """
    CollisionCount = 0
    CollisionIndicies = []
    lastval = True
    for i, val in enumerate(Collisions):
        if val == True and lastval == False:
            CollisionIndicies.append(i)
            CollisionCount += 1
        lastval = val
    return CollisionCount, CollisionIndicies


def parse_orgtable(lines):
    """
    Parse an org-table (input as a list of strings split by newline)
    into a Pandas data frame.

    Parameters
    ----------
    lines : string
        an org-table input as a list of strings split by newline
    
    Returns
    -------
    dataframe : pandas.DataFrame
        A data frame containing the org-table's data
    """
    def parseline(l):
        w = l.split('|')[1:-1]
        return [wi.strip() for wi in w]
    columns = parseline(lines[0])

    data = []
    for line in lines[2:]:
        data.append(map(str, parseline(line)))
    dataframe = _pd.DataFrame(data=data, columns=columns)
    dataframe.set_index("RunNo")
    return dataframe

def plot_3d_dist(Z, X, Y, N=1000, AxisOffset=0, Angle=-40, LowLim=None, HighLim=None, show_fig=True):
    """
    Plots Z, X and Y as a 3d scatter plot with heatmaps of each axis pair.

    Parameters
    ----------
    Z : ndarray
        Array of Z positions with time
    X : ndarray
        Array of X positions with time
    Y : ndarray
        Array of Y positions with time
    N : optional, int
        Number of time points to plot (Defaults to 1000)
    AxisOffset : optional, double
        Offset to add to each axis from the data - used to get a better view
        of the heat maps (Defaults to 0)
    LowLim : optional, double
        Lower limit of x, y and z axis
    HighLim : optional, double
        Upper limit of x, y and z axis
    show_fig : optional, bool
        Whether to show the produced figure before returning

    Returns
    -------
    fig : matplotlib.figure.Figure object
        The figure object created
    ax : matplotlib.axes.Axes object
        The subplot object created
    """
    angle = Angle
    fig = _plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    y = Z[0:N]
    x = X[0:N]
    z = Y[0:N]

    ax.scatter(x, y, z, alpha=0.3)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    if LowLim != None:
        lowLim = LowLim - AxisOffset
    else:
        lowLim = min([xlim[0], ylim[0], zlim[0]]) - AxisOffset
    if HighLim != None:
        highLim = HighLim + AxisOffset
    else:
        highLim = max([xlim[1], ylim[1], zlim[1]]) + AxisOffset
    ax.set_xlim([lowLim, highLim])
    ax.set_ylim([lowLim, highLim])
    ax.set_zlim([lowLim, highLim])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(30, angle)

    h, yedges, zedges = _np.histogram2d(y, z, bins=50)
    h = h.transpose()
    normalized_map = _plt.cm.Blues(h/h.max())
    yy, zz = _np.meshgrid(yedges, zedges)
    xpos = lowLim # Plane of histogram
    xflat = _np.full_like(yy, xpos) 
    p = ax.plot_surface(xflat, yy, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

    h, xedges, zedges = _np.histogram2d(x, z, bins=50)
    h = h.transpose()
    normalized_map = _plt.cm.Blues(h/h.max())
    xx, zz = _np.meshgrid(xedges, zedges)
    ypos = highLim # Plane of histogram
    yflat = _np.full_like(xx, ypos) 
    p = ax.plot_surface(xx, yflat, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

    h, yedges, xedges = _np.histogram2d(y, x, bins=50)
    h = h.transpose()
    normalized_map = _plt.cm.Blues(h/h.max())
    yy, xx = _np.meshgrid(yedges, xedges)
    zpos = lowLim # Plane of histogram
    zflat = _np.full_like(yy, zpos) 
    p = ax.plot_surface(xx, yy, zflat, facecolors=normalized_map, rstride=1, cstride=1, shade=False)
    if show_fig == True:
        _plt.show()
    return fig, ax

def multi_plot_3d_dist(ZXYData, N=1000, AxisOffset=0, Angle=-40, LowLim=None, HighLim=None, ColorArray=None, alphaLevel=0.3, show_fig=True):
    """
    Plots serveral Z, X and Y datasets as a 3d scatter plot with heatmaps of each axis pair in each dataset.

    Parameters
    ----------
    ZXYData : ndarray
        Array of arrays containing Z, X, Y data e.g. [[Z1, X1, Y1], [Z2, X2, Y2]]
    N : optional, int
        Number of time points to plot (Defaults to 1000)
    AxisOffset : optional, double
        Offset to add to each axis from the data - used to get a better view
        of the heat maps (Defaults to 0)
    LowLim : optional, double
        Lower limit of x, y and z axis
    HighLim : optional, double
        Upper limit of x, y and z axis
    show_fig : optional, bool
        Whether to show the produced figure before returning

    Returns
    -------
    fig : matplotlib.figure.Figure object
        The figure object created
    ax : matplotlib.axes.Axes object
        The subplot object created
    """
    if ZXYData.shape[1] != 3:
        raise ValueError("Parameter ZXYData should be an array of length-3 arrays containing arrays of Z, X and Y data")
    if ColorArray != None:
        if ZXYData.shape[0] != len(ColorArray):
            raise ValueError("Parameter ColorArray should be the same lenth as ZXYData")
    else:
        ColorArray = list(mcolours.BASE_COLORS.keys())
        #ColorArray = ['b', 'g', 'r']
        #    ColorMapArray = [_plt.cm.Blues, _plt.cm.Greens, _plt.cm.Reds]
        if ZXYData.shape[0] > len(ColorArray):
            raise NotImplementedError("Only {} datasets can be plotted with automatic colors".format(len(ColorArray)))
        
    angle = Angle
    fig = _plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for datindx, ZXY in enumerate(ZXYData):
        y = ZXY[0][0:N]
        x = ZXY[1][0:N]
        z = ZXY[2][0:N]
        ax.scatter(x, y, z, alpha=alphaLevel, color=ColorArray[datindx])
        
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    if LowLim != None:
        lowLim = LowLim - AxisOffset
    else:
        lowLim = min([xlim[0], ylim[0], zlim[0]]) - AxisOffset
    if HighLim != None:
        highLim = HighLim + AxisOffset
    else:
        highLim = max([xlim[1], ylim[1], zlim[1]]) + AxisOffset
    ax.set_xlim([lowLim, highLim])
    ax.set_ylim([lowLim, highLim])
    ax.set_zlim([lowLim, highLim])

    for datindx, ZXY in enumerate(ZXYData):
        y = ZXY[0][0:N]
        x = ZXY[1][0:N]
        z = ZXY[2][0:N]
        
        #h, yedges, zedges = _np.histogram2d(y, z, bins=50)
        #h = h.transpose()
        #normalized_map = ColorMapArray[datindx](h/h.max())
        #yy, zz = _np.meshgrid(yedges, zedges)
        xpos = lowLim # Plane of histogram
        #xflat = _np.full_like(yy, xpos) 
        #p = ax.plot_surface(xflat, yy, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)
        xflat = _np.full_like(y, xpos) 
        ax.scatter(xflat, y, z, color=ColorArray[datindx], alpha=alphaLevel)
        
        #h, xedges, zedges = _np.histogram2d(x, z, bins=50)
        #h = h.transpose()
        #normalized_map = ColorMapArray[datindx](h/h.max())
        #xx, zz = _np.meshgrid(xedges, zedges)
        ypos = highLim # Plane of histogram
        #yflat = _np.full_like(xx, ypos) 
        #p = ax.plot_surface(xx, yflat, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)
        yflat = _np.full_like(x, ypos) 
        ax.scatter(x, yflat, z, color=ColorArray[datindx], alpha=alphaLevel)
        
        #h, yedges, xedges = _np.histogram2d(y, x, bins=50)
        #h = h.transpose()
        #normalized_map = ColorMapArray[datindx](h/h.max())
        #yy, xx = _np.meshgrid(yedges, xedges)
        zpos = lowLim # Plane of histogram
        #zflat = _np.full_like(yy, zpos) 
        #p = ax.plot_surface(xx, yy, zflat, facecolors=normalized_map, rstride=1, cstride=1, shade=False)
        zflat = _np.full_like(y, zpos) 
        ax.scatter(x, y, zflat, color=ColorArray[datindx], alpha=alphaLevel)
    
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(30, angle)
    
    if show_fig == True:
        _plt.show()
    return fig, ax

# ------ Functions for extracting mass via potential comparision ---------------

def steady_state_potential(xdata,HistBins=100):
    """ 
    Calculates the steady state potential. Used in 
    fit_radius_from_potentials.

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
    import numpy as _np
    
    pops=_np.histogram(xdata,HistBins)[0]
    bins=_np.histogram(xdata,HistBins)[1]
    bins=bins[0:-1]
    bins=bins+_np.mean(_np.diff(bins))
    
    #normalise pops
    pops=pops/float(_np.sum(pops))
    
    return bins,-_np.log(pops)

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
    import numpy as _np
    adata = calc_acceleration(xdata, dt)
    xdata = xdata[2:] # removes first 2 values as differentiating twice means
    # we have acceleration[n] corresponds to position[n-2]
    
    z=_np.polyfit(xdata,adata,order)
    p=_np.poly1d(z)
    spring_pot=_np.polyint(p)
    return -spring_pot

def calc_acceleration(xdata, dt):
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
    acceleration = _np.diff(_np.diff(xdata))/dt**2
    return acceleration

def fit_radius_from_potentials(z, SampleFreq, Damping, HistBins=100, show_fig=False):
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
    fig : matplotlib.figure.Figure object
        figure showing fitted dynamical potential and stationary potential
    ax : matplotlib.axes.Axes object
        axes for above figure

    """
    dt = 1/SampleFreq
    boltzmann=Boltzmann
    temp=300 # why halved??
    density=1800
    SteadyStatePotnl = list(steady_state_potential(z, HistBins=HistBins))
    yoffset=min(SteadyStatePotnl[1])
    SteadyStatePotnl[1] -= yoffset

    SpringPotnlFunc = dynamical_potential(z, dt)
    SpringPotnl = SpringPotnlFunc(z)
    kBT_Gamma = temp*boltzmann*1/Damping
    
    DynamicPotentialFunc = make_dynamical_potential_func(kBT_Gamma, density, SpringPotnlFunc)
    FitSoln = _curve_fit(DynamicPotentialFunc, SteadyStatePotnl[0], SteadyStatePotnl[1], p0 = 50)
    print(FitSoln)
    popt, pcov = FitSoln
    perr = _np.sqrt(_np.diag(pcov))
    Radius, RadiusError = popt[0], perr[0]

    mass=((4/3)*pi*((Radius*10**-9)**3))*density
    yfit=(kBT_Gamma/mass)
    Y = yfit*SpringPotnl
    
    fig, ax = _plt.subplots()
    ax.plot(SteadyStatePotnl[0], SteadyStatePotnl[1], 'bo', label="Steady State Potential")
    _plt.plot(z,Y, 'r-', label="Dynamical Potential")
    ax.legend(loc='best')
    ax.set_ylabel('U ($k_{B} T $ Joules)')
    ax.set_xlabel('Distance (mV)')
    _plt.tight_layout()
    if show_fig == True:
        _plt.show()
    return Radius*1e-9, RadiusError*1e-9, fig, ax

def make_dynamical_potential_func(kBT_Gamma, density, SpringPotnlFunc):
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
        mass = ((4/3)*pi*((Radius*10**-9)**3))*density
        yfit=(kBT_Gamma/mass)
        Y = yfit*SpringPotnlFunc(xdata)
        return Y
    return PotentialFunc

# ------------------------------------------------------------

def calc_mean_amp(signal):
    """
    calculates the mean amplitude by calculating the RMS
    of the signal and then multiplying it by √2.
    
    Parameters
    ----------
    signal : ndarray
    array of floats containing an AC signal
    
    Returns
    -------
    mean_amplitude : float
        the mean amplitude of the signal
    """
    return _np.sqrt(2)*_np.sqrt(_np.mean(signal**2))

def calc_z0_and_conv_factor_from_ratio_of_harmonics(z, z2, NA=0.999):
    """
    Calculates the Conversion Factor and physical amplitude of motion in nms 
    by comparison of the ratio of the heights of the z signal and 
    second harmonic of z.

    Parameters
    ----------
    z : ndarray
        array containing z signal in volts
    z2 : ndarray
        array containing second harmonic of z signal in volts
    NA : float
        NA of mirror used in experiment

    Returns
    -------
    z0 : float
        Physical average amplitude of motion in nms
    ConvFactor : float
        Conversion Factor between volts and nms
    """
    V1 = calc_mean_amp(z)
    V2 = calc_mean_amp(z2)
    ratio = V2/V1
    beta = 4*ratio
    laserWavelength = 1550e-9 # in m
    k0 = (2*pi)/(laserWavelength)
    WaistSize = laserWavelength/(pi*NA)
    Zr = pi*WaistSize**2/laserWavelength
    z0 = beta/(k0 - 1/Zr)
    ConvFactor = V1/z0
    T0 = 300
    return z0, ConvFactor

def calc_mass_from_z0(z0, w0):
    """
    Calculates the mass of the particle using the equipartition
    from the angular frequency of the z signal and the average
    amplitude of the z signal in nms.

    Parameters
    ----------
    z0 : float
        Physical average amplitude of motion in nms
    w0 : float
        Angular Frequency of z motion

    Returns
    -------
    mass : float
        mass in kgs
    """
    T0 = 300
    mFromEquipartition = Boltzmann*T0/(w0**2 * z0**2)
    return mFromEquipartition

def calc_mass_from_fit_and_conv_factor(A, Damping, ConvFactor):
    """
    Calculates mass from the A parameter from fitting, the damping from 
    fitting in angular units and the Conversion factor calculated from 
    comparing the ratio of the z signal and first harmonic of z.

    Parameters
    ----------
    A : float
        A factor calculated from fitting
    Damping : float
        damping in radians/second calcualted from fitting
    ConvFactor : float
        conversion factor between volts and nms

    Returns
    -------
    mass : float
        mass in kgs
    """
    T0 = 300
    mFromA = 2*Boltzmann*T0/(pi*A) * ConvFactor**2 * Damping
    return mFromA

# -----------------------------------------------------------------------------

def get_time_slice(time, z, zdot=None, timeStart=None, timeEnd=None):
    """
    Get slice of time, z and (if provided) zdot from timeStart to timeEnd.

    Parameters
    ----------
    time : ndarray
        array of time values 
    z : ndarray
        array of z values
    zdot : ndarray, optional
        array of zdot (velocity) values.
    timeStart : float, optional
        time at which to start the slice.
        Defaults to beginnging of time trace
    timeEnd : float, optional
        time at which to end the slide.
        Defaults to end of time trace

    Returns
    -------
    time_sliced : ndarray
        array of time values from timeStart to timeEnd
    z_sliced : ndarray
        array of z values from timeStart to timeEnd
    zdot_sliced : ndarray
        array of zdot values from timeStart to timeEnd.
        None if zdot not provided

    """
    if timeStart == None:
        timeStart = time[0]
    if timeEnd == None:
        timeEnd = time[-1]

    StartIndex = _np.where(time == take_closest(time, timeStart))[0][0]
    EndIndex = _np.where(time == take_closest(time, timeEnd))[0][0]

    time_sliced = time[StartIndex:EndIndex]
    z_sliced = z[StartIndex:EndIndex]

    if zdot != None:
        zdot_sliced = zdot[StartIndex:EndIndex]
    else:
        zdot_sliced = None    
    
    return time_sliced, z_sliced, zdot_sliced

    
def calc_radius_from_mass(Mass):
    """
    Given the mass of a particle calculates the 
    radius, assuming a 1800 kg/m**3 density.

    Parameters
    ----------
    Mass : float
        mass in kgs
   
    Returns
    -------
    Radius : float
        radius in ms
    """
    density = 1800
    Radius = (3*Mass/(4*pi*density))**(1/3)
    return Radius
    
# ------------------------------------------------------------

def unit_conversion(array, unit_prefix, current_prefix=""):
    """
    Converts an array or value to of a certain 
    unit scale to another unit scale.

    Accepted units are:
    E - exa - 1e18
    P - peta - 1e15
    T - tera - 1e12
    G - giga - 1e9
    M - mega - 1e6
    k - kilo - 1e3
    m - milli - 1e-3
    u - micro - 1e-6
    n - nano - 1e-9
    p - pico - 1e-12
    f - femto - 1e-15
    a - atto - 1e-18

    Parameters
    ----------
    array : ndarray
        Array to be converted
    unit_prefix : string
        desired unit (metric) prefix (e.g. nm would be n, ms would be m)
    current_prefix : optional, string
        current prefix of units of data (assumed to be in SI units
        by default (e.g. m or s)

    Returns
    -------
    converted_array : ndarray
        Array multiplied such as to be in the units specified
    """
    UnitDict = {
        'E': 1e18,
        'P': 1e15,
        'T': 1e12,
        'G': 1e9,
        'M': 1e6,
        'k': 1e3,
        '': 1,
        'm': 1e-3,
        'u': 1e-6,
        'n': 1e-9,
        'p': 1e-12,
        'f': 1e-15,
        'a': 1e-18,
    }
    try:
        Desired_units = UnitDict[unit_prefix]
    except KeyError:
        raise ValueError("You entered {} for the unit_prefix, this is not a valid prefix".format(unit_prefix))
    try:
        Current_units = UnitDict[current_prefix]
    except KeyError:
        raise ValueError("You entered {} for the current_prefix, this is not a valid prefix".format(current_prefix))
    conversion_multiplication = Current_units/Desired_units
    converted_array = array*conversion_multiplication
    return converted_array

def audiate(data, filename):
    AudioFreq = int(30000/10e6*data.SampleFreq)
    _writewav(filename, AudioFreq, data.voltage)
    return None

# ----------------- WIGNER FUNCTIONS -------------------------------------------------------------

def extract_slices(z, freq, sample_freq, show_plot=False):
    """
    Iterates through z trace and pulls out slices of length period_samples
    and assigns them a phase from -180 to 180. Each slice then becomes a column
    in the 2d array that is returned. Such that the row (the first index) refers
    to phase (i.e. dat[0] are all the samples at phase = -180) and the column
    refers to the oscillation number (i.e. dat[:, 0] is the first oscillation).

    Parameters
    ----------
    z : ndarray
        trace of z motion
    freq : float
        frequency of motion
    sample_freq : float
        sample frequency of the z array
    show_plot : bool, optional (default=False)
        if true plots and shows the phase plotted against the positon
        for each oscillation built on top of each other.

    Returns
    -------
    phase : ndarray
        phase (in degrees) for each oscillation
    phase_slices : ndarray
        2d numpy array containing slices as detailed above.

    """
    dt = 1/sample_freq # dt between samples
    period = 1/freq # period of oscillation of motion
    period_samples = round(period/dt) # integer number of discrete samples in a period
    number_of_oscillations = int(_np.floor(len(z)/period_samples)) # number of oscillations in z trace


    phase_slices_untransposed = _np.zeros([number_of_oscillations-1, period_samples])

    phase = _np.linspace(-180, 180, period_samples) # phase assigned to samples

    if show_plot == True:
        fig, ax = _plt.subplots()

    for i in range(number_of_oscillations-1): 
        # loops through number of oscillations - 1 pulling out period_samples
        # slices and assigning them a phase from -180 to 180 degrees
        start = i*period_samples # start index of section
        end = (i+1)*period_samples # end index of section
        if show_plot == True:
            _plt.plot(phase, z[start:end]) 
        phase_slices_untransposed[i] = z[start:end] # enter z section as ith row
    
    phase_slices = phase_slices_untransposed.transpose() # swap rows and columns 

    if show_plot == True:
        _plt.show()
    return phase, phase_slices

def histogram_phase(phase_slices, phase, histbins=200, show_plot=False):
    """
    histograms the phase slices such as to build a histogram of the position
    distribution at each phase value.

    Parameters
    ----------
    phase_slices : ndarray
        2d array containing slices from many oscillations at each phase
    phase : ndarray
        1d array of phases corresponding to slices
    histbins : int, optional (default=200)
        number of bins to use in histogramming data
    show_plot : bool, optional (default=False)
        if true plots and shows the heatmap of the
        phase against the positon distribution

    Returns
    -------
    counts_array : ndarray
        2d array containing the number of counts varying with
        phase and position.
    bin_edges : ndarray
        positions of bin edges

    """
    counts_array = _np.zeros([len(phase), histbins])

    histedges = [phase_slices.min(), phase_slices.max()]
    for i, phase_slice in enumerate(phase_slices): # for each value of phase
        counts, bin_edges = _np.histogram(phase_slice, bins=histbins, range=histedges) # histogram the position distribution at that phase
        counts_array[i] = counts
    counts_array = _np.array(counts_array)
    counts_array_transposed = _np.transpose(counts_array).astype(float)

    if show_plot == True:
        fig = _plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.set_title('Phase Distribution')
        ax.set_xlabel("phase (°)")
        ax.set_ylabel("x")
        _plt.imshow(counts_array_transposed, cmap='hot', interpolation='nearest', extent=[phase[0], phase[-1], histedges[0], histedges[1]])
        ax.set_aspect('auto')
        _plt.show()

    return counts_array_transposed, bin_edges

def get_wigner(z, freq, sample_freq, histbins=200, show_plot=False):
    """
    Calculates an approximation to the wigner quasi-probability distribution
    by splitting the z position array into slices of the length of one period
    of the motion. This slice is then associated with phase from -180 to 180
    degrees. These slices are then histogramed in order to get a distribution
    of counts of where the particle is observed at each phase. The 2d array
    containing the counts varying with position and phase is then passed through
    the inverse radon transformation using the Simultaneous Algebraic 
    Reconstruction Technique approximation from the scikit-image package.

    Parameters
    ----------
    z : ndarray
        trace of z motion
    freq : float
        frequency of motion
    sample_freq : float
        sample frequency of the z array
    histbins : int, optional (default=200)
        number of bins to use in histogramming data for each phase
    show_plot : bool, optional (default=False)
        Whether or not to plot the phase distribution

    Returns
    -------
    iradon_output : ndarray
        2d array of size (histbins x histbins)
    bin_centres : ndarray
        positions of the bin centres

    """
    
    phase, phase_slices = extract_slices(z, freq, sample_freq, show_plot=False)

    counts_array, bin_edges = histogram_phase(phase_slices, phase, histbins, show_plot=show_plot)

    diff = bin_edges[1] - bin_edges[0]
    bin_centres = bin_edges[:-1] + diff

    iradon_output = _iradon_sart(counts_array, theta=phase)

    #_plt.imshow(iradon_output, extent=[bin_centres[0], bin_centres[-1], bin_centres[0], bin_centres[-1]])
    #_plt.show()

    return iradon_output, bin_centres

def plot_wigner3d(iradon_output, bin_centres, bin_centre_units="", cmap=_cm.cubehelix_r, view=(10, -45), figsize=(10, 10)):
    """
    Plots the wigner space representation as a 3D surface plot.

    Parameters
    ----------
    iradon_output : ndarray
        2d array of size (histbins x histbins)
    bin_centres : ndarray
        positions of the bin centres
    bin_centre_units : string, optional (default="")
        Units in which the bin_centres are given
    cmap : matplotlib.cm.cmap, optional (default=cm.cubehelix_r)
        color map to use for Wigner
    view : tuple, optional (default=(10, -45))
        view angle for 3d wigner plot
    figsize : tuple, optional (default=(10, 10))
        tuple defining size of figure created

    Returns
    -------
    fig : matplotlib.figure.Figure object
        figure showing the wigner function
    ax : matplotlib.axes.Axes object
        axes containing the object

    """
    fig = _plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    resid1 = iradon_output.sum(axis=0)
    resid2 = iradon_output.sum(axis=1)

    x = bin_centres # replace with x
    y = bin_centres # replace with p (xdot/omega)
    xpos, ypos = _np.meshgrid(x, y)
    X = xpos
    Y = ypos
    Z = iradon_output

    ax.set_xlabel("x ({})".format(bin_centre_units))
    ax.set_xlabel("y ({})".format(bin_centre_units))

    ax.scatter(_np.min(X)*_np.ones_like(y), y, resid2/_np.max(resid2)*_np.max(Z), alpha=0.7)
    ax.scatter(x, _np.max(Y)*_np.ones_like(x), resid1/_np.max(resid1)*_np.max(Z), alpha=0.7)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.view_init(view[0], view[1])

    return fig, ax


def plot_wigner2d(iradon_output, bin_centres, cmap=_cm.cubehelix_r, figsize=(6, 6)):
    """
    Plots the wigner space representation as a 2D heatmap.

    Parameters
    ----------
    iradon_output : ndarray
        2d array of size (histbins x histbins)
    bin_centres : ndarray
        positions of the bin centres
    cmap : matplotlib.cm.cmap, optional (default=cm.cubehelix_r)
        color map to use for Wigner
    figsize : tuple, optional (default=(6, 6))
        tuple defining size of figure created

    Returns
    -------
    fig : matplotlib.figure.Figure object
        figure showing the wigner function
    ax : matplotlib.axes.Axes object
        axes containing the object

    """
    xx, yy = _np.meshgrid(bin_centres, bin_centres)
    resid1 = iradon_output.sum(axis=0)
    resid2 = iradon_output.sum(axis=1)
    
    wigner_marginal_seperation = 0.001
    left, width = 0.2, 0.65-0.1 # left = left side of hexbin and hist_x
    bottom, height = 0.1, 0.65-0.1 # bottom = bottom of hexbin and hist_y
    bottom_h = height + bottom + wigner_marginal_seperation
    left_h = width + left + wigner_marginal_seperation
    cbar_pos = [0.03, bottom, 0.05, 0.02+width]

    rect_wigner = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    fig = _plt.figure(figsize=figsize)

    axWigner = _plt.axes(rect_wigner)
    axHistx = _plt.axes(rect_histx)
    axHisty = _plt.axes(rect_histy)

    pcol = axWigner.pcolor(xx, yy, iradon_output, cmap=cmap)
    binwidth = bin_centres[1] - bin_centres[0]
    axHistx.bar(bin_centres, resid2, binwidth)
    axHisty.barh(bin_centres, resid1, binwidth)

    _plt.setp(axHistx.get_xticklabels(), visible=False) # sets x ticks to be invisible while keeping gridlines
    _plt.setp(axHisty.get_yticklabels(), visible=False) # sets x ticks to be invisible while keeping gridlines
    for tick in axHisty.get_xticklabels():
        tick.set_rotation(-90)


    cbaraxes = fig.add_axes(cbar_pos)  # This is the position for the colorbar
    #cbar = _plt.colorbar(axp, cax = cbaraxes)
    cbar = fig.colorbar(pcol, cax = cbaraxes, drawedges=False) #, orientation="horizontal"
    cbar.solids.set_edgecolor("face")
    cbar.solids.set_rasterized(True)
    cbar.ax.set_yticklabels(cbar.ax.yaxis.get_ticklabels(), y=0, rotation=45)
    #cbar.set_label(cbarlabel, labelpad=-25, y=1.05, rotation=0)

    plotlimits = _np.max(_np.abs(bin_centres))
    axWigner.axis((-plotlimits, plotlimits, -plotlimits, plotlimits))
    axHistx.set_xlim(axWigner.get_xlim())
    axHisty.set_ylim(axWigner.get_ylim())

    return fig, axWigner, axHistx, axHisty, cbar


def fit_data(freq_array, S_xx_array, AGuess, OmegaTrap, GammaGuess, make_fig=True, show_fig=True):
    
    logPSD = 10 * _np.log10(S_xx_array) # putting S_xx in dB

    def calc_theory_PSD_curve_fit(freqs, A, TrapFreq, BigGamma):
        Theory_PSD = 10 * \
            _np.log10(PSD_fitting_eqn(A, TrapFreq, BigGamma, freqs)) # PSD in dB
        if A < 0 or TrapFreq < 0 or BigGamma < 0:
            return 1e9
        else:
            return Theory_PSD

    datax = _np.array(freq_array)*(2*_np.pi) # angular frequency
    datay = logPSD # S_xx data in dB

    p0 = _np.array([AGuess, OmegaTrap, GammaGuess])

    Params_Fit, Params_Fit_Err = fit_curvefit(p0,
                                              datax,
                                              datay,
                                              calc_theory_PSD_curve_fit)

    if make_fig == True:
        fig = _plt.figure(figsize=properties["default_fig_size"])
        ax = fig.add_subplot(111)

        PSDTheory_fit_initial = PSD_fitting_eqn(p0[0],
                                                 p0[1],
                                                 p0[2],
                                                 datax)

        PSDTheory_fit = PSD_fitting_eqn(Params_Fit[0],
                                         Params_Fit[1],
                                         Params_Fit[2],
                                         datax)

        ax.plot(datax / (2 * pi), S_xx_array,
                color="darkblue", label="Raw PSD Data", alpha=0.5)
        ax.plot(datax / (2 * pi), PSDTheory_fit_initial,
                '--', alpha=0.7, color="purple", label="initial vals")
        ax.plot(datax / (2 * pi), PSDTheory_fit,
                color="red", label="fitted vals")
        ax.semilogy()
        legend = ax.legend(loc="best", frameon = 1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("$S_{xx}$ ($V^2/Hz$)")
        if show_fig == True:
            _plt.show()
        return Params_Fit, Params_Fit_Err, fig, ax
    else:
        return Params_Fit, Params_Fit_Err, None, None
