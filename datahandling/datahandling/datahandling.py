import datahandling.LeCroy
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
import pandas as _pd
import fnmatch as _fnmatch


def load_data(Filepath):
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


def multi_load_data(Channel, RunNos, RepeatNos, directoryPath='.'):
    """
    Lets you load multiple datasets at once.

    Channel : int
        The channel you want to load
    RunNos : sequence
        Sequence of run numbers you want to load
    RepeatNos : sequence
        Sequence of repeat numbers you want to load


    """
    files = glob('{}/*'.format(directoryPath))
    files_CorrectChannel = []
    for file_ in files:
        if 'CH{}'.format(Channel) in file_:
            files_CorrectChannel.append(file_)
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
    data = []
    for filepath in files_CorrectRepeatNo:
        print(filepath)
        data.append(load_data(filepath))
    return data


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

        The following attributes are only assigned after get_fit has been called.

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
        self.filedir = self.filepath[0:-len(self.filename)]
        self.get_time_data()
        self.get_PSD()
        return None

    def get_time_data(self):
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
        f = open(self.filepath, 'rb')
        raw = f.read()
        f.close()
        self.waveDescription, self.time, self.Voltage, _ = \
            datahandling.LeCroy.InterpretWaveform(raw)
        self.SampleFreq = (1 / self.waveDescription["HORIZ_INTERVAL"])
        return self.time, self.Voltage

    def plot_time_data(self, timeStart, timeEnd, ShowFig=True):
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
        if timeStart == "Default":
            timeStart = self.time[0]
        if timeEnd == "Default":
            timeEnd = self.time[-1]

        StartIndex = list(self.time).index(test_closest(self.time, timeStart))
        EndIndex = list(self.time).index(test_closest(self.time, timeEnd))

        fig = _plt.figure(figsize=[10, 6])
        ax = fig.add_subplot(111)
        ax.plot(self.time[StartIndex:EndIndex],
                self.Voltage[StartIndex:EndIndex])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.set_xlim([timeStart, timeEnd])
        if ShowFig == True:
            _plt.show()
        return fig, ax

    def get_PSD(self, NPerSegment='Default', window="hann"):
        """
        Extracts the pulse spectral density (PSD) from the data.

        Parameters
        ----------
        NPerSegment : int, optional
            Length of each segment used in scipy.welch
            default = the Number of time points

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
        if NPerSegment == "Default":
            NPerSegment = len(self.time)
            if NPerSegment > 1e5:
                NPerSegment = int(1e5)
        self.freqs, self.PSD = scipy.signal.welch(self.Voltage, self.SampleFreq,
                                                  window=window, nperseg=NPerSegment)
        return self.freqs, self.PSD

    def plot_PSD(self, xlim="Default", ShowFig=True):
        """
        plot the pulse spectral density.

        Parameters
        ----------
        xlim : array_like, optional
            The x limits of the plotted PSD [LowerLimit, UpperLimit]
            Default value is [0, SampleFreq/2]
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
#        self.get_PSD()
        if xlim == "Default":
            xlim = [0, self.SampleFreq / 2]
        fig = _plt.figure(figsize=[10, 6])
        ax = fig.add_subplot(111)
        ax.semilogy(self.freqs, self.PSD, color="blue")
        ax.set_xlabel("Frequency Hz")
        ax.set_xlim(xlim)
        ax.grid(which="major")
        ax.set_ylabel("PSD ($v^2/Hz$)")
        if ShowFig == True:
            _plt.show()
        return fig, ax

    def get_fit(self, WidthOfPeakToFit, NMovAveToFit, TrapFreq, A_Initial=0.1e10, Gamma_Initial=400, ShowFig=True):
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
                The trapping frequency in the z axis (in angular frequency)
        Gamma : uncertainties.ufloat
                The damping factor Gamma = Γ = Γ_0 + δΓ
                where:
                        Γ_0 = Damping factor due to environment
                        δΓ = extra damping due to feedback
        """
        Params, ParamsErr, fig, ax = fit_PSD(
            self, WidthOfPeakToFit, NMovAveToFit, TrapFreq, A_Initial, Gamma_Initial, ShowFig)

        print("\n")
        print("A: {} +- {}% ".format(Params[0],
                                     ParamsErr[0] / Params[0] * 100))
        print(
            "Trap Frequency: {} +- {}% ".format(Params[1], ParamsErr[1] / Params[1] * 100))
        print(
            "Big Gamma: {} +- {}% ".format(Params[2], ParamsErr[2] / Params[2] * 100))

        self.A = _uncertainties.ufloat(Params[0], ParamsErr[0])
        self.Ftrap = _uncertainties.ufloat(Params[1], ParamsErr[1])
        self.Gamma = _uncertainties.ufloat(Params[2], ParamsErr[2])

        return self.A, self.Ftrap, self.Gamma, fig, ax

    def extract_parameters(self, P_mbar, P_Error):
        """
        Extracts the Radius  mass and Conversion factor for a particle.

        P_mbar : float 
            The pressure in mbar when the data was taken.
        P_Error : float
            The error in the pressure value (as a decimal e.g. 15% = 0.15)


        """

        [R, M, ConvFactor], [RErr, MErr, ConvFactorErr] = \
            extract_parameters(P_mbar, P_Error,
                                           self.A.n, self.A.std_dev,
                                           self.Gamma.n, self.Gamma.std_dev)
        self.Radius = _uncertainties.ufloat(R, RErr)
        self.Mass = _uncertainties.ufloat(M, MErr)
        self.ConvFactor = _uncertainties.ufloat(ConvFactor, ConvFactorErr)

        return self.Radius, self.Mass, self.ConvFactor

    def extract_ZXY_motion(self, ApproxZXYFreqs, uncertaintyInFreqs, ZXYPeakWidths, subSampleFraction):
        """
        Extracts the x, y and z signals (in volts) from the

        """
        [zf, xf, yf] = ApproxZXYFreqs
        zf, xf, yf = get_ZXY_freqs(
            self, zf, xf, yf, bandwidth=uncertaintyInFreqs)
        print(zf, xf, yf)
        [zwidth, xwidth, ywidth] = ZXYPeakWidths
        self.zVolts, self.xVolts, self.yVolts = get_ZXY_data(
            self, zf, xf, yf, subSampleFraction, zwidth, xwidth, ywidth)
        return self.zVolts, self.xVolts, self.yVolts

    def plot_phase_space(self, zf, xf=80000, yf=120000, FractionOfSampleFreq=4, zwidth=10000, xwidth=5000, ywidth=5000, ShowFig=True):
        """
        author: Markus Rademacher
        """
        Z, X, Y, Time = get_ZXY_data(
            self, zf, xf, yf, FractionOfSampleFreq, zwidth, xwidth, ywidth, ShowFig=False)

        conv = self.ConvFactor.n
        ZArray = Z / conv
        ZVArray = _np.diff(ZArray) * (self.SampleFreq / FractionOfSampleFreq)
        VarZ = _np.var(ZArray)
        VarZV = _np.var(ZVArray)
        MaxZ = _np.max(ZArray)
        MaxZV = _np.max(ZVArray)
        if MaxZ > MaxZV / (2 * _np.pi * zf):
            _plotlimit = MaxZ * 1.1
        else:
            _plotlimit = MaxZV / (2 * _np.pi * zf) * 1.1

        JP1 = _sns.jointplot(_pd.Series(ZArray[1:], name="$z$(m) \n filepath=%s" % (self.filepath)), _pd.Series(
            ZVArray / (2 * _np.pi * zf), name="$v_z$/$\omega$(m)"), stat_func=None, xlim=[-_plotlimit, _plotlimit], ylim=[-_plotlimit, _plotlimit])
        JP1.ax_joint.text(_np.mean(ZArray), MaxZV / (2 * _np.pi * zf) * 1.15,
                          r"$\sigma_z=$ %.2Em, $\sigma_v=$ %.2Em" % (
                              VarZ, VarZV),
                          horizontalalignment='center')
        JP1.ax_joint.text(_np.mean(ZArray), MaxZV / (2 * _np.pi * zf) * 1.6,
                          "filepath=%s" % (self.filepath),
                          horizontalalignment='center')
        if ShowFig == True:
            _plt.show()

        return VarZ, VarZV, JP1, self.Mass


def calc_temp(Data_ref, Data):
    #T = 300*(Data.A/Data.Gamma)/(Data_ref.A/Data_ref.Gamma)
    T = 300 * ((Data.A * Data_ref.Gamma) / (Data_ref.A * Data.Gamma))
    return T


def fit_curvefit(p0, datax, datay, function, yerr=None, **kwargs):

    pfit, pcov = \
        _curve_fit(function, datax, datay, p0=p0,
                   sigma=yerr, epsfcn=0.0001, **kwargs)
    error = []
    for i in range(len(pfit)):
        try:
            error.append(_np.absolute(pcov[i][i])**0.5)
        except:
            error.append(0.00)
    pfit_curvefit = pfit
    perr_curvefit = _np.array(error)
    return pfit_curvefit, perr_curvefit


def moving_average(a, n=3):
    ret = _np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def test_closest(myList, myNumber):
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
    # gamma = Big Gamma - damping (due to environment and feedback (if
    # feedback is on))
    return 10 * _np.log10(A / ((Omega0**2 - omega**2)**2 + (omega * gamma)**2))


def fit_PSD(Data, bandwidth, NMovAve, TrapFreqGuess, AGuess=0.1e10, GammaGuess=400, ShowFig=True):
    """
    Fits theory PSD to Data. Assumes highest point of PSD is the
    trapping frequency.

    Parameters
    ----------
    Data - data object to be fitted
    bandwidth - bandwidth around trapping frequency peak to
                fit the theory PSD to
    NMovAve - amount of moving averages to take before the fitting

    ShowFig - (defaults to True) if set to True this function plots the
        PSD of the data, smoothed data and theory peak from fitting.

    Returns
    -------
    ParamsFit - Fitted parameters:
        [A, TrappingFrequency, Gamma]
    ParamsFitErr - Error in fitted parameters:
        [AErr, TrappingFrequencyErr, GammaErr]

    """
    AngFreqs = 2 * _np.pi * Data.freqs
    Angbandwidth = 2 * _np.pi * bandwidth
    AngTrapFreqGuess = 2 * _np.pi * TrapFreqGuess

    ClosestToAngTrapFreqGuess = test_closest(AngFreqs, AngTrapFreqGuess)
    index_ftrap = _np.where(AngFreqs == ClosestToAngTrapFreqGuess)
    ftrap = AngFreqs[index_ftrap]

    f_fit_lower = test_closest(AngFreqs, ftrap - Angbandwidth / 2)
    f_fit_upper = test_closest(AngFreqs, ftrap + Angbandwidth / 2)

    indx_fit_lower = int(_np.where(AngFreqs == f_fit_lower)[0])
    indx_fit_upper = int(_np.where(AngFreqs == f_fit_upper)[0])

#    print(f_fit_lower, f_fit_upper)
#    print(AngFreqs[indx_fit_lower], AngFreqs[indx_fit_upper])

    # find highest point in region about guess for trap frequency - use that
    # as guess for trap frequency and recalculate region about the trap
    # frequency
    index_ftrap = _np.where(Data.PSD == max(
        Data.PSD[indx_fit_lower:indx_fit_upper]))

    ftrap = AngFreqs[index_ftrap]

#    print(ftrap)

    f_fit_lower = test_closest(AngFreqs, ftrap - Angbandwidth / 2)
    f_fit_upper = test_closest(AngFreqs, ftrap + Angbandwidth / 2)

    indx_fit_lower = int(_np.where(AngFreqs == f_fit_lower)[0])
    indx_fit_upper = int(_np.where(AngFreqs == f_fit_upper)[0])

    PSD_smoothed = moving_average(Data.PSD, NMovAve)
    freqs_smoothed = moving_average(AngFreqs, NMovAve)

    logPSD_smoothed = 10 * _np.log10(PSD_smoothed)

    def calc_theory_PSD_curve_fit(freqs, A, TrapFreq, BigGamma):
        Theory_PSD = PSD_Fitting(A, TrapFreq, BigGamma, freqs)
        if A < 0 or TrapFreq < 0 or BigGamma < 0:
            return 1e9
        else:
            return Theory_PSD

    datax = freqs_smoothed[indx_fit_lower:indx_fit_upper]
    datay = logPSD_smoothed[indx_fit_lower:indx_fit_upper]

    p0 = _np.array([AGuess, ftrap, GammaGuess])

    Params_Fit, Params_Fit_Err = fit_curvefit(p0,
                                              datax, datay, calc_theory_PSD_curve_fit)

    #    print("Params Fitted:", Params_Fit, "Error in Params:", Params_Fit_Err)
    fig = _plt.figure()
    ax = fig.add_subplot(111)

    PSDTheory_fit_initial = PSD_Fitting(p0[0], p0[1],
                                        p0[2], freqs_smoothed)

    PSDTheory_fit = PSD_Fitting(Params_Fit[0], Params_Fit[1],
                                Params_Fit[2], freqs_smoothed)

    ax.plot(AngFreqs / (2 * _np.pi), 10 * _np.log10(Data.PSD),
            color="darkblue", label="Raw PSD Data", alpha=0.5)
    ax.plot(freqs_smoothed / (2 * _np.pi), logPSD_smoothed,
            color='blue', label="smoothed", linewidth=1.5)
    ax.plot(freqs_smoothed / (2 * _np.pi), PSDTheory_fit_initial,
            '--', alpha=0.7, color="purple", label="initial vals")
    ax.plot(freqs_smoothed / (2 * _np.pi), PSDTheory_fit,
            color="red", label="fitted vals")
    ax.set_xlim([(ftrap - 5 * Angbandwidth) / (2 * _np.pi),
                 (ftrap + 5 * Angbandwidth) / (2 * _np.pi)])
    ax.plot([(ftrap - Angbandwidth) / (2 * _np.pi), (ftrap - Angbandwidth) / (2 * _np.pi)],
            [min(logPSD_smoothed), max(logPSD_smoothed)], '--',
            color="grey")
    ax.plot([(ftrap + Angbandwidth) / (2 * _np.pi), (ftrap + Angbandwidth) / (2 * _np.pi)],
            [min(logPSD_smoothed), max(logPSD_smoothed)], '--',
            color="grey")
    ax.legend(loc="best")
    if ShowFig == True:
        _plt.show()
    return Params_Fit, Params_Fit_Err, fig, ax


def extract_parameters(Pressure, PressureErr, A, AErr, Gamma0, Gamma0Err):
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
    Pressure = 100 * Pressure  # conversion to Pascals

    rho = 2200  # kgm^3
    dm = 0.372e-9  # m I'Hanlon, 2003
    T0 = 300  # kelvin
    kB = 1.38e-23  # m^2 kg s^-2 K-1
    eta = 18.27e-6  # Pa s, viscosity of air

    radius = (0.169 * 9 * _np.pi * eta * dm**2) / \
        (_np.sqrt(2) * rho * kB * T0) * (Pressure) / (Gamma0)
    err_radius = radius * \
        _np.sqrt(((PressureErr * Pressure) / Pressure)
                 ** 2 + (Gamma0Err / Gamma0)**2)
    mass = rho * ((4 * _np.pi * radius**3) / 3)
    err_mass = mass * 2 * err_radius / radius
    conversionFactor = _np.sqrt(A * _np.pi * mass / (kB * T0 * Gamma0))
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
    bandwidth : float
        The bandwidth around the approximate peak to look for the actual peak.

        Returns:
    trapfreqs : list
        List containing the trap frequencies in the following order (z, x, y)

    """
    trapfreqs = []
    for freq in [zfreq, xfreq, yfreq]:
        z_f_fit_lower = test_closest(Data.freqs, freq - bandwidth / 2)
        z_f_fit_upper = test_closest(Data.freqs, freq + bandwidth / 2)
        z_indx_fit_lower = int(_np.where(Data.freqs == z_f_fit_lower)[0])
        z_indx_fit_upper = int(_np.where(Data.freqs == z_f_fit_upper)[0])

        z_index_ftrap = _np.where(Data.PSD == max(
            Data.PSD[z_indx_fit_lower:z_indx_fit_upper]))
        # find highest point in region about guess for trap frequency
        # use that as guess for trap frequency and recalculate region
        # about the trap frequency
        z_ftrap = Data.freqs[z_index_ftrap]
        trapfreqs.append(z_ftrap)
    return trapfreqs


def get_ZXY_data(Data, zf, xf, yf, FractionOfSampleFreq,
               zwidth=10000, xwidth=5000, ywidth=5000,
               ztransition=10000, xtransition=5000, ytransition=5000,
               filterImplementation="filtfilt",
               timeStart="Default", timeEnd="Default",
               ShowFig=True):
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
    filterImplementation : string
        filtfilt or lfilter - use scipy.filtfilt or lfilter
        default: filtfilt
    timeStart : float
        Starting time for filtering
    timeEnd : float
        Ending time for filtering
    ShowFig : bool
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
    if timeStart == "Default":
        timeStart = Data.time[0]
    if timeEnd == "Default":
        timeEnd = Data.time[-1]

    StartIndex = list(Data.time).index(test_closest(Data.time, timeStart))
    EndIndex = list(Data.time).index(test_closest(Data.time, timeEnd))

    SAMPLEFREQ = Data.SampleFreq / FractionOfSampleFreq

    if filterImplementation == "filtfilt":
        ApplyFilter = scipy.signal.filtfilt
    elif filterImplementation == "lfilter":
        ApplyFilter = scipy.signal.lfilter
    else:
        raise ValueError("filterImplementation must be one of [filtfilt, lfilter] you entered: {}".format(
            filterImplementation))

    input_signal = Data.Voltage[StartIndex: EndIndex][0::FractionOfSampleFreq]

    bZ, aZ = IIRFilterDesign(zf, zwidth, ztransition, SAMPLEFREQ, GainStop=100)

    zdata = ApplyFilter(bZ, aZ, input_signal)

    if(_np.isnan(zdata).any()):
        raise ValueError(
            "Value Error: FractionOfSampleFreq must be higher, a sufficiently small sample frequency should be used to produce a working IIR filter.")

    bX, aX = IIRFilterDesign(xf, xwidth, xtransition, SAMPLEFREQ, GainStop=100)

    xdata = ApplyFilter(bX, aX, input_signal)

    if(_np.isnan(xdata).any()):
        raise ValueError(
            "Value Error: FractionOfSampleFreq must be higher, a sufficiently small sample frequency should be used to produce a working IIR filter.")

    bY, aY = IIRFilterDesign(yf, ywidth, ytransition, SAMPLEFREQ, GainStop=100)

    ydata = ApplyFilter(bY, aY, input_signal)

    if(_np.isnan(ydata).any()):
        raise ValueError(
            "Value Error: FractionOfSampleFreq must be higher, a sufficiently small sample frequency should be used to produce a working IIR filter.")

    if ShowFig == True:
        NPerSegment = len(Data.time)
        if NPerSegment > 1e5:
            NPerSegment = int(1e5)
        f, PSD = scipy.signal.welch(
            input_signal, SAMPLEFREQ, nperseg=NPerSegment)
        f_z, PSD_z = scipy.signal.welch(zdata, SAMPLEFREQ, nperseg=NPerSegment)
        f_y, PSD_y = scipy.signal.welch(ydata, SAMPLEFREQ, nperseg=NPerSegment)
        f_x, PSD_x = scipy.signal.welch(xdata, SAMPLEFREQ, nperseg=NPerSegment)
        _plt.plot(f, 10 * _np.log10(PSD))
        _plt.plot(f_z, 10 * _np.log10(PSD_z), label="z")
        _plt.plot(f_x, 10 * _np.log10(PSD_x), label="x")
        _plt.plot(f_y, 10 * _np.log10(PSD_y), label="y")
        _plt.legend(loc="best")
        _plt.xlim([zf - zwidth - ztransition, yf + ywidth + ytransition])
        _plt.show()

    timedata = Data.time[StartIndex: EndIndex][0::FractionOfSampleFreq]
    return zdata, xdata, ydata, timedata


def get_ZXY_data_IFFT(Data, zf, xf, yf,
                    zwidth=10000, xwidth=5000, ywidth=5000,
                    timeStart="Default", timeEnd="Default",
                    ShowFig=True):
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
    zwidth : float
        The width of the pass-band of the IIR filter to be generated to
        filter Z.
    xwidth : float
        The width of the pass-band of the IIR filter to be generated to
        filter X.
    ywidth : float
        The width of the pass-band of the IIR filter to be generated to
        filter Y.
    timeStart : float
        Starting time for filtering
    timeEnd : float
        Ending time for filtering
    ShowFig : bool
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
    if timeStart == "Default":
        timeStart = Data.time[0]
    if timeEnd == "Default":
        timeEnd = Data.time[-1]

    StartIndex = list(Data.time).index(test_closest(Data.time, timeStart))
    EndIndex = list(Data.time).index(test_closest(Data.time, timeEnd))

    SAMPLEFREQ = Data.SampleFreq

    input_signal = Data.Voltage[StartIndex: EndIndex]

    zdata = IFFT_filter(input_signal, SAMPLEFREQ, zf -
                       zwidth / 2, zf + zwidth / 2)

    xdata = IFFT_filter(input_signal, SAMPLEFREQ, xf -
                       zwidth / 2, xf + zwidth / 2)

    ydata = IFFT_filter(input_signal, SAMPLEFREQ, yf -
                       zwidth / 2, yf + zwidth / 2)

    if ShowFig == True:
        NPerSegment = len(Data.time)
        if NPerSegment > 1e5:
            NPerSegment = int(1e5)
        f, PSD = scipy.signal.welch(
            input_signal, SAMPLEFREQ, nperseg=NPerSegment)
        f_z, PSD_z = scipy.signal.welch(zdata, SAMPLEFREQ, nperseg=NPerSegment)
        f_y, PSD_y = scipy.signal.welch(ydata, SAMPLEFREQ, nperseg=NPerSegment)
        f_x, PSD_x = scipy.signal.welch(xdata, SAMPLEFREQ, nperseg=NPerSegment)
        _plt.plot(f, 10 * _np.log10(PSD))
        _plt.plot(f_z, 10 * _np.log10(PSD_z), label="z")
        _plt.plot(f_x, 10 * _np.log10(PSD_x), label="x")
        _plt.plot(f_y, 10 * _np.log10(PSD_y), label="y")
        _plt.legend(loc="best")
        _plt.xlim([zf - zwidth, yf + ywidth])
        _plt.xlabel('Frequency (Hz)')
        _plt.ylabel(r'$S_{xx}$')
        _plt.title("filepath = %s" % (Data.filepath))
        _plt.show()

    timedata = Data.time[StartIndex: EndIndex]
    return zdata, xdata, ydata, timedata


def animate(zdata, xdata, ydata,
            conversionFactor, timedata,
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
    conversionFactor : float
        conversion factor (in units of Volts/Metre)
    timedata : ndarray
        Array containing the time data in seconds.
    BoxSize : float
        The size of the box in which to animate the particle - in nm
    timeSteps : int
        Number of time steps to animate
    filename : string
        filename to create the mp4 under ({filename}.mp4)

    """
    timePerFrame = 0.203
    print("This will take ~ {} minutes".format(timePerFrame * timeSteps / 60))

    conv = conversionFactor * 1e-9

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
    myBitrate = 1e6

    fig = _plt.figure(figsize=(a, b))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("{} us".format(timedata[0] * 1e6))
    ax.set_xlabel('X (nm)')
    ax.set_xlim([XBoxStart, XBoxEnd])
    ax.set_ylabel('Y (nm)')
    ax.set_ylim([YBoxStart, YBoxEnd])
    ax.set_zlabel('Z (nm)')
    ax.set_zlim([ZBoxStart, ZBoxEnd])
    ax.view_init(20, -30)

    #ax.view_init(0, 0)

    def setup_plot():
        XArray = 1 / conv * xdata[0]
        YArray = 1 / conv * ydata[0]
        ZArray = 1 / conv * zdata[0]
        scatter = ax.scatter(XArray, YArray, ZArray)
        return scatter,

    def animate(i):
        # print "\r {}".format(i),
        print("Frame: {}".format(i), end="\r")
        ax.clear()
        ax.view_init(20, -30)
        ax.set_title("{} us".format(timedata[i] * 1e6))
        ax.set_xlabel('X (nm)')
        ax.set_xlim([XBoxStart, XBoxEnd])
        ax.set_ylabel('Y (nm)')
        ax.set_ylim([YBoxStart, YBoxEnd])
        ax.set_zlabel('Z (nm)')
        ax.set_zlim([ZBoxStart, ZBoxEnd])
        XArray = 1 / conv * xdata[i]
        YArray = 1 / conv * ydata[i]
        ZArray = 1 / conv * zdata[i]
        scatter = ax.scatter(XArray, YArray, ZArray)
        ax.scatter([XArray], [0], [-ZBoxEnd], c='k', alpha=0.9)
        ax.scatter([-XBoxEnd], [YArray], [0], c='k', alpha=0.9)
        ax.scatter([0], [YBoxEnd], [ZArray], c='k', alpha=0.9)

        Xx, Yx, Zx, Xy, Yy, Zy, Xz, Yz, Zz = [], [], [], [], [], [], [], [], []

        for j in range(0, 30):

            Xlast = 1 / conv * xdata[i - j]
            Ylast = 1 / conv * ydata[i - j]
            Zlast = 1 / conv * zdata[i - j]

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
                XCur = 1 / conv * xdata[i - j + 1]
                YCur = 1 / conv * ydata[i - j + 1]
                ZCur = 1 / conv * zdata[i - j + 1]
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


def IFFT_filter(Signal, SampleFreq, lowerFreq, upperFreq):
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

    Returns
    -------
    FilteredData : ndarray
        Array containing the filtered data
    """
    print("starting fft")
    Signalfft = scipy.fftpack.fft(Signal)
    print("starting freq calc")
    freqs = _np.fft.fftfreq(len(Signal)) * SampleFreq
    print("starting bin zeroing")
    for i, freq in enumerate(freqs):
        if freq < lowerFreq or freq > upperFreq:
            Signalfft[i] = 0
    print("starting ifft")
    FilteredSignal = 2 * scipy.fftpack.ifft(Signalfft)
    print("done")
    return FilteredSignal


def IIR_filter_design(CentralFreq, bandwidth, transitionWidth, SampleFreq, GainStop=40, GainPass=0.01):
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


def IIR_filter_design_New(Order, btype, CriticalFreqs, SampleFreq, StopbandAttenuation=40, ftype='cheby2'):
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
    NyquistFreq = SampleFreq / 2
    if len(CriticalFreqs) > 1:
        if (CriticalFreqs[1] > NyquistFreq):
            raise ValueError(
                "Need a higher Sample Frequency for this Frequency range")

        BandStartNormed = CriticalFreqs[0] / NyquistFreq
        BandStopNormed = CriticalFreqs[1] / NyquistFreq

        bandpass = [BandStartNormed, BandStopNormed]

        b, a = scipy.signal.iirfilter(Order, bandpass, rs=StopbandAttenuation,
                                      btype=btype, analog=False, ftype=ftype)
    else:
        CriticalFreq = CriticalFreqs[0]

        if (CriticalFreq > NyquistFreq):
            raise ValueError(
                "Need a higher Sample Frequency for this Critical Frequency")

        CriticalFreqNormed = CriticalFreq / NyquistFreq

        b, a = scipy.signal.iirfilter(Order, CriticalFreqNormed, rs=StopbandAttenuation,
                                      btype=btype, analog=False, ftype=ftype)

    return b, a


def get_freq_response(a, b, ShowFig=True, SampleFreq=(2 * _np.pi), NumOfFreqs=500, whole=False):
    """
    This function takes an array of coefficients and finds the frequency
    response of the filter using scipy.signal.freqz.
    ShowFig sets if the response should be plotted

    Parameters
    ----------
    b : array_like
        Coefficients multiplying the x values (inputs of the filter)

    a : array_like
        Coefficients multiplying the y values (outputs of the filter)
    ShowFig : bool
        Verbosity of function (i.e. whether to plot frequency and phase
        response or whether to just return the values.)
        Options (Default is 1):
        False - Do not plot anything, just return values
        True - Plot Frequency and Phase response and return values
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
    w, h = scipy.signal.freqz(b=b, a=a, worN=NumOfFreqs, whole=whole)
    freqList = w / (_np.pi) * SampleFreq / 2.0
    himag = _np.array([hi.imag for hi in h])
    GainArray = 20 * _np.log10(_np.abs(h))
    PhaseDiffArray = _np.unwrap(_np.arctan2(_np.imag(h), _np.real(h)))
    if ShowFig == True:
        fig1 = _plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(freqList, GainArray, '-', label="Specified Filter")
        ax.set_title("Frequency Response")
        if SampleFreq == 2 * _np.pi:
            ax.set_xlabel(("$\Omega$ - Normalized frequency "
                           "($\pi$=Nyquist Frequency)"))
        else:
            ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_xlim([0, SampleFreq / 2.0])
        _plt.show()
        fig2 = _plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(freqList, PhaseDiffArray, '-', label="Specified Filter")
        ax.set_title("Phase Response")
        if SampleFreq == 2 * _np.pi:
            ax.set_xlabel(("$\Omega$ - Normalized frequency "
                           "($\pi$=Nyquist Frequency)"))
        else:
            ax.set_xlabel("frequency (Hz)")

        ax.set_ylabel("Phase Difference")
        ax.set_xlim([0, SampleFreq / 2.0])
        _plt.show()

    return freqList, GainArray, PhaseDiffArray


def multi_plot_PSD(DataArray, xlim=[0, 500e3], LabelArray=[], ShowFig=True):
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
        LabelArray = ["DataSet {}".format(i)
                      for i in _np.arange(0, len(DataArray), 1)]
    fig = _plt.figure(figsize=[10, 6])
    ax = fig.add_subplot(111)

    for i, data in enumerate(DataArray):
        ax.semilogy(data.freqs, data.PSD, alpha=0.8, label=LabelArray[i])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(xlim)
    ax.grid(which="major")
    ax.legend(loc="best")
    ax.set_ylabel("PSD ($v^2/Hz$)")

    _plt.title('filedir=%s'%(DataArray[0].filedir))
    
    if ShowFig == True:
        _plt.show()
    return fig, ax


def multi_plot_Time(DataArray, SubSampleN=1, xlim="default", ylim="default", LabelArray=[], ShowFig=True):
    """
    plot the pulse spectral density.

    Parameters
    ----------
    DataArray : array-like
        array of DataObject instances for which to plot the PSDs
    SubSampleN : int
        Number of intervals between points to remove (to sub-sample data so
        that you effectively have lower sample rate to make plotting easier
        and quicker.
    xlim : array-like
        2 element array specifying the lower and upper x limit for which to
        plot the time signal
    LabelArray : array-like, optional
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
        LabelArray = ["DataSet {}".format(i)
                      for i in _np.arange(0, len(DataArray), 1)]
    fig = _plt.figure(figsize=[10, 6])
    ax = fig.add_subplot(111)

    for i, data in enumerate(DataArray):
        ax.plot(data.time[::SubSampleN], data.Voltage[::SubSampleN],
                alpha=0.8, label=LabelArray[i])
    ax.set_xlabel("time (s)")
    if xlim != "default":
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([DataArray[0].time[0], DataArray[0].time[-1]])
    if ylim != "default":
        ax.set_ylim(ylim)
    else:
        ax.set_xlim([min(DataArray[0].Voltage), max(DataArray[0].Voltage)])
    ax.grid(which="major")
    ax.legend(loc="best")
    ax.set_ylabel("Voltage (V)")
    if ShowFig == True:
        _plt.show()
    return fig, ax


def parse_orgtable(lines):
    """Parse an org-table (input as a list of strings split by newline) into a Pandas data frame."""
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


class PressureData():
    def __init__(self, filename):
        with open(filename, 'r') as file:
            fileContents = file.readlines()
        self.PressureData = parse_orgtable(fileContents)

    def get_pressure(self, RunNo):
        Pressure = float(self.PressureData[self.PressureData.RunNo == '{}'.format(
            RunNo)]['Pressure (mbar)'])
        return Pressure
