import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import leastsq as _leastsq
import pkg_resources as _pkg_resources
from time import strftime as _strftime
try:
    from numba import jit as _jit
except ImportError:
    print("numba module not present - this module will run faster using numba")
    def _jit(func): # this part defines a dectorator which does nothing
        def func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return func_wrapper 


def MakeSignal(freq, ampl, phase, vert_offset):
    """Returns a sinusoidal signal function, which is a function of t (time), 
    of the following form: ampl*sin(2*pi*freq*t + phase) + vert_offset

    Parameters
    ----------
    freq : float
        Frequency of the signal function to be created
    ampl : float
        Amplitude of the signal function to be created
    phase : float`
        Phase of the sinusoidal signal function to be created
    vert_offset : float
        vertical offset of the sinusoidal signal function to be created from the 0 axis
    
    Returns
    -------
    Signal : Function
        
    """
    def Signal(t):
        return ampl*_np.sin((2*_np.pi)*freq*t + phase) + vert_offset
    Signal.__doc__ = """Defines a signal of the form: ampl*sin(2*pi*freq*t + phase) + vert_offset
    Where:
    ampl = {}
    freq = {}
    phase = {}
    vert_offset = {}
    
    Parameters
    ----------
    t - float
    Time - time at which to evaluate the signal.
    
    Returns
    -------
    out : float
    The value of the signal at the time specified.""".format(freq, ampl, phase, vert_offset)
    return Signal

@_jit
def MakeSampledSignal(signal, sample_freq, sample_time):
    """
    Calculates the values of a signal sampled over a specified time with a specified
    Sample Frequency.
    
    Parameters
    ----------
    signal : function
        A function of the form f(t) where t is time which returns a single float,
        the value of the signal at time t.
    sample_freq : float
         frequency at which to sample the signal in Hz - i.e. take samples period of the 
         sample frequency - every (1/sample_freq seconds)
    sample_time : float
         Time over which the signal should be sampled in seconds

    Returns
    -------
    tArray : ndarray
        Array of times at which the signal has been evaluated.
    SampledSignal : ndarray
        Array of values of the signal at the corresponding times in tArray 
    """
    tArray = _np.arange(0,sample_time,1/sample_freq)
    SampledSignal = signal(tArray)
    return tArray, SampledSignal

def MakeFilter(coefArray):
    """Make a filter with the coefficient's specified. Returns a function which takes a single
    argument, an array of data to be filtered.
    
    Parameters
    ----------
    coefArray : array_like
        Array containing the coefficients for the FIR filter to create. First 
        element multiplies x[n] the next x[n-1] and so on.

    Returns
    -------
    filter_func : function
        A function which implements a the specified FIR filter, takes an array
        of floats as input and filters them.
    """
    def filter_func(x):
        y = _np.zeros_like(x)
        for i in _np.arange(0, len(x)-len(coefArray)):
            y[i] = 0
            for j in _np.arange(0, len(coefArray)):
                y[i] += coefArray[j]*x[i+j]
        return y
    filter_func.__doc__ = """A function implementing an FIR filter with the following coefficients
    {}
    Parameters
    ----------
    x : ndarray
        Array containing data to be filtered - should be a 1D array of the values of 
        signal as it varies with time.

    Returns
    -------
    y : ndarray
        The result of filtering the data with the filter specified above.
    """.format(coefArray)
    return filter_func

def _fittingFunc(params, time, signal):
    """
    Function used to fit a sine wave to a signal - returns the value of the sine wave
    at the point defined by the params and time arguments minus the value of the 
    signal parameter.

    Parameters
    ----------
    params : array_like
        Array containing the values of the following parameters in the corresponding order:
        [ Amplitude of sine wave
        frequency of sine wave
        phase of sine wave
        vertical offset of sine wave from the 0 axis ]
    time - float
        time at which to calculate the value of the sine wave.
    signal - float
        Value to be minused from the value of the signal - if it is a perfect fit it should 
        return a value equal to 0
    """
    Amp, freq, Phase, Offset = params[0], params[1], params[2], params[3]
    return Amp*_np.sin(2*_np.pi*freq*time+Phase) + Offset - signal

@_jit
def fitData(time, signal, guess_freq, guess_Amp, guess_phase, guess_Offset, Plotstuff):
    """
    Uses a least squares method to fit a sine wave to a signal measured in time.

    Parameters
    ----------
    time : ndarray
        Array containing the time values at which measurements were taken/simulated
    signal : ndarray
        Array containing the value of the signal at times corresponding to the time parameter
    guess_freq : float
        Guess for Frequency of the signal to be used in the leastsquares fit
    guess_Amp : float
        Guess for Amplitude of the signal to be used in the leastsquares fit
    guess_phase :
        Guess for Phase of the signal to be used in the leastsquares fit
    guess_Offset : float
         Guess for vertical offset of the signal to be used in the leastsquares fit
    Plotstuff - Boolean 
         Boolean stating whether or not to plot the time against the measured voltage and the model
    
    Returns
    -------
    est_freq : float
        Frequency fitted by the model
    est_Amp : float
        Amplitude fitted by the model
    est_phase : float
        Phase fitted by the model
    est_Offset : float
        Vertical (voltage) offset fitted by the model
    
    """
          
    est_Amp, est_freq, est_phase, est_Offset = _leastsq(_fittingFunc, [guess_Amp,
                                guess_freq, guess_phase, guess_Offset], args=(time, signal))[0]
    data_fit = est_Amp*_np.sin((est_freq*2*_np.pi)*time+est_phase) + est_Offset
    
    if Plotstuff == True:
     
        fig = _plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time, signal, label="Data")
        ax.plot(time, data_fit, label='Fitted')
        ax.set_title("".format(guess_freq))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("signal")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=3, fancybox=True, shadow=True)
        ax.grid(True)
    return est_freq, est_Amp, est_phase, est_Offset

@_jit
def filterData(FilterDesired_coefArray, data):
    """
    Makes a filter with the coefficients specified and filters the data provided
    with the specified filter.

    Parameters
    ----------
    coefArray : array_like
        Array containing the coefficients for the FIR filter to create. First 
        element multiplies x[n] the next x[n-1] and so on.
    data : ndarray
        Array containing data to be filtered - should be a 1D array of the values of 
        signal as it varies with time.
    
    returns
    -------
    result : ndarray
        The result of filtering the data with the filter specified above.
    """
    filter_perfect = MakeFilter(FilterDesired_coefArray) # Make perfect filter
    result_perfect = filter_perfect(data)
    result = result_perfect
    return result


def _MakeTruncFilter(NumOfBitsToShift):
    """
    Makes a function, which, when provided with an array of coefficients and some data, 
    will construct a truncated version of that filter (with the coefficients multiplied 
    by 2^NumOfBitsToShift and truncted to integers) and filter the data provided.

    Parameters
    ----------
    NumOfBitsToShift : int
        Number for the function generated to bits-shift 
    """
    def filterData_trunc(FilterDesired_coefArray, data):
        BitShiftMultiplier = 2**NumOfBitsToShift
        coefArray_x = _SHIFT_rmzeros(NumOfBitsToShift, FilterDesired_coefArray) # mutliply coefficient array by 
        # BitShiftMultiplier and trucate to integers and remove (only!) the zeroes at the end of the array

        filter_x = MakeFilter(coefArray_x) # Make a filter with the coefficients above
        result_x = filter_x(data) # filter s_int data with the shifted filter
        result_bitshift = result_x//BitShiftMultiplier # divide the result by BitShiftMultiplier using INTEGER DIVISION
        return result_bitshift
    filterData_trunc.__doc__ = """This function constructs a filter with coefficients
    equal to the specified coefficients multiplied by 2**{0} and then truncted down 
    to integers. It then uses this constructed filter to filter the data provided 
    in the data parameter and then divides the filtered data by 2**{0}. This is 
    how floating point coefficients must be utilized inside the FPGA since it 
    cannot implement floating point arithmetic.

    Parameters
    ----------
    FilterDescired_coefArray : array_like
        Array containing the desired coefficients for the FIR filter. First 
        element multiplies x[n] the next x[n-1] and so on. These will be
        multipled and truncated as detailed above and then used to filter the
        data which is then divided by the multiplier. This effectively truncates
        each element of the filter.
    data : ndarray
        Array containing data to be filtered - should be a 1D array of the values of 
        signal as it varies with time.
    
    returns
    -------
    result : ndarray
        The result of filtering the data with the truncated filter as specified above.
    """.format(NumOfBitsToShift)
    return filterData_trunc

@_jit
def _runFilter(filterDataFunc, FilterDesired_coefArray, NumOfFreqs = 500, FreqRange = "Default"):
    """
    This function simulates the filter by creating sinusoidal 16 bit input data
    at a range of linearly spaced frequencies and then sampling this data using a 20KHz
    sampling frequency (this is used as a compromise between execution time
    and the precision of the frequency and phase response. It returns the frequency and
    phase response in terms of the normalised frequency (0->pi) Where pi is the Nyquist
    frequency. It calculates the frequency and phase response by filtering the generated
    sinusoidal signals with a python implementation of the filter and then fits a sine
    wace to the output sine waves to calculate the phase and amplitude response. It
    then calculates the ratio of the amplitudes between filtered and unfiltered signals
    and the difference in phase between the 2 and returns the gain in dB and phase difference
    in radians.

    Parameters
    ----------
    filterDataFunc : function
        This function should be a function which takes as it's first argument
        an array of coefficients and as it's second an array of data to be filtered.
        The functions defined in this module (filterData and filterData_trunc) are
        of the form used by this function.
    FilterDescired_coefArray : array_like
        Array containing the desired coefficients for the FIR filter. First 
        element multiplies x[n] the next x[n-1] and so on. Depending on what
        function is passed as fitlerDataFunc it may implement this filter 
        directly or a truncated version of this filter. 
    NumOfFreqs : int
        Number of frequencies to use to simulate the frequency and phase response
        of the filter. Default is 500.
    FreqRange : array_like, length 2
        The frequency range (defined in normalised frequency 0->pi where pi 
        is the Nyquist frequency) over which to simulate the filter frequency and
        phase response. By default (when equal to the string "Default") it takes 
        the values [pi/100, pi].
    
    Returns
    -------
    freqListNorm : ndarray
        Array containing the normalised frequencies corresponding to the simulated frequencies simulated
    GainArray : ndarray 
        Array containing the gain in dB of the filter when simulated (10*log_10(A_out/A_in))
    PhaseDiffList : ndarray
        Array containing the phase difference between the input sine wave and output sine wave
    """
    
    SampleFreq = 20000 # lower than 10000 causes the first few points of frequency response to
    # be cutoff, a larger frequency works but slows down how long it takes to run due to 
    # the higher time resolution caused by using a higher sample frequency

    NyquistFreq = SampleFreq/2

    
    if FreqRange == sentinel:
        FreqRange=[SampleFreq/100 ,NyquistFreq]
    #elif FreqRange[0] < 0.0030:
    #    raise ValueError ("Lower bound on frequency range must be equal to or above 0.0030")
    else:
        FreqRange = NyquistFreq*(_np.array(FreqRange)/_np.pi)

    freqList = _np.linspace(FreqRange[0], FreqRange[1], NumOfFreqs)

    PhaseDiffList = _np.zeros_like(freqList)
    AmpRatioList = _np.zeros_like(freqList)

    PlotFits = False

    for freqNo, Freq in enumerate(freqList):
        GeneratedAmp = 2**16
        GeneratedPhase = 0
        GeneratedOffset = 2**16/2
        # Generate data that varies between 0 and 65536 in integer steps with frequency of Freq Hz {
        Signal = MakeSignal(Freq, GeneratedAmp, GeneratedPhase, GeneratedOffset)
        NumSecs = 5*(1/Freq)+0.05 # simulate 5 Oscillations and 0.05 seconds at this frequency to determine attenuation
        t, s = MakeSampledSignal(Signal, SampleFreq, NumSecs)# Sample signal at a rate of SampleFreq Hz for NumSecs seconds
        s_int = _np.trunc(s) # makes signal into integers between 0 and 65536 (2^16) => What you'd get inside the FPGA
        
        result = filterDataFunc(FilterDesired_coefArray, s_int) # function that filters the data as the FPFA would
        # and returns the filtered signal between 0 and 65536

        UpperLimForPlot = Freq*2

        stoppingPoint = len(FilterDesired_coefArray) # point after which the filtered signal means
        # nothing since it goes along multiplying the array of coefficients by the array of
        # simulated data. When the amount of data left is less than the coefficients the filter
        # can no longer be used on the data without some elements being zero and therefore
        # introducing an error - will cause problems when fitting as the end part appears attenuated.
        # We cut-off the last few points for this reason.
        
        
        time = t[0:-stoppingPoint]
        voltageCH1 = s_int[0:-stoppingPoint]
        voltageCH2 = result[0:-stoppingPoint]
        SampleInterval = 1/SampleFreq

        est_freq2, est_Amp2, est_phase2, est_Offset2 = fitData(time, voltageCH2, 
                                                                Freq, GeneratedAmp, GeneratedPhase, GeneratedOffset, PlotFits)
        phase_diff = GeneratedPhase - est_phase2
        PhaseDiffList[freqNo] = phase_diff
        AmpRatioList[freqNo] = est_Amp2/GeneratedAmp

    freqListNorm = (freqList/NyquistFreq*_np.pi)
    AmpRat = _np.abs(AmpRatioList)
    GainArray = 10*_np.log10(AmpRat)
    return freqListNorm, GainArray, PhaseDiffList


def ModelFilter(CoefArray, verbosity=1, NumOfFreqs=500):
    """
    This function takes an array of coefficients and finds the frequency response
    of the filter. Verbosity sets if the response should be plotted
    
    Parameters
    ----------
    CoefArray : array_like
        Coefficients of FIR filter to be modelled.
    verbosity : int
        Verbosity of function (i.e. whether to plot frequency and phase response or whether to just return the values.)
        Options (Default is 1): 
        0 - Do not plot anything, just return values
        1 - Plot Frequency and Phase response and return values
    NumOfFreqs : array_like
        Number of frequencies to use to simulate the frequency and phase response
        of the filter. Default is 500.
    
    Returns
    -------
    freqList : ndarray
        Array containing the normalised frequencies corresponding to the simulated frequencies simulated
    GainArray : ndarray 
        Array containing the gain in dB of the filter when simulated (10*log_10(A_out/A_in))
    PhaseDiffArray : ndarray
        Array containing the phase difference between the input sine wave and output sine wave

    """

    freqList, GainArray, PhaseDiffArray = _runFilter(filterData, CoefArray, NumOfFreqs)
    if verbosity == 1: 
        fig1 = _plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(freqList, GainArray, '-', label = "Specified Filter")
        ax.set_title("Frequency Response")
        ax.set_xlabel("$\Omega$ - Normalized frequency ($\pi$=Nyquist Frequency)")
        ax.set_ylabel("Gain (dB)")
        ax.set_xlim([0, _np.pi])
        fig2 = _plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(freqList, PhaseDiffArray, '-', label = "Specified Filter")
        ax.set_title("Phase Response")
        ax.set_xlabel("$\Omega$ - Normalized frequency ($\pi$=Nyquist Frequency)")
        ax.set_ylabel("Phase Difference")
        ax.set_xlim([0, _np.pi])

    _plt.show()
    return freqList, GainArray, PhaseDiffArray

def ModelFilterResponse(CoefArray, SampleFreq, FreqRange, verbosity=1, NumOfFreqs=500):
    """
    This function takes an array of coefficients and finds the frequency response
    of the filter over a certain range using a certain Sample Frequenct. Verbosity 
    sets if the response should be plotted
    
    Parameters
    ----------
    CoefArray : array_like
        Coefficients of FIR filter to be modelled.
    SampleFreq : float
        Sample frequency (in Hz) to simulate (used to convert frequency range to normalised frequency range)
    FreqRange : array_like, length 2
        The frequency range (in Hz) to simulate the response of the filter over
    verbosity : int
        Verbosity of function (i.e. whether to plot frequency and phase response or whether to just return the values.)
        Options (Default is 1): 
        0 - Do not plot anything, just return values
        1 - Plot Frequency and Phase response and return values
    NumOfFreqs : array_like
        Number of frequencies to use to simulate the frequency and phase response
        of the filter. Default is 500.
    
    Returns
    -------
    freqList : ndarray
        Array containing the normalised frequencies corresponding to the simulated frequencies simulated
    GainArray : ndarray 
        Array containing the gain in dB of the filter when simulated (10*log_10(A_out/A_in))
    PhaseDiffArray : ndarray
        Array containing the phase difference between the input sine wave and output sine wave
    """
    NormFreqRange = [_np.pi*FreqRange[0]/SampleFreq, _np.pi*FreqRange[1]/SampleFreq]
    NormfreqList, GainArray, PhaseDiffArray = _runFilter(filterData, CoefArray, NumOfFreqs, NormFreqRange)
    freqList = (NormfreqList/_np.pi)*SampleFreq
    if verbosity == 1: 
        fig1 = _plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(freqList, GainArray, '-', label = "Specified Filter")
        ax.set_title("Frequency Response")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_xlim([FreqRange[0], FreqRange[1]])
        fig2 = _plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(freqList, PhaseDiffArray, '-', label = "Specified Filter")
        ax.set_title("Phase Response")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("Phase Difference")
        ax.set_xlim([FreqRange[0], FreqRange[1]])

    _plt.show()
    return freqList, GainArray, PhaseDiffArray
    
@_jit
def _SHIFT_rmzeros(NumOfBitsToShift, Coefficients):
    """
    This function multiplies the coefficient array by 2^NumOfBitsToShift and truncates the resulting
    array to integers and if there are elements which are zero at the end of the truncted coefficient
    array it removes them.

    Parameters
    ----------
    NumOfBitsToShift : int
        The number of bits to shift the coefficients before truncating them to integers
    Coefficients : ndarray
        Array of coefficients to shift and truncate.
    
    Returns
    -------
    coefArray_Shifted_trailingzerosRemoved : ndarray
        Array of coefficients that have been bit-shifted and truncated to integers
        and had any trailing coefficients with value 0 removed
    """
    BitShiftMultiplier = 2**NumOfBitsToShift
    coefArray_Shifted = _np.zeros_like(Coefficients)
    for i in range(0, len(Coefficients)):
        coefArray_Shifted[i] = int(Coefficients[i]*BitShiftMultiplier)
    
    coefArray_Shifted_trailingzerosRemoved = coefArray_Shifted
    for i in range(len(coefArray_Shifted)-1, 0, -1):
        if(coefArray_Shifted[i] == 0):
            coefArray_Shifted_trailingzerosRemoved = _np.delete(coefArray_Shifted_trailingzerosRemoved, i)
        else:
            break
    return coefArray_Shifted_trailingzerosRemoved

@_jit
def CalcNumOfBitsForSumRegister(NumOfBitsOfOutput, coefArray):
    """
    This Function calculates from the coefficients used in the filter what 
    size register should store the summation of the coefficients multiplied by
    the last N inputs

    Parameters
    ----------
    NumOfBitsOfOutput : int
        Number of bits used for storing the output and the coefficients.
    coefArray : ndarray
        Array containing the FIR filter coefficients 

    Returns
    -------
    NumOfBitsForSumRegister : int
        Number of bits required to store any possible value of the sum of all 
        the coefficients multiplied by the inputs
    """
    maxinput = _np.ones_like(coefArray)*2**NumOfBitsOfOutput
    BigBadCoefArray = max(coefArray)*_np.ones_like(coefArray) # make an array the same size as the 
    # coefficient array where each element is equal to the largest coefficient in the array
    maxSum = sum(BigBadCoefArray*maxinput) # have to do this because the SUM(coefficients) can be 0 => the coefficient can 
    # cancel out when all multiplied by the same number
    NumOfBitsForSumRegister = 0
    while(2**NumOfBitsForSumRegister < maxSum):
        NumOfBitsForSumRegister += 1
    #print("Num Of Bits Required to store max Sum Result: ", NumOfBitsForSumRegister, "\n", 
    #      "max value of Sum Possible:", 2**NumOfBitsForSumRegister, "\n", 
    #     "max possbile with this filter: ", maxSum)
    return NumOfBitsForSumRegister

@_jit
def HowManyBitsToStore(num):
    """
    Calculates how many bits you need to store a number
    
    Parameters
    ----------
    num : int
        Number you want to store
    
    Returns
    -------
    n : int 
        Number of bits required to store the number parameter
    """
    n = 0
    while (2**n-1) < num:
        n += 1
    return n

@_jit
def FindHowManyBitsForCoefs(coefArray):
    """Calculates how many bits are needed to store the largest coefficient in the array
    of coefficients provided.

    Parameters
    ----------
    coefArray : array_like
        Array containing the coefficients
    Returns
    -------
    largestNumOfBits : int
        Number of bits required to store the largest coefficient in the parameter array
    """
    for coef in coefArray:
        largestNumOfBits = 0
        if HowManyBitsToStore(coef) > largestNumOfBits:
            largestNumOfBits = HowManyBitsToStore(coef)

    return(largestNumOfBits)

def ModelTrunctedFilter(FilterDesired_coefArray, NumOfBitsOfOutput, NumOfBitsToShift, 
    verbosity=0):
    """This function takes an array of coefficients, the number of bits in the input/output,
    the number of bits you wish to shift the coefficients before truncating them for integers
    to be used in the FPGA filter and a verbosity - if set to 0 it will just calculate and return
    the return parameters. If set to 1 it will do the above and also print out some values
    and produce a plot of the frequency response of the trunctated and desired filters.
    PLOT PHASE RESPONSE OF TRUNCATED AND IDEAL FITLERS?????!!!!!
    
    Parameters
    ----------
    FilterDesired_coefArray : ndarray
        Coefficients of the FIR filter you want to implement (not truncated)
    NumOfBitsOfOutput : int
        The number of bits of the FPGA output 
    NumOfBitsToShift : int
        The number of bits to shift the coefficients before truncating them
    verbosity : int
        Verbosity of function (i.e. whether to plot frequency and phase response or whether to just return the values.)
        Options (Default is 1): 
        0 - Do not plot anything, just return values
        1 - Plot Frequency and Phase response and return values

    Returns
    -------
    coefArray_trunced_ints : ndarray
        The truncated coefficient array used to produce the frequency repsonse.
    NumOfTruncedCoefs : int
        Length of coefArray_trunced_ints
    NumOfBitsForCoefs : int
        Number of bits required to store the largest coefficient in coefArray_trunced_ints
    NumBitsForSumRegister : int
        Number of bits required to store the summation of the coefficients in coefArray_trunced_ints multiplied by 
        the input values being filtered.
    freqList : ndarray
        Array containing the normalised frequencies corresponding to the simulated frequencies simulated
    GainArray : ndarray 
        Array containing the gain in dB of the ideal FIR filter when simulated (10*log_10(A_out/A_in))
    GainArrayTrunc : ndarray 
        Array containing the gain in dB of the truncated filter when simulated (10*log_10(A_out/A_in))
    """
    BitShiftMultiplier = 2**NumOfBitsToShift

    freqList, GainArray, PhaseDiffArray = _runFilter(filterData, FilterDesired_coefArray)
    filterData_trunc = _MakeTruncFilter(NumOfBitsToShift)
    freqList, GainArrayTrunc, PhaseDiffArray = _runFilter(filterData_trunc, FilterDesired_coefArray)

    coefArray_trunced = _SHIFT_rmzeros(NumOfBitsToShift, FilterDesired_coefArray) # mutliply coefficient array by 
    #print("Desired Coefs: ", FilterDesired_coefArray)
    #print("Coefficients that will be realised by filter: ", coefArray_trunced/2**NumOfBitsToShift)
    #print("Integer coefficients used in filter before power of 2 division: ", coefArray_trunced)

    # BitShiftMultiplier and trucate to integers and remove (only!) the zeroes at the end of the array
    NumBitsForSumRegister = CalcNumOfBitsForSumRegister(NumOfBitsOfOutput, coefArray_trunced)


    FindHowManyBitsForCoefs(coefArray_trunced)

    coefArray_trunced_ints = _np.ndarray.astype(coefArray_trunced, int)

    NumOfBitsForCoefs = FindHowManyBitsForCoefs(coefArray_trunced)

    NumOfTruncedCoefs = len(coefArray_trunced_ints)
    
    if verbosity == 1: 
        fig2 = _plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(freqList, GainArray, '-', label = "Specified Filter")
        ax.plot(freqList, GainArrayTrunc, 'r-', label = "{} bit shifted approx to filter".format(NumOfBitsToShift))
        ax.set_title("Frequency Response")
        ax.set_xlabel("$\Omega$ - Normalized frequency ($\pi$=Nyquist Frequency)")
        ax.set_ylabel("Gain (dB)")
        ax.set_xlim([0, _np.pi])
        ax.legend(bbox_to_anchor=(1.50, 1), loc='lower right', ncol=1)

        print(" Number of Bits of Output:", NumOfBitsOfOutput, "\n",
              "Number of bits to shift:", NumOfBitsToShift, "\n",
              "Number of integer (truncated) Coefficients:", NumOfTruncedCoefs, "\n",
              "Coefficient Array:", coefArray_trunced_ints, "\n",
              "Number of bits to required to store largest coefficient:", NumOfBitsForCoefs, "\n",
              "Number of bits required for SUM register:", NumBitsForSumRegister)
    return (coefArray_trunced_ints, NumOfTruncedCoefs, NumOfBitsForCoefs, 
                NumBitsForSumRegister, freqList, GainArray, GainArrayTrunc)

def StartBitShift(Coefs):
    """                                                                                                            
    Calculate the minimum bit shift such that when the coefficients are truncated none                             
    of the coefficients are truncated to 0.                                                                        
                                                                                                                   
    Parameters                                                                                                     
    ----------                                                                                                     
    Coefs : array_like                                                                                             
        Array of coefficients                                                                                      
                                                                                                                   
    Returns                                                                                                        
    -------                                                                                                        
    NumBitsToShift : int                                                                                           
        Minimum number of bits you can shift the coefficient array before truncating                               
        them such that all the coefficients in the array are non-zero after truncation                             
    """
    minCoef = min(Coefs)
    NumBitsToShift = 1
    while _SHIFT_rmzeros(NumBitsToShift, [minCoef]) == _np.array([0.0]):
        NumBitsToShift += 1
    return(NumBitsToShift)
    

@_jit
def ApproxShiftRequired(FilterDesired_coefArray, NumOfBitsOfOutput, verbosity=1, IdealityParameter = 0.1):
    """
    This function calculates approximately how large the bit-shift of the coefficients
    needs to be for the frequency response of the filter to be approximately the same.
    THIS IS ONLY AN APPROXIMATION. The threshold is when the average difference between
    the ideal and truncated FIR filter's frequency responses (Gain in dB) are less than 
    the IdealityParameter
    
    Parameters
    ----------
    FilterDesired_coefArray : ndarray
        Coefficients of the FIR filter you want to implement (not truncated)
    NumOfBitsOfOutput : int
        The number of bits of the FPGA output 
    verbosity : int
        Verbosity of function (i.e. whether to plot frequency response of the ideal and truncated FIR filter
        calculate to be adequate by this function or whether to just return the values.)
        Options (Default is 1): 
        0 - Do not plot anything, just return values
        1 - Plot Frequency and Phase response and return values
    IdealityParameter : float
        Parameter defining how close the average difference between
        the ideal and truncated FIR filter's frequency responses has 
        to be before the truncated filter is deemed adequate. By
        deault it takes value 0.1
    """
    NumOfBitsToShift = StartBitShift(FilterDesired_coefArray) # start from having the bitshift such that           
    # when truncated none of the coefficients are truncated below 0  

    (coefArray_trunced_ints, NumOfTruncedCoefs, NumOfBitsForCoefs, 
     NumBitsForSumRegister, FreqList, GainArray, GainArrayTrunc) = ModelTrunctedFilter(FilterDesired_coefArray, 
                                                        NumOfBitsOfOutput, NumOfBitsToShift, 
                                                        verbosity=0)
    while sum(abs(GainArray - GainArrayTrunc))/len(FreqList) > IdealityParameter:
        NumOfBitsToShift+=1
        (coefArray_trunced_ints, NumOfTruncedCoefs, NumOfBitsForCoefs, 
     NumBitsForSumRegister, FreqList, GainArray, GainArrayTrunc) = ModelTrunctedFilter(FilterDesired_coefArray, 
                                                        NumOfBitsOfOutput, NumOfBitsToShift, 
                                                        verbosity=0)
    
    print("The following filter can be achieved by shifting by {} bits".format(NumOfBitsToShift))
    (coefArray_trunced_ints, NumOfTruncedCoefs, NumOfBitsForCoefs, 
     NumBitsForSumRegister, _, _, _) = ModelTrunctedFilter(FilterDesired_coefArray, 
                                                        NumOfBitsOfOutput, NumOfBitsToShift,
                                                                      verbosity=verbosity)

    return NumOfBitsToShift

def _FindContainedKey(inputString, inputMap):
    """
    Looks in the input string to find a sub-string which is a key in the inputMap
    dictionary, it then returns the first key it finds in the input string
    
    Parameters
    ----------
    inputString : string
         String to be searched for keys to the inputMap dictionary
    inputMap : dict
         Dictionary containing key's which are strings

    Returns
    -------
    ContainsKey : bool
         True if string contains a sub-string which is itself a key to the dictionary 
             passed as a parameter
         False if string does not contain a sub-string which is itself a key to the 
               dictionary passed as a parameter
    key : string
        The string, which is itself a key to the dictionary passed as a parameter,
        which was found to be in the inputString parameter. Equal to None if no keys found.
    """
    ContainedKey = ""
    for key in inputMap.keys():
        if key in inputString:
            ContainedKey = key
            return True, key
    return False, None

def MakeVHDLFilter(NumOfBitsOfRawInput, NumOfBitsOfOutput, NumOfBitsToShift, NumOfCoefs, 
    coefArray_trunced_ints, NumOfBitsForCoefs, NumBitsForSumRegister,
                  filename):
    """
    Create the VHDL module code to implement a particular FIR filter, Currently the coefficients 
    must be less than 2**NumOfBitsOfOutput (i.e. NumOfBitsForCoefs < NumOfBitsOfOutput).

    Parameters
    ----------
    NumOfBitsOfRawInput : int
        Number of bits of FPGA's input
    NumOfBitsOfOutput : int
        Number of bits of FPGA's output
    NumOfBitsToShift : int
        Number of bits to shift coefficients by
    NumOfCoefs : int
        Number of (truncated) coefficients
    coefArray_trunced_ints : ndarray
        Truncated integer coefficients
    NumOfBitsForCoefs : int
        Number of bits required to store the largest coefficient
    NumBitsForSumRegister : int
        Number of bits required to store the summation of the coefficients multiplied by the inputs
    filename : string
        Filename to which the created VHDL FIR filter will be stored.
    
    Returns
    -------
    None
    """

    if NumOfBitsForCoefs > NumOfBitsOfOutput:
        print("ERROR: WILL NOT WORK - More Bits Needed For Coefficients")
        return None

    AdditionTemplate = "to_integer(unsigned(ProductArray(0))) +"
    FinalAdditiontemplate ="to_integer(unsigned(ProductArray(0)))"

    AdditionString = ''

    indentNewLine ='\n        '

    for i in range(0, NumOfCoefs):
        if i < NumOfCoefs-1 and coefArray_trunced_ints[i] != 0:
            AdditionString += AdditionTemplate.replace('0', '{}'.format(i))
            AdditionString += indentNewLine
        elif i == NumOfCoefs-1:
            AdditionString += FinalAdditiontemplate.replace('0', '{}'.format(i))

    InputsAndValues = {
        'NoOfRawInputBits_1 :' : str(NumOfBitsOfRawInput-1),
        'NoOfOutputBits_NoOfRawInputBits_1 :' : str(NumOfBitsOfOutput-NumOfBitsOfRawInput-1),
        'NoOfOutputBits :': str(NumOfBitsOfOutput),
        'NoOfOutputBits_1 :' : str(NumOfBitsOfOutput-1),
        'DoubleNoOfOutputBits_1 :' : str(2*NumOfBitsOfOutput-1),
        'NoOfBitsToShift :' : str(NumOfBitsToShift),
        'NoOfBitsToShift_1 :' : str(NumOfBitsToShift-1),
        'NoOfCoefs_1 :' : str(NumOfCoefs-1),
        'NoOfBitsForSumRegister :' : str(NumBitsForSumRegister),
        'NoOfBitsForSumRegister_1 :' : str(NumBitsForSumRegister-1),
        'Coefficients :' : str(tuple(coefArray_trunced_ints)),
        'ZerosArray :' : '0'*NumOfBitsToShift,
        'ZerosArrayRawInputToInput :' : '0'*(NumOfBitsOfOutput-NumOfBitsOfRawInput),
        'ADDITION_STUFF_GOES_HERE' : AdditionString
    }
    InputsAndTemplateVals = {
        'NoOfRawInputBits_1 :' : 'N0',
        'NoOfOutputBits_NoOfRawInputBits_1 :' : 'N0_5',
        'NoOfOutputBits :': 'N1',
        'NoOfOutputBits_1 :' : 'N2',
        'DoubleNoOfOutputBits_1 :' : 'N3',
        'NoOfBitsToShift :' : 'N4',
        'NoOfBitsToShift_1 :' : 'N5',
        'NoOfCoefs_1 :' : 'N6',
        'NoOfBitsForSumRegister :' : 'N7',
        'NoOfBitsForSumRegister_1 :' : 'N8',
        'Coefficients :' : 'N9',
        'ZerosArray :' : 'N10',
        'ZerosArrayRawInputToInput :' : 'N11',
        'ADDITION_STUFF_GOES_HERE' : 'ADDITION_STUFF_GOES_HERE'
    }

    resource_package = __name__  # Gets the module/package name.
    rawtemplate = _pkg_resources.resource_string(resource_package, "NonRecursiveFilterTemplate.VHD") # reads in the template file as a bytes object

    outputFile = open(filename, "w")
    currentDate = _strftime("%d/%m/%Y")
    currentTime = _strftime("%H:%M:%S")
    print("""-- This VHDL file, implementing an FIR filter, was created 
-- by the python FPGA_DSPy library on {} at {}""".format(currentDate, currentTime), file=outputFile)
    template = rawtemplate.decode(encoding='UTF-8') # decocdes the bytes object into a string object
    for i, line in enumerate(template.splitlines()): # splits the string into lines
        if line == "":
            line = " " # sets an empty line into a string containing a single white-space rather than a string with zero length
        if line[-1] == "\n":
            line = line[:-1]
        if (i < 33) or (i > 65 and i < 71) or i == 92: # only search the first 33 lines (up until end of constant declarations) - any futhur and the code could be messed up # also search line 63 to set the coefficients 
            #print(line)
            ContainsSubString, SubString =  _FindContainedKey(line, InputsAndValues)
            if ContainsSubString:
                line = line.replace(InputsAndTemplateVals[SubString], InputsAndValues[SubString])
                del InputsAndValues[SubString]
        print(line, file=outputFile)

    outputFile.close()
    return None


def SimpleMakeFilter(FilterDesired_coefArray, NumOfBitsOfInput, NumOfBitsOfOutput, 
                     filename, verbosity=1, IdealityParameter = 0.1):
    """
    Create the VHDL module code to implement a particular FIR filter. This function
    will produce a truncated set of coefficients that implement an FIR filter which
    approximates the specified FIR filter such that the average differnece between the
    frequency response (gain in dB) of the specified filter and the truncated filter 
    is less than the IdealityParameter.

    FilterDesired_coefArray : ndarray
        Coefficients of the FIR filter you want to implement (not truncated)
    NumOfBitsOInput : int
        Number of bits of FPGA's input
    NumOfBitsOfOutput : int
        Number of bits of FPGA's output
    filename : string
        Filename to which the created VHDL FIR filter will be stored.
    verbosity : int
        Verbosity of function (i.e. whether to plot frequency response of the ideal and truncated FIR filter
        calculate to be adequate by this function or whether to just return the values.)
        Options (Default is 1): 
        0 - Do not plot anything, just return values
        1 - Plot Frequency and Phase response and return values
    IdealityParameter : float
        Parameter defining how close the average difference between
        the ideal and truncated FIR filter's frequency responses has 
        to be before the truncated filter is deemed adequate. By
        deault it takes value 0.1
    Returns
    -------
    None
    """
    NumOfBitsToShift = ApproxShiftRequired(FilterDesired_coefArray, NumOfBitsOfOutput, verbosity, IdealityParameter)
    TruncatedFilterValues = ModelTrunctedFilter(FilterDesired_coefArray, 
        NumOfBitsOfOutput, NumOfBitsToShift, verbosity=0)
    (coefArray_trunced_ints, NumOfTruncedCoefs, NumOfBitsForCoefs, 
     NumBitsForSumRegister, freqList, GainArray, GainArrayTrunc) = TruncatedFilterValues 
    MakeVHDLFilter(NumOfBitsOfInput, NumOfBitsOfOutput, NumOfBitsToShift, NumOfTruncedCoefs, 
    coefArray_trunced_ints, NumOfBitsForCoefs, NumBitsForSumRegister, filename)
    return None

def _makeCoefficients(CentralFrequency, Bandwidth):
    """
    Function which returns a function which calculates the coefficients to 
    implement the bandpass filter specified by the coefficients.

    Parameters
    ----------
    CentralFrequency : float
        Central frequency of bandpass filter desired in normalised frequnecy.
    BandWidth : float
        BandWidth of bandpass filter desired in normalised frequnecy.
        
    Returns
    -------
    h - funtion which takes an argument n (any positive integer) and returns
        the corresponding coefficient/impulse response to implement the 
        "ideal" rectangular windowed bandpass filter.
    """
    Omega0 = CentralFrequency
    Omega1 = Bandwidth/2.0
    def h(n):
        if n == 0:
            return 0
        else:
            return( 1/(n*_np.pi)*_np.sin(n*Omega1)*_np.cos(n*Omega0) )
    h.__doc__ = """"""
    return h
    

def _HanningWeightFunction(n, NumberOfCoefs):
    w_n = 0.5 + 0.5*_np.cos(n*_np.pi/(NumberOfCoefs))
    return w_n

def _HammingWeightFunction(n, NumberOfCoefs):
    w_n = 0.54 + 0.46*_np.cos(n*_np.pi/(NumberOfCoefs-1))
    return w_n

def MakeBandPassFilterCoefficients_Normalised(CentralFrequency, Bandwidth,
                                   NumberOfCoefs, WindowFunction=None):
    """
    This function calculates the array of coefficients needed to implement a particular bandpass
    filter. The bandpass filter is specified by the input arguments.

    Parameters
    ----------
    CentralFrequency : float
         The frequency at the centre of the pass-band (in terms of the normalised
         frequency 0 to pi (where pi is the Nyquist frequency)
    BandWidth : flaot
         The width of the pass-band (in terms of the normalised frequency 0 to pi 
         (where pi is the Nyquist frequency)
    NumberOfCoefs : int
         The number of coefficients to implement the filter with
    WindowFunction : string
        The type of window function to use options are:
        None - Rectangular window - truncates coefficients generated by a rectangular "ideal" bandpass
        "Hanning" - a Hanning (or Von Hann) window
        "Hamming" - a Hamming window
    
    Returns
    -------
    WindowedhArray - ndarray
        Array containing the coefficients to implement the specified bandpass FIR filter
    """
    h = _makeCoefficients(CentralFrequency, Bandwidth)
    hArray = _np.zeros(NumberOfCoefs)
    WArray = _np.zeros(NumberOfCoefs)
    for n in range(0, NumberOfCoefs):
        hArray[n] = h(n)
    if WindowFunction == None:
        WArray = _np.ones(NumberOfCoefs)
    elif WindowFunction == 'Hanning':
        for n in range(0, NumberOfCoefs):
            WArray[n] = _HanningWeightFunction(n, NumberOfCoefs)
    elif WindowFunction == 'Hamming':
        for n in range(0, NumberOfCoefs):
            WArray[n] = _HammingWeightFunction(n, NumberOfCoefs)
    else:
        print("""Please enter a valid Window Function. Options are:
        None - Rectangular window - truncates coefficients generated by a rectangular "ideal" bandpass
        Hanning - a Hanning (or Von Hann) window
        Hamming - a Hamming window        
        """)
        return None
            
    WindowedhArray = hArray*WArray
    return WindowedhArray


def MakeBandPassFilterCoefficients(SampleFreq, CentralFrequency, Bandwidth,
                                   NumberOfCoefs, WindowFunction=None):
    NormCentralFrequency = _np.pi*CentralFrequency/SampleFreq
    NormBandwidth = _np.pi*Bandwidth/SampleFreq
    Coefs = MakeBandPassFilterCoefficients_Normalised(NormCentralFrequency, NormBandwidth, NumberOfCoefs, WindowFunction)
    return Coefs
