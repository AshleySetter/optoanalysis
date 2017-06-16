import matplotlib
matplotlib.use('agg', warn=False, force=True)
import pytest
import datahandling
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt

plot_similarity_tolerance = 30
float_relative_tolerance = 1e-3

def test_load_data():
    """
    Tests that load_data works and therefore that DataObject.__init__, DataObject.get_time_data and DataObject.getPSD work. Specifically it checks that the data loads and that it returns an object of type DataObject. It checks that the filepath points to the correct place. The data is sampled at the correct frequency and therefore that it has loaded the times correctly. It checks that the max frequency in the PSD is approximately equal to the Nyquist frequency for the test data. It also checks that the data returned by get_time_data matches the data loaded.
    """
    data = datahandling.load_data("testData.raw")
    assert type(data) == datahandling.datahandling.DataObject
    assert data.filename == "testData.raw"
    assert data.time[1]-data.time[0] == pytest.approx(1/data.SampleFreq, rel=float_relative_tolerance) 
    assert max(data.freqs) == pytest.approx(data.SampleFreq/2, rel=0.00001) # max freq in PSD is approx equal to Nyquist frequency
    t, V = data.load_time_data() 
    np.testing.assert_array_equal(t, data.time)
    np.testing.assert_array_equal(V, data.voltage)
    return None

GlobalData = datahandling.load_data("testData.raw", NPerSegmentPSD=int(1e5)) # Load data to be used in upcoming tests - so that it doesn't need to be loaded for each individual function to be tested

@pytest.mark.mpl_image_compare(tolerance=plot_similarity_tolerance) # this decorator compares the figure object returned by the following function to the baseline png image stored in tests/baseline
def test_plot_PSD():
    """
    This tests that the plot of the PSD produced by DataObject.plot_PSD is produced correctly and matches the baseline to a certain tolerance.
    """
    fig, ax = GlobalData.plot_PSD([0, 400], ShowFig=False)
    return fig

@pytest.mark.mpl_image_compare(tolerance=plot_similarity_tolerance) # this decorator compares the figure object returned by the following function to the baseline png image stored in tests/baseline
def test_get_fit():
    """
    Tests that DataObject.get_fit works and therefore tests fitPSD, fit_curvefit and PSD_Fitting as these are dependancies. It tests that the output values of the fitting are correct (both the values and thier errors) and that the plot looks the same as the baseline, within a certain tolerance.
    """
    A, F, Gamma, fig, ax = GlobalData.get_fit(75000, 10000, ShowFig=False)
    assert A.n == pytest.approx(584418711252, rel=float_relative_tolerance)
    assert F.n == pytest.approx(466604, rel=float_relative_tolerance)
    assert Gamma.n == pytest.approx(3951.716, rel=float_relative_tolerance)
    
    assert A.std_dev == pytest.approx(5827258935, rel=float_relative_tolerance)    
    assert F.std_dev == pytest.approx(50.3576, rel=float_relative_tolerance)     
    assert Gamma.std_dev == pytest.approx(97.5671, rel=float_relative_tolerance)
    
    return fig

def test_extract_parameters():
    """
    Tests that DataObject.extract_parameters works and returns the correct values.
    """
    with open("testDataPressure.dat", 'r') as file:
        for line in file:
            pressure = float(line.split("mbar")[0])
    R, M, ConvFactor = GlobalData.extract_parameters(pressure, 0.15)

    assert R.n == pytest.approx(3.27536e-8, rel=float_relative_tolerance)
    assert M.n == pytest.approx(3.23808e-19, rel=float_relative_tolerance)
    assert ConvFactor.n == pytest.approx(190629, rel=float_relative_tolerance)
    
    assert R.std_dev == pytest.approx(4.97914e-9, rel=float_relative_tolerance)    
    assert M.std_dev == pytest.approx(9.84496e-20, rel=float_relative_tolerance)     
    assert ConvFactor.std_dev == pytest.approx(58179.9, rel=float_relative_tolerance)

    return None

def test_get_time_data():
    """
    Tests that DataObject.get_time_data returns the correct number of values.
    """
    t, v = GlobalData.get_time_data(timeStart=0, timeEnd=1e-3)
    assert len(t) == len(v)
    assert len(t) == 10000
    return None

@pytest.mark.mpl_image_compare(tolerance=plot_similarity_tolerance)
def test_plot_time_data():
    """
    This tests that the plot of the time trace (from -1ms to 1ms) produced by DataObject.plot_time_data is produced correctly and matches the baseline to a certain tolerance.
    """
    fig, ax = GlobalData.plot_time_data(timeStart=-1e-3, timeEnd=1e-3, units='ms', ShowFig=False)
    return fig
    
def test_calc_area_under_PSD():
    """
    This tests that the calculation of the area under the PSD
    from 50 to 100 KHz, calculated by 
    DataObject.calc_area_under_PSD is unchanged.
    """
    TrueArea = 1.6900993420543872e-06
    area = GlobalData.calc_area_under_PSD(50e3, 100e3)
    assert area == pytest.approx(TrueArea, rel=float_relative_tolerance)
    return None

def test_get_fit_auto():
    """
    This tests that DataObect.get_fit_auto continues to return the same
    values as when the test was created, to within the set tolerance.
    """
    ATrue = 466612.80058291875
    AErrTrue = 54.936633293369404
    OmegaTrapTrue = 583205139563.28
    OmegaTrapErrTrue = 7359927227.585048
    BigGammaTrue = 3946.998785496495
    BigGammaErrTrue = 107.96706466271127
    A, OmegaTrap, BigGamma = GlobalData.get_fit_auto(70e3, ShowFig=False)
    assert A.n == pytest.approx(ATrue, rel=float_relative_tolerance)
    assert OmegaTrap.n == pytest.approx(OmegaTrapTrue, rel=float_relative_tolerance)
    assert BigGamma.n == pytest.approx(BigGammaTrue, rel=float_relative_tolerance)
    assert A.std_dev == pytest.approx(AErrTrue, rel=float_relative_tolerance)
    assert OmegaTrap.std_dev == pytest.approx(OmegaTrapErrTrue, rel=float_relative_tolerance)
    assert BigGamma.std_dev == pytest.approx(BigGammaErrTrue, rel=float_relative_tolerance)
    return None 

@pytest.mark.mpl_image_compare(tolerance=plot_similarity_tolerance)
def test_extract_motion():
    """
    Tests the DataObject.extract_motion function, and therefore
    the get_ZXY_data function and get_ZXY_freqs function.
    """
    expectedLength = int(np.floor(len(GlobalData.time)/3))
    z, x, y, t, fig, ax = GlobalData.extract_ZXY_motion([75e3, 167e3, 185e3], 5e3, [15e3, 15e3, 15e3], 3, NPerSegmentPSD=int(1e5), MakeFig=True, ShowFig=False)
    assert len(z) == len(t)
    assert len(z) == len(x)
    assert len(x) == len(y)
    assert len(z) == expectedLength
    return fig

@pytest.mark.mpl_image_compare(tolerance=plot_similarity_tolerance)
def test_plot_phase_space():
    """
    Test the plot_phase_space and therefore calc_phase_space function.
    """
    fig1, axscatter, axhistx, axhisty, cb = GlobalData.plot_phase_space(75e3, GlobalData.ConvFactor, PeakWidth=10e3, logscale=False, ShowPSD=False, ShowFig=False, FractionOfSampleFreq=3)
    return fig1
    
def test_multi_load_data():
    """
    Tests the multi_load_data function, checks that the data
    is loaded correctly by checking various properties are set.
    """
    data = datahandling.multi_load_data(1, [1, 36], [0])
    assert data[0].filename == "CH1_RUN00000001_REPEAT0000.raw"
    assert data[1].filename == "CH1_RUN00000036_REPEAT0000.raw"
    for dataset in data:
        assert type(dataset) == datahandling.datahandling.DataObject
        assert dataset.time[1]-dataset.time[0] == pytest.approx(1/dataset.SampleFreq, rel=float_relative_tolerance) 
        assert max(dataset.freqs) == pytest.approx(dataset.SampleFreq/2, rel=0.00001) # max freq in PSD is approx equal to Nyquist frequency
    return None

GlobalMultiData = datahandling.multi_load_data(1, [1, 36], [0], NPerSegmentPSD=int(1e5)) # Load data to be used in upcoming tests - so that it doesn't need to be loaded for each individual function to be tested

def test_calc_temp():
    """
    Tests calc_temp by calculating the temperature of the
    z degree of data from it's reference.
    """
    for dataset in GlobalMultiData:
        dataset.get_fit_auto(65e3, ShowFig=False)
    T = datahandling.calc_temp(GlobalMultiData[0], GlobalMultiData[1])
    assert T.n == pytest.approx(2.6031509367704735, rel=float_relative_tolerance)
    assert T.std_dev == pytest.approx(0.21312482508893446, rel=float_relative_tolerance)
    return None

@pytest.mark.mpl_image_compare(tolerance=plot_similarity_tolerance)
def test_multi_plot_PSD():
    """
    This tests that the plot of PSD for the 2 datasets (from 0 to 300KHz) 
    produced by DataObject.multi_plot_PSD is produced correctly and matches the
    baseline to a certain tolerance.
    """
    fig, ax = datahandling.multi_plot_PSD(GlobalMultiData, xlim=[0, 300], units="kHz", LabelArray=["Reference", "Cooled"], ColorArray=["red", "blue"], alphaArray=[0.8, 0.8], ShowFig=False)
    return fig

@pytest.mark.mpl_image_compare(tolerance=plot_similarity_tolerance)
def test_multi_plot_time():
    """
    This tests that the plot of time for the 2 datasets (from -1000us to 1000us) 
    produced by DataObject.multi_plot_time is produced correctly and matches the
    baseline to a certain tolerance.
    """
    fig, ax = datahandling.multi_plot_time(GlobalMultiData, SubSampleN=1, units='us', xlim=[-1000, 1000], LabelArray=["Reference", "Cooled"], ShowFig=False)
    return fig

@pytest.mark.mpl_image_compare(tolerance=plot_similarity_tolerance)
def test_multi_subplots_time():
    """
    This tests that the plots of time for the 2 datasets (from -1000us to 1000us) 
    produced by DataObject.multi_subplots_time is produced correctly and matches the
    baseline to a certain tolerance.
    """
    fig, ax = datahandling.multi_subplots_time(GlobalMultiData, SubSampleN=1, units='us', xlim=[-1000, 1000], LabelArray=["Reference", "Cooled"], ShowFig=False)
    return fig
