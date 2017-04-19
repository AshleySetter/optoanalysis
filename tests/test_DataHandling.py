import matplotlib
matplotlib.use('agg', warn=False, force=True)
import pytest
import datahandling
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt

def test_load_data():
    """
    Tests that load_data works and therefore that DataObject.__init__, DataObject.get_time_data and DataObject.getPSD work. Specifically it checks that the data loads and that it returns an object of type DataObject. It checks that the filepath points to the correct place. The data is sampled at the correct frequency and therefore that it has loaded the times correctly. It checks that the max frequency in the PSD is approximately equal to the Nyquist frequency for the test data. It also checks that the data returned by get_time_data matches the data loaded.
    """
    data = datahandling.load_data("testData.raw")
    assert type(data) == datahandling.datahandling.DataObject
    assert data.filename == "testData.raw"
    assert data.time[1]-data.time[0] == pytest.approx(1/data.SampleFreq, rel=0.0001) # approx equal to within 0.01%
    assert max(data.freqs) == pytest.approx(data.SampleFreq/2, rel=0.00001) # max freq in PSD is approx equal to Nyquist frequency within 0.001%
    t, V = data.get_time_data()
    np.testing.assert_array_equal(t, data.time)
    np.testing.assert_array_equal(V, data.voltage)
    
    return None

GlobalData = datahandling.load_data("testData.raw") # Load data to be used in upcoming tests - so that it doesn't need to be loaded for each individual function to be tested

@pytest.mark.mpl_image_compare(tolerance=20) # this decorator compares the figure object returned by the following function to the baseline png image stored in tests/baseline
def test_plot_PSD():
    """
    This tests that the plot of the PSD produced by DataObject.plot_PSD is produced correctly and matches the baseline to a certain tolerance.
    """
    fig, ax = GlobalData.plot_PSD([0, 400e3], ShowFig=False)
    return fig

@pytest.mark.mpl_image_compare(tolerance=20) # this decorator compares the figure object returned by the following function to the baseline png image stored in tests/baseline
def test_get_fit():
    """
    Tests that DataObject.get_fit works and therefore tests fitPSD, fit_curvefit and PSD_Fitting as these are dependancies. It tests that the output values of the fitting are correct (both the values and thier errors) and that the plot looks the same as the baseline, within a certain tolerance.
    """
    A, F, Gamma, fig, ax = GlobalData.get_fit(75000, 10000, ShowFig=False)
    assert A.n == pytest.approx(584418711252, rel=0.0001)
    assert F.n == pytest.approx(466604, rel=0.0001)
    assert Gamma.n == pytest.approx(3951.716, rel=0.0001)
    
    assert A.std_dev == pytest.approx(5827258935, rel=0.0001)    
    assert F.std_dev == pytest.approx(50.3576, rel=0.0001)     
    assert Gamma.std_dev == pytest.approx(97.5671, rel=0.0001)
    
    return fig

def test_extract_parameters():
    """
    Tests that DataObject.extract_parameters works and returns the correct values.
    """
    with open("testDataPressure.dat", 'r') as file:
        for line in file:
            pressure = float(line.split("mbar")[0])
    R, M, ConvFactor = GlobalData.extract_parameters(pressure, 0.15)

    assert R.n == pytest.approx(3.27536e-8, rel=0.0001)
    assert M.n == pytest.approx(3.23808e-19, rel=0.0001)
    assert ConvFactor.n == pytest.approx(190629, rel=0.0001)
    
    assert R.std_dev == pytest.approx(4.97914e-9, rel=0.0001)    
    assert M.std_dev == pytest.approx(9.84496e-20, rel=0.0001)     
    assert ConvFactor.std_dev == pytest.approx(58179.9, rel=0.0001)

    return None
