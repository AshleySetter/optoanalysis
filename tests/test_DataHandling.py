import matplotlib
matplotlib.use('agg', warn=False, force=True)
import pytest
import DataHandling
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt

def test_nothing():
    print("Nothing Tested")
    print(dir(DataHandling))

def test_LoadData():
    data = DataHandling.LoadData("testData.raw")
    assert type(data) == DataHandling.DataHandling.DataObject
    assert data.filename == "testData.raw"
    assert data.time[1]-data.time[0] == pytest.approx(data.SampleFreq, 0.0001*data.SampleFreq) # approx equal to within 0.01%
    assert max(data.freqs) == pytest.approx(data.SampleFreq/2, 0.00001*data.SampleFreq/2) # max freq in PSD is approx equal to Nyquist frequency within 0.001%
    t, V = data.getTimeData()
    np.testing.assert_array_equal(t, data.time)
    np.testing.assert_array_equal(V, data.Voltage)
    
    return None

@pytest.mark.mpl_image_compare(tolerance=20) # this decorator compares the figure object returned by the following function to the baseline png image stored in tests/baseline
def test_plotPSD():
    data = DataHandling.LoadData("testData.raw")
    fig, ax = data.plotPSD([0, 400e3], ShowFig=False)
    return fig
