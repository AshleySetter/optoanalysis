import matplotlib
matplotlib.use('agg')
import pytest

pytest.main(['-v', '--cov', 'DataHandling', '--mpl'])

# runs pytest in verbose mode, reports coverage of the DataHandling library and with pytest-mpl so that the figures created are compared to the baselines in tests/baseline. I am running it inside of python so that the matplotlib.use('agg') command which sets the matplotlib DISPLAY variable is set before running pytest - otherwise plots cannot be made on travis-CI
