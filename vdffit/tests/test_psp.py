from vdffit.net.psp import fetch_psp_files
from vdffit.io.psp import SPANL2CDF
from vdffit.fitting import BiMaxFitter
from datetime import datetime

from astropy.timeseries import TimeSeries


def test_psp():
    fetch_psp_files('2020-01-07', '2020-01-08')
    cdf = SPANL2CDF(datetime(2020, 1, 7))
    fitter = BiMaxFitter()
    result = fitter.fit_cdf(cdf)

    assert isinstance(result, TimeSeries)
