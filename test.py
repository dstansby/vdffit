from datetime import datetime

from vdffit.fitting import BiMaxFitter
from vdffit.io.psp import SPANL2CDF

# files = fetch_psp_files('2020-01-01', '2020-01-08')

cdf = SPANL2CDF(datetime(2020, 1, 7))
fitter = BiMaxFitter()
result = fitter.fit_cdf(cdf)
print(result)
