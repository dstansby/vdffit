from vdffit.io import SPANL2CDF
from vdffit.fitting import BiMaxFitter
from datetime import datetime

cdf = SPANL2CDF(datetime(2021, 6, 30))
fitter = BiMaxFitter()
result = fitter.fit_cdf(cdf)
print(result)
