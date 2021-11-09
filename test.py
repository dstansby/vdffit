from vdffit.io import SPANL2CDF
from vdffit.fitting import BiMaxFitter
from datetime import datetime

vdf = SPANL2CDF(datetime(2021, 6, 30))[0]
fitter = BiMaxFitter()
result = fitter.fit(vdf)
print(result)
