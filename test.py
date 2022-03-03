from vdffit.io import PASL2CDF
from vdffit.fitting import BiMaxFitter
from datetime import datetime

cdf = PASL2CDF(datetime(2021, 8, 13))
vdf = cdf[0]
exit()

print(vdf.velocities_instr_frame())
exit()

cdf = SPANL2CDF(datetime(2021, 6, 30))
fitter = BiMaxFitter()
result = fitter.fit_cdf(cdf)
print(result)
