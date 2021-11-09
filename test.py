from vdffit.io import SPANL2CDF
from datetime import datetime

vdf = SPANL2CDF(datetime(2021, 6, 30))[0]
print(vdf.vdf)
