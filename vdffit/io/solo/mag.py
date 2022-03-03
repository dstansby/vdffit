import glob
import pathlib

import numpy as np

from vdffit.io.cdf import CDFFile
from vdffit.util.vector import Vector

base_dir = pathlib.Path('/Volumes/Work/Data/solo/mag')

__all__ = ['MAGL2']


class MAGL2(CDFFile):
    def __init__(self, date):
        self.date = date
        # Calling this loads the CDF and checks that the file exists
        self.cdf
        self.mag_rtn = self.cdf.varget('B_RTN')
        self.times

    @property
    def path(self):
        date_str = self.date.strftime('%Y%m%d')
        fname = f'solo_L2_mag-rtn-normal_{date_str}_V*.cdf'
        fpath = base_dir / fname
        fpaths = sorted(glob.glob(str(fpath)))
        if len(fpaths):
            return pathlib.Path(fpaths[-1])

        raise FileNotFoundError(f'No MAG data for {self.date} in {base_dir}')

    def get_bvec(self, epoch):
        """
        Get the magnetic field vector closest to *time*.
        """
        idx = np.argmin(np.abs(epoch - self.epochs))
        return Vector(self.mag_rtn[idx, :])
