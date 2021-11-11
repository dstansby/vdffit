from datetime import date
import pathlib
from functools import cached_property

import cdflib
import numpy as np

from vdffit.io.cdf import CDFFile

from vdffit.util.vector import Vector


base_dir = pathlib.Path('/Volumes/Work/Data/psp/fields/l2/mag_sc_4_per_cycle')


class MAGL2(CDFFile):
    """
    A level 2 magnetic field data product, specifically
    'epoch_mag_SC_4_Sa_per_Cyc'.
    """
    epoch_var = 'epoch_mag_SC_4_Sa_per_Cyc'

    def __init__(self, date):
        self.date = date
        # Calling this loads the CDF and checks that the file exists
        self.cdf

    @property
    def path(self):
        date_str = self.date.strftime('%Y%m%d')
        fname = f'psp_fld_l2_mag_sc_4_sa_per_cyc_{date_str}_v02.cdf'

        return base_dir / self.date.strftime('%Y') / fname

    @cached_property
    def all_bvecs(self):
        return self.cdf.varget('psp_fld_l2_mag_SC_4_Sa_per_Cyc')

    def get_bvec(self, epoch):
        """
        Get the magnetic field vector closest to *time*.
        """
        idx = np.argmin(np.abs(epoch - self.epochs))
        return Vector(self.all_bvecs[idx, :])
