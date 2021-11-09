from datetime import date
import pathlib

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

    def get_bvec(self, dtime):
        """
        Get the magnetic field vector closest to *time*.
        """
        epochs = self.cdf.varget(self.epoch_var)
        dtime = cdflib.cdfepoch.compute([dtime.year, dtime.month, dtime.day,
                                         dtime.hour, dtime.minute,
                                         dtime.second,
                                         dtime.microsecond // 1000,
                                         dtime.microsecond % 1000,
                                         0])
        idx = np.argmin(np.abs(epochs - dtime))
        mag = self.cdf.varget('psp_fld_l2_mag_SC_4_Sa_per_Cyc',
                              startrec=idx, endrec=idx)
        return Vector(mag[0])
