from functools import cached_property
import pathlib

import astropy.constants as const
import numpy as np

from vdffit.vdf import PASDistribution
from vdffit.io.cdf import VDFCDF
from vdffit.io.solo.mag import MAGL2


base_dir = pathlib.Path('/Volumes/Work/Data/solo/swa/pas')


class PASL2CDF(VDFCDF):
    """
    A SWA PAS level 2 file.

    Each file corresponds to a single day.
    """
    def __init__(self, date):
        """
        Parameters
        ----------
        date : astropy.time.Time
            Date of file.
        """
        self.date = date
        self.mag_cdf = MAGL2(date)
        # Calling this loads the CDF and checks that the file exists
        self.cdf

    @property
    def path(self):
        """
        Path to CDF file.

        Returns
        -------
        pathlib.Path
        """
        date_str = self.date.strftime('%Y%m%d')
        fname = f'solo_L2_swa-pas-vdf_{date_str}_V02.cdf'
        return base_dir / fname

    def __getitem__(self, idx):
        """
        Get a single distribtuion function.

        Parameters
        ----------
        idx : int or Time

        Returns
        -------
        PASDistribution
        """
        if not isinstance(idx, int):
            time = idx
            if time not in self.times:
                raise ValueError(f'{time} not found in timestamps')
            idx = self.times.index(time)
        else:
            time = self.times[idx]

        epoch = self.epochs[idx]

        return PASDistribution(self.vdf[idx, ...],
                               self.energy,
                               self.theta,
                               self.phi,
                               time,
                               self.mag_cdf.get_bvec(epoch))

    @cached_property
    def vdf(self):
        """
        Velocity distribution function.

        Shape (ntime, ntheta, nphi, nenergy).
        """
        return self.varget('vdf')

    @cached_property
    def energy(self):
        """
        Energies.

        Shape (nenergy).
        """
        return self.varget('Energy')

    @cached_property
    def theta(self):
        """
        Theta values.

        Shape (ntheta, nphi, nenergy).
        """
        return self.varget('Full_elevation')

    @cached_property
    def phi(self):
        """
        Phi values.

        Shape (ntheta, nphi, nenergy).
        """
        return self.varget('Full_azimuth')
