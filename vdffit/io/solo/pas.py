from functools import cached_property
import pathlib

import astropy.constants as const
import astropy.units as u
import numpy as np

from vdffit.vdf import PASDistribution
from vdffit.io.cdf import VDFCDF
from vdffit.io.solo.mag import MAGL2


base_dir = pathlib.Path('/Volumes/Work/Data/solo/swa/pas')
u.add_enabled_units([u.def_unit('unitless', u.dimensionless_unscaled)])


__all__ = ['PASL2CDF']


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
        start_idx = [self.start_azimuth_idx[idx],
                     self.start_elevation_idx[idx],
                     self.start_energy_idx[idx]]
        shape = [self.n_azimuth[idx],
                 self.n_elevation[idx],
                 self.n_energy[idx]]

        return PASDistribution(self.vdf[idx, ...],
                               self.energy,
                               self.theta,
                               self.phi,
                               start_idx,
                               shape,
                               time,
                               self.mag_cdf.get_bvec(epoch))

    @cached_property
    def vdf(self):
        """
        Velocity distribution function.

        Shape (ntime, nphi, ntheta, nenergy).
        """
        return self.varget('vdf')

    @cached_property
    def energy(self):
        """
        Energies.

        Shape (nenergy,).
        """
        return self.varget('Energy')

    @cached_property
    def theta(self):
        """
        Theta values.

        Shape (nphi, ntheta,).
        """
        return self.varget('Full_elevation')[:, :, 1]

    @cached_property
    def phi(self):
        """
        Phi values.

        Shape (nphi, ntheta,).
        """
        return self.varget('Full_azimuth')[:, :, 1]

    @cached_property
    def elevation_correction(self):
        """
        Elevation correction.

        Shape (nenergy,)
        """
        return self.varget('Elevation_correction')

    @cached_property
    def start_energy_idx(self):
        """
        Index of the first energy bin.

        Shape (ntime,)
        """
        return self.varget('start_energy').value.astype(int)

    @cached_property
    def n_energy(self):
        """
        Number of energy bins.

        Shape (ntime,)
        """
        return self.varget('nb_energy').value.astype(int)

    @cached_property
    def start_elevation_idx(self):
        """
        Index of the first elevation bin.

        Shape (ntime,)
        """
        return self.varget('start_elevation').value.astype(int)

    @cached_property
    def n_elevation(self):
        """
        Number of energy bins.

        Shape (ntime,)
        """
        return self.varget('nb_elevation').value.astype(int)

    @cached_property
    def start_azimuth_idx(self):
        """
        Index of the first azimuth bin.

        Shape (ntime,)
        """
        return self.varget('start_CEM').value.astype(int)

    @cached_property
    def n_azimuth(self):
        """
        Number of energy bins.

        Shape (ntime,)
        """
        return self.varget('nb_CEM').value.astype(int)
