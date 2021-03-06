from functools import cached_property

import astropy.constants as const

from vdffit import data_dir
from vdffit.io.cdf import VDFCDF
from vdffit.io.psp.mag import MAGL2
from vdffit.vdf import SPANDistribution

__all__ = ['SPANL2CDF']


class SPANL2CDF(VDFCDF):
    """
    A SWEAP SPAN level 2 file.

    Each file corresponds to a single day.

    Notes
    -----
    The energy and angle matrices are 2D. The first index is time, the second
    index is bin in parameter space. The second index is always length 2048,
    which can be reshaped into (8, 32, 8) to get the (phi, E, theta) bins.
    """
    def __init__(self, date, species='p'):
        """
        Parameters
        ----------
        date : astropy.time.Time
            Date of file.
        species : str, optional
            Species. Can be 'p' for protons or 'a' for alphas.
        """
        self.species = species
        self.date = date
        self.mag_cdf = MAGL2(date)
        # Calling this loads the CDF and checks that the file exists
        self.cdf

    @property
    def path(self):
        date_str = self.date.strftime('%Y%m%d')
        if self.species == 'p':
            fname = f'psp_swp_spi_sf00_l2_8dx32ex8a_{date_str}_v04.cdf'
            return data_dir / fname
        elif self.species == 'a':
            raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Get a single distribtuion function.

        Parameters
        ----------
        idx : int or Time

        Returns
        -------
        SPAN_distribution
        """
        if not isinstance(idx, int):
            time = idx
            if time not in self.times:
                raise ValueError(f'{time} not found in timestamps')
            idx = self.times.index(time)
        else:
            time = self.times[idx]

        epoch = self.epochs[idx]

        if self.species == 'p':
            mass = const.m_p
        elif self.species == 'a':
            mass = 4 * const.m_p
        return SPANDistribution(self.eflux[idx, :],
                                self.energy[idx, :],
                                self.theta[idx, :],
                                self.phi[idx, :],
                                mass,
                                time,
                                self.mag_cdf.get_bvec(epoch),
                                self.species)

    @cached_property
    def eflux(self):
        """
        Differential energy flux.
        """
        return self.varget('EFLUX')

    @cached_property
    def energy(self):
        """
        Energies.
        """
        return self.varget('ENERGY')

    @cached_property
    def theta(self):
        """
        Theta values.
        """
        return self.varget('THETA')

    @cached_property
    def phi(self):
        """
        Phi values.
        """
        return self.varget('PHI')
