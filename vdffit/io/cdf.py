import abc
from functools import cached_property

import astropy.units as u
import cdflib


class CDFFile(abc.ABC):
    """
    Interface to a single CDF file.
    """
    epoch_var = 'Epoch'

    @property
    @abc.abstractmethod
    def path(self):
        """
        Path to CDF file - implemented by sub-classes.
        """

    @cached_property
    def cdf(self):
        """
        cdflib.CDF object.
        """
        if not self.path.exists():
            raise FileNotFoundError(f'path {self.path} does not exist')
        return cdflib.CDF(self.path)

    @property
    def info(self):
        """
        CDF information.
        """
        return self.cdf.cdf_info()

    @property
    def zvars(self):
        """
        z-var information.
        """
        return self.info['zVariables']

    def varget(self, var_str):
        """
        Get an inidividual variable.

        Parameters
        ----------
        var_str : str
            Variable name.

        Returns
        -------
        astropy.units.Quantity
        """
        var = self.cdf.varget(var_str)
        if var_str == 'PHI':
            # PHI is missing any attributes in the alpha data
            return var * u.Unit('deg')
        units = self.cdf.varattsget(var_str)['UNITS']
        if units == 'eV/cm2-s-ster-eV':
            # Can actually ignore steradians apparently...
            units = 'eV/(cm2 s eV)'
        elif units == 'Degrees':
            units = 'deg'
        return var * u.Unit(units)

    @cached_property
    def times(self):
        """
        Times.

        Returns
        -------
        list[datetime.datetime]
        """
        epochs = self.cdf.varget(self.epoch_var)
        times = cdflib.cdfepoch.to_datetime(epochs)
        return times

    def __len__(self):
        return len(self.times)


class VDFCDF(CDFFile):
    """
    A CDF file containing velocity distribution functions.
    """
    @abc.abstractmethod
    def __getitem__(self, i):
        """
        Get the ith distribution funciton. Must return a VDFBase instance.
        """
