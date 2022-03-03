import abc

import astropy.units as u
import numpy as np
from joblib import Parallel, delayed

__all__ = ['FitterBase']


class FitterBase(abc.ABC):
    vunit = u.km / u.s
    vdfunit = u.s**3 / u.m**6

    def fit_cdf(self, cdf):
        """
        Fit all velocity distribution functions in a CDF file.

        Parameters
        ----------
        cdf : vdffit.io.CDFFile
        """
        times = cdf.times
        params = Parallel(n_jobs=1, verbose=1)(
            delayed(self.fit_single)(cdf[t]) for t in times)
        params = self.post_fit_process(params)
        return params

    def fit_single(self, dist):
        """
        Fit a single velocity distribution function.

        Derived classes should **not** override this, but instead should
        implement ``run_single_fit()``.


        Parameters
        ----------
        dist : vdffit.vdf.VDFBase
            A single velocity distribution function.

        Returns
        -------
        params : dict
            Fit parameters.
        """
        # Strip units
        velocities = dist.velocities.to_value(self.vunit)
        vdf = dist.vdf.to_value(self.vdfunit)
        # Apply mask
        velocities = velocities[dist.mask, :]
        vdf = vdf[dist.mask]
        # Pass to fitting method
        status, params = self.run_single_fit(velocities, vdf, dist.bvec)
        if status != 1:
            params = [np.nan] * len(self.fit_param_names)
        params = {k: v for k, v in zip(self.fit_param_names, params)}
        params['fit status'] = status
        params['quality flag'] = dist.quality_flag()
        params['Time'] = dist.time
        return params

    @abc.abstractproperty
    def fit_param_names(self):
        """
        Return a list of parameter names fitted by the fitter.
        """

    @abc.abstractmethod
    def run_single_fit(self, velocities, vdf, bvec):
        """
        Fit a single distribution funciton.

        Parameters
        ----------
        velocities : numpy.ndarray
            Velocity array, shape (n, 3).
        vdf : numpy.ndarray
            VDF array, shape (n, )

        Returns
        -------
        status : int
            Fitting status code. 1 for a succseful fit.
        params : dict
            Fit parameters.
        """

    @abc.abstractmethod
    def status_info(self):
        """
        Return a `dict` containing information about the status codes that the
        fitting method can return.
        """

    @abc.abstractmethod
    def post_fit_process(self, params):
        """
        Parameters
        ----------
        params : dict[datetime: list]

        Returns
        -------
        astropy.table.QTable
        """
