import abc

import astropy.units as u


class FitterBase(abc.ABC):
    vunit = u.km / u.s
    vdfunit = u.s**3 / u.m**6

    def fit(self, dist):
        """
        Derived classes should **not** override this, but instead should
        implement ``run_single_fit()``.

        Returns
        -------
        status : int
            Fitting status. 1 for a succesful fit.
        params : dict
            Fit parameters.
        """
        # Strip units
        velocities = dist.velocities.to_value(self.vunit)
        vdf = dist.vdf.to_value(self.vdfunit)
        # Apply mask
        print(dist.mask)
        velocities = velocities[dist.mask, :]
        vdf = vdf[dist.mask]
        # Rotate velocities into field aligned frame
        R = dist.bvec.rotation_matrix
        velocities = np.einsum('ij,kj->ki', R, velocities)
        # Pass to fitting method
        status, params = self.run_single_fit(velocities, vdf)

    @abc.abstractproperty
    def fit_param_names(self):
        """
        Return a list of parameter names fitted by the fitter.
        """

    @abc.abstractmethod
    def run_single_fit(self, velocities, vdf):
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
