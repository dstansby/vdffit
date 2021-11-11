import numpy as np
import scipy.optimize as opt

from .base import FitterBase


class BiMaxFitter(FitterBase):
    @property
    def fit_param_names(self):
        return ['A', 'vx', 'vy', 'vz', 'vth_perp', 'vth_par']

    def status_info(self):
        return {2: "Less than 12 points available for fit.",
                3: "Velocity at peak VDF is non-finite."
                4: "Fitted velocity is out of the VDF bounds."}

    @staticmethod
    def bi_maxwellian_3D(vx, vy, vz, A, vbx, vby, vbz, vth_z, vth_perp):
        '''
        Return distribution function at (vx, vy, vz),
        given 6 distribution parameters.
        '''
        # Put in bulk frame
        vx = vx - vbx
        vy = vy - vby
        vz = vz - vbz
        exponent = (vx / vth_perp)**2 + (vy / vth_perp)**2 + (vz / vth_z)**2
        return A * np.exp(-exponent)

    def run_single_fit(self, vs, vdf, bvec):
        """
        Fit a bi-Maxwellian distribution function.

        Parameters
        ----------
        vs : numpy.ndarray
        vdf : numpy.ndarray
        bvec : Vector

        Returns
        -------
        fit_params : fitting.BiMaxwellParams
        """
        if len(vdf) < 12:
            return 2, {}

        # Rotate velocities into field aligned frame
        R = bvec.rotation_matrix
        vs = np.einsum('ij,kj->ki', R, vs)

        # Residuals to minimize
        def resid(maxwell_params, vs, vdf):
            fit = self.bi_maxwellian_3D(vs[:, 0], vs[:, 1],
                                        vs[:, 2], *maxwell_params)
            return vdf - fit

        guesses = self.initial_guesses(vs, vdf)
        if np.any(np.isnan([guesses[1], guesses[2], guesses[3]])):
            return 3, {}

        # Do fitting
        fitout = opt.least_squares(resid, guesses,
                                   args=(vs, vdf), method='lm',
                                   ftol=1e-6, xtol=1e-14)

        fitparams = fitout.x

        v_bulk = fitparams[1:4]
        out_of_bounds = [(v_bulk[i] < np.min(vs[:, i]) or
                          v_bulk[i] > np.max(vs[:, i]))
                         for i in range(3)]
        out_of_bounds = np.any(out_of_bounds)
        if out_of_bounds:
            return 4, {}

        # Transform bulk velocity out of field aligned frame
        fitparams[1:4] = np.einsum('ij,j->i', R.T, v_bulk)
        return 1, {k: v for k, v in zip(self.fit_param_names, fitparams)}

    def initial_guesses(self, vs, vdf):
        """
        Initial gueses for a bimaxwellian fit.

        Parameters
        ----------
        vs : numpy.ndarray
        vdf : numpy.ndarray
        """
        peak_idx = np.nanargmax(vdf)
        A0 = vdf[peak_idx]
        v0 = vs[peak_idx, :]
        return [A0, v0[0], v0[1], v0[2], 40, 40]
