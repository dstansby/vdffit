import scipy.optimize as opt

from .base import FitterBase


class BiMaxFitter(FitterBase):
    @property
    def fit_param_names(self):
        return ['A', 'vx', 'vy', 'vz', 'vth_perp', 'vth_par']

    def status_info(self):
        return {2: "Less than 12 points available for fit.",
                4: "Fitted velocity is out of the VDF bounds."}

    def run_single_fit(self, vs, vdf):
        """
        Fit a bi-Maxwellian distribution function.

        Returns
        -------
        fit_params : fitting.BiMaxwellParams
        """
        # Rotate velocities into field aligned frame
        v_b_frame = self._v_b_frame

        if len(vdf) < 12:
            return 2, {}

        # Residuals to minimize
        def resid(maxwell_params, v_b_frame, vdf):
            fit = fitting.bi_maxwellian_3D(v_b_frame[:, 0], v_b_frame[:, 1],
                                           v_b_frame[:, 2], *maxwell_params)
            return vdf - fit

        guesses = self._initial_guesses
        if np.any(np.isnan(guesses.fit_guess_tuple[1:4])):
            return 5, {}
        fitout = opt.least_squares(resid, guesses.fit_guess_tuple,
                                   args=(v_b_frame, vdf), method='lm',
                                   ftol=1e-6, xtol=1e-14)

        fitparams = fitout.x

        v = fitparams[1:4]
        out_of_bounds = [(v[i] < np.min(v_b_frame[:, i]) or
                          v[i] > np.max(v_b_frame[:, i]))
                         for i in range(3)]
        out_of_bounds = np.any(out_of_bounds)
        if out_of_bounds:
            return 4, {}

        # Put v back into the spacecraft frame
        v = np.dot(self.bvec.rotation_matrix.T, v)
        fit_params = fitting.BiMaxwellParams(fitparams[0] * vdfunit,
                                             v * vunit,
                                             fitparams[4] * vunit,
                                             fitparams[5] * vunit,
                                             self.bvec.vec * Bunit)

        return fit_params, 1
