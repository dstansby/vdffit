from functools import cached_property

import astropy.constants as const
import astropy.units as u
import numpy as np

from .vdf import VDFBase

__all__ = ['PASDistribution']


class PASDistribution(VDFBase):
    """
    A single distribution measured by PAS.
    """

    def __init__(self, vdf, energy, theta, phi, start_idx, shape, time, bvec):
        self.start_idx = start_idx
        end_idx = [start + s for start, s in zip(start_idx, shape)]
        slc = tuple(slice(s, e) for s, e in zip(start_idx, end_idx))

        self._vdf = vdf[slc]
        self._theta = theta[slc[:2]]
        self._phi = phi[slc[:2]]

        self._time = time
        self._bvec = bvec
        # Assume proton mass
        self.mass = const.m_p

        self._modv = np.sqrt(2 * energy / self.mass).to(u.km / u.s)

    def vdf(self):
        return self._vdf

    @property
    def time(self):
        return self._time

    @cached_property
    def peak_vdf(self):
        """
        Return index and value of peak VDF.
        """
        idx = np.nanargmax(self.vdf)
        return idx, self.vdf[idx]

    # Quality checks
    @cached_property
    def peak_idx(self):
        idx, _ = self.peak_vdf
        return np.unravel_index(idx, self.shape)

    '''
    def max_vdf_on_edge(self):
        """
        Return True if the peak of the VDF is at the edge of the angular bins.
        """
        return (self.peak_idx[0] in [0, 7]) or (self.peak_idx[2] in [0, 7])

    def has_angular_resolution(self):
        """
        Return True if all of the adjacent angular bins to the peak vdf has
        data.
        """
        peak_idx = self.peak_idx
        vdf = self.vdf.reshape(self.shape)
        for i, j in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            n_positive = np.sum(vdf[peak_idx[0]+i, :, peak_idx[2]+j] > 0)
            if n_positive == 0:
                return False

        peak_vels = vdf[peak_idx[0], peak_idx[1]-1:peak_idx[1]+2, peak_idx[2]]
        if not np.all(peak_vels > 0):
            return False

        return True
    '''

    def quality_flag_info(self):
        return {}

    def quality_flag(self):
        return 1

    @property
    def bvec(self):
        return self._bvec

    def velocities_instr_frame(self):
        """
        Velocities in the instrument frame.

        The last index is the velocity component.
        """
        modv = self._modv
        vx = modv * np.cos(self._theta) * np.cos(self._phi)
        vy = modv * np.cos(self._theta) * np.sin(self._phi)
        vz = modv * np.sin(self._theta)
        v = np.stack([vx, vy, vz], axis=-1)
        return v

    @cached_property
    def velocities(self):
        """
        Velocity in the spacecraft frame.

        This is different from the instrument frame by a flip in the x and y
        directions.
        """
        vinstr = self.velocities_instr_frame()
        phi = 20 * u.deg
        vx = -np.cos(phi) * vinstr[:, 1] - np.sin(phi) * vinstr[:, 2]
        vy = np.sin(phi) * vinstr[:, 1] - np.cos(phi) * vinstr[:, 2]
        vz = vinstr[:, 0]
        # vx = -vinstr[:, 1]
        # vy = -vinstr[:, 2]
        vz = vinstr[:, 0]
        return np.stack([vx, vy, vz], axis=-1)

    '''
    @property
    def mask(self):
        _, peak_val = self.peak_vdf
        # Only select values within 1% of peak VDF value
        return (self.vdf > 0.01 * peak_val).astype(bool)
    '''
