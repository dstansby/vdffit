import numpy as np

__all__ = ['Vector']


class Vector:
    """
    A single vector.

    Parameters
    ----------
    vec : numpy.ndarray
        Vector data, must be shape (3, ).
    """
    def __init__(self, vec):
        assert vec.shape == (3, ), "Vector must have shape (3, )"
        self.vec = vec

    def __str__(self):
        return self.vec.__str__()

    def __repr__(self):
        return f'MAGVector, {self.vec.__repr__()}'

    @property
    def rotation_matrix(self):
        """
        The 3x3 rotation matrix that maps this vector on to the z-axis.
        """
        norm = np.linalg.norm(self.vec)
        zaxis = np.array([0, 0, 1])

        # Calculate orthogonal axis
        orthvec = np.cross(zaxis, self.vec)
        phi = np.arccos(self.vec[2] / norm)

        R = self._rotationmatrixangle(orthvec, -phi)
        return R

    @staticmethod
    def _rotationmatrixangle(axis, theta):
        """
        Return the rotation matrix about a given axis.

        The rotation is taken to be counterclockwise about the given axis. Uses the
        Euler-Rodrigues formula.

        Parameters
        ----------
            axis : array_like
                Axis to rotate about.
            theta : float
                Angle through which to rotate in radians.

        Returns
        -------
            R : array_like
                Rotation matrix resulting from rotation about given axis.
        """
        assert axis.shape == (3, ), 'Axis must be a single 3 vector'
        assert np.dot(axis, axis) != 0, 'Axis has zero length'

        normaxis = axis / (np.sqrt(np.dot(axis, axis)))

        a = np.cos(theta / 2)
        b, c, d = -normaxis * np.sin(theta / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        out = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
        return out
