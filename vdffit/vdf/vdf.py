import abc

__all__ = ['VDFBase']


class VDFBase(abc.ABC):
    """
    A single velocity distribution function.
    """
    @abc.abstractmethod
    def quality_flag(self):
        """
        Return an integer quality flag, indicating if this distribution is
        suitable for fitting.

        If it is suitible for fitting, must return ``1``.
        Otherwise must return a number >``1``,
        """

    @abc.abstractmethod
    def quality_flag_info(self):
        """
        Return a `dict` which maps integer quality flags to a short description
        of each quality flag.
        """

    @abc.abstractproperty
    def time(self):
        return self._time

    @abc.abstractproperty
    def bvec(self):
        """
        Magnetic field vector associated with this distribution function.
        """

    @abc.abstractmethod
    def velocities(self):
        """
        Velocities at the VDF samples.

        These must be in the same frame as the magnetic field vector.
        """

    @abc.abstractmethod
    def vdf(self):
        """
        Velocity distribution function values.

        Must be in units equivalent to seconds**3 / meters**6.
        """

    @property
    def mask(self):
        """
        An optional mask to indicate bad data values. By default all values
        are marked as good.
        """
        return np.ones(self.vdf.shape)
