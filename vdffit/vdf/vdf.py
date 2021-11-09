import abc


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

    @abc.abstractmethod
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
