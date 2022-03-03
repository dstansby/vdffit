from .bimax import BiMaxFitter

__all__ = ['PSPProtonCoreFitter']


class PSPProtonCoreFitter(BiMaxFitter):
    """
    """
    def post_fit_process(self, params):
        super().post_fit_process(params)
