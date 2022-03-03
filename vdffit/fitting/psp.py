from .bimax import BiMaxFitter


class PSPProtonCoreFitter(BiMaxFitter):
    """
    """
    def post_fit_process(self, params):
        ts = super().post_fit_process(params)
