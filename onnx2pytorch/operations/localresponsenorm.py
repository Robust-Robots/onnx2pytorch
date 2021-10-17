import warnings

from torch.nn.modules.normalization import LocalResponseNorm


class LocalResponseNormUnsafe(LocalResponseNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_input_dim(self, input):
        return
