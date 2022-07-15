from ml_swissknife import utils


class BackwardHookMode(metaclass=utils.ContainerMeta):
    ghost_norm = "ghost_norm"
    ghost_grad = "ghost_grad"
    default = "default"


class ClippingMode(metaclass=utils.ContainerMeta):
    default = "default"  # Global fixed.
    ghost = "ghost"  # Global fixed clipping with ghost clipping.
    per_layer = "per_layer"  # Per layer fixed clipping.
    per_layer_percentile = "per_layer_percentile"  # Clip gradient per-layer based on gradient norm percentile.


class AccountingMode(metaclass=utils.ContainerMeta):
    rdp = "rdp"
    glw = "glw"
    all_ = "all"
