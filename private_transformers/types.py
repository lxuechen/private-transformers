import enum


class BackwardHookMode(enum.Enum):
    ghost_norm = "ghost_norm"
    ghost_grad = "ghost_grad"
    default = "default"

    @staticmethod
    def all():
        return tuple(map(lambda mode: mode.value, BackwardHookMode))
