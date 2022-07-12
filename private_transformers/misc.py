"""Miscellaneous helpers."""
import warnings


def handle_unused_kwargs(unused_kwargs, msg=None):
    if len(unused_kwargs) > 0:
        if msg is not None:
            warnings.warn(f"{msg}: Unexpected arguments {unused_kwargs}")
        else:
            warnings.warn(f"Unexpected arguments {unused_kwargs}")
