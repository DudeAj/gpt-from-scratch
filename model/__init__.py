# model/__init__.py
#
# Exposes the public API of the model package.
# Consumers can now write:
#
#   from model import GPT
#
# instead of reaching into the internal submodule:
#
#   from model.gpt import GPT  (still works, but less clean)
#
# __all__ explicitly declares what `from model import *` exports,
# and serves as documentation for what this package is meant to provide.

from model.gpt import GPT

__all__ = ["GPT"]
