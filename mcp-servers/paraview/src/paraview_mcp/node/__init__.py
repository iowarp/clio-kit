"""The node half: the scene engine and trame view that run inside ParaView's Python.

This package is copied to the machine where ParaView runs and started there
with that machine's ``pvpython``. It imports ParaView only inside functions, so
the front can ship it without ParaView installed.
"""

NODE_PROTOCOL = 1
