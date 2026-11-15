"""
An OpenGL context created via GLFW.
"""
from __future__ import annotations
import contextlib as contextlib
import glfw as glfw
__all__: list[str] = ['GLContext', 'contextlib', 'glfw']
class GLContext:
    """
    An OpenGL context created via GLFW.
    """
    @staticmethod
    def as_current(*args, **kwds):
        """
        Temporarily makes this context current, restoring any prior context.
        """
    def __del__(self):
        ...
    def __init__(self, max_width, max_height):
        ...
    def free(self):
        ...
    def make_current(self):
        ...
