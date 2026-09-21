"""Serve a composite as a page: levers over its handles, its wiring with a
signal trace, and a saved calibration run.

>>> from hallsim.view import page_for, serve
>>> serve(page_for(comp, registry=REGISTRY, t_end=14.0))

Needs the ``app`` extra.
"""

from hallsim.view._app import build_app, serve
from hallsim.view._model import ModelBank, ViewModel
from hallsim.view._page import Lever, Page, Panel, Shade, page_for, panel

__all__ = [
    "Lever",
    "ModelBank",
    "Page",
    "Panel",
    "Shade",
    "ViewModel",
    "build_app",
    "page_for",
    "panel",
    "serve",
]
