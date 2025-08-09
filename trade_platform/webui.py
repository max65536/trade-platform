"""
Backward-compat entry module for WebUI.

The WebUI has been modularized under trade_platform.webui_app.*
This stub keeps existing imports working and delegates to the new server.
"""

from .webui_app.server import main  # re-export for console script

__all__ = ["main"]

