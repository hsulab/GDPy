"""Deprecated potential-manager namespace.

Use :mod:`gdpx.providers` for new integrations.  ``REGISTER`` remains for one
minor release so existing configuration and imports continue to work.
"""

from gdpx.providers.compat_registry import REGISTER

__all__ = ["REGISTER"]
