"""Compatibility module; use ``geoharmony-splash`` or ``python -m geoharmony``."""

from geoharmony.splash.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
