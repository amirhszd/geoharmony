"""Supported command-line entry point for the SPLASH browser workflow."""

import argparse
import json
import sys
from pathlib import Path

from geoharmony.splash.config import SplashConfig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SPLASH: register orthorectified VNIR and SWIR imagery using an MSI reference."
    )
    parser.add_argument(
        "--version", action="version", version="GeoHarmony SPLASH 0.1.0"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON settings; relative paths are resolved from this file",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate settings, raster metadata and acquisition-row bands without processing",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run a configuration using matching saved manual homographies",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print browser URLs without opening a browser automatically",
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Local browser port (default 8050)"
    )
    commands = parser.add_subparsers(dest="command")
    prepare = commands.add_parser(
        "prepare-rows",
        help="Append acquisition-row IDs to a native waterfall BEFORE orthorectification",
    )
    prepare.add_argument("input", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser(
        "demo",
        help="Generate synthetic inputs and exercise all three stages without manual clicking",
    )
    demo.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        if args.command == "prepare-rows":
            from geoharmony.splash.create_index_raster_from_waterfall import (
                prepare_rows,
            )

            print(prepare_rows(args.input, args.output))
            return 0
        if args.command == "demo":
            from geoharmony.splash.demo import run_demo

            run_demo(args.output_dir)
            return 0
        if args.validate_only and not args.config:
            raise ValueError("--validate-only requires --config")
        if args.no_gui and not args.config:
            raise ValueError(
                "--no-gui requires --config and saved manual transformations"
            )
        config = SplashConfig.load(args.config) if args.config else SplashConfig()
        if args.validate_only:
            print(json.dumps(config.validate_inputs(), indent=2))
            print(
                "Configuration and input validation passed. No registration was performed."
            )
            return 0
        if not args.no_gui:
            from geoharmony.rigid_coregisteration.launchers import get_splash_inputs

            config = SplashConfig.from_dict(
                get_splash_inputs(args.port, config.to_dict(), not args.no_browser)
            )
        from geoharmony.splash.pipeline import run_splash

        outputs = run_splash(
            config, port=args.port, open_browser=not args.no_browser, no_gui=args.no_gui
        )
        print("SPLASH finished:\n" + json.dumps(outputs, indent=2))
        return 0
    except KeyboardInterrupt:
        print("SPLASH cancelled.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"SPLASH: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
