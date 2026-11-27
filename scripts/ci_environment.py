"""Derive a CI environment without duplicating or changing dependency pins."""

import argparse
import re
from pathlib import Path

SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12")
PYTHON_ENTRY = re.compile(
    r"(?m)^([ \t]*-[ \t]*python=)(\d+\.\d+(?:\.\d+)?)([ \t]*(?:#.*)?)$"
)


def environment_for_python(source, version):
    """Replace only the interpreter pin, retaining its patch for the same minor."""
    if version not in SUPPORTED_PYTHONS:
        raise ValueError(f"Unsupported CI Python version: {version}")
    matches = list(PYTHON_ENTRY.finditer(source))
    if len(matches) != 1:
        raise ValueError("Expected exactly one plain python=... dependency pin")
    entry = matches[0]
    current_pin = entry.group(2)
    current_minor = ".".join(current_pin.split(".")[:2])
    pin = current_pin if current_minor == version else version
    return source[: entry.start(2)] + pin + source[entry.end(2) :]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-version", required=True, choices=SUPPORTED_PYTHONS)
    parser.add_argument("--source", type=Path, default=Path("environment.yml"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.source.resolve() == args.output.resolve():
        parser.error("The CI environment must not overwrite the production recipe")
    try:
        source = args.source.read_text(encoding="utf-8")
        rendered = environment_for_python(source, args.python_version)
        args.output.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Created {args.output} for Python {args.python_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
