"""Smoke tests — confirm the package imports and the CLI parser is wired.

Real pipeline tests (Week 7) will run against synthetic meshes in examples/.
"""

import fitting_cad
from fitting_cad.cli import build_parser


def test_version():
    assert fitting_cad.__version__


def test_parser_builds():
    parser = build_parser()
    args = parser.parse_args(["view", "scan.stl"])
    assert args.command == "view"
