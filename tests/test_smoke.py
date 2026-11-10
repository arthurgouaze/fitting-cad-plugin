"""Smoke tests: the package imports and the CLI parser is wired."""

import limbform
from limbform.cli import build_parser


def test_version():
    assert limbform.__version__


def test_parser_builds():
    parser = build_parser()
    args = parser.parse_args(["view", "scan.stl"])
    assert args.command == "view"
