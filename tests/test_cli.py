# tests/test_cli.py

from typer.testing import CliRunner

from oadr_cpep.cli import app

runner = CliRunner()

EXPECTED_COMMANDS = [
    "select-features",
    "fit-models",
    "apply-coefficients",
    "consensus-features",
    "aggregate-vectors",
]


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_all_commands_registered():
    result = runner.invoke(app, ["--help"])
    for command in EXPECTED_COMMANDS:
        assert command in result.output


def test_subcommand_help():
    for command in EXPECTED_COMMANDS:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
