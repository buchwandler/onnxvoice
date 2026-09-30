from pathlib import Path

from onnxvoice.cli import build_parser

EXPECTED_DOCUMENTED_COMMANDS = {
    "list",
    "voices",
    "installed",
    "install",
    "updates",
    "update",
    "info",
    "doctor",
    "path",
    "show",
    "remove",
    "verify",
    "cache",
    "import",
    "catalog",
}


def test_cli_commands_match_the_documentation_coverage_set():
    parser = build_parser()
    command_action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(command_action.choices) == EXPECTED_DOCUMENTED_COMMANDS

    cli_docs = Path("docs/cli.md").read_text(encoding="utf-8")
    documented_commands = {
        line.split("`onnxvoice ", 1)[1].split("`", 1)[0]
        for line in cli_docs.splitlines()
        if line.startswith("| `onnxvoice ")
    }
    assert documented_commands == EXPECTED_DOCUMENTED_COMMANDS
