import ast
import re
from pathlib import Path

PYTHON_BLOCK = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


def test_documentation_python_examples_are_syntactically_valid():
    for path in [Path("README.md"), *Path("docs").rglob("*.md")]:
        text = path.read_text(encoding="utf-8")
        for index, match in enumerate(PYTHON_BLOCK.finditer(text), start=1):
            ast.parse(match.group(1), filename=f"{path}:example-{index}")
