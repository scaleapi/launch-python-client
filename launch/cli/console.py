from typing import Any, Optional

import rich
from rich.console import Console as RichConsole

rich_console = RichConsole(highlight=False)


def spinner(message: str):
    """
    Shows a spinner until the with scope exits.
    """
    return rich_console.status(f"[bold green]{message}")


def pretty_print(message: Any, style: Optional[str] = None, markup: Optional[bool] = None) -> None:
    """
    Pretty prints to the console.
    """
    if style or markup is not None:
        rich_console.print(message, style=style, markup=markup)
    else:
        rich.print(message)
