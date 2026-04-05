"""Allow running as: python -m active_memory

Defaults to the proxy — the primary way to use active-memory.
Use `python -m active_memory chat` for the interactive CLI.
"""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "chat":
    sys.argv.pop(1)  # remove subcommand so argparse sees the rest
    from active_memory.cli import main
else:
    from active_memory.proxy import main

main()
