"""MAIT command-line entry point."""
import argparse
from . import __version__, export_demo


def main():
    parser = argparse.ArgumentParser(description="Prepare an original MAIT demo (analysis environment required to run it).")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("export-demo", help="Export script and data to a new directory")
    demo.add_argument("directory")
    args = parser.parse_args()
    try:
        script = export_demo(args.directory)
    except (OSError, ValueError, SyntaxError) as exc:
        parser.exit(1, f"Could not export demo: {exc}\n")
    print(f"Exported {script}\nAnalysis has not been run. The full MAIT analysis environment is required.")
