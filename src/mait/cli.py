"""MAIT command-line entry point."""
import argparse
import os
import subprocess
import sys
from . import __version__, export_demo


def main():
    parser = argparse.ArgumentParser(description="Prepare an original MAIT demo (analysis environment required to run it).")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("export-demo", help="Export script and data to a new directory")
    demo.add_argument("directory")
    run = commands.add_parser("demo", help="Export and run the bundled CPU classification demo")
    run.add_argument("directory", help="New directory for example inputs and results")
    args = parser.parse_args()
    try:
        script = export_demo(args.directory)
    except (OSError, ValueError, SyntaxError) as exc:
        parser.exit(1, f"Could not export demo: {exc}\n")
    if args.command == "demo":
        environment = os.environ.copy()
        environment.setdefault("MPLBACKEND", "Agg")
        print(f"Running demo. Results will be saved beside {script}", flush=True)
        result = subprocess.run([sys.executable, "-u", str(script)], env=environment)
        if result.returncode:
            print("Demo failed. Install the analysis extra in this Python environment: "
                  'python -m pip install "mait-pipeline[analysis]"', file=sys.stderr)
        raise SystemExit(result.returncode)
    print(f"Exported {script}\nAnalysis has not been run. The full MAIT analysis environment is required.")
