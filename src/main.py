# Main entry point for the dataset generation script.
import argparse
import os
import sys

# Ensure the script can find modules in the 'src' directory
# This is important if running main.py directly from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.generator import run_generation
except ImportError:
    # Fallback if running from within src directory (less common)
    from generator import run_generation

def main():
    """
    Parses command-line arguments and starts the generation process.
    """
    parser = argparse.ArgumentParser(
        description="Generate an image dataset using Automatic1111 API based on a config file."
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file.",
        metavar="CONFIG_FILE" # More descriptive name in help message
    )

    args = parser.parse_args()

    # Basic check if the config file exists before passing it on
    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file not found at '{args.config_file}'")
        sys.exit(1) # Exit with a non-zero code to indicate failure

    print(f"Using configuration file: {args.config_file}")

    try:
        run_generation(args.config_file)
        print("\nScript finished.")
    except Exception as e:
        # Catch any unexpected errors bubbling up to the top level
        print(f"\nAn critical error occurred: {e}")
        # Consider adding more detailed logging or traceback here if needed
        # import traceback
        # traceback.print_exc()
        sys.exit(1) # Indicate failure

if __name__ == "__main__":
    main()
