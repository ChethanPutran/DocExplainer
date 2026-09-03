#!/usr/bin/env python3
"""
Doc Explainer Application
Main entry point for the GUI application
"""

import sys
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Doc Explainer - Intelligent Document Explanation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                    Examples:
                    %(prog)s document.pdf                    # Open PDF document
                    %(prog)s --config custom.json document.pdf  # Use custom config
                    %(prog)s --debug --no-splash document.html  # Debug mode without splash
                    %(prog)s --theme dark document.txt           # Open with dark theme
                    %(prog)s --log-level DEBUG document.pdf      # Set log level to DEBUG
                    %(prog)s --profile document.pdf              # Enable performance profiling
                    %(prog)s --reset-config                     # Reset configuration to defaults
                    %(prog)s --clear-cache                     # Clear application cache
                """
    )

    parser.add_argument(
        'document',
        nargs='?',
        help='Document to open (PDF, TXT, HTML, etc.)'
    )

    parser.add_argument(
        '--config',
        '-c',
        help='Configuration file path'
    )

    parser.add_argument(
        '--debug',
        '-d',
        action='store_true',
        help='Enable debug mode'
    )

    parser.add_argument(
        '--no-splash',
        action='store_true',
        help='Disable splash screen'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )

    parser.add_argument(
        '--version',
        '-v',
        action='version',
        version='Doc Explainer 1.0.0'
    )

    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )

    parser.add_argument(
        '--reset-config',
        action='store_true',
        help='Reset configuration to defaults'
    )

    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear application cache'
    )

    return parser



def main():
    """Main entry point"""
    
    parser = create_parser()
    args = parser.parse_args()

    try:

        from .ui.gui.app import DocExplainerApp

        # Create and run application
        app = DocExplainerApp(
            config=args.config,
            log_level=args.log_level,
            reset_config=args.reset_config,
            clear_cache=args.clear_cache
        )

        # Handle profiling
        if args.profile:
            import cProfile
            import pstats
            from io import StringIO

            profiler = cProfile.Profile()
            profiler.enable()

            exit_code = app.run(args.document)

            profiler.disable()

            # Save profile stats
            profiler.dump_stats(Path.home() / '.doc_explainer' / 'profile.stats')

            # Print top 20 functions
            s = StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            stats.print_stats(20)
            print(s.getvalue())

        else:
            exit_code = app.run(args.document)

        # Cleanup
        app.cleanup()
    except Exception as e:
        logger.exception("An error occurred while running the application.")
        print(f"Error: {e}")
        exit_code = 1
        
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
