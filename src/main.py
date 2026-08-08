#!/usr/bin/env python3
"""
Doc Explainer Application
Main entry point for the GUI application
"""

import sys
import argparse
import logging
from pathlib import Path
import shutil 

from config import UIConfig, BackendConfig, LLMConfig

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
        '--uiconfig',
        '-c',
        help='UI configuration file path'
    )

    parser.add_argument(
        '--llmconfig',
        '-l',
        help='LLM configuration file path'
    )

    parser.add_argument(
        '--backendconfig',
        '-b',
        help='Backend configuration file path'
    )

    parser.add_argument(
        '--theme',
        '-t',
        choices=['light', 'dark', 'high_contrast', 'sepia'],
        help='Theme to use'
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
    def handle_reset_config():
        """Reset configuration to defaults"""
        config_file = Path.home() / '.doc_explainer' / 'config' / 'ui_config.json'
        if config_file.exists():
            config_file.unlink()
            logger.info("Configuration reset to defaults")
        else:
            logger.info("No configuration file found")


    def handle_clear_cache():
        """Clear application cache"""
        cache_dir = Path.home() / '.doc_explainer' / 'cache'
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)
            logger.info("Cache cleared")
        else:
            logger.info("No cache directory found")


    # Create necessary directories if they don't exist
    base_dir = Path.home() / '.doc_explainer'

    config_dir = base_dir / 'config' 
    log_dir = base_dir / 'logs' 
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)


    resource_dir = base_dir / "resources" 

    config_file = config_dir / 'ui_config.json'
    log_file = log_dir / 'app.log'


    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Copy the asset if it doesn't exist
    if not resource_dir.exists():
        # Source could be a global 'assets' folder in your project root
        resource_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = Path(__file__).parents[3] / "assets" 
        
        if source_path.exists():
            shutil.copytree(source_path, resource_dir, dirs_exist_ok=True)
            logger.info(f"Copied assets to {resource_dir}")
        else:
            logger.warning("Source assets not found in root assets folder!")
    else:
        logger.info(f"Assets already exist at {resource_dir}")
    
    parser = create_parser()
    args = parser.parse_args()

    # Handle special commands
    if args.reset_config:
        handle_reset_config()
        return 0

    if args.clear_cache:
        handle_clear_cache()
        return 0

    # Set logging level    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Override theme from command line
    if args.theme:
        # This would override the config theme
        pass


    from ui.gui.app import DocExplainerApp
    
    # Create and run application
    app = DocExplainerApp(
        config_paths={
            'ui': args.uiconfig,
            'llm': args.llm_config,
            'backend': args.backend_config
        },
        debug=args.debug
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

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
