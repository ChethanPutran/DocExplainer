# Run with default settings
python -m src.ui.gui.app

# Open a specific document
python -m src.ui.gui.app /path/to/your/document.pdf

# Run with a specific theme
python -m src.ui.gui.app --theme dark document.pdf

# Run with custom config file
python -m src.ui.gui.app --config /path/to/config.json document.pdf

# Enable debug mode
python -m src.ui.gui.app --debug document.pdf

# Full Command Line Options
python -m src.ui.gui.app --help