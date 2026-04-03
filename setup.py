#!/usr/bin/env python3
"""
Setup script for Doc Explainer
"""

from setuptools import setup, find_packages
import os

# Read long description from README if available
long_description = "Doc Explainer - Intelligent Document Explanation System"
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Read version from file or use default
__version__ = "1.0.0"
if os.path.exists("src/version.py"):
    with open("src/version.py", "r", encoding="utf-8") as f:
        exec(f.read())

setup(
    name="doc-explainer",
    version=__version__,
    author="Doc Explainer Team",
    author_email="team@docexplainer.com",
    description="Intelligent Document Explanation System with AI-powered features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/doc-explainer",
    project_urls={
        "Bug Tracker": "https://github.com/ChethanPutran/DocExplainer/issues",
        "Documentation": "https://github.com/ChethanPutran/DocExplainer/tree/main/docs",
        "Source Code": "https://github.com/ChethanPutran/DocExplainer",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
    ],
    python_requires=">=3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    
    # Core dependencies
    install_requires=[
        # GUI Framework
        'PySide6>=6.5.0',
        
        # LLM and AI Framework
        'langchain>=0.1.0',
        'langchain-community>=0.0.10',
        'langchain-core>=0.1.0',
        
        # LLM Providers
        'langchain-google-genai>=0.0.5',
        'langchain-openai>=0.0.2',
        'google-generativeai>=0.3.0',
        'openai>=1.0.0',
        
        # Document Processing
        'PyMuPDF>=1.23.0',  # fitz
        'pdfplumber>=0.10.0',
        'python-pptx>=0.6.21',
        'python-docx>=0.8.11',
        'markdown>=3.5.0',
        'beautifulsoup4>=4.12.0',
        'lxml>=4.9.0',
        
        # NLP and Text Processing
        'spacy>=3.7.0',
        'nltk>=3.8.0',
        'transformers>=4.35.0',
        'sentence-transformers>=2.2.0',
        
        # Vector Databases and Search
        'chromadb>=0.4.0',
        'faiss-cpu>=1.7.0',
        'scikit-learn>=1.3.0',
        
        # Knowledge Graph
        'networkx>=3.1',
        'graphviz>=0.20.1',
        'plotly>=5.17.0',
        
        # Voice and Audio
        'speechrecognition>=3.10.0',
        'pyttsx3>=2.90',
        'pyaudio>=0.2.11',
        'sounddevice>=0.4.6',
        'whisper>=1.1.10',
        
        # Data Processing
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'pydantic>=2.0.0',
        'pydantic-settings>=2.0.0',
        
        # Utilities
        'python-dotenv>=1.0.0',
        'pyyaml>=6.0',
        'tqdm>=4.66.0',
        'requests>=2.31.0',
        'aiohttp>=3.9.0',
        'asyncio>=3.4.3',
        
        # Caching
        'redis>=5.0.0',
        'diskcache>=5.6.0',
        
        # Monitoring and Logging
        'loguru>=0.7.0',
        'psutil>=5.9.0',
        
        # Export and Serialization
        'jsonlines>=4.0.0',
        'orjson>=3.9.0',
        
        # Image Processing
        'pillow>=10.0.0',
        'opencv-python>=4.8.0',
        
        # Web Scraping (for online resources)
        'selenium>=4.15.0',
        'webdriver-manager>=4.0.0',
    ],
    
    # Optional dependencies for different features
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-qt>=4.2.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
            'pre-commit>=3.5.0',
            'tox>=4.11.0',
            'sphinx>=7.2.0',
            'sphinx-rtd-theme>=1.3.0',
            'twine>=4.0.0',
            'wheel>=0.41.0',
        ],
        'gpu': [
            'faiss-gpu>=1.7.0',
            'cudatoolkit>=11.8',
        ],
        'mlx': [
            'mlx>=0.0.5',  # Apple Silicon optimization
        ],
        'ocr': [
            'pytesseract>=0.3.10',
            'easyocr>=1.7.0',
        ],
        'viz': [
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
        ],
        'full': [
            'faiss-gpu>=1.7.0',
            'pytesseract>=0.3.10',
            'easyocr>=1.7.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
        ],
    },
    
    # Entry points for console scripts
    entry_points={
        'console_scripts': [
            'doc-explainer=src.ui.gui.app:main',
            'doc-explainer-cli=src.cli.main:main',
        ],
        'gui_scripts': [
            'doc-explainer-gui=src.ui.gui.app:main',
        ],
    },
    
    # Package data files
    package_data={
        'doc-explainer': [
            'resources/**/*',
            'config/*.yaml',
            'config/*.json',
        ],
    },
    
    # Data files outside the package
    data_files=[
        ('share/applications', ['doc-explainer.desktop']),
        ('share/icons/hicolor/256x256/apps', ['icons/doc-explainer.png']),
        ('share/doc/doc-explainer', ['README.md', 'LICENSE', 'CHANGELOG.md']),
    ],
    
    # License
    license="MIT",
    keywords="document, explanation, ai, llm, knowledge-graph, education",
    platforms=["any"],
    zip_safe=False,
)