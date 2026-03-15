#!/usr/bin/env python3
"""
Document Visualization Tool
Usage: python visualize_document.py <json_file> [output_html]
"""

import sys
import os
from src.core.document.html.generator import generate_document_html
import webbrowser

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_document.py <json_file> [output_html]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found")
        sys.exit(1)
    
    output_html = sys.argv[2] if len(sys.argv) > 2 else json_path.replace('.json', '_visualization.html')
    
    print(f"Generating HTML from {json_path}...")
    html_path = generate_document_html(json_path, output_html)
    
    print(f"HTML generated: {html_path}")
    
    # Ask to open in browser
    response = input("Open in browser? (y/n): ").lower()
    if response == 'y':
        webbrowser.open(f'file://{os.path.abspath(html_path)}')

if __name__ == "__main__":
    main()