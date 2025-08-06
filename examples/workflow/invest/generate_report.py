#!/usr/bin/env python3
"""
Simple wrapper script to generate HTML stock analysis reports from comprehensive reports.
Usage: python generate_report.py <output_path> <comprehensive_md_file> <technical_chart> <price_volume_chart>
"""

import sys
import os
from pathlib import Path
from html_report_generator import HTMLGenerator

def main():
    if len(sys.argv) != 5:
        print("Usage: python generate_report.py <output_path> <comprehensive_md_file> <technical_chart> <price_volume_chart>")
        print("Example: python generate_report.py reports/300750_report.html 300750/reports/comprehensive_report_300750_20250725.md charts/technical.png charts/price_volume.png")
        sys.exit(1)
    
    output_path = sys.argv[1]
    comprehensive_md_file = sys.argv[2]
    technical_chart = sys.argv[3]
    price_volume_chart = sys.argv[4]
    
    # Validate input files
    if not os.path.exists(comprehensive_md_file):
        print(f"Error: Comprehensive markdown file not found: {comprehensive_md_file}")
        sys.exit(1)
    
    if not os.path.exists(technical_chart):
        print(f"Warning: Technical chart not found: {technical_chart}")
        technical_chart = ""
    
    if not os.path.exists(price_volume_chart):
        print(f"Warning: Price/volume chart not found: {price_volume_chart}")
        price_volume_chart = ""
    
    try:
        generator = HTMLGenerator(output_path)
        output_file = generator.generate_report(comprehensive_md_file, technical_chart, price_volume_chart)
        print(f"✅ HTML report generated successfully: {output_file}")
        print(f"📁 Assets folder: {Path(output_file).parent / 'assets'}")
        print(f"🌐 Open the HTML file in your browser to view the comprehensive report")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 