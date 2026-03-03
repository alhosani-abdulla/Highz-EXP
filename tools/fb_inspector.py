#!/usr/bin/env python3
"""
Interactive inspector for state spectra across cycles.
View all spectra from a specific state in each cycle, navigate between cycles with arrow keys.
"""

import argparse
import sys
from pathlib import Path

# Import functions from waterfall_plotter
from src.filterbank.visualization.spec_inspector import SpectrumInspector

def main():
    parser = argparse.ArgumentParser(
        description='Interactive inspector for state spectra across cycles',
        epilog='''
Navigation: Use arrow keys (← →) or buttons to move between cycles. Press Q to quit.

Reference spectrum format:
  --reference-spectrum cycle_010:1    (use spectrum 1 from cycle_010...)
  --reference-spectrum cycle_001:2    (use spectrum 2 from cycle_001...)
        ''')
    parser.add_argument('day_dir', type=Path,
                        help='Path to day directory (e.g., 20251102)')
    parser.add_argument('--state', type=str, required=True,
                        help='State number to inspect')
    parser.add_argument('--filter', type=int, default=10,
                        help='Filter number to plot (0-20, default: 10)')
    parser.add_argument('--reference-spectrum', type=str, metavar='CYCLE:IDX',
                        help='Global reference spectrum (e.g., cycle_010:1 or cycle_001_11022025_000200:2)')
    
    args = parser.parse_args()
    
    if not args.day_dir.exists():
        print(f"Error: Directory not found: {args.day_dir}")
        return 1
    
    if args.filter < 0 or args.filter > 20:
        print(f"Error: Filter number must be 0-20")
        return 1
    
    inspector = SpectrumInspector(args.day_dir, args.state, args.filter, args.reference_spectrum)
    inspector.show()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
