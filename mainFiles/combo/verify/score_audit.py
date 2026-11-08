#!/usr/bin/env python3
"""Score audit ONCE, as combo-sai. A failed attempt also consumes the marker."""
import argparse
from common import score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    score(parser.parse_args().csv, 'audit')
