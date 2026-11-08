#!/usr/bin/env python3
"""Compare a validation forecast to the pinned TM_COMBINED3 forecast by query_id."""
import argparse
from common import REFERENCE, compare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--reference', default=str(REFERENCE))
    parser.add_argument('--tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    compare(args.csv, args.reference, args.tolerance)
