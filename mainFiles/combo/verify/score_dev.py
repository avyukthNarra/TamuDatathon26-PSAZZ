#!/usr/bin/env python3
"""Score only the 72 dev episodes with the existing harness and its bootstrap CI."""
import argparse
from common import score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    score(parser.parse_args().csv, 'dev')
