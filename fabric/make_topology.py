#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a validated example ring; does not configure interfaces or contact hosts."""
import argparse
import json
from ring import generate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranks', type=int, choices=(4, 8), required=True)
    p.add_argument('--subnet-base', default='10.100.224.0/24')
    a = p.parse_args()
    print(json.dumps(generate(a.ranks, a.subnet_base), indent=2))
