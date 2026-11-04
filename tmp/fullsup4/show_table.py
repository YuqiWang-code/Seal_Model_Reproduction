import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'papers')

name = sys.argv[1]
anchor = sys.argv[2]
before = int(sys.argv[3]) if len(sys.argv) > 3 else 5
after = int(sys.argv[4]) if len(sys.argv) > 4 else 60

text = open(os.path.join(OUT, name + '.txt'), encoding='utf-8').read()
lines = text.splitlines()
for i, ln in enumerate(lines):
    if anchor in ln:
        print(f'--- match at line {i} ---')
        lo, hi = max(0, i - before), min(len(lines), i + after)
        for j in range(lo, hi):
            print(f'{j}: {lines[j]}')
        print()
