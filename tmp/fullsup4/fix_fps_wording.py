"""Fix the FPS wording in the 12 baseline records (best-of-3 rounds, not median-of-medians)."""
import io
import os

DOCS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'docs')
OLD = '（thop MACs、100 次×3 轮取中位）'
NEW = '（thop MACs；三轮 × 每轮 100 次计时，取最好一轮的中位数）'

for name in ('DNANet', 'MSHNet', 'UIU-Net', 'SCTransNet', 'RPCANet', 'HDNet', 'ILNet',
             'RPCANet++', 'L2SKNet', 'BGM', 'DRPCA-Net', 'DURD'):
    path = os.path.join(DOCS, f'{name}_Seal实验记录.md')
    text = io.open(path, encoding='utf-8').read()
    if OLD not in text:
        print(f'skip {name}: pattern absent')
        continue
    with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text.replace(OLD, NEW))
    print(f'ok {name}')
