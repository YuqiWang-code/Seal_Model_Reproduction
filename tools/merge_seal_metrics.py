#!/usr/bin/env python3
"""Merge the four-metric and small-target JSONs of one method into a single record."""
import json
import sys
from pathlib import Path


def main():
    name, root = sys.argv[1], Path(sys.argv[2])
    m4 = json.loads((root / 'fullsup_metrics.json').read_text(encoding='utf-8'))['Seal']
    small = json.loads((root / 'fullsup_small_metrics.json').read_text(encoding='utf-8'))
    record = {
        'method': name,
        'protocol': 'Seal train full-GT supervised retraining -> Seal test',
        'eight_metrics': {
            'S': m4['S'], 'Fw': m4['F'], 'E': m4['E'], 'MAE': m4['MAE'],
            'IoU': small['metrics']['IoU'], 'F1': small['metrics']['F1'],
            'Pd': small['metrics']['Pd'], 'Fa': small['metrics']['Fa'],
        },
        'prediction_dir': small['prediction_dir'],
        'gt_dir': small['gt_dir'],
        'four_metric_source': str((root / 'fullsup_metrics.json').resolve()),
        'small_metric_source': str((root / 'fullsup_small_metrics.json').resolve()),
        'small_metric_counts': small['counts'],
        'small_metric_protocol': small['protocol'],
    }
    out = root / 'eight_metrics.json'
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record['eight_metrics'], indent=2))


if __name__ == '__main__':
    main()
