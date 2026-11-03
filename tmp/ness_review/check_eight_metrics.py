import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as E

base=Path('docs/NESS-Net_Seal评估产物')
small=json.loads((base/'small_target_metrics.json').read_text())
test=json.loads((base/'test_metrics.json').read_text())
selection=json.loads((base/'selection.json').read_text())
assert test['checkpoint']==selection['selected']['checkpoint']
assert selection['selected']['epoch']==5
expected={**test['test'],**small['metrics']}
expected['Fw']=expected.pop('F')
c=small['counts']
assert small['metrics']=={'IoU':c['tp']/(c['tp']+c['fp']+c['fn']),
 'F1':2*c['tp']/(2*c['tp']+c['fp']+c['fn']),
 'Pd':c['n_detected']/c['n_gt'],'Fa':c['false_alarm_pixels']/c['total_pixels']}
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with ZipFile('docs/Seal量化SOTA表.xlsx') as z:
 strings=[''.join(n.itertext()) for n in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
 sheet=E.fromstring(z.read('xl/worksheets/sheet1.xml'))
 rows={}
 for row in sheet.findall('.//s:row',ns):
  cells={}
  for cell in row:
   ref=cell.attrib['r'];col=''.join(x for x in ref if x.isalpha())
   value=cell.find('s:v',ns)
   if value is not None:
    kind=cell.attrib.get('t')
    cells[col]=strings[int(value.text)] if kind=='s' else value.text if kind=='str' else float(value.text)
   elif cell.attrib.get('t')=='inlineStr':cells[col]=''.join(cell.find('s:is',ns).itertext())
  rows[int(row.attrib['r'])]=cells
 headers={col:str(value).replace('↑','').replace('↓','').strip() for col,value in rows[2].items()}
 candidates=[row for row in rows.values() if str(row.get('A','')).startswith('NESS-Net')]
 assert len(candidates)==1
 row=candidates[0]
 for key,value in expected.items():
  col=next(col for col,label in headers.items() if label==key)
  assert row[col]==value,(key,row[col],value)
 assert not any(values for num,values in rows.items() if num>20)
 print('NESS-Net table: all eight metrics exactly match JSON; one row; no data after row 20.')
 print('Prediction:',small['prediction_dir'])
 print('Checkpoint:',test['checkpoint'])
