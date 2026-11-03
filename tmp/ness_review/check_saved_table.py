from zipfile import ZipFile
import xml.etree.ElementTree as E
import json

ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with ZipFile('docs/Seal量化SOTA表.xlsx') as z:
    x=E.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows=x.findall('.//s:row',ns)
    assert max(int(r.attrib['r']) for r in rows)==20
    row=next(r for r in rows if r.attrib['r']=='17')
    cells={c.attrib['r']:c for c in row}
    metric=json.load(open('tmp/ness_review/published_metrics.json'))['refinement_namlab']
    for col,key in zip('EFGH',['S','F','E','MAE']):
        assert float(cells[col+'17'].find('s:v',ns).text)==metric[key]
print('Saved workbook: 20 rows; NESS primary metrics match original-function result.')
