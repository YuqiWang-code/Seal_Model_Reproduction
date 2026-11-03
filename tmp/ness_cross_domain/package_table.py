import json,re,sys
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as E
from xml.sax.saxutils import escape

ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def values(z):
    strings=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        strings=[''.join(n.itertext()) for n in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
    result={}
    for c in E.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:c',ns):
        v=c.find('s:v',ns);kind=c.attrib.get('t')
        if v is not None:
            result[c.attrib['r']]=strings[int(v.text)] if kind=='s' else v.text if kind=='str' else float(v.text)
        elif kind=='inlineStr':result[c.attrib['r']]=''.join(c.find('s:is',ns).itertext())
    return result

b=Path('tmp/ness_cross_domain')
edits=json.loads((b/'edits.json').read_text())
with ZipFile(b/'authored.xlsx') as z:
    authored=values(z)
    assert all(authored[k]==v for k,v in edits.items())
with ZipFile('docs/Seal量化SOTA表.xlsx') as z:
    before=values(z)
    parts=[(info,z.read(info.filename)) for info in z.infolist()]
name='xl/worksheets/sheet1.xml'
index=next(i for i,(info,data) in enumerate(parts) if info.filename==name)
source=parts[index][1].decode()
row_pattern=r'<(?P<prefix>\w+:)?row\b[^>]*\br="17"[^>]*>.*?</(?:\w+:)?row>'
match=re.search(row_pattern,source,re.S)
old_row=match[0];row=old_row;prefix=match['prefix'] or ''
for ref in edits:
    pattern=rf'<(?:\w+:)?c\b[^>]*\br="{ref}"[^>]*(?:/>|>.*?</(?:\w+:)?c>)'
    found=re.search(pattern,row,re.S)
    assert found,ref
    style=re.search(r'\bs="([^"]+)"',found[0]);style=f' s="{style[1]}"' if style else ''
    value=authored[ref]
    if isinstance(value,str):
        cell=f'<{prefix}c r="{ref}"{style} t="inlineStr"><{prefix}is><{prefix}t xml:space="preserve">{escape(value)}</{prefix}t></{prefix}is></{prefix}c>'
    else:cell=f'<{prefix}c r="{ref}"{style}><{prefix}v>{repr(value)}</{prefix}v></{prefix}c>'
    row=row[:found.start()]+cell+row[found.end():]
updated=source[:match.start()]+row+source[match.end():]
parts[index]=(parts[index][0],updated.encode())
target=Path('docs/Seal量化SOTA表.xlsx') if '--commit' in sys.argv else b/'final.xlsx'
with ZipFile(target,'w') as z:
    for info,data in parts:z.writestr(info,data)
with ZipFile(target) as z:
    after=values(z)
    assert {k:v for k,v in before.items() if k not in edits}=={k:v for k,v in after.items() if k not in edits}
    assert all(after[k]==v for k,v in edits.items())
    assert not any(int(re.search(r'\d+',k)[0])>20 and v not in ('',None) for k,v in after.items())
    for info,data in parts:
        if info.filename!=name:assert z.read(info.filename)==data
print('Only NESS-Net row changed; eight values exact; all other cells and package parts preserved; no result after row 20.')
