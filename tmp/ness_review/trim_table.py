from zipfile import ZipFile
from pathlib import Path
import re

p=Path('docs/Seal量化SOTA表.xlsx')
with ZipFile(p) as z:
    parts=[(info,z.read(info.filename)) for info in z.infolist()]
for index,(info,data) in enumerate(parts):
    if info.filename=='xl/worksheets/sheet1.xml':
        text=data.decode('utf-8')
        text=re.sub(r'<(?:x:)?row\b[^>]*\br="(\d+)"[^>]*>.*?</(?:x:)?row>',lambda m:'' if int(m[1])>20 else m[0],text,flags=re.S)
        text=re.sub(r'sqref="([^"]+)"',lambda m:'sqref="'+re.sub(r'([A-Z]+)(\d+)',lambda a:a[1]+str(min(int(a[2]),20)),m[1])+'"',text)
        parts[index]=(info,text.encode('utf-8'))
temp=p.with_suffix('.tmp.xlsx')
with ZipFile(temp,'w') as z:
    for info,data in parts:z.writestr(info,data)
temp.replace(p)
