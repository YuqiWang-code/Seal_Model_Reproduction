import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const base='F:/Code_Repositories_2/CursorCode/Seal_Model_Reproduction';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(`${base}/docs/Seal量化SOTA表.xlsx`));
if(process.argv.includes('--edit')){
 const report=JSON.parse(await fs.readFile(`${base}/docs/FCT-SAM-WSCOD_诊断_20261103/eight_metric_results.json`,'utf8'));
 const metrics=report.experiments.find(e=>e.name==='cross_domain').metrics;
 const sheet=wb.worksheets.getItemAt(0);
 const before=sheet.getUsedRange().values;
 const normalize=x=>String(x??'').replace(/[↑↓\s]/g,'');
 const headerRow=before.findIndex(r=>r.some(x=>normalize(x)==='IoU')&&r.some(x=>normalize(x)==='Fw'));
 const headers=before[headerRow].map(normalize);
 const methodCol=before[0].findIndex(x=>normalize(x)==='Method');
 const rows=before.map((r,i)=>({row:i,value:r[methodCol]})).filter(x=>x.row>headerRow&&x.row<20&&/FCT.?SAM/i.test(String(x.value)));
 if(rows.length!==1)throw new Error('Expected one FCT row in first 20 rows');
 const row=rows[0].row;
 await fs.copyFile(`${base}/docs/Seal量化SOTA表.xlsx`,`${base}/tmp/fct_diagnosis/workbook_before_edit.xlsx`);
 const allowed=new Set();
 for(const key of ['S','Fw','E','MAE','IoU','F1','Pd','Fa']){
  const col=headers.indexOf(key);if(col<0)throw new Error(`Missing header ${key}`);
  sheet.getCell(row,col).values=[[metrics[key]]];allowed.add(`${row}:${col}`);
 }
 const statusCol=headers.indexOf('重训/测试');if(statusCol<0)throw new Error('Missing status header');
 sheet.getCell(row,statusCol).values=[['公开权重跨域测试']];allowed.add(`${row}:${statusCol}`);
 sheet.getCell(row,methodCol).values=[['FCT-SAM-WSCOD']];allowed.add(`${row}:${methodCol}`);
 if(before.slice(20).some(r=>r.some(x=>x!==null&&x!=='')))throw new Error('Unexpected populated rows after row 20');
 wb.recalculate();
 const preview=await wb.render({sheetName:sheet.name,range:`A${row}:L${row+2}`,scale:1.2,format:'png'});
 await fs.writeFile(`${base}/tmp/fct_diagnosis/workbook_after.png`,new Uint8Array(await preview.arrayBuffer()));
 const status=await wb.render({sheetName:sheet.name,range:`O${row}:O${row+2}`,scale:1.2,format:'png'});
 await fs.writeFile(`${base}/tmp/fct_diagnosis/workbook_status.png`,new Uint8Array(await status.arrayBuffer()));
 const output=await SpreadsheetFile.exportXlsx(wb);await output.save(`${base}/docs/Seal量化SOTA表.xlsx`);
 const saved=await SpreadsheetFile.importXlsx(await FileBlob.load(`${base}/docs/Seal量化SOTA表.xlsx`));
 const after=saved.worksheets.getItemAt(0).getUsedRange().values;
 const changes=[];
 for(let r=0;r<Math.max(before.length,after.length);r++)for(let c=0;c<before[0].length;c++){
  const a=before[r]?.[c]??null,b=after[r]?.[c]??null;
  if(r>=20&&b!==null&&b!=='')throw new Error(`Nonempty value after row20: ${r+1}`);
  if(JSON.stringify(a)!==JSON.stringify(b)){
   if(!allowed.has(`${r}:${c}`))throw new Error(`Unrelated value changed ${r+1}:${c+1}`);
   changes.push([r+1,c+1]);
  }
 }
 console.log(JSON.stringify({method:after[row][methodCol],locatedRow:row+1,locatedMetricColumns:headers,
   changedCells:changes,first20Only:true,otherValuesPreserved:true}));
 process.exit(0);
}
if(process.argv.includes('--help-delete')){
 console.log(wb.help('range.delete',{include:'index,examples,notes',maxChars:2200}).ndjson);
 process.exit(0);
}
for(const s of wb.worksheets.items){
 const v=s.getUsedRange().values;
 console.log(JSON.stringify({sheet:s.name,rows:v.length,nonemptyAfter20:v.map((r,i)=>({row:i+1,values:r})).filter(x=>x.row>20&&x.values.some(y=>y!==null&&y!=='')),header:v.slice(0,2),fct:v.map((r,i)=>({row:i+1,values:r})).filter(x=>String(x.values[0]).includes('FCT'))}));
 const p=await wb.render({sheetName:s.name,range:'A12:L14',scale:1,format:'png'});
 await fs.writeFile(`${base}/tmp/fct_diagnosis/workbook_before.png`,new Uint8Array(await p.arrayBuffer()));
}
