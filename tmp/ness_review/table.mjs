import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const path='F:/Code_Repositories_2/CursorCode/Seal_Model_Reproduction/docs/Seal量化SOTA表.xlsx';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path));
console.log((await wb.inspect({kind:'sheet',include:'id,name'})).ndjson);
const sh=wb.worksheets.getItemAt(0);
console.log(JSON.stringify(sh.getRange('A1:K24').values));
await fs.writeFile('tmp/ness_review/table_before.png',new Uint8Array(await (await wb.render({sheetName:sh.name,range:'A1:K20',scale:1.5})).arrayBuffer()));
if(process.argv.includes('--edit')){
 const m=JSON.parse(await fs.readFile('tmp/ness_review/published_metrics.json','utf8')).refinement_namlab;
 const rows=sh.getRange('A1:K20').values;
 const index=rows.findIndex(row=>row.some(v=>String(v).includes('NESS')));
 if(index<0)throw Error('NESS row absent');
 console.log('NESS_ROW',index+1,rows[index]);
 // The existing table metric columns are identified from its header.
 const header=rows.find(row=>row.some(v=>String(v).includes('MAE')));
 for(const [label,key] of [['S','S'],['Fw','F'],['E','E'],['MAE','MAE']]){
  const col=header.findIndex(v=>String(v).replace(/[↑↓\s]/g,'')===label);
  if(col<0)throw Error('Metric header missing '+label);
  sh.getCell(index,col).values=[[m[key]]];
 }
 sh.getCell(index,10).values=[['refinement+NAMLab\n热图替代深度\n原函数公开权重测试']];
 sh.getCell(index,10).format.wrapText=true;
 sh.getCell(index,10).format.columnWidth=25;
 sh.getCell(index,10).format.rowHeight=48;
 wb.recalculate();
 console.log((await wb.inspect({kind:'table',range:`${sh.name}!A${index+1}:K${index+1}`,include:'values,formulas',tableMaxRows:1,tableMaxCols:11})).ndjson);
 await fs.writeFile('tmp/ness_review/table_after.png',new Uint8Array(await (await wb.render({sheetName:sh.name,range:'A1:K20',scale:1.5})).arrayBuffer()));
 await (await SpreadsheetFile.exportXlsx(wb)).save(path);
}
