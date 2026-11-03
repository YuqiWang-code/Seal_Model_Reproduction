import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const file='docs/Seal量化SOTA表.xlsx';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
const sh=wb.worksheets.getItemAt(0);
const rows=sh.getRange('A1:O20').values;
const index=rows.findIndex(row=>String(row[0]).startsWith('NESS-Net'));
if(index<0)throw Error('Missing NESS-Net row');
console.log(JSON.stringify({sheet:sh.name,row:index+1,headers:rows[1],values:rows[index]}));
await fs.writeFile('tmp/ness_cross_domain/before.png',new Uint8Array(await(await wb.render({sheetName:sh.name,range:'A14:O20',scale:1.5})).arrayBuffer()));
if(process.argv.includes('--edit')){
 await fs.copyFile(file,'tmp/ness_cross_domain/original.xlsx');
 const record=JSON.parse(await fs.readFile('docs/NESS-Net_Seal评估产物/cross_domain_metrics.json','utf8'));
 const edits={};
 function write(col,value){sh.getCell(index,col).values=[[value]];edits[String.fromCharCode(65+col)+(index+1)]=value;}
 write(0,'NESS-Net');write(2,'RGB-D');write(3,'Scribble');
 for(const [key,value]of Object.entries(record.metrics)){
  const col=rows[1].findIndex(v=>String(v).replace(/[↑↓\s]/g,'')===key);
  if(col<0)throw Error('Missing metric '+key);
  write(col,value);
 }
 const protocol=rows[1].findIndex(v=>String(v)==='重训/测试');
 write(protocol,'公开refinement+NAMLab\n源域Scribble→Seal\nT替代深度');
 sh.getCell(index,protocol).format.wrapText=true;
 wb.recalculate();
 await fs.writeFile('tmp/ness_cross_domain/edits.json',JSON.stringify(edits));
 await fs.writeFile('tmp/ness_cross_domain/after.png',new Uint8Array(await(await wb.render({sheetName:sh.name,range:'A14:O20',scale:1.5})).arrayBuffer()));
 await(await SpreadsheetFile.exportXlsx(wb)).save('tmp/ness_cross_domain/authored.xlsx');
}
