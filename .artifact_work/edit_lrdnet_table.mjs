import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const [inputPath, outputPath, scodPath, smallPath] = process.argv.slice(2);
const scod = JSON.parse(await fs.readFile(scodPath, 'utf8')).Seal;
const small = JSON.parse(await fs.readFile(smallPath, 'utf8')).metrics;
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const sheet = workbook.worksheets.getItem('Quantity');
const rows = sheet.getRange('A1:AD20').values;
const row = rows.findIndex(r => String(r[0]).includes('LRDNet'));
if (row < 2) throw new Error('LRDNet row missing');
const headers = rows[1].map(v => String(v ?? '').replace(/[↑↓]/g, '').trim());
const values = {S:scod.S,Fw:scod.F,E:scod.E,MAE:scod.MAE,...small};
for (const [key,value] of Object.entries(values)) {
  const col=headers.indexOf(key);
  if (col<0) throw new Error(`Missing metric header ${key}`);
  sheet.getRangeByIndexes(row,col,1,1).values=[[value]];
}
sheet.getRangeByIndexes(row,0,1,1).values=[['LRDNet-S-COD']];
sheet.getRangeByIndexes(row,headers.indexOf('重训/测试'),1,1).values=[['源域重训后跨域测试']];

workbook.recalculate();
const result = await workbook.inspect({
  kind: 'table',
  range: `Quantity!A${row+1}:O${row+1}`,
  include: 'values,formulas',
  tableMaxRows: 14,
  tableMaxCols: 11,
  maxChars: 3500,
});
console.log(result.ndjson);
const preview=await workbook.render({sheetName:'Quantity',range:'A12:O15',scale:1,format:'png'});
await fs.writeFile('.artifact_work/cross_domain_after.png',new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
