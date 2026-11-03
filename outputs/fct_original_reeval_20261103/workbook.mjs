import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';
const base = 'F:/Code_Repositories_2/CursorCode/Seal_Model_Reproduction';
const out = `${base}/outputs/fct_original_reeval_20261103`;
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(`${base}/docs/Seal量化SOTA表.xlsx`));
if (process.argv.includes('--edit')) {
  try { await fs.access(`${out}/workbook_before_edit.xlsx`); }
  catch { await fs.copyFile(`${base}/docs/Seal量化SOTA表.xlsx`, `${out}/workbook_before_edit.xlsx`); }
  const before = wb.worksheets.items.map(s=>({name:s.name,values:s.getUsedRange().values,formulas:s.getUsedRange().formulas}));
  const cross = JSON.parse(await fs.readFile(`${base}/docs/FCT-SAM-WSCOD_原函数评估_20261103/cross_domain_metrics_full.json`,'utf8')).Seal;
  const adapt = JSON.parse(await fs.readFile(`${base}/docs/FCT-SAM-WSCOD_原函数评估_20261103/seal_adaptation_metrics_full.json`,'utf8')).Seal;
  const sheet = wb.worksheets.getItem('Quantity');
  for (const [row,metrics] of [[13,cross],[24,adapt]]) {
    if (sheet.getRange(`A${row}`).values[0][0] !== 'FCT-SAM-WSCOD') throw new Error('FCT row changed');
    sheet.getRange(`E${row}:H${row}`).values = [[metrics.S,metrics.F,metrics.E,metrics.MAE]];
    sheet.getRange(`E${row}:H${row}`).setNumberFormat('0.00000');
    sheet.getRange(`K${row}`).values = [['原函数复算']];
  }
  wb.recalculate();
  console.log((await wb.inspect({kind:'table',range:'Quantity!E13:K13',include:'values,formulas',tableMaxRows:1,tableMaxCols:7,maxChars:1600})).ndjson);
  console.log((await wb.inspect({kind:'table',range:'Quantity!E24:K24',include:'values,formulas',tableMaxRows:1,tableMaxCols:7,maxChars:1600})).ndjson);
  for (const [label,range] of [['main','A12:H14'],['adaptation','A23:H25'],['status','K12:K25']]) {
    const preview=await wb.render({sheetName:'Quantity',range,scale:1.5,format:'png'});
    await fs.writeFile(`${out}/after_${label}.png`,new Uint8Array(await preview.arrayBuffer()));
  }
  const output = await SpreadsheetFile.exportXlsx(wb);
  await output.save(`${base}/docs/Seal量化SOTA表.xlsx`);
  const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(`${base}/docs/Seal量化SOTA表.xlsx`));
  const allowed = new Set(['13:5','13:6','13:7','13:8','13:11','24:5','24:6','24:7','24:8','24:11']);
  const changes=[];
  for (const original of before) {
    const current=reopened.worksheets.getItem(original.name).getUsedRange();
    const values=current.values, formulas=current.formulas;
    for (let r=0;r<original.values.length;r++) for (let c=0;c<original.values[r].length;c++) {
      const key=`${r+1}:${c+1}`;
      if (JSON.stringify(original.values[r][c]) !== JSON.stringify(values[r][c])) {
        if (original.name!=='Quantity' || !allowed.has(key)) throw new Error(`Unexpected cell change ${original.name}:${key}`);
        changes.push(key);
      }
      if ((original.formulas[r]?.[c]??'') !== (formulas[r]?.[c]??'')) throw new Error(`Formula changed ${key}`);
    }
  }
  console.log(JSON.stringify({changedCells:changes,unrelatedValuesAndFormulasPreserved:true}));
  process.exit(0);
}
for (const [label,range] of [['main','A12:H14'],['adaptation','A23:H25']]) {
  const preview = await wb.render({sheetName:'Quantity',range,scale:1.5,format:'png'});
  await fs.writeFile(`${out}/before_${label}.png`,new Uint8Array(await preview.arrayBuffer()));
}
console.log('Rendered the existing FCT ranges before editing.');
