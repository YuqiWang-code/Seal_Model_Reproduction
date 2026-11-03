import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const [inputPath, previewPath] = process.argv.slice(2);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const rows = await workbook.inspect({
  kind: 'table',
  range: 'Quantity!A12:K26',
  include: 'values,formulas',
  tableMaxRows: 15,
  tableMaxCols: 11,
  maxChars: 3000,
});
console.log(rows.ndjson);
const preview = await workbook.render({ sheetName: 'Quantity', range: 'A12:K26', scale: 1, format: 'png' });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
