import {Workbook} from '@oai/artifact-tool';
const w=Workbook.create();w.worksheets.add('S');
console.log(w.help('worksheet.deleteRows',{include:'index,examples,notes',maxChars:2500}).ndjson);
