// записать модель после singSurgery приложения: node lab/surg.mjs in.onnx out.onnx
import fs from 'node:fs';
const html = fs.readFileSync('app/index.html', 'utf8'), a = html.indexOf('function singSurgery(u8) {'), b = html.indexOf("info: dX + ' / ' + fX", a);
const singSurgery = new Function(html.slice(a, html.indexOf('\n}', b) + 2) + '\nreturn singSurgery;')();
const r = singSurgery(new Uint8Array(fs.readFileSync(process.argv[2])));
fs.writeFileSync(process.argv[3], Buffer.concat(r.parts.map((p) => Buffer.from(p.buffer, p.byteOffset, p.byteLength))));
console.log(r.info, r.f0Elem);
