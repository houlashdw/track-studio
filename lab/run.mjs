// Лаборатория голоса: настоящая модель Kokoro (HF) + onnxruntime-web (wasm, как на телефоне) в Node.
// Ищем строки, на которых модель отдаёт NaN, сравниваем варианты (fp16/fp32, тембры, правки графа).
// node lab/run.mjs <out.json>   (модели в lab/models/, варианты — lab/variants.json)
import fs from 'node:fs';
import * as ort from 'onnxruntime-web';

const html = fs.readFileSync('app/index.html', 'utf8');
const grab = (start, endMark) => { const a = html.indexOf(start); if (a < 0) throw new Error('нет ' + start); const b = html.indexOf(endMark, a); return html.slice(a, b + endMark.length); };
// singSurgery из приложения (или из lab/surgery.js, если есть — для проверки правок графа)
let surgSrc = fs.existsSync('lab/surgery.js') ? fs.readFileSync('lab/surgery.js', 'utf8') : (() => { const a = html.indexOf('function singSurgery(u8) {'), b = html.indexOf("info: dX + ' / ' + fX", a); return html.slice(a, html.indexOf('\n}', b) + 2); })();
const singSurgery = new Function(surgSrc + '\nreturn singSurgery;')();
const KVOCAB = new Function('return ' + grab('{";": 1,', '}'))();
const VP = new Function(grab('const VOICE_PACK = {', '};') + '\nreturn VOICE_PACK;')();
const U = (() => { const b = Buffer.from(VP.b64, 'base64'); return new Uint16Array(b.buffer, b.byteOffset, b.length / 2); })();
const h2f = (h) => { const sg = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? sg * m * 2 ** -24 : e === 31 ? sg * Infinity : sg * (1 + m / 1024) * 2 ** (e - 15); };
const vpack = (id) => { const k = VP.ids.indexOf(id); if (k < 0) throw new Error('голоса нет в паке: ' + id); const n = VP.rows * 256, o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = h2f(U[k * n + i]); return o; };
const blend = (mix) => { let o = null, ws = 0; for (const [id, w] of mix) { const v = vpack(id); if (!o) o = new Float32Array(v.length); for (let i = 0; i < v.length; i++) o[i] += v[i] * w; ws += w; } for (let i = 0; i < o.length; i++) o[i] /= ws; return o; };
const style = (tim, pro, n) => { const t = blend(tim), p = blend(pro), rows = Math.floor(Math.min(t.length, p.length) / 256), off = Math.min(n, rows - 1) * 256, st = new Float32Array(256);
  for (let i = 0; i < 128; i++) st[i] = t[off + i]; for (let i = 128; i < 256; i++) st[i] = p[off + i]; return st; };
const fb = new Float32Array(1), ib = new Uint32Array(fb.buffer);
const f2h = (arr) => { const o = new Uint16Array(arr.length); for (let i = 0; i < arr.length; i++) { fb[0] = arr[i]; const x = ib[0], sg = (x >>> 16) & 0x8000; let e = ((x >>> 23) & 0xff) - 112, m = x & 0x7fffff; if (e <= 0) o[i] = sg; else if (e >= 31) o[i] = sg | 0x7c00; else o[i] = sg | (e << 10) | ((m + 0x1000) >> 13); } return o; };

ort.env.wasm.numThreads = 1;
const cases = JSON.parse(fs.readFileSync(process.env.LAB_CASES || 'lab/cases.json', 'utf8'));
const variants = JSON.parse(fs.readFileSync('lab/variants.json', 'utf8'));
const out = { started: new Date().toISOString(), ort: ort.env.versions, results: [] };
const save = () => fs.writeFileSync(process.argv[2] || 'lab-out/results.json', JSON.stringify(out, null, 1));
const sessions = new Map();
async function session(model) {
  if (sessions.has(model)) return sessions.get(model);
  for (const k of [...sessions.keys()]) { try { await sessions.get(k).s.release(); } catch (e) { } sessions.delete(k); }
  const u8 = new Uint8Array(fs.readFileSync('lab/models/' + model)), r = singSurgery(u8);
  const bytes = Buffer.concat(r.parts.map((p) => Buffer.from(p.buffer, p.byteOffset, p.byteLength)));
  const t0 = Date.now(), s = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  const rec = { s, elem: r.f0Elem, info: r.info, nMix: !!r.nMix, loadMs: Date.now() - t0 }; sessions.set(model, rec); return rec;
}
const wavs = fs.existsSync('lab-out') ? 'lab-out' : null;
for (const v of variants) {
  const S = await session(v.model);
  for (const c of cases) for (let rep = 0; rep < (v.rep || 1); rep++) {
    if (v.only && !v.only.includes(c.mode)) continue;
    const ids = [0]; for (const ch of c.ipa) { const q = KVOCAB[ch]; if (q != null) ids.push(q); } ids.push(0); const T = ids.length;
    const i64 = (a, d) => new ort.Tensor('int64', BigInt64Array.from(a, (x) => BigInt(Math.round(x))), d);
    const fT = (arr, d) => (S.elem === 10 ? new ort.Tensor('float16', f2h(arr), d) : new ort.Tensor('float32', Float32Array.from(arr), d));
    const f0 = c.f0.map((x) => x * 2 ** ((v.up || 0) / 12));
    const feeds = { input_ids: i64(ids, [1, T]), style: new ort.Tensor('float32', style(c.tim && !v.forceTim ? c.tim : v.tim, c.pro && !v.forceTim ? c.pro : v.pro, T - 2), [1, 256]), speed: new ort.Tensor('float32', [1], [1]),
      dur_in: i64(c.dur, [1, T]), dur_mix: i64([1], [1]), f0_in: fT(f0, [1, 1, f0.length]), f0_mix: fT([c.f0mix], [1]) };
    if (S.nMix) feeds.n_mix = fT([v.nmix || 0], [1]);
    const t0 = Date.now(); let res;
    try {
      const o = await S.s.run(feeds), w = o.waveform, d = w.data;
      let nf = 0, e = 0; const x = new Float32Array(d.length); for (let i = 0; i < d.length; i++) { const y = w.type === 'float16' ? h2f(d[i]) : d[i]; x[i] = y; if (!Number.isFinite(y)) nf++; else e += y * y; }
      res = { name: v.name, mode: c.mode, tx: c.tx, rep, n: d.length, nf, rms: +Math.sqrt(e / d.length).toFixed(5), ms: Date.now() - t0 };
      if (wavs && v.wav) { const pcm = Buffer.alloc(x.length * 2); for (let i = 0; i < x.length; i++) pcm.writeInt16LE(Math.max(-32767, Math.min(32767, Math.round((Number.isFinite(x[i]) ? x[i] : 0) * 32767))), i * 2);
        const hdr = Buffer.alloc(44); hdr.write('RIFF', 0); hdr.writeUInt32LE(36 + pcm.length, 4); hdr.write('WAVEfmt ', 8); hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(1, 22); hdr.writeUInt32LE(24000, 24); hdr.writeUInt32LE(48000, 28); hdr.writeUInt16LE(2, 32); hdr.writeUInt16LE(16, 34); hdr.write('data', 36); hdr.writeUInt32LE(pcm.length, 40);
        fs.mkdirSync('lab-out/wav', { recursive: true }); fs.writeFileSync(`lab-out/wav/${v.name}__${c.mode}__${c.tx.replace(/[^a-z]+/gi, '_').slice(0, 30)}.wav`, Buffer.concat([hdr, pcm])); }
    } catch (er) { res = { name: v.name, mode: c.mode, tx: c.tx, err: String(er && er.message || er).slice(0, 300) }; }
    out.results.push(res); console.log(JSON.stringify(res)); save();
  }
  out[v.name + '_load_ms'] = S.loadMs;
}
out.done = new Date().toISOString(); save();
