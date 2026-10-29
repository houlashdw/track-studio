// прогон расширенной модели в onnxruntime-web: где впервые появляется NaN/Inf
import fs from 'node:fs';
import * as ort from 'onnxruntime-web';
const html = fs.readFileSync('app/index.html', 'utf8');
const grab = (start, endMark) => { const a = html.indexOf(start); const b = html.indexOf(endMark, a); return html.slice(a, b + endMark.length); };
const KVOCAB = new Function('return ' + grab('{";": 1,', '}'))();
const VP = new Function(grab('const VOICE_PACK = {', '};') + '\nreturn VOICE_PACK;')();
const U = (() => { const b = Buffer.from(VP.b64, 'base64'); return new Uint16Array(b.buffer, b.byteOffset, b.length / 2); })();
const h2f = (h) => { const sg = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? sg * m * 2 ** -24 : e === 31 ? (m ? NaN : sg * Infinity) : sg * (1 + m / 1024) * 2 ** (e - 15); };
const vpack = (id) => { const k = VP.ids.indexOf(id), n = VP.rows * 256, o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = h2f(U[k * n + i]); return o; };
const blend = (mix) => { let o = null, ws = 0; for (const [id, w] of mix) { const v = vpack(id); if (!o) o = new Float32Array(v.length); for (let i = 0; i < v.length; i++) o[i] += v[i] * w; ws += w; } for (let i = 0; i < o.length; i++) o[i] /= ws; return o; };
const CEN = ['af_heart', 'af_bella', 'af_sarah', 'af_nicole', 'if_sara', 'am_michael'];
const center = () => { const vs = CEN.map(vpack), c = new Float32Array(vs[0].length); for (const v of vs) for (let i = 0; i < c.length; i++) c[i] += v[i] / vs.length; return c; };
const fb = new Float32Array(1), ib = new Uint32Array(fb.buffer);
const f2h = (arr) => { const o = new Uint16Array(arr.length); for (let i = 0; i < arr.length; i++) { fb[0] = arr[i]; const x = ib[0], sg = (x >>> 16) & 0x8000; let e = ((x >>> 23) & 0xff) - 112, m = x & 0x7fffff; if (e <= 0) o[i] = sg; else if (e >= 31) o[i] = sg | 0x7c00; else o[i] = sg | (e << 10) | ((m + 0x1000) >> 13); } return o; };
ort.env.wasm.numThreads = 1;
const [model, orderF, caseF, idx, outF] = process.argv.slice(2);
const c = JSON.parse(fs.readFileSync(caseF, 'utf8'))[+idx], order = JSON.parse(fs.readFileSync(orderF, 'utf8'));
const s = await ort.InferenceSession.create(fs.readFileSync(model), { executionProviders: ['wasm'], graphOptimizationLevel: process.env.OPT || 'all' });
const ids = [0]; for (const ch of c.ipa) { const q = KVOCAB[ch]; if (q != null) ids.push(q); } ids.push(0); const T = ids.length;
const t = blend(c.tim), p = blend(c.pro), cc = center(), a = c.alpha || 1, off = Math.min(T - 2, VP.rows - 1) * 256, st = new Float32Array(256);
for (let i = 0; i < 128; i++) st[i] = t[off + i]; for (let i = 128; i < 256; i++) { const q = p[off + i]; st[i] = Math.abs(a - 1) > 0.01 ? cc[off + i] + a * (q - cc[off + i]) : q; }
const i64 = (x, d) => new ort.Tensor('int64', BigInt64Array.from(x, (v) => BigInt(Math.round(v))), d);
const feeds = { input_ids: i64(ids, [1, T]), style: new ort.Tensor('float32', st, [1, 256]), speed: new ort.Tensor('float32', [1], [1]), dur_in: i64(c.dur, [1, T]), dur_mix: i64([1], [1]),
  f0_in: new ort.Tensor('float16', f2h(c.f0), [1, 1, c.f0.length]), f0_mix: new ort.Tensor('float16', f2h([c.f0mix]), [1]) };
const o = await s.run(feeds), res = [];
const val = (tn) => { const d = tn.data; if (tn.type === 'float16') return Array.from(d, h2f); if (tn.type === 'float32' || tn.type === 'float64') return Array.from(d); return null; };
for (const [name, op, node] of order) { const tn = o[name]; if (!tn) continue; const v = val(tn); if (!v) continue; let nan = 0, inf = 0, mx = 0; for (const x of v) { if (Number.isNaN(x)) nan++; else if (!Number.isFinite(x)) inf++; else if (Math.abs(x) > mx) mx = Math.abs(x); }
  res.push({ name, op, node, n: v.length, nan, inf, max: +mx.toPrecision(4), dims: tn.dims }); }
const first = res.find((r) => r.nan || r.inf), big = res.filter((r) => r.max > 30000).slice(0, 20);
fs.writeFileSync(outF, JSON.stringify({ tx: c.tx, wav_nan: (() => { let k = 0; for (const x of val(o.waveform)) if (!Number.isFinite(x)) k++; return k; })(), first, firstIdx: res.indexOf(first), around: first ? res.slice(Math.max(0, res.indexOf(first) - 12), res.indexOf(first) + 4) : null, big, n: res.length }, null, 1));
console.log('first NaN/Inf:', JSON.stringify(first));
