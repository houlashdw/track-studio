"""Пак для сборщика в Reaper из скачанных битов (free for profit, уже без тегов): demucs → стемы →
по каждому биту 1–2 восьмитактовые петли: мелодия (other+vocals), бас и барабаны аудио + разметка в сетке:
удары кика/снейра/хэта по 1/16 (хэт по 1/32) и ноты 808 (по басу). Итог в томе ts-ace: /pack/<out>/ + pack.json."""
import os, sys, json, glob, subprocess
import modal

image = (modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg', 'libsndfile1')
         .pip_install('torch==2.5.1', 'torchaudio==2.5.1', 'demucs==4.0.1', 'librosa==0.10.2.post1', 'soundfile', 'numpy<2'))
vol = modal.Volume.from_name('ts-ace', create_if_missing=True)
app = modal.App('track-studio-pack', image=image)
VOL = '/vol'
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]; MIN = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def analyze(s, sd, OUT, nloops=2):
    import numpy as np, librosa, soundfile as sf
    SR = 44100; ld = lambda n: sf.read(f'{sd}/{n}.wav', dtype='float32', always_2d=True)[0]
    dr, ba, ot, vo = ld('drums'), ld('bass'), ld('other'), ld('vocals'); mel = ot + vo
    mono = lambda x: x.mean(1); R = 22050; rs = lambda x: librosa.resample(mono(x), orig_sr=SR, target_sr=R)
    yd, yb, ym = rs(dr), rs(ba), rs(mel); y = yd + yb + ym
    g = s['genre']; lo, hi = (70, 105) if g == 'lofi' else (105, 180)
    # 1) сетка: бит сделан в DAW — темп постоянный. Ищем темп, чья сетка 1/16 лучше всего ловит удары (векторная сила фаз)
    oe = librosa.onset.onset_strength(y=yd, sr=R, hop_length=256); fo = librosa.onset.onset_detect(onset_envelope=oe, sr=R, hop_length=256)
    if len(fo) < 40: return []
    on = librosa.frames_to_time(fo, sr=R, hop_length=256); w = oe[fo]
    def vs(bs):
        st = 15 / bs[:, None]; z = (w * np.exp(2j * np.pi * on[None, :] / st)).sum(1) / w.sum(); return z
    bs = np.arange(lo, hi, .02); z = vs(bs); b0 = bs[np.abs(z).argmax()]
    bs = np.arange(b0 - .03, b0 + .03, .001); z = vs(bs); k = np.abs(z).argmax(); bpm = float(bs[k]); fit = float(np.abs(z[k]))
    st = 15 / bpm; T = 4 * st; t0 = (np.angle(z[k]) / (2 * np.pi) * st) % st
    # 2) огибающие ударов по полосам (стем барабанов)
    H = 128; S = np.abs(librosa.stft(yd, n_fft=1024, hop_length=H)) ** 2; fr = librosa.fft_frequencies(sr=R, n_fft=1024)
    def env(a, b):
        e = np.log1p(1e4 * S[(fr >= a) & (fr < b)].mean(0)); o = np.maximum(0, np.diff(e, prepend=e[0])); return o
    E = {'k': env(30, 120), 's': np.sqrt(env(200, 1000) * env(1500, 5000)), 'h': env(7000, 11000)}   # снейр/клэп: и тело, и треск одновременно
    tt = lambda t: int(round(t * R / H))
    EN = {nm: e / (np.percentile(e[e > 0], 99.5) + 1e-9) for nm, e in E.items()}
    PK = {nm: librosa.util.peak_pick(e, pre_max=4, post_max=4, pre_avg=12, post_avg=12, delta=.08, wait=3) for nm, e in EN.items()}
    LAT = {}   # задержка пиков огибающей относительно сетки (своя у каждой полосы) — медиана смещения от ближайшей 1/16
    for nm in PK:
        tp = PK[nm][EN[nm][PK[nm]] > .3] * H / R; r = (tp - t0) / st; dv = (r - np.round(r)) * st
        LAT[nm] = float(np.median(dv)) if len(dv) > 8 else 0.0
    def at(e, t, w): a, b = max(0, tt(t - w)), tt(t + w) + 1; return float(e[a:b].max()) if b > a and a < len(e) else 0.0
    # начало такта: одна из 16 шестнадцатых — кик на «раз», снейр на 3-й доле (халфтайм) или на 2 и 4
    nbar = int((len(y) / R - t0) / (4 * T)) - 1
    if nbar < 10: return []
    def bscore(q):
        a = lambda j, o: t0 + (q + 16 * j + o) * st
        return sum(at(E['k'], a(j, 0), .03) + max(at(E['s'], a(j, 8), .03), .5 * (at(E['s'], a(j, 4), .03) + at(E['s'], a(j, 12), .03))) for j in range(nbar))
    q = max(range(16), key=bscore); t0 = t0 + q * st; ph = 0
    nb = int((len(y) / R - t0) / T); bars = nb // 4
    if bars < 10: return []
    mrms = lambda a, b, x: float(np.sqrt((x[int(a * R):int(b * R)] ** 2).mean() + 1e-12))
    C = librosa.feature.chroma_cqt(y=ym, sr=R, hop_length=512); ct = lambda t: min(C.shape[1] - 1, int(t * R / 512))
    cand = []
    for j in range(0, bars - 8 + 1, 2):
        a = t0 + (ph + 4 * j) * T; b = a + 32 * T
        c1 = C[:, ct(a):ct(a + 16 * T)].mean(1); c2 = C[:, ct(a + 16 * T):ct(b)].mean(1); sim = float(c1 @ c2 / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-9))
        cand.append((np.sqrt(mrms(a, b, ym) * mrms(a, b, yd)) * (.5 + .5 * sim), j, a, b))
    cand.sort(reverse=True); pick = []
    for c in cand:
        if all(abs(c[1] - p[1]) >= 8 for p in pick): pick.append(c)
        if len(pick) >= nloops: break
    out = []; base = os.path.splitext(s['filename'])[0]
    for li, (sc, j, a, b) in enumerate(pick):
        # барабаны: 128 шагов по 1/16, хэт 256 по 1/32
        st = {}
        for nm, n in (('k', 128), ('s', 128), ('h', 256)):
            d = 32 * T / n; e = EN[nm]; best = {}
            for f_ in PK[nm]:   # пики огибающей → ближайший шаг сетки (если удар не дальше трети шага)
                t = f_ * H / R - a - LAT[nm]; i = int(round(t / d))
                if 0 <= i < n and abs(t - i * d) < .4 * d and e[f_] > (.3 if nm != 'h' else .2): best[i] = max(best.get(i, 0), float(e[f_]))
            if nm == 'h': best = {i: v for i, v in best.items() if i % 2 == 0 or v > .5}   # 1/32 вне сетки 1/16 — только явные роллы
            cap = {'k': 6 * 8, 's': 3 * 8}.get(nm)   # больше 6 киков / 3 снейров на такт — ловится лишнее
            if cap and len(best) > cap: th = sorted(best.values(), reverse=True)[cap - 1]; best = {i: v for i, v in best.items() if v >= th}
            st[nm] = [[i, round(min(1.0, max(.3, v)), 2)] for i, v in sorted(best.items())]
        ks = {i for i, _ in st['k']}; st['s'] = [x for x in st['s'] if x[0] not in ks]   # щелчок кика в полосе снейра
        # 808 по басу: высота по шагам 1/16 (pyin), новая нота — смена высоты или удар
        seg = yb[int(a * R):int(b * R)]; d16 = 2 * T / 8
        f0, vf, _ = librosa.pyin(seg, fmin=27.5, fmax=220, sr=R, frame_length=4096, hop_length=256)
        rm = librosa.feature.rms(y=seg, frame_length=2048, hop_length=256)[0]; ob = librosa.onset.onset_strength(y=seg, sr=R, hop_length=256); ob = ob / (np.percentile(ob, 98) + 1e-9)
        fps = R / 256; notes = []; cur = None; rmax = np.percentile(rm, 95) + 1e-9
        for i in range(128):
            fa, fb = int(i * d16 * fps), max(int(i * d16 * fps) + 1, int((i + 1) * d16 * fps))
            fv = f0[fa:fb][vf[fa:fb]] if fb <= len(f0) else []
            act = fb <= len(rm) and rm[fa:fb].mean() > .2 * rmax and len(fv) > 0
            if not act: cur = None; continue
            m = int(round(librosa.hz_to_midi(np.median(fv)))); hit = ob[fa:min(fb + 1, len(ob))].max() > .5 if fa < len(ob) else False
            if cur is None or m != cur[2] or hit: cur = [i, 1, m]; notes.append(cur)
            else: cur[1] += 1
        notes = [n for n in notes if n[1] >= 1]
        ch = C[:, ct(a):ct(b)].mean(1); kk = max([(np.corrcoef(np.roll(MAJ, q), ch)[0, 1], q, 'major') for q in range(12)] + [(np.corrcoef(np.roll(MIN, q), ch)[0, 1], q, 'minor') for q in range(12)])
        lid = f'{base}_L{li}'; fs = {}
        for nm, x in (('mel', mel), ('bass', ba), ('drums', dr)):
            f = f'{OUT}/{lid}_{nm}.ogg'; seg2 = x[int(a * SR):int(b * SR)]
            sf.write(f, seg2 if nm == 'mel' else seg2.mean(1, keepdims=True), SR, format='OGG', subtype='VORBIS'); fs[nm] = os.path.basename(f)
        out.append(dict(id=lid, genre=g, bpm=round(bpm, 3), fit=round(fit, 3), key=NOTES[kk[1]], mode=kk[2], score=round(float(sc), 4), files=fs, steps=st, b808=notes,
                        rms=dict(mel=mrms(a, b, ym), bass=mrms(a, b, yb), drums=mrms(a, b, yd))))
    return out


@app.function(gpu='T4', volumes={VOL: vol}, timeout=3 * 3600)
def proc(src: str, out: str, items: list):
    D = f'{VOL}/data/{src}'; OUT = f'{VOL}/pack/{out}'; os.makedirs(OUT, exist_ok=True); os.makedirs('/tmp/c', exist_ok=True)
    fl = []
    for s in items:   # 100 с из середины: петли там, а demucs в 2 раза быстрее
        c = f"/tmp/c/{s['filename'].replace('.mp3', '.wav')}"
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '12', '-t', '100', '-i', f"{D}/{s['filename']}", '-ar', '44100', c]); fl.append(c)
    subprocess.run([sys.executable, '-m', 'demucs', '-n', 'htdemucs', '-d', 'cuda', '-o', '/tmp/sep'] + fl, check=False)
    res = []
    for s in items:
        sd = f"/tmp/sep/htdemucs/{s['filename'].replace('.mp3', '')}"
        try: r = analyze(s, sd, OUT); res += r; print(s['filename'], [(x['id'], round(x['bpm']), x['key'] + x['mode'][:3], len(x['steps']['k']), len(x['b808'])) for x in r], flush=True)
        except Exception as e: print('⚠', s['filename'], repr(e)[:200], flush=True)
    vol.commit(); return json.loads(json.dumps(res, default=float))


@app.function(volumes={VOL: vol})
def save(out: str, res: list):
    json.dump(res, open(f'{VOL}/pack/{out}/pack.json', 'w')); vol.commit(); return len(res)


@app.function(volumes={VOL: vol})
def listing(src: str):
    return json.load(open(f'{VOL}/data/{src}/dataset.json'))['samples']


@app.local_entrypoint()
def main(src: str = 'ts-beatprep2', out: str = 'pack1', per: int = 24, limit: int = 0):
    ds = listing.remote(src)
    if limit: ds = ds[:limit]
    bs = [ds[i:i + per] for i in range(0, len(ds), per)]
    res = []
    for r in proc.starmap([(src, out, b) for b in bs]): res += r
    print('петель:', save.remote(out, res), {g: sum(1 for x in res if x['genre'] == g) for g in sorted(set(x['genre'] for x in res))})
