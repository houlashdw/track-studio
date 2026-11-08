"""v2: атаки ударных → класс по спектру (кик/снейр/хэт) → сетка по найденному темпу/долям → паттерны 16 шагов"""
import json, collections, numpy as np, librosa
D = '/home/claude/ins'; meta = json.load(open(D + '/meta.json')); SR = 22050
out = collections.defaultdict(list); rep = {}
for key, m in meta.items():
    y, _ = librosa.load(f'{D}/{key}_drums.mp3', sr=SR)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=256)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=256, start_bpm=m['bpm'], tightness=400)
    tempo = float(np.atleast_1d(tempo)[0])
    if tempo < m['bpm'] * 0.7: tempo *= 2
    if tempo > m['bpm'] * 1.4: tempo /= 2
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=256)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=256, backtrack=False, units='time', delta=0.07)
    hits = []
    for t in on:
        a = int(t * SR); w = y[a:a + int(.06 * SR)]
        if len(w) < 256: continue
        F = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2; f = np.fft.rfftfreq(len(w), 1 / SR)
        lo, md, hi = F[(f > 30) & (f < 150)].sum(), F[(f > 180) & (f < 2500)].sum(), F[(f > 6000)].sum(); tot = lo + md + hi + 1e-12
        amp = np.sqrt(np.mean(w ** 2))
        c = 'k' if lo / tot > .55 else ('h' if hi / tot > .45 and md / tot < .35 else 's')
        hits.append((t, c, amp))
    import scipy.signal as sg
    yh = sg.sosfilt(sg.butter(4, 6500, 'hp', fs=SR, output='sos'), y); eh = librosa.onset.onset_strength(y=yh, sr=SR, hop_length=256)
    for t in librosa.onset.onset_detect(onset_envelope=eh, sr=SR, hop_length=256, units='time', delta=0.04):
        a = int(t * SR); hits.append((t, 'h', float(np.sqrt(np.mean(yh[a:a + int(.03 * SR)] ** 2)))))
    hits = [x for x in hits if x[1] != 'h' or True]
    if len(bt) < 8 or not hits: continue
    step = 60 / tempo / 4; t0 = bt[0]
    # шаг каждой атаки по сетке от первой доли
    grid = collections.defaultdict(lambda: collections.defaultdict(float))
    for t, c, a in hits:
        s = int(round((t - t0) / step))
        if s < 0 or abs((t - t0) - s * step) > step * .35: continue
        grid[s][c] = max(grid[s][c], a)
    nst = max(grid) + 1 if grid else 0
    amps = {c: np.percentile([a for t, cc, a in hits if cc == c] or [1], 80) for c in 'ksh'}
    def row(c, a0):
        return ''.join(('x' if grid[a0 + i][c] > .7 * amps[c] else 'o' if grid[a0 + i][c] > .4 * amps[c] else 'g' if c != 'k' and grid[a0 + i][c] > .2 * amps[c] else '.') for i in range(16))
    rot = max(range(16), key=lambda r: sum(2 * grid[r + 16 * j]['k'] / amps['k'] + max(grid[r + 16 * j + 8]['s'], (grid[r + 16 * j + 4]['s'] + grid[r + 16 * j + 12]['s']) / 2) / amps['s'] for j in range(nst // 16)))
    bars = [(row('k', rot + 16 * j), row('s', rot + 16 * j), row('h', rot + 16 * j)) for j in range(1, (nst - rot) // 16 - 1)]
    ok = lambda b: b[0][0] in 'xo' and (b[1][8] in 'xo' or (b[1][4] in 'xo' and b[1][12] in 'xo')) and b[1].count('x') + b[1].count('o') <= 5 and b[2].count('.') <= 13
    bars = [b for b in bars if ok(b)]
    for p, c in collections.Counter(bars).most_common(2):
        out[m['genre']].append([p[0], p[1], p[2], '................'])
    rep[key] = (round(tempo), collections.Counter(bars).most_common(1)[0][0] if bars else None)
for k, v in rep.items(): print(k, v)
json.dump(out, open('/home/claude/ana/grooves_ace.json', 'w')); print({g: len(v) for g, v in out.items()})
