"""Kaggle GPU: биты (сгенерированные дообученным ACE и/или настоящие) → стемы demucs → одиночные кики/снейры/хэты/808 → OGG + kits.json.
Вход: выходы ноутбуков (ksrc) с *.mp3. Итог: /kaggle/working/kit/*.ogg, kits.json (src='tsai')."""
import os, sys, glob, json, subprocess, collections
W = '/kaggle/working/kit'; os.makedirs(W, exist_ok=True); SR = 44100
subprocess.run(f'{sys.executable} -m pip install -q demucs librosa soundfile', shell=True)
import numpy as np, librosa, soundfile as sf
PAT = os.environ.get('PAT', '*.mp3'); MAXF = int(os.environ.get('MAXF', '120'))
files = sorted(f for f in glob.glob('/kaggle/input/**/' + PAT, recursive=True) if 'no_vocals' not in f)[:MAXF]
print('файлов', len(files), flush=True)
for i in range(0, len(files), 30): subprocess.run([sys.executable, '-m', 'demucs', '-n', 'htdemucs', '-o', '/tmp/sep'] + files[i:i + 30])
LEN = {'kick': .55, 'snare': .45, 'hat': .16}; per = collections.Counter(); idx = collections.defaultdict(list); CAP = int(os.environ.get('CAP', '2'))


def save(seg, c, tag, extra=None):
    n = len(seg); fo = np.ones(n); k = int(n * .25); fo[-k:] = np.linspace(1, 0, k) ** 2; seg = seg * fo; seg = seg / (np.abs(seg).max() + 1e-9) * .9
    nm = f'{c}_tsai_{len(idx[c]):03d}.ogg'; sf.write('/tmp/os.wav', seg.astype(np.float32), SR)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', '/tmp/os.wav', '-c:a', 'libvorbis', '-q:a', '5', f'{W}/{nm}'], check=True)
    it = {'f': nm, 'src': 'tsai', 'fam': 'trap', 'len': round(n / SR, 2), 'from': tag}; it.update(extra or {}); idx[c].append(it)


def f0(x):
    seg = x[int(.06 * SR):int(.6 * SR)]; S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 16)); fr = np.fft.rfftfreq(1 << 16, 1 / SR)
    m = (fr > 28) & (fr < 220); return float(fr[m][np.argmax(S[m])])


for f in files:
    b = os.path.splitext(os.path.basename(f))[0]; dd = f'/tmp/sep/htdemucs/{b}'
    if not os.path.exists(dd + '/drums.wav'): continue
    y, _ = librosa.load(dd + '/drums.wav', sr=SR, mono=True)
    on = librosa.onset.onset_detect(y=y, sr=SR, hop_length=512, units='time', delta=0.08, backtrack=True); cand = []
    for i, t in enumerate(on):
        a = int(t * SR); w = y[a:a + int(.06 * SR)]
        if len(w) < 512: continue
        F = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2; fq = np.fft.rfftfreq(len(w), 1 / SR)
        lo, md, hi = F[(fq > 30) & (fq < 150)].sum(), F[(fq > 180) & (fq < 2500)].sum(), F[fq > 6000].sum(); tot = lo + md + hi + 1e-12
        c = 'kick' if lo / tot > .7 else 'hat' if hi / tot > .6 and md / tot < .25 else 'snare' if md / tot > .45 and lo / tot < .3 else None
        if not c: continue
        nxt = on[i + 1] - t if i + 1 < len(on) else 9; prv = t - on[i - 1] if i else 9
        if nxt < LEN[c] * .85 or prv < .1: continue
        seg = y[max(0, a - int(.004 * SR)): a + int(LEN[c] * SR)].copy()
        if len(seg) >= int(LEN[c] * SR * .9): cand.append((float(np.abs(seg).max()), c, seg))
    for c in LEN:
        for amp, _, seg in sorted([x for x in cand if x[1] == c], key=lambda x: -x[0])[:CAP]:
            save(seg, c, b, {'hz': round(f0(seg), 1)} if c == 'kick' else None)
    # 808: длинные ноты в стеме баса
    yb, _ = librosa.load(dd + '/bass.wav', sr=SR, mono=True); ob = librosa.onset.onset_detect(y=yb, sr=SR, units='time', delta=0.1, backtrack=True); got = 0
    for i, t in enumerate(ob):
        nxt = ob[i + 1] - t if i + 1 < len(ob) else 9
        if nxt < 1.4 or got >= 1: continue
        a = int(t * SR); seg = yb[max(0, a - 64): a + int(min(nxt, 2.8) * SR)].copy()
        if len(seg) < int(1.3 * SR) or np.abs(seg).max() < .05: continue
        h = f0(seg)
        if 28 < h < 120: save(seg, '808', b, {'hz': round(h, 1)}); got += 1
json.dump(idx, open(W + '/kits.json', 'w'), ensure_ascii=False, indent=0); print({c: len(v) for c, v in idx.items()})
