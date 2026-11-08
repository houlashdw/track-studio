"""одиночные удары (кик/снейр/хэт) из стемов ударных ACE → OGG в банк smp/ (семья по жанру)"""
import json, collections, subprocess, os, numpy as np, librosa, soundfile as sf
D = '/home/claude/ins'; OUT = '/home/claude/track-studio/app/smp'; meta = json.load(open(D + '/meta.json')); SR = 44100
TRAPF = {'trap', 'drill', 'plugg', 'rage', 'phonk'}; LEN = {'kick': .5, 'snare': .38, 'hat': .14}; per = collections.Counter(); idx_add = collections.defaultdict(list)
for key, m in meta.items():
    g = m['genre']; y, _ = librosa.load(f'{D}/{key}_drums.mp3', sr=SR)
    on = librosa.onset.onset_detect(y=y, sr=SR, hop_length=512, units='time', delta=0.08, backtrack=True)
    cand = []
    for i, t in enumerate(on):
        a = int(t * SR); w = y[a:a + int(.06 * SR)]
        if len(w) < 512: continue
        F = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2; f = np.fft.rfftfreq(len(w), 1 / SR)
        lo, md, hi = F[(f > 30) & (f < 150)].sum(), F[(f > 180) & (f < 2500)].sum(), F[(f > 6000)].sum(); tot = lo + md + hi + 1e-12
        c = 'kick' if lo / tot > .7 else 'hat' if hi / tot > .6 and md / tot < .25 else 'snare' if md / tot > .45 and lo / tot < .3 else None
        if not c: continue
        nxt = on[i + 1] - t if i + 1 < len(on) else 9; prv = t - on[i - 1] if i else 9
        if nxt < LEN[c] * (0.8 if c != 'kick' else 0.9) or prv < .1: continue
        seg = y[max(0, a - int(.004 * SR)): a + int(LEN[c] * SR)].copy()
        if len(seg) < int(LEN[c] * SR * .9): continue
        cand.append((float(np.abs(seg).max()), c, seg))
    for c in ('kick', 'snare', 'hat'):
        L = sorted([x for x in cand if x[1] == c], key=lambda x: -x[0])[:1]
        for amp, c_, seg in L:
            if per[(g, c)] >= 2: continue
            n = len(seg); fo = np.ones(n); k = int(n * .3); fo[-k:] = np.linspace(1, 0, k) ** 2; seg = seg * fo; seg = seg / (np.abs(seg).max() + 1e-9) * .9
            nm = f'{c}_ace_{g}_{per[(g, c)]}.ogg'; sf.write('/tmp/claude-0/os.wav', seg.astype(np.float32), SR)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', '/tmp/claude-0/os.wav', '-c:a', 'libvorbis', '-q:a', '5', f'{OUT}/{nm}'], check=True)
            it = {'f': nm, 'src': 'ace-step-1.5 (generated)', 'len': round(n / SR, 2)}
            if g in TRAPF: it['fam'] = 'trap'
            if c == 'kick':
                F = np.abs(np.fft.rfft(seg[int(.02 * SR):int(.2 * SR)])); f = np.fft.rfftfreq(len(seg[int(.02 * SR):int(.2 * SR)]), 1 / SR); mk = (f > 35) & (f < 120); it['hz'] = round(float(f[mk][np.argmax(F[mk])]), 1)
            idx_add[c].append(it); per[(g, c)] += 1
print({c: len(v) for c, v in idx_add.items()}); json.dump(idx_add, open('/home/claude/ana/oneshots_idx.json', 'w'), indent=0)
