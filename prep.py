"""Вокал → куски 3 с → эмбеддинги голоса (Resemblyzer) → кластеры «один голос». Отчёт без аудио."""
import sys, json, numpy as np, soundfile as sf, librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
voc, out = sys.argv[1], sys.argv[2]
y, sr = librosa.load(voc, sr=16000, mono=True)
W, H = 3 * 16000, int(1.5 * 16000); segs = []
rms_all = np.sqrt(np.mean(y ** 2))
for a in range(0, len(y) - W, H):
    c = y[a:a + W]; r = np.sqrt(np.mean(c ** 2))
    if r > 0.35 * rms_all: segs.append((a / 16000, c))
enc = VoiceEncoder('cpu'); E = np.array([enc.embed_utterance(preprocess_wav(c, source_sr=16000)) for _, c in segs])
print('segments', len(segs))
best = None
for th in [0.30, 0.35, 0.40]:
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric='cosine', linkage='average').fit(E)
    lab = cl.labels_; cnt = np.bincount(lab)
    if best is None: best = (th, lab)
    print('th', th, 'clusters', len(cnt), 'top', sorted(cnt)[::-1][:6])
rep = {}
for th in [0.30, 0.35, 0.40]:
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric='cosine', linkage='average').fit(E).labels_
    cs = []
    for k in np.argsort(-np.bincount(lab))[:8]:
        idx = np.where(lab == k)[0]; t = [segs[i][0] for i in idx]
        cen = E[idx].mean(0); cen /= np.linalg.norm(cen); coh = float(np.mean(E[idx] @ cen))
        # куски по песням: склеиваем подряд идущие времена
        ranges = []; 
        for tt in sorted(t):
            if ranges and tt - ranges[-1][1] <= 4.5: ranges[-1][1] = tt + 3
            else: ranges.append([tt, tt + 3])
        cs.append({'cluster': int(k), 'minutes': round(len(idx) * 1.5 / 60, 1), 'coherence': round(coh, 3), 'ranges': [[round(a, 1), round(b, 1)] for a, b in ranges]})
    rep[str(th)] = cs
np.save(out + '/emb.npy', E); json.dump({'times': [s[0] for s in segs]}, open(out + '/times.json', 'w'))
json.dump(rep, open(out + '/report.json', 'w'), ensure_ascii=False, indent=1)
