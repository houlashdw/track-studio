"""Подготовка вокала перед Reaper (по ТЗ «вокалоид под бит»):
1) квантизация слогов к сетке 1/16 проекта (мягкая: сдвигаем куски между атаками, не растягиваем — тембр не страдает);
2) срез гула ниже 120 Гц; 3) мягкий дезноиз (не убивает синтетический характер); 4) динамический дебриссинг 5–9 кГц (только громкие «с», «ш»).
python3 voxprep.py in.mp3 out.wav --bpm 140 [--strength .6] [--t0 0]   → печатает, насколько слоги легли на сетку до и после"""
import sys, argparse, json
import numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfilt, sosfiltfilt

SR = 44100


def onsets(x, sr=SR):
    oe = librosa.onset.onset_strength(y=x, sr=sr, hop_length=256, aggregate=np.median)
    fr = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, hop_length=256, backtrack=True, delta=.12, wait=4)
    return librosa.frames_to_time(fr, sr=sr, hop_length=256), oe[fr] if len(fr) else np.array([])


def grid_fit(x, bpm, t0=0.0, sr=SR):
    """доля слогов, чья атака лежит в пределах ±15 мс от узла 1/16, и среднее отклонение в мс"""
    on, w = onsets(x, sr)
    if len(on) < 4: return 0.0, 0.0
    st = 15 / bpm; r = (on - t0) / st; dv = (r - np.round(r)) * st * 1000
    return float((np.abs(dv) < 15).mean()), float(np.abs(dv).mean())


def quantize(x, bpm, t0=0.0, strength=.6, maxshift=.04, sr=SR):
    on, w = onsets(x, sr)
    if len(on) < 4: return x
    st = 15 / bpm; tgt = np.round((on - t0) / st) * st + t0; d = np.clip((tgt - on) * strength, -maxshift, maxshift)
    cuts = list(on) + [len(x) / sr]; X = int(.008 * sr); out = np.zeros(len(x) + int(sr), np.float32)
    # кусок i: [on_i, on_{i+1}) едет на d_i; концы чуть растягиваем реальным продолжением источника, чтобы не было дыр
    first = int(max(0, on[0] - .005) * sr); out[:first] += x[:first]
    for i in range(len(on)):
        a = int(on[i] * sr); b = int(cuts[i + 1] * sr) + X; b = min(b, len(x)); s = int(round(d[i] * sr))
        seg = x[a:b].copy(); n = len(seg)
        if n < 4 * X: out[a + s:a + s + n] += seg; continue
        env = np.ones(n, np.float32); env[:X] = np.linspace(0, 1, X); env[-X:] = np.linspace(1, 0, X)
        if i == 0: env[:X] = 1
        out[max(0, a + s):max(0, a + s) + n] += seg * env
    return out[:len(x)]


def hpf(x, f=120, sr=SR): return sosfiltfilt(butter(4, f, 'high', fs=sr, output='sos'), x).astype(np.float32)


def denoise(x, sr=SR, depth=.5):
    """мягкое спектральное гейтирование: шумовой пол — 10-й процентиль по частотам, максимум −6 дБ"""
    S = librosa.stft(x, n_fft=2048, hop_length=512); m = np.abs(S)
    fl = np.percentile(m, 10, axis=1, keepdims=True) * 1.6
    g = np.clip((m - fl) / (m + 1e-9), depth, 1.0)
    from scipy.ndimage import uniform_filter
    g = uniform_filter(g, size=(3, 5))
    return librosa.istft(S * g, hop_length=512, length=len(x)).astype(np.float32)


def deess(x, sr=SR, lo=5000, hi=9500, reduce=.6):
    band = sosfiltfilt(butter(2, [lo, hi], 'band', fs=sr, output='sos'), x).astype(np.float32)
    n = int(.004 * sr); e = np.sqrt(np.convolve(band ** 2, np.ones(n) / n, 'same') + 1e-12)
    thr = np.percentile(e[e > 1e-4], 85) if (e > 1e-4).any() else 1.0
    g = np.where(e > thr, (thr / e) ** reduce, 1.0); g = np.convolve(g, np.ones(n) / n, 'same').astype(np.float32)
    return x - band + band * g


def prep(x, bpm, t0=0.0, strength=.6, raw=False):
    f0, m0 = grid_fit(x, bpm, t0)
    if raw:   # только срез гула: сравнить со всей обработкой (подгонка к сетке, дезноиз, дебриссинг)
        y = hpf(x); pk = np.abs(y).max(); y = y * (.9 / pk) if pk > .9 else y
        return y.astype(np.float32), dict(grid_before=round(f0, 2), ms_before=round(m0, 1), raw=True)
    y = quantize(x, bpm, t0, strength); f1, m1 = grid_fit(y, bpm, t0)
    y = deess(denoise(hpf(y)))
    pk = np.abs(y).max(); y = y * (.9 / pk) if pk > .9 else y
    return y.astype(np.float32), dict(grid_before=round(f0, 2), ms_before=round(m0, 1), grid_after=round(f1, 2), ms_after=round(m1, 1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--bpm', type=float, required=True)
    ap.add_argument('--t0', type=float, default=0.0); ap.add_argument('--strength', type=float, default=.6); ap.add_argument('--raw', action='store_true')
    a = ap.parse_args()
    x, sr = librosa.load(a.inp, sr=SR, mono=True)
    y, info = prep(x, a.bpm, a.t0, a.strength, a.raw); sf.write(a.out, y, SR, subtype='PCM_24')
    print('вокал:', json.dumps(info), 'длина %.1f с' % (len(y) / SR))
