"""Баланс вокал/бит: на отрезках, где поёт вокал, сравнить громкость микса с голым битом (после выравнивания общей громкости).
python3 balance.py beat.wav mix.wav vox.wav [--want 1.5] → печатает, на сколько дБ вокал стоит выше/ниже цели (минус = убавить)"""
import sys, argparse, numpy as np, librosa
ap = argparse.ArgumentParser(); ap.add_argument('beat'); ap.add_argument('mix'); ap.add_argument('vox'); ap.add_argument('--want', type=float, default=1.5); a = ap.parse_args()
SR = 22050; ld = lambda f: librosa.load(f, sr=SR, mono=True)[0]
b, m, v = ld(a.beat), ld(a.mix), ld(a.vox); n = min(len(b), len(m), len(v)); b, m, v = b[:n], m[:n], v[:n]
H = int(.4 * SR); k = n // H
rms = lambda x: np.sqrt((x[:k * H].reshape(k, H) ** 2).mean(1) + 1e-12)
rb, rm, rv = rms(b), rms(m), rms(v); act = rv > np.percentile(rv, 60) * .5; act &= rv > 1e-3
gm = 20 * np.log10(rm[~act].mean() / rb[~act].mean()) if (~act).sum() > 3 else 0.0   # поправка общего уровня по тихим местам (там вокала нет)
d = 20 * np.log10(rm[act].mean() / rb[act].mean()) - gm
print(round(float(d - a.want), 2))
