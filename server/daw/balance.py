"""Уровни до мастера: bpre.wav — бит, vpre.wav — вокал после цепочки (оба без мастер-обработки, фейдер вокала = F0).
Печатает новый фейдер вокала: активный вокал на --want дБ выше среднего уровня бита.   python3 balance.py bpre.wav vpre.wav F0 [--want 1.5]"""
import argparse, numpy as np, librosa
ap = argparse.ArgumentParser(); ap.add_argument('bpre'); ap.add_argument('vpre'); ap.add_argument('f0', type=float); ap.add_argument('--want', type=float, default=1.5); a = ap.parse_args()
SR = 22050; ld = lambda f: librosa.load(f, sr=SR, mono=True)[0]
b, v = ld(a.bpre), ld(a.vpre); H = int(.4 * SR)
def wr(x): k = len(x) // H; return np.sqrt((x[:k * H].reshape(k, H) ** 2).mean(1) + 1e-12)
rb, rv = wr(b), wr(v); lb = 20 * np.log10(rb[rb > rb.max() * .05].mean()); lv = 20 * np.log10(rv[rv > rv.max() * .15].mean())
print(round(float(a.f0 + (lb + a.want) - lv), 1))
