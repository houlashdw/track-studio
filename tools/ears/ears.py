"""«Уши»: оценки качества звука моделями, обученными на оценках живых слушателей.
DNSMOS (Microsoft): SIG — чистота самого голоса (искажения), BAK — фон/шум, OVRL — общее (1..5).
NISQA: MOS — общее, NOI — шумность, COL — окраска (тембр «не тот»), DIS — разрывы/щелчки, LOUD — громкость (5 = нет проблемы);
NISQA-TTS: NAT — естественность синтезированного голоса.
Audiobox Aesthetics (Meta, обучена на оценках музыки/речи людьми, 1..10): PQ — качество продакшна, CE — удовольствие, CU — полезность, PC — сложность.
Свои замеры: CLIP — доля клиппинга, HF — «грязь» выше 10 кГц (дБ к сигналу).
usage: ears.py file1 [file2 ...] [--seg 6]  (wav/mp3/flac или *.f32 моно 44.1 кГц; --seg N — ещё и по кускам N с)"""
import sys, os, subprocess, tempfile, numpy as np, soundfile as sf, librosa
sys.path.insert(0, '/home/claude/ears/sm')
from speechmos import dnsmos

ND = '/home/claude/ears/nisqa'

def load(p):
    if p.endswith('.f32'):
        return np.fromfile(p, np.float32).astype(np.float64), 44100
    y, sr = librosa.load(p, sr=None, mono=True)
    return y.astype(np.float64), sr

def dns(y, sr):
    x = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    r = dnsmos.run(x.astype(np.float32), 16000)
    return {'SIG': r['sig_mos'], 'BAK': r['bak_mos'], 'OVRL': r['ovrl_mos']}

def nisqa(files):
    out = {}
    for model, cols in [('nisqa.tar', ['mos_pred', 'noi_pred', 'col_pred', 'dis_pred', 'loud_pred']), ('nisqa_tts.tar', ['mos_pred'])]:
        d = tempfile.mkdtemp()
        for i, (y, sr) in enumerate(files):
            x = librosa.resample(y, orig_sr=sr, target_sr=48000) if sr != 48000 else y
            sf.write(f'{d}/{i:03d}.wav', (x / (np.abs(x).max() + 1e-9) * 0.9).astype(np.float32), 48000)
        o = tempfile.mkdtemp()
        subprocess.run([sys.executable, 'run_predict.py', '--mode', 'predict_dir', '--pretrained_model', 'weights/' + model, '--data_dir', d, '--output_dir', o, '--bs', '4'],
                       cwd=ND, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        import pandas as pd
        df = pd.read_csv(o + '/NISQA_results.csv')
        for _, row in df.iterrows():
            i = int(row['deg'].split('.')[0]); r = out.setdefault(i, {})
            if model == 'nisqa.tar':
                r.update({'MOS': row['mos_pred'], 'NOI': row['noi_pred'], 'COL': row['col_pred'], 'DIS': row['dis_pred'], 'LOUD': row['loud_pred']})
            else:
                r['NAT'] = row['mos_pred']
    return out

_AES = []
def aes(items):
    if not _AES:
        from audiobox_aesthetics.infer import initialize_predictor
        _AES.append(initialize_predictor('/home/claude/ears/aesm/checkpoint.pt'))
    import torch
    batch = []
    for y, sr in items:
        x = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        batch.append({'path': torch.tensor(x, dtype=torch.float32)[None], 'sample_rate': 16000})
    return _AES[0].forward(batch)

def own(y, sr):
    pk = np.abs(y).max() + 1e-9; clip = float(np.mean(np.abs(y) > 0.985 * pk)) if pk > 0.95 else 0.0
    S = np.abs(np.fft.rfft(y[: sr * 20] * np.hanning(min(len(y), sr * 20)))) ** 2; f = np.fft.rfftfreq(min(len(y), sr * 20), 1 / sr)
    hf = 10 * np.log10(S[f > 10000].sum() / (S[(f > 300) & (f < 4000)].sum() + 1e-12) + 1e-12)
    return {'CLIP': clip * 100, 'HF': hf}

def chunks(y, sr, seg):
    n = int(seg * sr); res = []
    for a in range(0, len(y) - n // 2, n):
        c = y[a:a + n]
        if np.sqrt(np.mean(c ** 2)) > 0.01 * np.abs(y).max(): res.append((a / sr, c))
    return res

if __name__ == '__main__':
    args = sys.argv[1:]; seg = None
    if '--seg' in args: i = args.index('--seg'); seg = float(args[i + 1]); del args[i:i + 2]
    items = []
    for p in args:
        y, sr = load(p); items.append((os.path.basename(p), None, y, sr))
        if seg:
            for t, c in chunks(y, sr, seg): items.append((os.path.basename(p), t, c, sr))
    nq = nisqa([(y, sr) for _, _, y, sr in items])
    try: ae = aes([(y, sr) for _, _, y, sr in items])
    except Exception as e: print('audiobox:', e); ae = [{} for _ in items]
    K = ['SIG', 'BAK', 'OVRL', 'MOS', 'NOI', 'COL', 'DIS', 'LOUD', 'NAT', 'PQ', 'CE', 'PC', 'HF']
    print('%-28s %6s ' % ('файл', 'с') + ' '.join('%5s' % k for k in K))
    for i, (name, t, y, sr) in enumerate(items):
        r = {**dns(y, sr), **nq.get(i, {}), **ae[i], **own(y, sr)}
        print('%-28s %6s ' % (name[:28], '' if t is None else '%.0f' % t) + ' '.join('%5.2f' % r.get(k, float('nan')) for k in K))
