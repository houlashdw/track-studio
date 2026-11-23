"""Ядро голоса ACE «lego» без Modal: для Kaggle (и любого GPU). Копия логики AceLego.lego из server/modal_ace.py.
load() → (dit, sep, has_voice); lego(ctx, q, beat_path) → mp3 голоса"""
import os, sys, time, glob, base64, subprocess


def load(root, adapter_glob=''):
    import torch
    os.chdir(root); sys.path.insert(0, root)
    from acestep import gpu_config
    gpu_config.cuda_supports_bfloat16 = lambda *a, **k: True          # T4/P100: bf16 вместо fp16 (иначе NaN)
    import acestep.core.generation.handler.memory_utils as _mu; _mu.cuda_supports_bfloat16 = gpu_config.cuda_supports_bfloat16
    from acestep.model_downloader import ensure_main_model, ensure_dit_model
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    t0 = time.time(); ensure_main_model(); ensure_dit_model('acestep-v15-base')
    dit = AceStepHandler(); print(dit.initialize_service(project_root=root, config_path='acestep-v15-base', device='cuda', offload_to_cpu=False), flush=True)
    lm = LLMHandler()
    ad = sorted(glob.glob(adapter_glob)) if adapter_glob else []; has_voice = False
    if ad: print(dit.load_lora(ad[-1]), flush=True); has_voice = True
    from demucs.pretrained import get_model
    sep = get_model('htdemucs').cuda().eval()
    print('lego загрузка %.0f с, голос vc2: %s' % (time.time() - t0, has_voice), flush=True)
    return type('Ctx', (), dict(dit=dit, lm=lm, sep=sep, has_voice=has_voice))()


def lego(self, q, beat_path):
    import subprocess, numpy as np, torch, soundfile as sf
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    from demucs.apply import apply_model
    T0 = time.time(); subprocess.run(['cp', beat_path, '/tmp/beat.mp3'], check=True)
    ref = None   # референс тембра: только наш же ACE-голос (понравившийся прошлый вокал), не запись реального человека
    if q.get('ref_b64'): open('/tmp/vref.mp3', 'wb').write(base64.b64decode(q['ref_b64'])); ref = '/tmp/vref.mp3'
    voice = float(q.get('voice', 1.0)) if self.has_voice else 0.0
    if self.has_voice:
        self.dit.set_use_lora(voice > 0)
        if voice > 0: self.dit.set_lora_scale(voice)
    seed0 = int(q.get('seed', -1))
    f0max = float(q.get('f0max', 0)); tries = int(q.get('tries', 2))
    for att in range(tries):   # женский голос вышел низким (мужским) или слишком высоким («писк») → ещё раз с другим сидом
        seed = seed0 if att == 0 else (abs(seed0) + 7919 * att if seed0 >= 0 else -1)
        p = GenerationParams(task_type='lego', src_audio='/tmp/beat.mp3', instruction='Generate the vocals track based on the audio context:',
                             caption=('vc2, ' if voice > 0 else '') + q.get('caption', 'female vocal in Russian'), lyrics=q.get('lyrics', ''), bpm=q.get('bpm') or None,
                             vocal_language=q.get('lang', 'ru'), keyscale=q.get('key') or '', repainting_start=0.0, repainting_end=-1, inference_steps=int(q.get('steps', 30)), seed=seed, reference_audio=ref)
        r = generate_music(self.dit, self.lm, p, GenerationConfig(batch_size=1, audio_format='wav', use_random_seed=seed < 0), save_dir='/tmp/lg')
        if not (r.success and r.audios): raise RuntimeError(str(getattr(r, 'error', r)))
        print('lego: генерация %.0f с (шагов %s)' % (time.time() - T0, q.get('steps', 30)), flush=True)
        if q.get('full'):   # «наш бит, сведённый ACE»: весь ответ ACE (наш бит в его звучании + голос) — склеено как у ACE
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', r.audios[0]['path'], '-b:a', '192k', '/tmp/full.mp3'], check=True); return open('/tmp/full.mp3', 'rb').read()
        x, sr = sf.read(r.audios[0]['path'], dtype='float32', always_2d=True)
        if sr != 44100:
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', r.audios[0]['path'], '-ar', '44100', '/tmp/lg44.wav'], check=True); x, sr = sf.read('/tmp/lg44.wav', dtype='float32', always_2d=True)
        if x.shape[1] == 1: x = np.repeat(x, 2, 1)
        with torch.no_grad():
            st = apply_model(self.sep, torch.from_numpy(x.T.copy())[None].cuda(), shifts=1, split=True, overlap=0.25)[0]
        v = st[self.sep.sources.index('vocals')].cpu().numpy().T; print('lego: +demucs %.0f с' % (time.time() - T0), flush=True)
        if q.get('full') or q.get('sex', 'f') != 'f': break
        f0 = vox_f0(v, sr); print('lego: высота голоса %.0f Гц (попытка %d)' % (f0, att + 1), flush=True)
        if f0 == 0 or (f0 >= 175 and not (f0max and f0 > f0max)): break
    # сдвиг ACE относительно нашего бита (огибающие атак, ±0.5 с): голос подвигаем, чтобы шёл ровно с битом
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', '/tmp/beat.mp3', '-ac', '1', '-ar', '44100', '/tmp/beat.wav'], check=True)
        b0, _ = sf.read('/tmp/beat.wav', dtype='float32'); m0 = st.sum(0).mean(0).cpu().numpy() - v.mean(1)   # музыка из ответа ACE (без голоса)
        H = 441
        def env(z):
            n = len(z) // H; e = np.sqrt((z[:n * H].reshape(n, H) ** 2).mean(1) + 1e-9); d = np.maximum(np.diff(np.log(e)), 0); return (d - d.mean()) / (d.std() + 1e-9)
        eb, em = env(b0), env(m0); n = min(len(eb), len(em), 3000); eb, em = eb[:n], em[:n]
        lags = range(-50, 51); cc = [float(np.dot(eb[max(0, -k):n - max(0, k)], em[max(0, k):n - max(0, -k)])) for k in lags]
        lag = list(lags)[int(np.argmax(cc))] * H   # >0: ACE запаздывает
        print('сдвиг ACE: %+.0f мс' % (lag / 44.1), flush=True)
        if lag > 0: v = np.concatenate([v[lag:], np.zeros((lag, 2), np.float32)])
        elif lag < 0: v = np.concatenate([np.zeros((-lag, 2), np.float32), v[:lag]])
    except Exception as e: print('сдвиг не вышел', e, flush=True)
    if q.get('deq', True): v = deboxx(v, sr)
    sf.write('/tmp/voc.wav', v, sr)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', '/tmp/voc.wav', '-b:a', '192k', '/tmp/voc.mp3'], check=True)
    print('lego: всего %.0f с' % (time.time() - T0), flush=True)
    return open('/tmp/voc.mp3', 'rb').read()



def vox_f0(v, sr):
    """медиана основного тона голоса (автокорреляция по громким кускам), 0 — не понять. Жен. пение обычно 190–500 Гц, муж. 90–170"""
    import numpy as np
    m = v.mean(1); n = int(.04 * sr); hop = n; fs = []
    thr = np.percentile(np.abs(m), 90) * .3
    for i in range(0, len(m) - n, hop * 3):
        x = m[i:i + n]
        if np.abs(x).mean() < thr: continue
        x = x - x.mean(); c = np.correlate(x, x, 'full')[n - 1:]; lo, hi = int(sr / 600), int(sr / 70)
        if c[0] <= 0: continue
        k = lo + int(np.argmax(c[lo:hi])); r = c[k] / c[0]
        if r > .5: fs.append(sr / k)
    return float(np.median(fs)) if len(fs) > 10 else 0.0



def deboxx(v, sr):
    """голос из lego выходит «в бутылке»: верх (2–8 кГц) на 20–35 дБ ниже середины. Подтягиваем полосы к обычному наклону вокала
    (адаптивно по измерению, +0…+12 дБ, плавная кривая), чуть убираем коробочную середину 300–700 Гц."""
    import numpy as np
    m = v.mean(1); n = 1 << 13; hop = n // 4
    if len(m) < n * 2: return v
    fr = np.fft.rfftfreq(n, 1 / sr); win = np.hanning(n)
    fr_e = np.array([np.abs(np.fft.rfft(m[i:i + n] * win)) ** 2 for i in range(0, len(m) - n, hop * 4)])
    loud = fr_e.sum(1) > np.percentile(fr_e.sum(1), 60); P = fr_e[loud].mean(0) + 1e-12
    band = lambda a, b: 10 * np.log10(P[(fr >= a) & (fr < b)].mean())
    ref = band(200, 1000)
    tgt = [(1500, 1000, 2000, -7), (3000, 2000, 4000, -12), (6000, 4000, 8000, -18), (10000, 8000, 12000, -23), (14000, 12000, 16000, -30)]
    pts = [(20, 0.0), (250, 0.0)]; boxy = band(300, 700) - ref
    pts.append((500, -min(3.0, max(0.0, boxy + 1))))   # середина толще остального — срезаем до 3 дБ
    for c, a, b, t in tgt: pts.append((c, float(np.clip(t - (band(a, b) - ref), 0, 6))))   # больше +6 дБ подъёма — металлический звон артефактов
    pts.append((sr / 2, pts[-1][1]))
    xs = np.log10([p[0] for p in pts]); ys = [p[1] for p in pts]
    g = 10 ** (np.interp(np.log10(np.maximum(fr, 20)), xs, ys) / 20)
    print('деEQ (дБ):', ', '.join('%d Гц %+.1f' % (p[0], p[1]) for p in pts[2:-1]), flush=True)
    out = np.zeros_like(v); wsum = np.zeros(len(v))
    for i in range(0, len(v) - n, hop):
        w2 = win
        for ch in range(v.shape[1]): out[i:i + n, ch] += np.fft.irfft(np.fft.rfft(v[i:i + n, ch] * w2) * g, n) * w2
        wsum[i:i + n] += w2 * w2
    out /= np.maximum(wsum, 1e-3)[:, None]
    pk = np.abs(out).max(); pv = np.abs(v).max()
    return (out * (pv / pk if pk > pv else 1)).astype('float32')
