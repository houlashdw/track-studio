"""Обучение своего голоса (RVC v2) на бесплатной видеокарте Kaggle.
Ячейка ноутбука (Internet: On, Accelerator: GPU T4 / P100, секрет KEYSTORE_PASS в Add-ons → Secrets):
    from kaggle_secrets import UserSecretsClient; import os
    os.environ['P'] = UserSecretsClient().get_secret('KEYSTORE_PASS')
    !wget -q https://raw.githubusercontent.com/houlashdw/track-studio/main/colab/kaggle_voice.py -O kv.py && python kv.py
Итог в /kaggle/working (вкладка Output): voice_best.onnx.enc (зашифрован тем же паролем) + demo_*.mp3 + scores.txt.
Шаги: отдельное окружение (как на GitHub) → расшифровка треков → вокал (Demucs) → нарезка → признаки → обучение с сохранением каждые 10 эпох →
для каждой сохранённой эпохи: ONNX + проверка разборчивости (Whisper) → лучшая эпоха."""
import os, sys, subprocess, glob, json, time, shutil

REPO = 'https://raw.githubusercontent.com/houlashdw/track-studio'
SRC = os.environ.get('SRC', 'src2.mp3.enc')
EPOCHS = int(os.environ.get('EPOCHS', '200')); EVERY = int(os.environ.get('EVERY', '10')); BATCH = os.environ.get('BATCH', '8')
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('kv_out')
os.environ['MPLBACKEND'] = 'Agg'   # в Kaggle стоит matplotlib_inline — в нашем окружении его нет, обучение падало на import matplotlib
T = '/tmp/kv'; VENV = T + '/venv'; PY = VENV + '/bin/python'; RVC = T + '/rvc'


def sh(cmd, **kw):
    print('$', cmd if isinstance(cmd, str) else ' '.join(cmd), flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def orchestrate():
    P = os.environ.get('P')
    if not P:
        try:
            from kaggle_secrets import UserSecretsClient
            P = UserSecretsClient().get_secret('KEYSTORE_PASS')
        except Exception:
            pass
    if not P: raise SystemExit('нет пароля: добавь секрет KEYSTORE_PASS (Add-ons → Secrets) и включи его для ноутбука')
    os.makedirs(T, exist_ok=True); os.makedirs(WORK, exist_ok=True); t0 = time.time()
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv || true')
    # 1) окружение как на GitHub (numpy<2, torch 2.6) — не трогаем системные пакеты Kaggle
    if not os.path.exists(PY):
        sh(f'{sys.executable} -m pip install -q uv')
        sh(f'{sys.executable} -m uv venv -q -p 3.11 {VENV} || {sys.executable} -m uv venv -q {VENV}')
        sh(f'{sys.executable} -m uv pip install -q -p {PY} torch==2.6.0 torchaudio==2.6.0 "numpy<2" scipy soundfile librosa==0.10.2.post1 '
           f'transformers==4.46.3 onnx onnxruntime tensorboard matplotlib demucs==4.0.1 lameenc')
    # 2) RVC 2.2 + те же правки, что в train-voice.yml
    if not os.path.isdir(RVC):
        sh(f'git clone -q --depth 1 --branch 2.2.231006 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git {RVC}')
        sh(f"""cd {RVC} && sed -i 's/                x = flow(x, x_mask, g=g, reverse=reverse)/                x, _ = flow(x, x_mask, g=g, reverse=reverse)/' infer/lib/infer_pack/models_onnx.py && \
sed -i 's/np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")/np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy().reshape(-1)/' infer/lib/train/utils.py && \
sed -i 's/        logger.warning("NO GPU DETECTED: falling back to CPU - this may take a while")/        print("CPU")/' infer/modules/train/train.py && \
mkdir -p assets/pretrained_v2 assets/rmvpe assets/weights""")
        HF = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main'
        for a, b in [('pretrained_v2/f0G40k.pth', 'assets/pretrained_v2/f0G40k.pth'), ('pretrained_v2/f0D40k.pth', 'assets/pretrained_v2/f0D40k.pth'), ('rmvpe.pt', 'assets/rmvpe/rmvpe.pt')]:
            sh(f'wget -nv --tries=8 --waitretry=20 -O {RVC}/{b} {HF}/{a}')
        for f in ['renri_rvc.py', 'onnx_half.py', 'demo_ids.json', 'kaggle_voice.py']:
            sh(f'wget -q -O {RVC}/{f} {REPO}/main/colab/{f}')
        sh(f'mkdir -p {T}/piper && cd {T}/piper && wget -q https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ru-irinia-medium.tar.gz && tar xzf voice-ru-irinia-medium.tar.gz')
    # 3) треки → вокал → куски
    if not os.path.exists(T + '/ds.ok'):
        sh(f'wget -q -O {T}/src.enc {REPO}/vprep-in/{SRC}')
        subprocess.run(['openssl', 'enc', '-d', '-aes-256-cbc', '-pbkdf2', '-iter', '200000', '-in', T + '/src.enc', '-out', T + '/src.mp3', '-pass', 'env:P'], check=True, env=dict(os.environ, P=P))
        sh(f'cd {T} && {PY} -m demucs --two-stems=vocals -n htdemucs --mp3 --mp3-bitrate 320 -o sep src.mp3')
        sh(f'{PY} {RVC}/kaggle_voice.py slice {T}/sep/htdemucs/src/vocals.mp3 {T}/ds')
        open(T + '/ds.ok', 'w').write('1')
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', SAVE_WEIGHTS='1', SAVE_EVERY=str(EVERY), LOG_INTERVAL='50',
               TRAIN_SECONDS=os.environ.get('TRAIN_SECONDS', '28800'))
    env.pop('P', None)
    for st in ['prep ' + T + '/ds', 'f0', 'feat']:
        sh(f'cd {RVC} && {PY} renri_rvc.py {st}', env=env)
    # 4) обучение (видеокарта); каждые EVERY эпох — маленькие веса в assets/weights
    sh(f'cd {RVC} && {PY} -u renri_rvc.py train {EPOCHS} {BATCH} 2>&1 | grep --line-buffered -vE "Warning|warn\\(|return_complex|_VF.stft|torch.load\\(|autocast\\(|WeightNorm|GradScaler"', env=env)
    print('обучение: %.0f мин' % ((time.time() - t0) / 60), flush=True)
    # 5) выбор лучшей эпохи и выдача
    try:
        sh(f'cd {RVC} && {PY} kaggle_voice.py eval {T}/piper {WORK}', env=env)
        best = open(WORK + '/best.txt').read().split()
    except Exception as e:   # проверка упала — не теряем обучение: отдаём веса нескольких эпох (их переведу в ONNX сам)
        print('⚠ проверка эпох упала:', e, flush=True)
        ws = sorted(glob.glob(RVC + '/assets/weights/renri_e*_s*.pth'), key=os.path.getmtime)[-6:]
        for f in ws:
            subprocess.run(['openssl', 'enc', '-aes-256-cbc', '-pbkdf2', '-iter', '200000', '-salt', '-in', f, '-out', WORK + '/' + os.path.basename(f) + '.enc', '-pass', 'env:P'], check=True, env=dict(os.environ, P=P))
        print('отдал веса:', [os.path.basename(f) for f in ws]); return
    for i, f in enumerate(best[:2]):
        out = WORK + ('/voice_best.onnx.enc' if i == 0 else '/voice_second.onnx.enc')
        subprocess.run(['openssl', 'enc', '-aes-256-cbc', '-pbkdf2', '-iter', '200000', '-salt', '-in', f, '-out', out, '-pass', 'env:P'], check=True, env=dict(os.environ, P=P))
    print(open(WORK + '/scores.txt').read())
    print('✅ готово за %.0f мин. Вкладка Output → скачай voice_best.onnx.enc (и demo_*.mp3 послушать)' % ((time.time() - t0) / 60))


def slice_vocals(src, dst):
    """вокал → фразы: кадры 50 мс выше порога, склейка пауз < 0.35 с, фразы ≥ 1.2 с; тихие куски (подпевки/хвосты эха) — мимо"""
    import numpy as np, librosa, soundfile as sf
    y, sr = librosa.load(src, sr=40000, mono=True); os.makedirs(dst, exist_ok=True)
    H = int(0.05 * sr); n = len(y) // H
    r = np.sqrt(np.mean(y[:n * H].reshape(n, H) ** 2, 1) + 1e-12); db = 20 * np.log10(r / (np.percentile(r, 95) + 1e-9))
    on = db > -28; seg = []; i = 0
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or (j + 7 < n and on[j:j + 7].any())): j += 1
            seg.append((i, j)); i = j
        else: i += 1
    k = 0; tot = 0
    for a, b in seg:
        if (b - a) * 0.05 < 1.2: continue
        a = max(0, a - 2); b = min(n, b + 2)
        sf.write(f'{dst}/v_{k:04d}.wav', y[a * H:b * H], sr); k += 1; tot += (b - a) * 0.05
    print('нарезка: фраз', k, 'минут %.1f' % (tot / 60), flush=True)


def evaluate(piper_dir, work):
    """каждая сохранённая эпоха → ONNX → строки Piper (женский русский, высота ×1.5) → Whisper → похожесть на исходный текст"""
    import numpy as np, torch, re, difflib, soundfile as sf, onnxruntime as ort
    sys.path.insert(0, os.getcwd()); import renri_rvc as R
    from onnx_half import half_store
    R.patch_attn()
    from infer.modules.onnx.export import export_onnx
    from transformers import pipeline
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    asr = pipeline('automatic-speech-recognition', model='openai/whisper-medium', device=0 if dev == 'cuda' else -1, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32)
    tr = lambda x, sr: asr({'raw': R.resample(x, sr, 16000), 'sampling_rate': 16000}, generate_kwargs={'language': 'russian', 'task': 'transcribe'})['text']
    norm = lambda t: re.sub(r'[^а-я]', '', t.lower().replace('ё', 'е'))
    cv = R.load_cv().to(dev)
    pv = ort.InferenceSession(glob.glob(os.path.join(piper_dir, '*.onnx'))[0])
    lines = []
    for ids in json.load(open('demo_ids.json')):
        ids = np.array([ids], np.int64)
        a = pv.run(None, {'input': ids, 'input_lengths': np.array([ids.shape[1]], np.int64), 'scales': np.array([0.667, 1.0, 0.8], np.float32)})[0].reshape(-1)
        x = R.resample(a, 22050, 16000)
        with torch.no_grad(): fe = cv(torch.from_numpy(x)[None].to(dev)).last_hidden_state[0].float().cpu().numpy()
        fe = np.repeat(fe, 2, axis=0); f0 = R.f0_autocorr(x) * 1.5; L = min(len(fe), len(f0))
        lines.append((fe[:L], f0[:L], norm(tr(a, 22050))))
    ws = sorted(glob.glob('assets/weights/renri_e*_s*.pth'), key=lambda f: int(re.search(r'_e(\d+)_', f).group(1)))
    res = []
    for w in ws:
        ep = int(re.search(r'_e(\d+)_', w).group(1)); onx = f'/tmp/kv/e{ep}.onnx'
        try: export_onnx(w, onx + '.f32'); half_store(onx + '.f32', onx); os.remove(onx + '.f32')
        except Exception as e: print('эпоха', ep, 'экспорт не вышел:', e, flush=True); continue
        rv = ort.InferenceSession(onx, providers=['CPUExecutionProvider']); rs = np.random.RandomState(0); outs = []; sc = []
        for fe, f0, ref in lines:
            L = len(fe)
            y = rv.run(None, {'phone': fe[None].astype(np.float32), 'phone_lengths': np.array([L], np.int64), 'pitch': R.coarse(f0.astype(np.float64))[None],
                              'pitchf': f0[None].astype(np.float32), 'ds': np.array([0], np.int64), 'rnd': rs.randn(1, 192, L).astype(np.float32) * 0.66666})[0].reshape(-1)
            outs += [y, np.zeros(R.SR // 3, np.float32)]
            sc.append(difflib.SequenceMatcher(None, norm(tr(y, R.SR)), ref).ratio())
        s = float(np.mean(sc)); res.append((s, ep, onx)); print('эпоха', ep, 'разборчивость %.3f' % s, flush=True)
        sf.write('/tmp/kv/d.wav', np.concatenate(outs), R.SR)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', '/tmp/kv/d.wav', '-b:a', '128k', f'{work}/demo_e{ep:03d}.mp3'], check=False)
    if not res: raise SystemExit('нет ни одной сохранённой эпохи — обучение не дошло до сохранения')
    res.sort(key=lambda r: (-round(r[0], 2), r[1]))   # лучшая разборчивость; при равенстве — более ранняя эпоха (меньше переобучения)
    open(work + '/scores.txt', 'w').write('\n'.join('эпоха %d: %.3f' % (e, s) for s, e, _ in sorted(res, key=lambda r: r[1])) + '\nлучшая: %d\n' % res[0][1])
    open(work + '/best.txt', 'w').write('\n'.join(o for _, _, o in res[:2]))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'slice': slice_vocals(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == 'eval': evaluate(sys.argv[2], sys.argv[3])
    else: orchestrate()
