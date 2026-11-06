"""Дообучение ACE-Step 1.5 (LoKr) на треках vc2 — на бесплатной T4 Kaggle, затем пробные песни с дообученной моделью.
Ячейка (Internet On, GPU T4, секрет KEYSTORE_PASS в Add-ons → Secrets):
    from kaggle_secrets import UserSecretsClient; import os
    os.environ['P'] = UserSecretsClient().get_secret('KEYSTORE_PASS')
    !wget -q https://raw.githubusercontent.com/houlashdw/track-studio/main/colab/kaggle_ace_lora.py -O al.py && python al.py
Итог в Output: lora_*.mp3 (пробные песни) + vc2_lokr.zip.enc (дообученная часть, зашифрована)."""
import os, sys, subprocess, json, time, glob, shutil

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('al_out')
R = '/tmp/ace'; RAW = 'https://raw.githubusercontent.com/houlashdw/track-studio'
EPOCHS = os.environ.get('EPOCHS', '300'); EVERY = os.environ.get('EVERY', '50'); TLIM = int(os.environ.get('TLIM', str(5 * 3600)))
TAG = 'vc2'


def sh(c, check=True, env=None):
    print('$', c, flush=True); return subprocess.run(c, shell=True, check=check, env=env).returncode


def dec(src, dst, P):
    subprocess.run(['openssl', 'enc', '-d', '-aes-256-cbc', '-pbkdf2', '-iter', '200000', '-in', src, '-out', dst, '-pass', 'env:P'], check=True, env=dict(os.environ, P=P))


GEN = r'''
import os, sys, json, time, shutil
sys.path.insert(0, os.getcwd())
from acestep import gpu_config
gpu_config.cuda_supports_bfloat16 = lambda *a, **k: True          # T4: bf16 вместо fp16 (иначе NaN)
import acestep.core.generation.handler.memory_utils as _mu; _mu.cuda_supports_bfloat16 = gpu_config.cuda_supports_bfloat16
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
ROOT = os.getcwd(); OUT, AD = sys.argv[1], sys.argv[2]
dit = AceStepHandler(); print(dit.initialize_service(project_root=ROOT, config_path='acestep-v15-turbo', device='cuda', offload_to_cpu=False), flush=True)
print(dit.load_lora(AD), flush=True)
lm = LLMHandler()
for name, cap, bpm, lyr in json.load(open('/tmp/ace_songs.json')):
    t0 = time.time()
    p = GenerationParams(task_type='text2music', thinking=False, caption='vc2, ' + cap, lyrics=lyr, bpm=bpm, duration=110, vocal_language='ru', seed=7)
    r = generate_music(dit, lm, p, GenerationConfig(batch_size=1, audio_format='mp3', mp3_bitrate='192k', use_random_seed=False), save_dir='/tmp/ace_gen')
    if r.success and r.audios:
        dst = f'{OUT}/lora_{name}.mp3'; shutil.copy(r.audios[0]['path'], dst); print('✅', dst, '%.0f с' % (time.time() - t0), flush=True)
    else: print('⚠', name, getattr(r, 'error', r), flush=True)
'''

if __name__ == '__main__':
    t0 = time.time(); os.makedirs(WORK, exist_ok=True)
    KIN = None   # закрытый датасет Kaggle — пароль не нужен
    for dp, dn, fn in os.walk('/kaggle/input'):
        print('input:', dp, fn[:12], flush=True)
        if 'dataset.json' in fn and not KIN: KIN = os.path.join(dp, 'dataset.json')
    P = os.environ.get('P')
    if not P and not KIN:
        try:
            from kaggle_secrets import UserSecretsClient; P = UserSecretsClient().get_secret('KEYSTORE_PASS')
        except Exception: pass
    if not P and not KIN: raise SystemExit('нет данных: ни датасета Kaggle, ни пароля (Add-ons → Secrets → KEYSTORE_PASS)')
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv || true')
    if not os.path.isdir(R): sh(f'git clone -q --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {R}')
    sh(f'{sys.executable} -m pip install -q uv'); sh(f'cd {R} && {sys.executable} -m uv sync -q')
    UV = f'cd {R} && MPLBACKEND=Agg {sys.executable} -m uv run python'
    # 1) данные: 10 треков vc2 + разметка Gemini (текст с метками частей, описание, bpm, тональность)
    A = R + '/al_data/audio'; os.makedirs(A, exist_ok=True)
    if KIN:
        for f in glob.glob(os.path.dirname(KIN) + '/*.mp3'): shutil.copy(f, A)
        ds = json.load(open(KIN))
    else:
        for i in range(10):
            sh(f'wget -q -O /tmp/t.enc {RAW}/vprep-in/ace_ds/vc2_{i:02d}.mp3.enc'); dec('/tmp/t.enc', f'{A}/vc2_{i:02d}.mp3', P)
        sh(f'wget -q -O /tmp/ds.enc {RAW}/annot-out/dataset.json.enc'); dec('/tmp/ds.enc', '/tmp/ds_raw.json', P)
        ds = json.load(open('/tmp/ds_raw.json'))
    for s in ds['samples']:
        s['audio_path'] = f"{A}/{s['filename']}"; s['custom_tag'] = TAG; s['bpm'] = int(s.get('bpm') or 0) or None
    ds['metadata'] = {'custom_tag': TAG, 'tag_position': 'prepend', 'genre_ratio': 0}
    json.dump(ds, open('/tmp/ds.json', 'w'), ensure_ascii=False, indent=1)
    print('песен в разметке:', len(ds['samples']), flush=True)
    # 2) модели + подготовка тензоров
    sh(f'{UV} -c "from acestep.model_downloader import ensure_main_model; print(ensure_main_model())"')
    CK = R + '/checkpoints'; T = R + '/al_data/tensors'; O = R + '/al_data/out'   # внутри папки ACE: тренер не пускает пути вне неё
    sh(f'{UV} -m acestep.training_v2.cli.train_fixed --preprocess --dataset-json /tmp/ds.json --audio-dir {A} --tensor-output {T} '
       f'--checkpoint-dir {CK} --model-variant turbo --max-duration 240 --precision bf16')
    # 3) обучение LoKr (быстрый адаптер); если bf16 на T4 не пойдёт — fp32
    base = (f'-m acestep.training_v2.cli.train_fixed --checkpoint-dir {CK} --model-variant turbo --dataset-dir {T} --output-dir {O} '
            f'--yes --plain --adapter-type lokr --lr 0.03 --epochs {EPOCHS} --save-every {EVERY} --batch-size 1 --gradient-accumulation 4 --gradient-checkpointing')
    rc = sh(f'cd {R} && MPLBACKEND=Agg timeout {TLIM} {sys.executable} -m uv run python {base} --precision bf16', check=False)
    if rc not in (0, 124) and not glob.glob(O + '/checkpoints/*'):
        print('⚠ bf16 не пошёл, пробую fp32', flush=True); shutil.rmtree(O, ignore_errors=True)
        sh(f'cd {R} && MPLBACKEND=Agg timeout {TLIM} {sys.executable} -m uv run python {base} --precision fp32', check=False)
    cands = [d for d in [O + '/final'] + sorted(glob.glob(O + '/checkpoints/epoch_*'), key=lambda d: int(d.split('epoch_')[1].split('_')[0]), reverse=True)
             if os.path.exists(d + '/lokr_weights.safetensors')]
    print('адаптеры:', cands[:4], flush=True)
    if not cands: raise SystemExit('обучение не сохранило ни одного адаптера — смотри лог выше')
    AD = cands[0]
    shutil.make_archive(WORK + '/vc2_lokr', 'zip', AD)   # выход ноутбука закрытый; в GitHub он уходит зашифрованным
    # 4) пробные песни с дообученной моделью (те же 4 текста, что в пробе базовой модели)
    sh(f'wget -q -O /tmp/kace.py {RAW}/main/colab/kaggle_ace.py')
    import importlib.util
    sp = importlib.util.spec_from_file_location('kace', '/tmp/kace.py'); k = importlib.util.module_from_spec(sp); sp.loader.exec_module(k)
    json.dump(k.SONGS, open('/tmp/ace_songs.json', 'w'), ensure_ascii=False)
    open(R + '/gen_lora.py', 'w').write(GEN)
    sh(f'{UV} gen_lora.py {WORK} {AD}', check=False)
    print('готово за %.0f мин → Output: lora_*.mp3 и vc2_lokr.zip.enc (адаптер: %s)' % ((time.time() - t0) / 60, os.path.basename(AD)))
