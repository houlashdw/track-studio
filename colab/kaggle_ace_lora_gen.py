"""Пробные песни ACE-Step с дообученной частью vc2 при разной силе (0 / 0.4 / 0.7), с «умной» частью (LM) — чтобы понять, откуда шёпот и шум.
Адаптер берётся из выхода ноутбука ts-lora (kernel_sources)."""
import os, sys, subprocess, json, time, glob, zipfile
WORK = '/kaggle/working'; R = '/tmp/ace'; RAW = 'https://raw.githubusercontent.com/houlashdw/track-studio'
def sh(c, check=True): print('$', c, flush=True); return subprocess.run(c, shell=True, check=check).returncode
GEN = r'''
import os, sys, json, time, shutil
sys.path.insert(0, os.getcwd())
from acestep import gpu_config
gpu_config.cuda_supports_bfloat16 = lambda *a, **k: True
import acestep.core.generation.handler.memory_utils as _mu; _mu.cuda_supports_bfloat16 = gpu_config.cuda_supports_bfloat16
from acestep.model_downloader import ensure_main_model, ensure_lm_model
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
ROOT = os.getcwd(); CK = ROOT + '/checkpoints'; OUT, AD = sys.argv[1], sys.argv[2]
ensure_main_model(); ensure_lm_model('acestep-5Hz-lm-1.7B')
dit = AceStepHandler(); print(dit.initialize_service(project_root=ROOT, config_path='acestep-v15-turbo', device='cuda', offload_to_cpu=False), flush=True)
lm = LLMHandler(); print(lm.initialize(checkpoint_dir=CK, lm_model_path='acestep-5Hz-lm-1.7B', backend='pt', device='cuda', offload_to_cpu=False), flush=True)
print(dit.load_lora(AD), flush=True)
songs = {n: (c, b, l) for n, c, b, l in json.load(open('/tmp/ace_songs.json'))}
for name in ['ballad', 'dance']:
    cap, bpm, lyr = songs[name]
    for sc in [0.0, 0.4, 0.7]:
        print(dit.set_use_lora(sc > 0), dit.set_lora_scale(sc) if sc > 0 else '', flush=True)
        t0 = time.time()
        p = GenerationParams(task_type='text2music', thinking=True, caption=('vc2, ' if sc > 0 else '') + cap, lyrics=lyr, bpm=bpm, duration=110, vocal_language='ru', seed=11)
        r = generate_music(dit, lm, p, GenerationConfig(batch_size=1, audio_format='mp3', mp3_bitrate='192k', use_random_seed=False), save_dir='/tmp/g')
        if r.success and r.audios: shutil.copy(r.audios[0]['path'], f'{OUT}/s{int(sc*10)}_{name}.mp3'); print('✅', name, sc, '%.0f с' % (time.time() - t0), flush=True)
        else: print('⚠', name, sc, getattr(r, 'error', r), flush=True)
'''
if __name__ == '__main__':
    t0 = time.time()
    z = (glob.glob('/kaggle/input/**/vc2_lokr.zip', recursive=True) + [None])[0]; print('адаптер:', z, flush=True)
    if not z: raise SystemExit('нет vc2_lokr.zip во входах')
    AD = '/tmp/ace_ad'; zipfile.ZipFile(z).extractall(AD)
    if not os.path.isdir(R): sh(f'git clone -q --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {R}')
    sh(f'{sys.executable} -m pip install -q uv'); sh(f'cd {R} && {sys.executable} -m uv sync -q')
    sh(f'wget -q -O /tmp/kace.py {RAW}/main/colab/kaggle_ace.py')
    import importlib.util
    sp = importlib.util.spec_from_file_location('kace', '/tmp/kace.py'); k = importlib.util.module_from_spec(sp); sp.loader.exec_module(k)
    json.dump(k.SONGS, open('/tmp/ace_songs.json', 'w'), ensure_ascii=False)
    open(R + '/gen2.py', 'w').write(GEN)
    sh(f'cd {R} && MPLBACKEND=Agg {sys.executable} -m uv run python gen2.py {WORK} {AD}', check=False)
    print('готово за %.0f мин' % ((time.time() - t0) / 60))
