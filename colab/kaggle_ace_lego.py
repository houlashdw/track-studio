"""Наш инструментал + голос ACE-Step: режим «lego» — ACE-Step дописывает вокальную дорожку поверх готового бита из приложения.
Ячейка (Internet On, GPU T4):
    !wget -q https://raw.githubusercontent.com/houlashdw/track-studio/main/colab/kaggle_ace_lego.py -O lego.py && python lego.py
Итог в Output: lego_<жанр>_mix.mp3 (наш бит + голос ACE) и lego_<жанр>_vocal.mp3 (только голос)."""
import os, sys, subprocess, json, time

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('lego_out')
R = '/tmp/ace'; RAW = 'https://raw.githubusercontent.com/houlashdw/track-studio'


def sh(c, check=True):
    print('$', c, flush=True); return subprocess.run(c, shell=True, check=check).returncode


BEATS = {   # жанр → (описание голоса, bpm)
    'ballad': ('soft emotional female vocal in Russian, intimate, breathy, melancholic, clear diction', 80),
    'trap': ('confident female rap vocal in Russian with melodic hook, rhythmic flow, slight autotune', 140),
    'dance': ('bright catchy female pop vocal in Russian, energetic, euphoric chorus, clear diction', 124),
}

GEN = r'''
import os, sys, json, time, shutil
sys.path.insert(0, os.getcwd())
from acestep import gpu_config
gpu_config.cuda_supports_bfloat16 = lambda *a, **k: True          # T4: bf16 вместо fp16 (иначе NaN)
import acestep.core.generation.handler.memory_utils as _mu; _mu.cuda_supports_bfloat16 = gpu_config.cuda_supports_bfloat16
from acestep.model_downloader import ensure_main_model, ensure_dit_model
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
ROOT = os.getcwd(); OUT = sys.argv[1]
print(ensure_main_model(), flush=True); print(ensure_dit_model('acestep-v15-base'), flush=True)
dit = AceStepHandler(); print(dit.initialize_service(project_root=ROOT, config_path='acestep-v15-base', device='cuda', offload_to_cpu=False), flush=True)
lm = LLMHandler()
for name, (cap, bpm, lyr) in json.load(open('/tmp/lego.json')).items():
    for take in (1, 2):
        t0 = time.time()
        p = GenerationParams(task_type='lego', src_audio=f'/tmp/beats/{name}.mp3', instruction='Generate the vocals track based on the audio context:',
                             caption=cap, lyrics=lyr, bpm=bpm, vocal_language='ru', repainting_start=0.0, repainting_end=-1, inference_steps=50, seed=100 + take)
        try:
            r = generate_music(dit, lm, p, GenerationConfig(batch_size=1, audio_format='wav', use_random_seed=False), save_dir='/tmp/lego_gen')
        except Exception as e:
            print('⚠', name, e, flush=True); continue
        if r.success and r.audios:
            dst = f'/tmp/lego_{name}_{take}.wav'; shutil.copy(r.audios[0]['path'], dst); print('✅', dst, '%.0f с' % (time.time() - t0), flush=True)
        else: print('⚠', name, getattr(r, 'error', r), flush=True)
'''

if __name__ == '__main__':
    t0 = time.time(); os.makedirs(WORK, exist_ok=True); os.makedirs('/tmp/beats', exist_ok=True)
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv || true')
    if not os.path.isdir(R): sh(f'git clone -q --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {R}')
    sh(f'{sys.executable} -m pip install -q uv'); sh(f'cd {R} && {sys.executable} -m uv sync -q')
    sh(f'wget -q -O /tmp/kace.py {RAW}/main/colab/kaggle_ace.py')
    import importlib.util
    sp = importlib.util.spec_from_file_location('kace', '/tmp/kace.py'); k = importlib.util.module_from_spec(sp); sp.loader.exec_module(k)
    L = {n: l for n, _, _, l in k.SONGS}
    for n in BEATS: sh(f'wget -q -O /tmp/beats/{n}.mp3 {RAW}/vprep-in/beats/{n}.mp3')
    json.dump({n: (c, b, L[n]) for n, (c, b) in BEATS.items()}, open('/tmp/lego.json', 'w'), ensure_ascii=False)
    open(R + '/gen_lego.py', 'w').write(GEN)
    sh(f'cd {R} && MPLBACKEND=Agg {sys.executable} -m uv run python gen_lego.py {WORK}', check=False)
    # сведение: наш бит + голос ACE (голос чуть впереди), плюс голос отдельно
    for n in BEATS:
        for take in (1, 2):
            v = f'/tmp/lego_{n}_{take}.wav'
            if not os.path.exists(v): continue
            sh(f'ffmpeg -v error -y -i {v} -b:a 192k {WORK}/lego_{n}_{take}_out.mp3', check=False)
            sh(f'ffmpeg -v error -y -i /tmp/beats/{n}.mp3 -i {v} -filter_complex "[0:a]volume=0.8[b];[1:a]volume=1.0[v];[b][v]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95" -b:a 192k {WORK}/lego_{n}_{take}_mix.mp3', check=False)
    print('готово за %.0f мин → Output: lego_*_out.mp3 (что выдал ACE) и lego_*_mix.mp3 (с нашим битом)' % ((time.time() - t0) / 60))
