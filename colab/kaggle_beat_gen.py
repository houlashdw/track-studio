"""Kaggle GPU: генерация битов уже обученным адаптером (без обучения).
Вход: выход ноутбука с *.zip адаптера (ksrc). Разные сиды на жанр, сила адаптера SCALES (0 = базовая модель для сравнения)."""
import os, sys, glob, json, subprocess, shutil, zipfile, time
R = '/tmp/ace'; W = '/kaggle/working'
SCALES = [float(x) for x in os.environ.get('SCALES', '0,0.5,1').split(',')]
NS = int(os.environ.get('NSEED', '1')); GEN = os.environ.get('GENRES', 'trap,drill,plugg,rage,phonk').split(',')
G = {'trap': ('dark melodic trap beat, heavy 808, rolling hi-hats, hard kick, crisp snare', 140, 'C# minor'),
     'drill': ('UK drill beat, sliding 808 bass, syncopated hi-hats, dark piano', 142, 'F minor'),
     'plugg': ('plugg beat, airy dreamy bells and soft synth melody, smooth bouncy 808, sparse drums, triplet hi-hats, mellow half-time feel', 150, 'E major'),
     'rage': ('rage beat, hyper aggressive distorted supersaw synth lead, blown-out distorted 808, fast hi-hats, energetic, dark futuristic', 160, 'G minor'),
     'phonk': ('drift phonk beat, loud cowbell melody, heavily distorted 808, memphis drum samples, aggressive', 130, 'A minor'),
     'jerk': ('jerk beat, fast bouncy drums, synth bass, playful', 150, 'D minor'),
     'pluggnb': ('pluggnb beat, smooth chords, bouncy 808, melodic', 140, 'F# major'),
     'detroit': ('detroit trap beat, hard punchy drums, bouncy 808, piano loop', 100, 'B minor'),
     'atlanta': ('Atlanta trap beat, hard 808, rolling triplet hi-hats, dark synths', 140, 'D# minor'),
     'lofi': ('lofi hip hop beat, dusty drums, warm keys, vinyl crackle', 85, 'A major')}


def sh(c, check=True):
    print('$', c, flush=True); return subprocess.run(c, shell=True, check=check).returncode


COVER = [float(x) for x in os.environ.get('COVER', '').split(',') if x]   # сила референса: 0.3 — только стиль, 0.7 — близко к оригиналу
REFS = {}
if COVER:
    for g in GEN:
        rs = sorted(glob.glob(f'/kaggle/input/**/ds/{g}_*.mp3', recursive=True))[:8]; REFS[g] = []
        for k, f in enumerate(rs[:int(os.environ.get('NSEED', '1'))]):
            o = f'/tmp/ref_{g}_{k}.mp3'; subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '30', '-t', '40', '-i', f, '-b:a', '192k', o]); REFS[g].append(o)
    print('референсы:', REFS, flush=True)
zs = sorted(glob.glob('/kaggle/input/**/*lokr*.zip', recursive=True) + glob.glob('/kaggle/input/**/*lora*.zip', recursive=True))
print('адаптеры:', zs, flush=True)
if not zs: raise SystemExit('нет адаптера во входе')
AD = '/tmp/adapter'; zipfile.ZipFile(zs[-1]).extractall(AD)
sh(f'git clone -q --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {R}')
sh(f'{sys.executable} -m pip install -q uv')
for i in range(6):
    if sh(f'cd {R} && {sys.executable} -m uv sync -q', check=False) == 0: break
    time.sleep(30)
os.environ['UV_NO_SYNC'] = '1'; os.environ['UV_FROZEN'] = '1'
code = f'''
import os, json, shutil, time
import sys; sys.path.insert(0, os.getcwd())
from acestep import gpu_config
gpu_config.cuda_supports_bfloat16 = lambda *a, **k: True
import acestep.core.generation.handler.memory_utils as _mu; _mu.cuda_supports_bfloat16 = gpu_config.cuda_supports_bfloat16
from acestep.model_downloader import ensure_main_model, ensure_dit_model
ensure_main_model(); ensure_dit_model('acestep-v15-turbo')
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
dit = AceStepHandler(); print(dit.initialize_service(project_root=os.getcwd(), config_path='acestep-v15-turbo', device='cuda', offload_to_cpu=False), flush=True)
lm = LLMHandler(); print(dit.load_lora({AD!r}), flush=True)
G = {json.dumps(G)}; GT = {os.environ.get('GTAG') == '1'}
for gi, g in enumerate({GEN!r}):
    cap, bpm, key = G[g]
    for sd in range({NS}):
     for sc in ({COVER!r} or {SCALES!r}):
        CV = {COVER!r}; REFS = {REFS!r}
        if CV and not REFS.get(g, [])[sd:sd + 1]: continue
        print(dit.set_lora_scale(1.0 if CV else sc), flush=True)
        kw = dict(task_type='cover', src_audio=REFS[g][sd], audio_cover_strength=sc) if CV else dict(task_type='text2music')
        p = GenerationParams(**kw, caption=((('tsbeat ' + g + ', ') if GT else 'tsbeat, ') if (sc > 0 or CV) else '') + cap + ', instrumental', lyrics='[Instrumental]', bpm=bpm, keyscale=key,
                             duration=60, vocal_language='unknown', seed=1000 + 37 * gi + 101 * sd, inference_steps=8, shift=3.0)
        r = generate_music(dit, lm, p, GenerationConfig(batch_size=1, audio_format='mp3', mp3_bitrate='192k', use_random_seed=False), save_dir='/tmp/g')
        if r.success and r.audios:
            d = '{W}/%s_%s_%d_s%02d.mp3' % ('cov' if CV else 'gen', g, sd, int(sc * 10)); shutil.copy(r.audios[0]['path'], d); print('OK', d, flush=True)
        else: print('FAIL', g, sc, getattr(r, 'error', ''), flush=True)
'''
open(R + '/gen_beats.py', 'w').write(code)
sh(f'cd {R} && MPLBACKEND=Agg {sys.executable} -m uv run python gen_beats.py', check=False)
print('готово', sorted(os.path.basename(f) for f in glob.glob(W + '/*.mp3')))
