"""Проба ACE-Step 1.5 (открытый генератор песен, MIT) на русском — на бесплатной видеокарте Kaggle.
Ячейка ноутбука (Internet: On, Accelerator: GPU T4):
    !wget -q https://raw.githubusercontent.com/houlashdw/track-studio/main/colab/kaggle_ace.py -O ace.py && python ace.py
Итог: /kaggle/working/ace_*.mp3 (вкладка Output)."""
import os, sys, subprocess, json, time

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('ace_out')
R = '/tmp/ace'


def sh(c):
    print('$', c, flush=True); subprocess.run(c, shell=True, check=True)


SONGS = [
    ('ballad', 'sad emotional russian pop ballad, soft piano, warm pads, gentle drums, intimate female vocal, melancholic, night city', 80, """[Verse 1]
Город спит, а я не сплю опять
Звёзды падают в мою тетрадь
Я иду по улице одна
Сердце бьётся, ночь моя длинна

[Chorus]
Мы горим, мы горим до утра
Небо светит, начинается игра
Ты со мной, и больше нет преград
Этот голос не вернуть назад

[Verse 2]
Фонари рисуют на стене
Тени тех, кто не пришёл ко мне
Я пишу тебе, но не отправлю
Всё, что было, в памяти оставлю

[Chorus]
Мы горим, мы горим до утра
Небо светит, начинается игра
Ты со мной, и больше нет преград
Этот голос не вернуть назад"""),
    ('trap', 'modern russian trap, hard 808 bass, rolling hi-hats, dark synth melody, confident female rap vocal with melodic hook, autotune', 140, """[Intro]
Йе, а

[Verse 1]
Я на районе, свет фонарей
Всё, что хотела, теперь у дверей
Мама, не плачь, я стала сильней
Деньги не главное, главное — с ней

[Hook]
Ночью не сплю, я пишу этот трек
Каждая строчка — один мой побег
Город горит, а внутри только снег
Я не сдаюсь, я сильнее, чем все

[Verse 2]
Холод в груди, но я не стою
Всё, что моё, я сама заберу
Сотни шагов по пустому двору
Завтра проснусь, и я снова в бою

[Hook]
Ночью не сплю, я пишу этот трек
Каждая строчка — один мой побег
Город горит, а внутри только снег
Я не сдаюсь, я сильнее, чем все"""),
    ('dance', 'upbeat russian dance pop, four on the floor, bright synths, funky bass, catchy female vocal, summer, euphoric chorus', 124, """[Verse 1]
Лето в окнах, музыка в груди
Всё плохое где-то позади
Мы танцуем прямо до утра
Это наша лучшая пора

[Pre-Chorus]
Ближе, ближе, громче этот бит

[Chorus]
Лови меня, лови этот момент
Мы вместе, и не нужен комплимент
Танцуй со мной, пока горит рассвет
Нам вместе никаких запретов нет

[Verse 2]
Солнце красит улицы в огонь
Ты держи меня, держи ладонь
Мы летим над крышами домов
Без ненужных и красивых слов

[Chorus]
Лови меня, лови этот момент
Мы вместе, и не нужен комплимент
Танцуй со мной, пока горит рассвет
Нам вместе никаких запретов нет"""),
    ('rock', 'energetic russian pop rock, distorted electric guitars, live drums, driving bass, powerful female vocal, anthem chorus', 150, """[Verse 1]
Сквозь бетон и провода
Я иду туда, где не была
Ветер бьёт в лицо, но я стою
Песню громкую свою пою

[Chorus]
Это мой огонь, это мой полёт
Кто со мной — тот вместе с ним поёт
Пусть весь мир вокруг сойдёт с ума
Я свою дорогу выбрала сама

[Bridge]
Выше, выше, выше облаков

[Chorus]
Это мой огонь, это мой полёт
Кто со мной — тот вместе с ним поёт
Пусть весь мир вокруг сойдёт с ума
Я свою дорогу выбрала сама"""),
]

GEN = r'''
import os, sys, json, time, shutil, glob
sys.path.insert(0, os.getcwd())
from acestep.model_downloader import ensure_main_model, ensure_lm_model
from acestep import gpu_config
# T4 (до Ampere): ACE-Step сам берёт float16 → переполнение, NaN, тишина. bfloat16 на T4 работает (LM уже в нём) и не переполняется.
gpu_config.cuda_supports_bfloat16 = lambda *a, **k: os.environ.get('ACE_DT', 'bf16') == 'bf16'
import acestep.core.generation.handler.memory_utils as _mu; _mu.cuda_supports_bfloat16 = gpu_config.cuda_supports_bfloat16
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
ROOT = os.getcwd(); CK = os.path.join(ROOT, 'checkpoints'); OUT = sys.argv[1]; LM = os.environ.get('ACE_LM', 'acestep-5Hz-lm-1.7B')
print(ensure_main_model(), flush=True); print(ensure_lm_model(LM), flush=True)
dit = AceStepHandler(); print(dit.initialize_service(project_root=ROOT, config_path='acestep-v15-turbo', device='cuda', offload_to_cpu=False), flush=True)
lm = LLMHandler(); THINK = True
try: print(lm.initialize(checkpoint_dir=CK, lm_model_path=LM, backend='pt', device='cuda', offload_to_cpu=False), flush=True)
except Exception as e: print('⚠ LM не завёлся, без него:', e, flush=True); THINK = False
for name, cap, bpm, lyr in json.load(open('/tmp/ace_songs.json')):
    for take in (1, 2):
        t0 = time.time()
        c = GenerationConfig(batch_size=1, audio_format='mp3', mp3_bitrate='192k')
        try:
            r = generate_music(dit, lm, GenerationParams(task_type='text2music', thinking=THINK, caption=cap, lyrics=lyr, bpm=bpm, duration=110, vocal_language='ru'), c, save_dir='/tmp/ace_gen')
        except Exception as e:
            print('⚠ с LM упало, пробую без него:', e, flush=True); THINK = False
            r = generate_music(dit, lm, GenerationParams(task_type='text2music', thinking=False, caption=cap, lyrics=lyr, bpm=bpm, duration=110, vocal_language='ru'), c, save_dir='/tmp/ace_gen')
        if r.success and r.audios:
            for a in r.audios:
                dst = f'{OUT}/ace_{name}_{take}.mp3'; shutil.copy(a['path'], dst); print('✅', dst, '%.0f с' % (time.time() - t0), flush=True)
        else:
            print('⚠', name, getattr(r, 'error', r), flush=True)
'''

if __name__ == '__main__':
    t0 = time.time(); os.makedirs(WORK, exist_ok=True)
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv || true')
    if not os.path.isdir(R):
        sh(f'git clone -q --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {R}')
    sh(f'{sys.executable} -m pip install -q uv')
    sh(f'cd {R} && {sys.executable} -m uv sync -q')
    json.dump(SONGS, open('/tmp/ace_songs.json', 'w'), ensure_ascii=False)
    open(R + '/gen_ru.py', 'w').write(GEN)
    env = 'MPLBACKEND=Agg'
    sh(f'cd {R} && {env} {sys.executable} -m uv run python gen_ru.py {WORK}')
    print('готово за %.0f мин → вкладка Output, файлы ace_*.mp3' % ((time.time() - t0) / 60))
