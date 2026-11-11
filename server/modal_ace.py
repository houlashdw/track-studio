"""Сервер «Трек Студии» на Modal: ACE-Step 1.5 (+ дообученная часть vc2) по запросу приложения.
POST /generate  (заголовок X-Key = секрет ts-key)  JSON:
  {"lyrics": "...", "caption": "...", "bpm": 120, "duration": 110, "voice": 1.0,   # voice: сила голоса vc2 (0 = без дообучения)
   "task": "text2music" | "lego", "beat_b64": "<mp3 нашего бита, для lego>", "seed": -1}
→ audio/mpeg (готовый трек).
Деплой: GitHub → workflow modal (cmd=deploy). Модели и адаптеры лежат в томе ts-ace (кэш между запусками)."""
import os, io, json, time, base64, glob, shutil
import modal
from typing import Optional
from fastapi import Header, HTTPException, Response

ACE = '/ace'
image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git', 'ffmpeg', 'libsndfile1')
    .pip_install('uv')
    .run_commands(
        f'git clone -q --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {ACE}',
        f'cd {ACE} && uv pip install --system --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 -e .',
    )
    .pip_install('fastapi[standard]', 'demucs==4.0.1')
)
vol = modal.Volume.from_name('ts-ace', create_if_missing=True)
app = modal.App('track-studio-ace', image=image)
VOL = '/vol'


GPU = os.environ.get('TS_GPU', 'T4')   # L4 на этом аккаунте не стартовал (303 по таймауту) — T4


@app.cls(gpu=GPU, volumes={VOL: vol}, secrets=[modal.Secret.from_name('ts-key')], timeout=900, scaledown_window=120)
class Ace:
    @modal.enter()
    def load(self):
        import sys
        os.makedirs(VOL + '/checkpoints', exist_ok=True)
        if not os.path.islink(ACE + '/checkpoints'):
            shutil.rmtree(ACE + '/checkpoints', ignore_errors=True); os.symlink(VOL + '/checkpoints', ACE + '/checkpoints')
        os.chdir(ACE); sys.path.insert(0, ACE)
        from acestep import gpu_config   # T4 (до Ampere): ACE сам берёт fp16 → NaN; bf16 работает
        gpu_config.cuda_supports_bfloat16 = lambda *a, **k: True
        import acestep.core.generation.handler.memory_utils as _mu; _mu.cuda_supports_bfloat16 = gpu_config.cuda_supports_bfloat16
        from acestep.model_downloader import ensure_main_model, ensure_lm_model
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        t0 = time.time(); ensure_main_model(); ensure_lm_model('acestep-5Hz-lm-1.7B'); vol.commit()
        self.dit = AceStepHandler(); print(self.dit.initialize_service(project_root=ACE, config_path='acestep-v15-turbo', device='cuda', offload_to_cpu=False))
        self.lm = LLMHandler(); print(self.lm.initialize(checkpoint_dir=ACE + '/checkpoints', lm_model_path='acestep-5Hz-lm-1.7B', backend='pt', device='cuda', offload_to_cpu=False))
        ad = sorted(glob.glob(VOL + '/adapters/turbo/*'))
        self.has_voice = False
        if ad:
            print(self.dit.load_lora(ad[-1])); self.has_voice = True
        print('загрузка %.0f с, голос vc2: %s' % (time.time() - t0, self.has_voice))

    def _gen(self, q):
        from acestep.inference import GenerationParams, GenerationConfig, generate_music
        voice = float(q.get('voice', 1.0)) if self.has_voice else 0.0
        if self.has_voice:
            self.dit.set_use_lora(voice > 0)
            if voice > 0: self.dit.set_lora_scale(voice)
        if not (q.get('lyrics') or '').strip(): q['lyrics'] = self._write(q.get('caption', ''), q.get('lang', 'ru'))['lyrics']   # нет текста — пишет сам
        cap = ('vc2, ' if voice > 0 else '') + q.get('caption', 'russian pop song, female vocal')
        seed = int(q.get('seed', -1))
        p = GenerationParams(task_type='text2music', thinking=True, caption=cap, lyrics=q.get('lyrics', ''), bpm=q.get('bpm') or None,
                             duration=float(q.get('duration', 110)), vocal_language=q.get('lang', 'ru'), inference_steps=8, shift=3.0, seed=seed)
        c = GenerationConfig(batch_size=1, audio_format='mp3', mp3_bitrate='192k', use_random_seed=seed < 0)
        r = generate_music(self.dit, self.lm, p, c, save_dir='/tmp/gen')
        if not (r.success and r.audios): raise RuntimeError(str(getattr(r, 'error', r)))
        return open(r.audios[0]['path'], 'rb').read()

    def _write(self, caption, lang='ru'):
        """текст песни по описанию: «простой режим» ACE (его языковая модель пишет куплеты/припев с метками)"""
        from acestep.inference import create_sample
        r = create_sample(self.lm, query=caption, instrumental=False, vocal_language=lang)
        return {'lyrics': getattr(r, 'lyrics', '') or '', 'caption': getattr(r, 'caption', '') or caption, 'bpm': getattr(r, 'bpm', None)}

    @modal.method()
    def write(self, q: dict) -> dict:
        return self._write(q.get('caption', ''), q.get('lang', 'ru'))

    @modal.method()
    def generate(self, q: dict) -> bytes:
        return self._gen(q)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        api = FastAPI(docs_url=None, redoc_url=None)
        # приложение открывается с https://appassets.androidplatform.net — нужен CORS (ключ всё равно обязателен)
        api.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['POST', 'GET', 'OPTIONS'], allow_headers=['*'])

        @api.get('/ping')
        def ping(): return {'ok': True, 'voice': self.has_voice}

        @api.post('/generate')
        async def gen(request: Request, x_key: Optional[str] = Header(default=None)):
            if not x_key or x_key != os.environ.get('TS_KEY'): raise HTTPException(status_code=401, detail='bad key')
            q = await request.json()
            try: b = self._gen(q)
            except Exception as e: raise HTTPException(status_code=500, detail=str(e)[:300])
            return Response(content=b, media_type='audio/mpeg')
        @api.post('/lyrics')
        async def lyrics(request: Request, x_key: Optional[str] = Header(default=None)):
            if not x_key or x_key != os.environ.get('TS_KEY'): raise HTTPException(status_code=401, detail='bad key')
            q = await request.json()
            try: return self._write(q.get('caption', ''), q.get('lang', 'ru'))
            except Exception as e: raise HTTPException(status_code=500, detail=str(e)[:300])

        @api.post('/lego')   # долго (минуты) → задание; приложение спрашивает /job/<id>
        async def lego(request: Request, x_key: Optional[str] = Header(default=None)):
            if not x_key or x_key != os.environ.get('TS_KEY'): raise HTTPException(status_code=401, detail='bad key')
            q = await request.json()
            if not q.get('beat_b64'): raise HTTPException(status_code=400, detail='нет бита')
            return {'id': AceLego().lego.spawn(q).object_id}

        @api.get('/job/{jid}')
        def job(jid: str, x_key: Optional[str] = Header(default=None), k: Optional[str] = None):
            key = x_key or k
            if not key or key != os.environ.get('TS_KEY'): raise HTTPException(status_code=401, detail='bad key')
            try: b = modal.FunctionCall.from_id(jid).get(timeout=0)
            except TimeoutError: return Response(status_code=202, content=b'{"state":"running"}', media_type='application/json')
            except Exception as e: raise HTTPException(status_code=500, detail=str(e)[:300])
            return Response(content=b, media_type='audio/mpeg')
        return api


@app.cls(gpu=GPU, volumes={VOL: vol}, timeout=1200, scaledown_window=120)
class AceLego:
    """наш бит (из приложения) → ACE-Step «lego» допевает вокал (полная модель base) → Demucs вырезает только голос → приложение кладёт его на свой бит"""
    @modal.enter()
    def load(self):
        import sys, torch
        os.makedirs(VOL + '/checkpoints', exist_ok=True); os.environ['TORCH_HOME'] = VOL + '/torch'
        if not os.path.islink(ACE + '/checkpoints'):
            shutil.rmtree(ACE + '/checkpoints', ignore_errors=True); os.symlink(VOL + '/checkpoints', ACE + '/checkpoints')
        os.chdir(ACE); sys.path.insert(0, ACE)
        from acestep import gpu_config
        gpu_config.cuda_supports_bfloat16 = lambda *a, **k: True
        import acestep.core.generation.handler.memory_utils as _mu; _mu.cuda_supports_bfloat16 = gpu_config.cuda_supports_bfloat16
        from acestep.model_downloader import ensure_main_model, ensure_dit_model
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        t0 = time.time(); ensure_main_model(); ensure_dit_model('acestep-v15-base')
        self.dit = AceStepHandler(); print(self.dit.initialize_service(project_root=ACE, config_path='acestep-v15-base', device='cuda', offload_to_cpu=False))
        self.lm = LLMHandler()
        ad = sorted(glob.glob(VOL + '/adapters/base/*')); self.has_voice = False
        if ad: print(self.dit.load_lora(ad[-1])); self.has_voice = True
        from demucs.pretrained import get_model
        self.sep = get_model('htdemucs').cuda().eval(); vol.commit()
        print('lego загрузка %.0f с, голос vc2: %s' % (time.time() - t0, self.has_voice))

    @modal.method()
    def stems(self, mp3: bytes) -> dict:
        """инструментал → дорожки Demucs (drums/bass/other) в mp3 — для разбора паттернов и звуков"""
        import subprocess, numpy as np, torch, soundfile as sf
        from demucs.apply import apply_model
        open('/tmp/in.mp3', 'wb').write(mp3); subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', '/tmp/in.mp3', '-ac', '2', '-ar', '44100', '/tmp/in.wav'], check=True)
        x, sr = sf.read('/tmp/in.wav', dtype='float32', always_2d=True)
        with torch.no_grad(): st = apply_model(self.sep, torch.from_numpy(x.T.copy())[None].cuda(), shifts=1, split=True, overlap=0.25)[0].cpu().numpy()
        out = {}
        for i, n in enumerate(self.sep.sources):
            if n == 'vocals': continue
            sf.write('/tmp/s.wav', st[i].T, sr); subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', '/tmp/s.wav', '-b:a', '128k', '/tmp/s.mp3'], check=True); out[n] = open('/tmp/s.mp3', 'rb').read()
        return out

    @modal.method()
    def lego(self, q: dict) -> bytes:
        import subprocess, numpy as np, torch, soundfile as sf
        from acestep.inference import GenerationParams, GenerationConfig, generate_music
        from demucs.apply import apply_model
        open('/tmp/beat.mp3', 'wb').write(base64.b64decode(q['beat_b64']))
        ref = None   # референс тембра: только наш же ACE-голос (понравившийся прошлый вокал), не запись реального человека
        if q.get('ref_b64'): open('/tmp/vref.mp3', 'wb').write(base64.b64decode(q['ref_b64'])); ref = '/tmp/vref.mp3'
        voice = float(q.get('voice', 1.0)) if self.has_voice else 0.0
        if self.has_voice:
            self.dit.set_use_lora(voice > 0)
            if voice > 0: self.dit.set_lora_scale(voice)
        seed = int(q.get('seed', -1))
        p = GenerationParams(task_type='lego', src_audio='/tmp/beat.mp3', instruction='Generate the vocals track based on the audio context:',
                             caption=('vc2, ' if voice > 0 else '') + q.get('caption', 'female vocal in Russian'), lyrics=q.get('lyrics', ''), bpm=q.get('bpm') or None,
                             vocal_language=q.get('lang', 'ru'), repainting_start=0.0, repainting_end=-1, inference_steps=int(q.get('steps', 30)), seed=seed, reference_audio=ref)
        r = generate_music(self.dit, self.lm, p, GenerationConfig(batch_size=1, audio_format='wav', use_random_seed=seed < 0), save_dir='/tmp/lg')
        if not (r.success and r.audios): raise RuntimeError(str(getattr(r, 'error', r)))
        if q.get('full'):   # «наш бит, сведённый ACE»: весь ответ ACE (наш бит в его звучании + голос) — склеено как у ACE
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', r.audios[0]['path'], '-b:a', '192k', '/tmp/full.mp3'], check=True); return open('/tmp/full.mp3', 'rb').read()
        x, sr = sf.read(r.audios[0]['path'], dtype='float32', always_2d=True)
        if sr != 44100:
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', r.audios[0]['path'], '-ar', '44100', '/tmp/lg44.wav'], check=True); x, sr = sf.read('/tmp/lg44.wav', dtype='float32', always_2d=True)
        if x.shape[1] == 1: x = np.repeat(x, 2, 1)
        with torch.no_grad():
            st = apply_model(self.sep, torch.from_numpy(x.T.copy())[None].cuda(), shifts=1, split=True, overlap=0.25)[0]
        v = st[self.sep.sources.index('vocals')].cpu().numpy().T
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
            self.last_lag = lag / 44.1
        except Exception as e: print('сдвиг не вышел', e, flush=True)
        sf.write('/tmp/voc.wav', v, sr)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', '/tmp/voc.wav', '-b:a', '192k', '/tmp/voc.mp3'], check=True)
        return open('/tmp/voc.mp3', 'rb').read()


@app.local_entrypoint()
def main(out: str = 'test_out.mp3'):
    t0 = time.time()
    b = Ace().generate.remote({'lyrics': '[Verse]\nГород спит, а я не сплю опять\nЗвёзды падают в мою тетрадь\n\n[Chorus]\nМы горим, мы горим до утра\nНебо светит, начинается игра',
                               'caption': 'sad emotional russian pop ballad, soft piano, intimate female vocal', 'bpm': 80, 'duration': 45, 'seed': 11})
    open(out, 'wb').write(b); print('тест: %d КБ за %.0f с' % (len(b) // 1024, time.time() - t0))


STYLE_TESTS = [   # проверка, слушается ли ACE слов про подачу/тембр/высоту/перегруз (одни и те же текст и сид)
    ('rap_trap', 'modern russian trap, hard 808, rolling hi-hats, confident female rap vocal in Russian, rhythmic fast flow, punchy delivery'),
    ('rap_dark_low', 'dark russian drill, sliding 808, low husky female rap vocal in Russian, aggressive, deep voice'),
    ('pop_high_airy', 'dreamy russian pop, bright synths, high-pitched airy breathy female vocal in Russian, soft and gentle'),
    ('pop_powerful', 'russian pop rock anthem, big drums, powerful belting female vocal in Russian, strong chest voice'),
    ('pop_distorted', 'russian alt rock, distorted guitars, gritty distorted saturated female vocal in Russian, raw and edgy'),
    ('pop_clean', 'russian pop ballad, piano, clean clear polished female vocal in Russian, crisp diction'),
]
LYR_RAP = '[Verse]\nЯ на районе, свет фонарей\nВсё, что хотела, теперь у дверей\nМама, не плачь, я стала сильней\nДеньги не главное, главное — с ней\n\n[Hook]\nНочью не сплю, я пишу этот трек\nКаждая строчка — один мой побег'
LYR_POP = '[Verse]\nГород спит, а я не сплю опять\nЗвёзды падают в мою тетрадь\n\n[Chorus]\nМы горим, мы горим до утра\nНебо светит, начинается игра'


@app.local_entrypoint()
def styles(outdir: str = 'styles_out', voice: float = 1.0):
    os.makedirs(outdir, exist_ok=True); a = Ace()
    for name, cap in STYLE_TESTS:
        t0 = time.time()
        b = a.generate.remote({'caption': cap, 'lyrics': LYR_RAP if name.startswith('rap') else LYR_POP, 'duration': 40, 'seed': 21, 'voice': voice})
        open(f'{outdir}/{name}_v{int(voice * 10)}.mp3', 'wb').write(b); print(name, '%.0f с' % (time.time() - t0), flush=True)


GENRES = {   # жанр приложения → описание для ACE (инструментал) и темп
    'trap': ('modern trap instrumental, hard 808 bass, rolling hi-hats, snare on 3, dark synth melody', 140),
    'drill': ('UK drill instrumental, sliding 808 bass, syncopated hi-hats, dark piano', 142),
    'plugg': ('plugg instrumental, soft bells, airy synth pads, bouncy 808, light hi-hats', 150),
    'rage': ('rage trap instrumental, distorted supersaw synth lead, heavy 808, aggressive', 155),
    'phonk': ('drift phonk instrumental, cowbell melody, distorted 808, aggressive drums', 130),
    'pop': ('modern pop instrumental, punchy drums, bright synths, warm bass, catchy chords', 118),
    'house': ('deep house instrumental, four on the floor kick, offbeat hi-hats, groovy bassline, chords', 124),
    'rock': ('pop rock instrumental, live drums, distorted electric guitars, bass guitar', 135),
    'lofi': ('lofi hip hop instrumental, dusty drums, jazzy electric piano chords, vinyl crackle, mellow bass', 85),
    'rnb': ('modern r&b instrumental, smooth electric piano, deep sub bass, crisp snaps, lush pads', 95),
}


@app.local_entrypoint()
def instr(outdir: str = 'instr_out', per: int = 4):
    """учебный набор для нашего инструментала: ACE делает инструменталы по жанрам → Demucs раскладывает на ударные/бас/остальное"""
    import json
    os.makedirs(outdir, exist_ok=True); a = Ace(); lg = AceLego(); meta = {}
    for g, (cap, bpm) in GENRES.items():
        for k in range(per):
            t0 = time.time()
            try:
                b = a.generate.remote({'caption': cap, 'lyrics': '[Instrumental]', 'bpm': bpm, 'duration': 30, 'seed': 1000 + k, 'voice': 0})
                st = lg.stems.remote(b)
            except Exception as e: print(g, k, 'ошибка', e, flush=True); continue
            open(f'{outdir}/{g}_{k}_mix.mp3', 'wb').write(b)
            for n, v in st.items(): open(f'{outdir}/{g}_{k}_{n}.mp3', 'wb').write(v)
            meta[f'{g}_{k}'] = {'genre': g, 'bpm': bpm, 'caption': cap}; print(g, k, '%.0f с' % (time.time() - t0), flush=True)
    json.dump(meta, open(f'{outdir}/meta.json', 'w'), indent=1)
