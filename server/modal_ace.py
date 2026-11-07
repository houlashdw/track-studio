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
    .pip_install('fastapi[standard]')
)
vol = modal.Volume.from_name('ts-ace', create_if_missing=True)
app = modal.App('track-studio-ace', image=image)
VOL = '/vol'


GPU = os.environ.get('TS_GPU', 'T4')


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
        cap = ('vc2, ' if voice > 0 else '') + q.get('caption', 'russian pop song, female vocal')
        seed = int(q.get('seed', -1))
        p = GenerationParams(task_type='text2music', thinking=True, caption=cap, lyrics=q.get('lyrics', ''), bpm=q.get('bpm') or None,
                             duration=float(q.get('duration', 110)), vocal_language=q.get('lang', 'ru'), inference_steps=8, shift=3.0, seed=seed)
        c = GenerationConfig(batch_size=1, audio_format='mp3', mp3_bitrate='192k', use_random_seed=seed < 0)
        r = generate_music(self.dit, self.lm, p, c, save_dir='/tmp/gen')
        if not (r.success and r.audios): raise RuntimeError(str(getattr(r, 'error', r)))
        return open(r.audios[0]['path'], 'rb').read()

    @modal.method()
    def generate(self, q: dict) -> bytes:
        return self._gen(q)

    @modal.fastapi_endpoint(method='POST', docs=False)
    def web(self, q: dict, x_key: Optional[str] = Header(default=None)):
        if not x_key or x_key != os.environ.get('TS_KEY'): raise HTTPException(status_code=401, detail='bad key')
        try: b = self._gen(q)
        except Exception as e: raise HTTPException(status_code=500, detail=str(e)[:300])
        return Response(content=b, media_type='audio/mpeg')


@app.local_entrypoint()
def main(out: str = 'test_out.mp3'):
    t0 = time.time()
    b = Ace().generate.remote({'lyrics': '[Verse]\nГород спит, а я не сплю опять\nЗвёзды падают в мою тетрадь\n\n[Chorus]\nМы горим, мы горим до утра\nНебо светит, начинается игра',
                               'caption': 'sad emotional russian pop ballad, soft piano, intimate female vocal', 'bpm': 80, 'duration': 45, 'seed': 11})
    open(out, 'wb').write(b); print('тест: %d КБ за %.0f с' % (len(b) // 1024, time.time() - t0))
