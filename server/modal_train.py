"""Дообучение ACE-Step (LoRA битов) на GPU Modal — когда недельная квота Kaggle кончилась.
Данные: том ts-ace /data/<src> (ds/*.mp3 + dataset.json, залиты workflow modal cmd=train).
Итог: /train/<out>/ в томе — *.zip адаптера + пробные mp3 по жанрам."""
import os, sys, subprocess, glob, shutil
import modal

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
    .apt_install('wget', 'curl').pip_install('soundfile')
    .add_local_file(os.path.join(os.path.dirname(__file__), '..', 'colab', 'kaggle_ace_lora.py'), '/al.py')
)
vol = modal.Volume.from_name('ts-ace', create_if_missing=True)
app = modal.App('track-studio-train', image=image)
VOL = '/vol'
# torchcodec в образе не грузится (Could not load libtorchcodec_*.so) → torchaudio.load/save через soundfile
PATCH = '''
try:
    import torch, torchaudio, soundfile as sf
    def _load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, *a, **k):
        d, sr = sf.read(str(path), dtype='float32', always_2d=True, start=frame_offset, frames=num_frames)
        t = torch.from_numpy(d.T.copy()); return (t if channels_first else t.T), sr
    def _save(path, src, sample_rate, channels_first=True, *a, **k):
        x = src.detach().float().cpu().numpy(); sf.write(str(path), x.T if channels_first else x, sample_rate)
    torchaudio.load = _load; torchaudio.save = _save
except Exception as e: print('patch torchaudio:', e)
'''


@app.function(gpu=os.environ.get('TS_GPU_TRAIN', 'L40S'), volumes={VOL: vol}, timeout=6 * 3600)
def train(src: str, out: str, env: str):
    os.makedirs(VOL + '/checkpoints', exist_ok=True)
    if not os.path.islink(ACE + '/checkpoints'):
        shutil.rmtree(ACE + '/checkpoints', ignore_errors=True); os.symlink(VOL + '/checkpoints', ACE + '/checkpoints')
    W = f'{VOL}/train/{out}'; shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    os.makedirs('/patch', exist_ok=True); open('/patch/sitecustomize.py', 'w').write(PATCH)
    e = dict(os.environ, SYSPY='1', INP=f'{VOL}/data/{src}', WORK=W, PYTHONPATH='/patch')
    e.update(dict(kv.split('=', 1) for kv in env.split() if '=' in kv))
    f = sorted(glob.glob(f'{VOL}/data/{src}/*.mp3'))[0]
    if subprocess.run([sys.executable, '-c', f'import torchaudio; w, sr = torchaudio.load({f!r}); print("проба чтения mp3:", tuple(w.shape), sr)'], env=e).returncode:
        return 2
    rc = subprocess.run([sys.executable, '/al.py'], env=e).returncode
    for lg in glob.glob(ACE + '/**/sidestep*.log', recursive=True) + glob.glob('/root/**/sidestep*.log', recursive=True):
        print('==== лог', lg); print(open(lg, errors='replace').read()[-6000:])
    vol.commit()
    print('код', rc, 'итог:', sorted(os.path.basename(f) for f in glob.glob(W + '/*')))
    return rc


@app.local_entrypoint()
def main(src: str = 'beatprep2', out: str = 'beatlora4', env: str = ''):
    print('обучение →', train.remote(src, out, env))
