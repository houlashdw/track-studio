"""Трек без Modal: бит и сведение — здесь (раннер GitHub, Reaper на CPU), голос ACE — на бесплатной видеокарте Kaggle.
python3 tools/ktrack.py <pack_dir> <out_dir> <seed> [genre] [lyrics.txt]
Окружение: KAGGLE_USERNAME/KAGGLE_KEY (секреты), PUSH_URL — адрес git для ветки с зашифрованным заданием."""
import os, sys, json, base64, time, subprocess, tempfile, secrets, glob, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'server')); sys.path.insert(0, os.path.join(ROOT, 'server', 'daw'))
import modal_ace, track

KU = os.environ['KAGGLE_USERNAME']; KREF = KU + '/ts-voice'; REPO = 'houlashdw/track-studio'


def run(c, **kw): return subprocess.run(c, shell=isinstance(c, str), check=True, text=True, capture_output=True, **kw).stdout


def kaggle_lego_many(qs):
    """все дубли голоса — одним запуском ноутбука Kaggle (по очереди на одной T4)"""
    t0 = time.time(); d = tempfile.mkdtemp()
    open(d + '/beat.mp3', 'wb').write(base64.b64decode(qs[0]['beat_b64']))
    json.dump([{k: v for k, v in q.items() if k != 'beat_b64'} for q in qs], open(d + '/job.json', 'w'), ensure_ascii=False)
    fl = ['beat.mp3', 'job.json']
    if os.path.exists('/tmp/adapters.tar'): shutil.copy('/tmp/adapters.tar', d); fl.append('adapters.tar')   # адаптер голоса едет внутри задания (датасет к ноутбуку подвешивал очередь)
    run(['tar', '-czf', d + '/job.tgz', '-C', d] + fl)
    P = secrets.token_hex(24)   # пароль задания живёт только в коде приватного ноутбука и в памяти этого процесса
    run(['openssl', 'enc', '-aes-256-cbc', '-pbkdf2', '-iter', '200000', '-salt', '-in', d + '/job.tgz', '-out', d + '/job.enc', '-pass', 'pass:' + P])
    g = d + '/git'; os.makedirs(g); shutil.copy(d + '/job.enc', g)
    run('git init -q -b kjob && git config user.name gha && git config user.email gha@users.noreply.github.com && git add -A && git commit -q -m job', cwd=g)
    run(['git', 'push', '-q', '-f', os.environ['PUSH_URL'], 'kjob'], cwd=g); sha = run('git rev-parse HEAD', cwd=g).strip()
    url = f'https://raw.githubusercontent.com/{REPO}/{sha}/job.enc'
    k = d + '/k'; os.makedirs(k)
    open(k + '/main.py', 'w').write('import os; os.environ.update(JOB_URL=%r, JOB_PASS=%r)\n' % (url, P) + open(os.path.join(ROOT, 'colab', 'kaggle_voice_job.py')).read())
    json.dump({'id': KREF, 'title': 'ts-voice', 'code_file': 'main.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
               'enable_gpu': True, 'enable_internet': True, 'accelerator': 'NvidiaTeslaT4', 'dataset_sources': [],
               'competition_sources': [], 'kernel_sources': []}, open(k + '/kernel-metadata.json', 'w'))
    def push():
        try: print(run(['kaggle', 'kernels', 'push', '-p', k, '--accelerator', 'NvidiaTeslaT4'])[-300:], flush=True)
        except subprocess.CalledProcessError: print(run(['kaggle', 'kernels', 'push', '-p', k])[-300:], flush=True)
    push(); time.sleep(30); st = ''; tq = time.time(); repushed = 0
    while time.time() - t0 < 40 * 60:
        try: st = run(['kaggle', 'kernels', 'status', KREF]).strip().splitlines()[-1]
        except subprocess.CalledProcessError as e: st = 'status? ' + (e.stdout or '')[-100:]
        print('%4.0f с  %s' % (time.time() - t0, st), flush=True)
        if any(x in st.lower() for x in ('complete', 'error', 'cancel')): break
        if 'queued' in st.lower() and time.time() - tq > 360 and repushed < 2: print('очередь затянулась — отправляю ноутбук заново', flush=True); push(); tq = time.time(); repushed += 1
        elif 'queued' not in st.lower(): tq = time.time()
        time.sleep(20)
    o = d + '/out'; os.makedirs(o); subprocess.run(['kaggle', 'kernels', 'output', KREF, '-p', o], capture_output=True, text=True)
    for lg in glob.glob(o + '/*.log'): print(open(lg, errors='replace').read()[-2500:], flush=True)
    takes = [open(f, 'rb').read() for f in sorted(glob.glob(o + '/take*.mp3'))]
    print('Kaggle: дублей %d за %.0f с' % (len(takes), time.time() - t0), flush=True)
    return takes


if __name__ == '__main__':
    pack, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(out, exist_ok=True)
    q = dict(seed=seed, genre=sys.argv[4] if len(sys.argv) > 4 else 'auto', lang='auto')
    if len(sys.argv) > 5 and os.path.exists(sys.argv[5]): q['lyrics'] = open(sys.argv[5]).read()
    work = tempfile.mkdtemp(prefix='trk')
    mp3, info, V, seed = modal_ace.build_track(q, kaggle_lego_many, pack, os.path.join(ROOT, 'app', 'smp'), work, track)
    open(f'{out}/track_s{seed}.mp3', 'wb').write(mp3)
    os.makedirs(out + '/st', exist_ok=True)
    for n in ('beat.mp3', 'vox_raw.mp3', 'vox_fx.mp3', 'mix.log'):
        if os.path.exists(os.path.join(work, n)): shutil.copy(os.path.join(work, n), f'{out}/st/s{seed}_{n}')
    json.dump(dict(info, voice=V['timbre'], lang=V['lang']), open(f'{out}/st/s{seed}_info.json', 'w'), ensure_ascii=False, indent=1)
    print('готово:', f'{out}/track_s{seed}.mp3', flush=True)
