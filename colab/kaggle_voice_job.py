"""Kaggle (GPU T4): голос ACE «lego» для одного трека — замена AceLego с Modal.
Задание: зашифрованный архив (beat.mp3 + job.json со списком заданий lego) по ссылке JOB_URL, пароль JOB_PASS — оба в коде приватного ноутбука.
Адаптер голоса — приватный датасет Kaggle ts-adapters. Выход в /kaggle/working: take0.mp3, take1.mp3 …, done.json"""
import os, sys, subprocess, json, time

WORK = '/kaggle/working'; R = '/tmp/ace'; RAW = 'https://raw.githubusercontent.com/houlashdw/track-studio/main'
T0 = time.time()


def sh(c, check=True):
    print('$', c[:160], flush=True); return subprocess.run(c, shell=True, check=check).returncode


sh('nvidia-smi --query-gpu=name,memory.total --format=csv || true')
if not os.path.isdir(R): sh(f'git clone -q --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {R}')
sh(f'{sys.executable} -m pip install -q uv'); sh(f'cd {R} && {sys.executable} -m uv sync -q')
sh(f'cd {R} && {sys.executable} -m uv pip install -q --python {R}/.venv/bin/python demucs soundfile')
sh(f'wget -q -O {R}/lego_core.py {RAW}/server/lego_core.py')
sh(f'wget -q -O /tmp/job.enc "{os.environ["JOB_URL"]}"')
sh('openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -in /tmp/job.enc -out /tmp/job.tgz -pass env:JOB_PASS && mkdir -p /tmp/job && tar -xzf /tmp/job.tgz -C /tmp/job')
sh('mkdir -p /tmp/ad && for t in $(find /kaggle/input /tmp/job -name adapters.tar 2>/dev/null); do tar xf $t -C /tmp/ad; done; find /tmp/ad -name adapter_config.json || true', check=False)
print('установка %.0f с' % (time.time() - T0), flush=True)

RUN = r'''
import os, sys, json, time, glob
sys.path.insert(0, os.getcwd()); import lego_core
ad = glob.glob('/tmp/ad/**/adapter_config.json', recursive=True) + glob.glob('/kaggle/input/**/adapter_config.json', recursive=True)
ctx = lego_core.load(os.getcwd(), os.path.dirname(ad[0]) if ad else '')
jobs = json.load(open('/tmp/job/job.json')); res = []
for i, q in enumerate(jobs):
    t = time.time()
    try:
        open('%s/take%d.mp3' % (sys.argv[1], i), 'wb').write(lego_core.lego(ctx, q, '/tmp/job/beat.mp3')); res.append(dict(i=i, ok=True, t=round(time.time() - t)))
    except Exception as e:
        import traceback; traceback.print_exc(); res.append(dict(i=i, ok=False, err=str(e)[:300]))
json.dump(res, open(sys.argv[1] + '/done.json', 'w'))
print('готово', res, flush=True)
'''
open(f'{R}/run_job.py', 'w').write(RUN)
sh(f'cd {R} && MPLBACKEND=Agg {sys.executable} -m uv run python run_job.py {WORK}', check=False)
print('всего %.0f с' % (time.time() - T0), flush=True)
