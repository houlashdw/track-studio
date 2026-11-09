"""Скачивает бесплатные драм-киты (ссылки MediaFire / zip), режет и нормализует one-shot'ы → OGG + kits.json.
Запуск на раннере GitHub (там есть интернет): python fetch_kits.py out_dir"""
import os, re, sys, json, glob, zipfile, subprocess, urllib.request, numpy as np, soundfile as sf

KITS = [  # (имя, ссылка) — бесплатные киты со страниц soundpacks.com и GitHub
    ('supreme', 'https://www.mediafire.com/?hhperp91vejpy52'),
    ('tsk', 'https://www.mediafire.com/?6qawlkpdlh753t5'),
    ('trippy', 'https://www.mediafire.com/?eai2yh75cucniz6'),
    ('aplus', 'https://www.mediafire.com/?zgbl66e88nv24k6'),
    ('free808', 'https://github.com/GareBear99/Free-808-Producer-Kit/archive/refs/heads/main.zip'),
]
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36'}
CAP = {'kick': 14, 'snare': 12, 'clap': 10, 'hat': 12, 'ohat': 8, 'perc': 10, '808': 14, 'crash': 3}
LEN = {'kick': 1.0, 'snare': .9, 'clap': .9, 'hat': .45, 'ohat': 1.2, 'perc': .9, '808': 3.2, 'crash': 2.5}
SR = 44100


def get(url, dst):
    if 'mediafire.com' in url:
        qk = url.split('?')[-1]; direct = None
        for api in [f'https://www.mediafire.com/api/1.5/file/get_links.php?quick_key={qk}&link_type=direct_download&response_format=json',
                    f'https://www.mediafire.com/api/1.4/file/get_links.php?quick_key={qk}&link_type=direct_download&response_format=json']:
            try:
                j = json.loads(subprocess.run(['curl', '-sL', '-A', UA['User-Agent'], api], capture_output=True, text=True, timeout=60).stdout)
                direct = j['response']['links'][0].get('direct_download'); print('api', bool(direct), flush=True)
                if direct: break
            except Exception as e: print('api ⚠', e, flush=True)
        if not direct:
            html = subprocess.run(['curl', '-sL', '-A', UA['User-Agent'], '-H', 'Accept: text/html', '-H', 'Accept-Language: en-US,en', f'https://www.mediafire.com/file/{qk}'], capture_output=True, text=True, timeout=60).stdout
            m = re.search(r'href="(https?://download[^"]+)"', html); print('page', len(html), bool(m), html[:200].replace('\n', ' '), flush=True)
            direct = m and m.group(1)
        if not direct: raise RuntimeError('нет прямой ссылки MediaFire')
        subprocess.run(['curl', '-sL', '-A', UA['User-Agent'], '-o', dst, direct], check=True, timeout=900); return
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(dst, 'wb') as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)


def cat(p):
    q = p.lower().replace('\\', '/')
    if re.search(r'loop|bpm_synced|bass_loops|warped_loops|midi|risers|impacts|game_ui|mutants|growls|\.sf2', q): return None
    n = q.split('/')[-1]; d = '/'.join(q.split('/')[-3:])
    if re.search(r'open|\boh\b|_oh|oh_', n) and re.search(r'hat|hh|\boh', d): return 'ohat'
    if re.search(r'hi.?hat|hihat|\bhh|hats', d): return 'hat'
    if '808' in d and not re.search(r'snare|clap|hat|perc|cowbell', d): return '808'
    if re.search(r'clap|snap', d): return 'clap'
    if re.search(r'snare|snr|\brim', d): return 'snare'
    if re.search(r'kick|kik|\bbd\b', d): return 'kick'
    if re.search(r'cymbal|crash', d): return 'crash'
    if re.search(r'perc|shaker|tamb|cowbell|bell|tom', d): return 'perc'
    return None


def f0(x):
    seg = x[int(.06 * SR):int(.6 * SR)]
    if seg.size < 4096: seg = x
    S = np.abs(np.fft.rfft(seg * np.hanning(seg.size), 1 << 16)); fr = np.fft.rfftfreq(1 << 16, 1 / SR)
    m = (fr > 28) & (fr < 220); return float(fr[m][np.argmax(S[m])])


def proc(path, c):
    try: x, sr = sf.read(path, always_2d=True)
    except Exception: return None
    x = x.mean(1)
    if sr != SR:
        t = np.arange(int(len(x) * SR / sr)) * sr / SR; x = np.interp(t, np.arange(len(x)), x)
    pk = np.abs(x).max()
    if pk < 1e-4: return None
    on = np.argmax(np.abs(x) > pk * .02); x = x[max(0, on - 32):]
    env = np.abs(x) > pk * 10 ** (-55 / 20); last = len(env) - np.argmax(env[::-1]) if env.any() else len(x)
    x = x[:min(last, int(LEN[c] * SR))]
    if len(x) < int(.03 * SR): return None
    fl = min(len(x) // 4, int(.03 * SR)); x[-fl:] *= np.linspace(1, 0, fl)
    return x / np.abs(x).max() * 10 ** (-1 / 20)


if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); idx = {}; rep = []
    for ki, (name, url) in enumerate(KITS):
        z = f'/tmp/{name}.zip'; d = f'/tmp/k_{name}'
        try: get(url, z); zipfile.ZipFile(z).extractall(d)
        except Exception as e: print('⚠', name, e, flush=True); continue
        files = sorted(glob.glob(d + '/**/*.wav', recursive=True) + glob.glob(d + '/**/*.WAV', recursive=True))
        by = {}
        for p in files:
            if '__MACOSX' in p: continue
            c = cat(os.path.relpath(p, d))
            if c: by.setdefault(c, []).append(p)
        print(name, len(files), 'wav', {c: len(v) for c, v in by.items()}, flush=True)
        for c, L in by.items():
            L = [L[int(i * len(L) / min(CAP[c], len(L)))] for i in range(min(CAP[c], len(L)))]   # равномерно по киту
            n = 0
            for p in L:
                x = proc(p, c)
                if x is None: continue
                fn = f'{c}_{name}_{n:02d}.ogg'; n += 1
                sf.write('/tmp/t.wav', x.astype(np.float32), SR)
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', '/tmp/t.wav', '-c:a', 'libvorbis', '-q:a', '5', f'{out}/{fn}'], check=True)
                it = {'f': fn, 'src': name, 'fam': 'trap', 'len': round(len(x) / SR, 2), 'orig': os.path.relpath(p, d)[-80:]}
                if c in ('kick', '808'): it['hz'] = round(f0(x), 1)
                idx.setdefault(c, []).append(it)
    json.dump(idx, open(f'{out}/kits.json', 'w'), ensure_ascii=False, indent=0)
    print({c: len(v) for c, v in idx.items()}, 'MB %.1f' % (sum(os.path.getsize(f) for f in glob.glob(out + '/*.ogg')) / 1e6))
