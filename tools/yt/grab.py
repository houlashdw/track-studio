"""Бесплатные биты «free for profit» с YouTube по жанрам → mp3 + meta.json (для дообучения ACE-Step).
python grab.py out N  (жанры — ниже; берём только ролики, где в названии есть free/for profit и нет «tagged»/«prod by» платных тегов)"""
import os, sys, json, subprocess, re
OUT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 10; os.makedirs(OUT, exist_ok=True)
GEN = os.environ.get('GENRES', 'trap,drill,plugg,rage,phonk,jerk,pluggnb,detroit,hyperpop,lofi').split(',')
Q = {'trap': 'free for profit dark trap beat', 'drill': 'free for profit drill type beat', 'plugg': 'free for profit plugg type beat',
     'rage': 'free for profit rage type beat', 'phonk': 'free for profit phonk beat', 'jerk': 'free for profit jerk type beat',
     'pluggnb': 'free for profit pluggnb type beat', 'detroit': 'free for profit detroit type beat', 'hyperpop': 'free for profit hyperpop type beat',
     'lofi': 'free for profit lofi hip hop beat', 'atlanta': 'free for profit atlanta trap type beat', 'rnb': 'free for profit rnb type beat'}
meta = json.load(open(OUT + '/meta.json')) if os.path.exists(OUT + '/meta.json') else {}
for g in GEN:
    q = Q.get(g, f'free for profit {g} type beat')
    r = subprocess.run(['yt-dlp', '--flat-playlist', '-J', f'ytsearch{N * 3}:{q}'], capture_output=True, text=True)
    if r.returncode: print('⚠ поиск', g, r.stderr[-400:], flush=True); continue
    ents = json.loads(r.stdout).get('entries', []); got = 0
    for e in ents:
        if got >= N: break
        t = (e.get('title') or ''); d = e.get('duration') or 0; vid = e.get('id')
        if not re.search(r'free', t, re.I) or not (80 <= d <= 330) or vid in meta: continue
        dst = f'{OUT}/{g}_{vid}.mp3'
        r2 = subprocess.run(['yt-dlp', '-q', '--no-warnings', '-f', 'bestaudio', '-x', '--audio-format', 'mp3', '--audio-quality', '192K', '-o', f'{OUT}/{g}_{vid}.%(ext)s', f'https://www.youtube.com/watch?v={vid}'], capture_output=True, text=True)
        if r2.returncode or not os.path.exists(dst): print('⚠', vid, r2.stderr[-300:], flush=True); continue
        meta[vid] = {'genre': g, 'title': t, 'channel': e.get('channel') or e.get('uploader'), 'dur': d, 'file': os.path.basename(dst)}; got += 1
        print('✅', g, t[:70], flush=True)
    json.dump(meta, open(OUT + '/meta.json', 'w'), ensure_ascii=False, indent=1)
print('всего', len(meta))
