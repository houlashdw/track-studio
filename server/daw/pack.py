"""Бит в Reaper из пака (нарезка скачанных битов, server/modal_pack.py):
мелодия — петля из бита A (растянута в сетку), кик/снейр/808 — по разметке A, хэты — по груву A или другого бита B того же жанра;
звучат наши киты (one-shot), 808 — наш сэмпл по нотам баса A. Аранжировка, сведение и мастер — в Reaper (build.lua).
python3 pack.py --pack <папка с pack.json> --genre trap --seed 3 --smp app/smp --wav out.wav --lua out.lua"""
import os, sys, json, random, argparse
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from daw import to_lua

LVL = {'kick': -13, 'snare': -17, 'hat': -25, 'ohat': -28, '808': -16, 'mel': -19, 'bassloop': -17}   # дБ RMS там, где звучит
NEAR = {'atlanta': 'trap', 'detroit': 'trap', 'pluggnb': 'plugg', 'jerk': 'plugg'}
KITS = {'rage': ['tsairage', 'hard', 'supreme'], 'phonk': ['tsaiphonk', 'hard', 'tsk'], 'plugg': ['tsaiplugg', 'aplus', 'supreme'],
        'drill': ['hard', 'tsk', 'trap'], 'trap': ['hard', 'supreme', 'aplus', 'trap'], 'lofi': ['aplus', 'trippy', 'trap']}
FORM = [('intro', 4), ('hook', 8), ('verse', 8), ('hook', 8), ('outro', 4)]


def arms(f, win=.35):
    x, sr = sf.read(f, dtype='float32', always_2d=True); x = x.mean(1)
    if win: x = x[:int(win * sr)]
    a = np.abs(x) > np.abs(x).max() * .05
    return float(np.sqrt((x[a] ** 2).mean())) if a.any() else 1.0


def pick_kit(smp, genre, R):
    idx = json.load(open(os.path.join(smp, 'index.json'))); out = {}
    for c in ['kick', 'snare', 'hat', 'ohat', '808']:
        pool = []
        for s in KITS.get(genre, KITS['trap']) + ['tsai', 'trap']:
            pool = [e for e in idx.get(c, []) if e.get('src') == s]
            if c == '808': pool = [e for e in pool if e.get('hz') and 28 < e['hz'] < 80]
            if pool and R.random() < .75: break
        pool = pool or [e for e in idx.get(c, []) if (c != '808' or (e.get('hz') and 28 < e['hz'] < 80))]
        if pool: out[c] = R.choice(pool)
    return out


def build(pack_dir, genre, seed, smp, out_wav):
    R = random.Random(seed); P = json.load(open(os.path.join(pack_dir, 'pack.json')))
    L = [x for x in P if x['genre'] == genre] or [x for x in P if x['genre'] == NEAR.get(genre, 'trap')]
    L = [x for x in L if len(x['steps']['k']) >= 6 and len(x['b808']) >= 4] or L
    L.sort(key=lambda x: -x['score']); A = R.choice(L[:max(3, len(L) // 2)])
    Bs = [x for x in L if x['id'].rsplit('_L', 1)[0] != A['id'].rsplit('_L', 1)[0] and abs(x['bpm'] / A['bpm'] - 1) < .1]
    B = R.choice(Bs) if Bs and R.random() < .6 else A
    bpm = round(A['bpm']); T = 60 / bpm; S16 = T / 4; BAR = 4 * T
    rate = bpm / A['bpm']   # петля в сетку проекта (≈1, без слышимой растяжки)
    kit = pick_kit(smp, genre, R); f = lambda c: os.path.abspath(os.path.join(smp, kit[c]['f']))
    tr = {}
    def hit(name, t, v=1.0, d=None, rate_=1.0):
        tr.setdefault(name, []).append(dict(t=round(t, 5), d=d or min(sf.info(f(name)).duration, 2.0), v=v, rate=rate_))
    mel = os.path.abspath(os.path.join(pack_dir, A['files']['mel'])); loops = []
    k8 = kit.get('808'); hz8 = (k8 or {}).get('hz') or 44.0; d8 = sf.info(f('808')).duration if k8 else 0
    use_bassloop = genre == 'drill' or not k8   # скольжения drill нотами не повторить — берём живой бас из бита
    bar0 = 0
    for sec, nb in FORM:
        t0 = bar0 * BAR
        # мелодия: интро/аутро — первые 4 такта петли, остальное — петля целиком
        loops.append(dict(t=t0, d=nb * BAR, v=.7 if sec in ('intro', 'outro') else (.85 if sec == 'verse' else 1.0), rate=rate, pp=True, fi=.005 if sec != 'intro' else 1.5 * BAR,
                          fo=2 * BAR if sec == 'outro' else .01, loop=True))
        if sec in ('hook', 'verse'):
            for b in range(nb):
                lb = b % 8; tb = t0 + b * BAR; last = b == nb - 1
                for i, v in A['steps']['k']:
                    if i // 16 == lb and not (last and i % 16 >= 12): hit('kick', tb + (i % 16) * S16, v)
                for i, v in A['steps']['s']:
                    if i // 16 == lb: hit('snare', tb + (i % 16) * S16, v)
                H = (B if sec == 'verse' else A)['steps']['h']
                for i, v in H:
                    if i // 32 == lb: hit('hat', tb + (i % 32) * S16 / 2, .55 + .45 * v)
                if not use_bassloop:
                    for i, ln, m in A['b808']:
                        if i // 16 != lb: continue
                        while m > 47: m -= 12
                        while m < 28: m += 12
                        r8 = 440 * 2 ** ((m - 69) / 12) / hz8; d = min(d8 / r8, ln * S16 + .04)
                        if last and i % 16 >= 12: continue
                        hit('808', tb + (i % 16) * S16, 1.0, d, r8)
        if sec == 'outro': hit('ohat', t0, .8) if 'ohat' in kit else None
        bar0 += nb
    total = bar0 * BAR + 1.5
    tracks = []
    for name in ['kick', 'snare', 'hat', 'ohat', '808']:
        if name in tr and name in kit:
            tracks.append(dict(name=name, kind='audio', file=f(name), items=tr[name], lvl=LVL[name], pan={'hat': .12, 'ohat': -.15}.get(name, 0)))
    tracks.append(dict(name='mel', kind='audio', file=mel, items=loops, lvl=LVL['mel'], pan=0))
    if use_bassloop:
        bl = os.path.abspath(os.path.join(pack_dir, A['files']['bass'])); bi = [dict(x, v=1.0, fi=.005, fo=.01) for x in loops[1:-1]]
        tracks.append(dict(name='bassloop', kind='audio', file=bl, items=bi, lvl=LVL['bassloop'], pan=0))
    for t in tracks:   # фейдер: RMS звучащей части → целевая громкость
        r = arms(t['file'], None if t['name'] in ('mel', 'bassloop') else .35)
        t['vol'] = float(np.clip(t['lvl'] - 20 * np.log10(r + 1e-9), -40, 18))
    info = dict(A=A['id'], B=B['id'], bpm=bpm, key=A['key'] + ' ' + A['mode'], kit={c: kit[c]['f'] for c in kit}, bassloop=use_bassloop)
    return dict(genre=genre, bpm=bpm, total=total, out=out_wav, tracks=tracks, info=info,
                drive={'rage': .7, 'phonk': .6}.get(genre, .3), loud={'rage': -7, 'phonk': -7, 'drill': -8}.get(genre, -9))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pack', required=True); ap.add_argument('--genre', default='trap'); ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--smp', default='app/smp'); ap.add_argument('--wav', required=True); ap.add_argument('--lua', required=True)
    a = ap.parse_args(); Rp = build(a.pack, a.genre, a.seed, a.smp, a.wav)
    open(a.lua, 'w').write('return ' + to_lua(Rp) + '\n')
    print('пак → Reaper:', a.genre, json.dumps(Rp['info'], ensure_ascii=False), 'дорожки:', [(t['name'], len(t['items'])) for t in Rp['tracks']])
