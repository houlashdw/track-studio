"""Бит в Reaper из пака (нарезка скачанных битов, server/modal_pack.py) — АРАНЖИРОВЩИК со случайностью на каждый сид:
структура (интро / куплет / брейк / хук / аутро — разной длины), темп ±6%, сдвиг тональности, разные петли мелодии по секциям (подогнаны по ключу и темпу),
свинг и «живость» хэтов, филлы и дропы перед хуками, райзеры и удары на переходах, вариации 808, слои на втором хуке, случайные уровни и FX шины.
Кик/снейр/808 — по разметке бита A, хэты — по груву A или B; звучат наши киты, 808 — наш сэмпл по нотам. Сведение и мастер — в Reaper (build.lua).
python3 pack.py --pack <папка с pack.json> --genre trap --seed 3 --smp app/smp --wav out.wav --lua out.lua"""
import os, sys, json, random, argparse
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from daw import to_lua

LVL = {'drumloop': -14, 'kick': -13, 'snare': -17, 'hat': -25, 'ohat': -28, '808': -16, 'mel': -19, 'mel2': -24, 'bassloop': -17, 'fx': -22}   # дБ RMS там, где звучит
NEAR = {'atlanta': 'trap', 'detroit': 'trap', 'pluggnb': 'plugg', 'jerk': 'plugg'}
KITS = {'rage': ['tsairage', 'hard', 'supreme'], 'phonk': ['tsaiphonk', 'hard', 'tsk'], 'plugg': ['tsaiplugg', 'aplus', 'supreme'],
        'drill': ['hard', 'tsk', 'trap'], 'trap': ['hard', 'supreme', 'aplus', 'trap'], 'lofi': ['aplus', 'trippy', 'trap']}
SWING = {'trap': [0, 0, .05, .09], 'plugg': [0, .05, .09, .12], 'rage': [0, 0, 0, .04], 'phonk': [0, .05, .1], 'drill': [0, 0, .04], 'lofi': [.08, .12, .16]}
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
GENRES = ['plugg', 'rage', 'phonk', 'trap', 'drill']


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


def semis(a, b): return (NOTES.index(b) - NOTES.index(a) + 6) % 12 - 6   # на сколько полутонов сдвинуть петлю в тональности b, чтобы она встала в a


def fx_wavs(dirn, BAR, R):
    """райзер (шум с нарастанием, 2 такта) и удар-импакт (спад 808 + шум) — генерируются под темп трека"""
    sr = 44100; n = int(2 * BAR * sr); rng = np.random.default_rng(R.randrange(1 << 30)); t = np.arange(n) / sr
    ns = rng.standard_normal((2, n)).astype('float32'); ns = np.diff(ns, axis=1, prepend=0)   # приподнять верх
    env = (t / t[-1]) ** 2.2; ph = np.cumsum(220 * 2 ** (t / t[-1] * 2.3)) / sr; sw = np.sin(2 * np.pi * ph) * .35
    ris = (ns * .5 + sw) * env; ris[:, -int(.03 * sr):] *= np.linspace(1, 0, int(.03 * sr))
    n2 = int(1.6 * sr); t2 = np.arange(n2) / sr; f = 30 + 110 * np.exp(-t2 * 9); sub = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t2 * 2.2)
    imp = np.stack([sub + rng.standard_normal(n2) * np.exp(-t2 * 7) * .25] * 2)
    out = {}
    for nm, x in (('riser', ris), ('impact', imp)):
        x = x / (np.abs(x).max() + 1e-9) * .8; p = os.path.join(dirn, f'fx_{nm}_{seed_tag(R)}.wav'); sf.write(p, x.T.astype('float32'), sr); out[nm] = p
    return out


def seed_tag(R): return '%x' % (id(R) & 0xffff)


def make_form(R, genre):
    """случайная структура: список секций (имя, такты, барабаны, 808, мелодия, вокал, слой); интро/хуки/куплеты разной длины, иногда брейк и билд"""
    S = lambda name, bars, **k: dict(name=name, bars=bars, drums=k.get('drums', 'full'), b808=k.get('b808', True), mel=k.get('mel', 1.0), vox=k.get('vox', False), layer=k.get('layer', False), loop=k.get('loop', 0), hv=k.get('hv', 'A'))
    F = [S('intro', R.choice([2, 4, 4]), drums=R.choice(['hats', 'light']), b808=R.random() < .5, mel=.9)]
    two_verses = R.random() < .65
    if R.random() < .75: F.append(S('verse', R.choice([8, 8, 12]), drums='light', mel=.8, vox=True, loop=1, hv=R.choice(['sparse', 'B'])))
    F.append(S('hook', R.choice([8, 8, 12, 16]), vox=True, hv=R.choice(['A', 'A', 'roll'])))
    if R.random() < .6: F.append(S('break', R.choice([2, 4]), drums='none', b808=False, mel=.9, vox=True, loop=0))
    if two_verses and F[1]['name'] == 'verse': F.append(S('verse', R.choice([8, 8]), drums='light', mel=.8, vox=True, loop=R.choice([1, 2]), hv=R.choice(['sparse', 'B', 'roll'])))
    F.append(S('hook', R.choice([8, 12, 16]), vox=True, layer=R.random() < .6, loop=0, hv=R.choice(['roll', 'A'])))
    F.append(S('outro', R.choice([2, 4, 4, 8]), drums=R.choice(['kick', 'hats', 'none']), b808=R.random() < .5, mel=.6))
    tot = lambda: sum(x['bars'] for x in F)
    for nm in ('break', 'verse'):   # длина трека для ACE-голоса: до ~48 тактов (зависит от темпа)
        while tot() > 48 and any(x['name'] == nm for x in F): F.pop(max(i for i, x in enumerate(F) if x['name'] == nm))
    return F


def build(pack_dir, genre, seed, smp, out_wav, drums='kit', vox=None, voxvol=0.0):
    R = random.Random(seed)
    if genre == 'auto': genre = R.choice(GENRES)
    P = json.load(open(os.path.join(pack_dir, 'pack.json')))
    L = [x for x in P if x['genre'] == genre] or [x for x in P if x['genre'] == NEAR.get(genre, 'trap')]
    L = [x for x in L if len(x['steps']['k']) >= 6 and len(x['b808']) >= 4] or L
    L.sort(key=lambda x: -x['score']); A = R.choice(L[:max(3, len(L) // 2)])
    base = lambda x: x['id'].rsplit('_L', 1)[0]
    Bs = [x for x in L if base(x) != base(A) and abs(x['bpm'] / A['bpm'] - 1) < .1]
    B = R.choice(Bs) if Bs and R.random() < .6 else A
    # другие петли мелодии для разных секций: тот же лад, близкий темп — подгоняем по ключу (полутона) и темпу (playrate)
    Cs = [x for x in L if base(x) not in (base(A), base(B)) and x['mode'] == A['mode'] and abs(x['bpm'] / A['bpm'] - 1) < .1]; R.shuffle(Cs)
    MEL = [A] + Cs[:2]
    bpm = round(A['bpm'] * R.uniform(.95, 1.05)); T = 60 / bpm; S16 = T / 4; BAR = 4 * T
    shift = R.choice([-3, -2, -1, 0, 0, 1, 2, 3])   # общий сдвиг тональности (петли и 808)
    sw = R.choice(SWING.get(genre, [0, .05])) * S16 * 2   # на сколько запаздывают нечётные шестнадцатые (свинг)
    kit = pick_kit(smp, genre, R); f = lambda c: os.path.abspath(os.path.join(smp, kit[c]['f']))
    FM = make_form(R, genre)
    tr = {}
    def hit(name, t, v=1.0, d=None, rate_=1.0):
        tr.setdefault(name, []).append(dict(t=round(max(0, t), 5), d=d or min(sf.info(f(name)).duration, 2.0), v=round(float(min(1.2, v)), 3), rate=rate_))
    k8 = kit.get('808'); hz8 = (k8 or {}).get('hz') or 44.0; d8 = sf.info(f('808')).duration if k8 else 0
    use_bassloop = genre == 'drill' or not k8   # скольжения drill нотами не повторить — берём живой бас из бита
    K = [x for x in A['steps']['k']]; SN = [x for x in A['steps']['s']]
    if len(SN) < 6: SN = [[16 * b + q, .9] for b in range(8) for q in ([4, 12] if genre == 'lofi' else [8])]
    sset = {i for i, _ in SN}; K = [x for x in K if x[0] not in sset]
    if len(K) > 36: th = sorted((v for _, v in K), reverse=True)[35]; K = [x for x in K if x[1] >= th or x[0] % 16 == 0]
    pa = [None] * 128   # 808: высота — из баса A (гармония мелодии), ритм — из A или (в режиме loop) из B
    for i, ln, m in sorted(A['b808']):
        for j in range(i, 128): pa[j] = m
    first = sorted(A['b808'])[0][2] if A['b808'] else 36; pa = [m if m is not None else first for m in pa]
    N808 = [[i, ln, pa[i]] for i, ln, m in B['b808']] if drums == 'loop' and B is not A and len(B['b808']) >= 4 else A['b808']
    HAT = {'A': A['steps']['h'], 'B': B['steps']['h']}
    fxw = fx_wavs(os.path.dirname(os.path.abspath(out_wav)) or '.', BAR, R)
    mels = {}; lay = []; dl = []; fxi = []; bl = []; vitems = []; hooks = []; dbl = []
    drumloop = os.path.abspath(os.path.join(pack_dir, B['files']['drums'])); rateB = bpm / B['bpm']
    bar0 = 0; nh = 0
    for si, sec in enumerate(FM):
        t0 = bar0 * BAR; nb = sec['bars']; nxt = FM[si + 1]['name'] if si + 1 < len(FM) else None
        if sec['name'] == 'hook': hooks.append((t0, nb * BAR))
        if sec['vox']: vitems.append((t0, nb * BAR, sec['name']))
        # мелодия: петля нужной секции (loop → индекс в MEL, если петель меньше — берём A); ключ и темп подгоняем
        M = MEL[min(sec['loop'], len(MEL) - 1)] if sec['name'] not in ('hook', 'break') else MEL[0]
        if sec['name'] == 'hook' and nh == 1 and len(MEL) > 1 and R.random() < .4: M = MEL[1]   # второй хук иногда на другой петле
        pitch = shift + (semis(A['key'], M['key']) if M is not A else 0)
        mf = os.path.abspath(os.path.join(pack_dir, M['files']['mel']))
        it = dict(t=t0, d=nb * BAR, v=sec['mel'], rate=bpm / M['bpm'], pp=True, pitch=pitch, fi=(nb * BAR * .2 if sec['name'] == 'intro' else .01), fo=(nb * BAR * .8 if sec['name'] == 'outro' else .02), loop=True, off=(R.choice([0, 0, 4 * 4 * (60 / M['bpm'])]) if sec['name'] != 'hook' else 0))
        mels.setdefault(mf, []).append(it)
        if sec['layer'] and len(MEL) > 1:   # слой: вторая петля тихо поверх на большом хуке
            L2 = MEL[-1] if MEL[-1] is not M else MEL[1]; lay.append((os.path.abspath(os.path.join(pack_dir, L2['files']['mel'])),
                dict(t=t0, d=nb * BAR, v=.55, rate=bpm / L2['bpm'], pp=True, pitch=shift + semis(A['key'], L2['key']) + (12 if R.random() < .3 else 0), fi=.2, fo=.05, loop=True)))
        if sec['drums'] in ('full', 'light') and drums == 'loop':
            dl.append(dict(t=t0, d=nb * BAR, v=1.0 if sec['name'] == 'hook' else .9, rate=rateB, pp=True, fi=.003, fo=.02, loop=True))
        if use_bassloop and sec['b808']:
            bl.append(dict(t=t0, d=nb * BAR, v=1.0, rate=bpm / A['bpm'], pp=True, pitch=shift, fi=.005, fo=.02, loop=True))
        # переход перед хуком: райзер на 2 такта + дроп барабанов; удар на 1-й доле хука
        if nxt in ('hook', 'verse') and nb >= 2: fxi.append(('riser', t0 + (nb - 2) * BAR, 2 * BAR))
        if sec['name'] == 'hook':
            fxi.append(('impact', t0, 1.6)); nh += 1
            if 'ohat' in kit: hit('ohat', t0, .9)
        if sec['drums'] in ('full', 'light', 'kick', 'hats') and drums == 'kit':
            for b in range(nb):
                lb = b % 8; tb = t0 + b * BAR; last = b == nb - 1; pre = nxt == 'hook' and b >= nb - 2
                phrase_end = (b % 4 == 3) or last
                fill = phrase_end and sec['drums'] in ('full', 'light') and (last or R.random() < .55)
                drop = pre and b == nb - 1 and R.random() < .5   # последний такт перед хуком — тишина под райзер (только хэт)
                kicks = [(i % 16, v) for i, v in K if i // 16 == lb]; sns = [(i % 16, v) for i, v in SN if i // 16 == lb]
                hh = [(i % 32, v) for i, v in HAT[sec['hv'] if sec['hv'] in HAT else 'A'] if i // 32 == lb]
                if sec['hv'] == 'sparse': hh = [x for x in hh if x[0] % 4 == 0 or R.random() < .35]
                if sec['hv'] == 'roll' and b % 2 == 1: hh += [(j, .5 + .4 * (j - 24) / 8) for j in range(24, 32) if R.random() < .85]
                if sec['drums'] == 'kick': sns = []; hh = []
                if sec['drums'] == 'hats': kicks = []; sns = []
                if sec['drums'] == 'light': sns = [x for x in sns]; kicks = [x for x in kicks if x[0] % 16 < 12 or R.random() < .5]
                if fill:
                    kind = R.choice(['roll', 'double', 'hatroll'])
                    kicks = [x for x in kicks if x[0] < 12]
                    if kind == 'roll': sns = [x for x in sns if x[0] < 12] + [(s, .55 + .15 * (s - 12)) for s in (12, 13, 14, 15)]
                    elif kind == 'double': sns = [x for x in sns if x[0] < 8] + [(s, .7 + .05 * k_) for k_, s in enumerate((8, 10, 12, 13, 14, 15))]
                    else: hh = [x for x in hh if x[0] < 16] + [(j, .45 + .5 * (j - 16) / 16) for j in range(16, 32)]
                if drop: kicks = [x for x in kicks if x[0] < 4]; sns = []; hh = [x for x in hh if x[0] % 4 == 0]
                if sec['name'] == 'hook' and R.random() < .3 and not drop:   # призрачный кик в хуке
                    g = R.choice([3, 6, 7, 11, 14]);
                    if all(x[0] != g for x in sns + kicks): kicks.append((g, .65))
                for i, v in kicks:
                    hit('kick', tb + i * S16 + (sw if i % 2 and i not in (0, 4, 8, 12) else 0), v * R.uniform(.92, 1.05))
                for i, v in sns: hit('snare', tb + i * S16 + R.uniform(-.002, .004), v * R.uniform(.9, 1.0))
                for j, v in hh:
                    off = (j % 2) * S16 / 2 + (j // 2) * S16 * 0   # 1/32 внутри шестнадцатой
                    tt = tb + (j / 2) * S16 + (sw if (j // 2) % 2 and j % 2 == 0 else 0) + R.uniform(-.004, .004)
                    hit('hat', tt, (.55 + .45 * v) * R.uniform(.82, 1.0) * (.85 if sec['drums'] == 'light' else 1))
                if sec['name'] == 'hook' and 'ohat' in kit and b % 2 == 1 and R.random() < .35 and not fill: hit('ohat', tb + 14 * S16, .7)
                if sec['b808'] and not use_bassloop and sec['drums'] != 'hats':
                    for i, ln, m in N808:
                        if i // 16 != lb: continue
                        m += shift
                        while m > 50: m -= 12
                        while m < 28: m += 12
                        if (fill or drop) and i % 16 >= 12: continue
                        if sec['name'] == 'hook' and b % 4 == 3 and R.random() < .25 and m + 12 <= 52: m += 12   # октавный «поп» в хуке
                        r8 = 440 * 2 ** ((m - 69) / 12) / hz8; d = min(d8 / r8, ln * S16 + .04)
                        hit('808', tb + (i % 16) * S16, 1.0, d, r8)
        bar0 += nb
    total = bar0 * BAR + 2.5
    tracks = []
    for name in ['kick', 'snare', 'hat', 'ohat', '808']:
        if name in tr and name in kit:
            tracks.append(dict(name=name, kind='audio', file=f(name), items=tr[name], lvl=LVL[name] + R.uniform(-1.2, 1.2), pan={'hat': R.choice([.08, .15, -.1]), 'ohat': R.choice([-.15, .2])}.get(name, 0)))
    byfile = mels
    for fn, items in byfile.items():   # каждая петля — своя дорожка (разные файлы), обработка одна ('mel')
        tracks.append(dict(name='mel', kind='audio', file=fn, items=items, lvl=LVL['mel'] + R.uniform(-1.5, 1.0), pan=0))
    for fn, it in lay: tracks.append(dict(name='mel2', kind='audio', file=fn, items=[it], lvl=LVL['mel2'], pan=R.choice([-.35, .35])))
    if dl: tracks.append(dict(name='drumloop', kind='audio', file=os.path.abspath(os.path.join(pack_dir, B['files']['drums'])), items=dl, lvl=LVL['drumloop'], pan=0))
    if use_bassloop and bl: tracks.append(dict(name='bassloop', kind='audio', file=os.path.abspath(os.path.join(pack_dir, A['files']['bass'])), items=bl, lvl=LVL['bassloop'], pan=0))
    for nm in ('riser', 'impact'):
        its = [dict(t=t_, d=d_, v=1.0, rate=1.0, fi=.02, fo=.02) for k_, t_, d_ in fxi if k_ == nm]
        if its: tracks.append(dict(name='fx', kind='audio', file=fxw[nm], items=its, lvl=LVL['fx'] + (3 if nm == 'impact' else 0), pan=0))
    if vox:   # вокал по секциям (интро/аутро — тишина), второй хук — с октавным дублем; копия на хуках идёт лишь на плейт-реверб
        vd = sf.info(vox).duration; dur = min(vd, total - .5)
        # фразы голоса (между паузами ≥ .25 с); в секции берём ЦЕЛЫЕ фразы, чей старт попал в окно секции (с заходом до такта назад) — никаких обрывов; в куплетах иногда пропускаем фразу, чтобы голос дышал
        xv, _sr = sf.read(vox, always_2d=True); W_ = int(.03 * _sr); ev = np.sqrt(np.convolve(xv.mean(1) ** 2, np.ones(W_) / W_, 'same')); H = int(.02 * _sr)
        ev = ev[:len(ev) // H * H].reshape(-1, H).mean(1); on_ = ev > max(ev.max() * .05, 1e-5); ph = []; i = 0; n_ = len(on_)
        while i < n_:
            if on_[i]:
                j = i; gap = 0
                while j < n_ and gap < 13: gap = 0 if on_[j] else gap + 1; j += 1
                ph.append((i * .02, (j - gap) * .02)); i = j
            else: i += 1
        RV = random.Random(seed * 31 + 5); vl = []
        for t_, d_, nm in vitems:
            for k, (p0, p1) in enumerate(ph):
                if not (t_ - BAR * .5 <= p0 < t_ + d_ - .3) or p1 - p0 < .25: continue
                t2 = max(0, p0 - .06); e2 = min(p1 + .35, dur)
                vl.append((dict(t=t2, d=e2 - t2, v=1.0 if nm == 'hook' else .85, rate=1.0, off=t2, fi=.05, fo=.3), nm))   # куплет на ~1.5 дБ тише хука
        tracks.append(dict(name='vox', kind='audio', file=os.path.abspath(vox), items=[x for x, _ in vl], lvl=0, vol=voxvol, pan=0))
        hk = [dict(t=t_, d=min(d_, dur - t_), v=1.0, rate=1.0, off=t_, fi=.04, fo=.5) for t_, d_ in hooks if t_ < dur - 1]  # копия-плейт на хуках
        tracks.append(dict(name='voxhook', kind='audio', file=os.path.abspath(vox), nomain=True, lvl=0, vol=voxvol, pan=0, items=hk))
        hv_ = [x for x, nm in vl if nm == 'hook' and hooks and x['t'] >= hooks[-1][0] - BAR * .5]   # октавный дубль голоса — на последнем хуке
        # стак на хуках: два дубля голоса жёстко влево/вправо, чуть расстроены (±9…14 центов) и опаздывают на 14…30 мс — как вторая/третья дорожка в студии
        hk_ph = [x for x, nm in vl if nm == 'hook']
        if hk_ph:
            for side in (-1, 1):
                dl_ = RV.uniform(.014, .030); cents = side * RV.uniform(.09, .14)
                tracks.append(dict(name='voxstk', kind='audio', file=os.path.abspath(vox), lvl=0, vol=voxvol - RV.uniform(6.5, 8.5), pan=side * .85,
                                   items=[dict(x, t=x['t'] + dl_, v=1.0, pitch=cents, fi=.06) for x in hk_ph]))
    for t in tracks:   # фейдер: RMS звучащей части → целевая громкость
        if t['name'] in ('vox', 'voxhook', 'voxdbl', 'voxstk'): continue
        r = arms(t['file'], None if t['name'] in ('mel', 'mel2', 'bassloop', 'drumloop', 'fx') else .35)
        t['vol'] = float(np.clip(t['lvl'] - 20 * np.log10(r + 1e-9), -40, 18))
    struct = ' '.join('%s%d%s' % (x['name'][0] + ('H' if x['name'] == 'hook' else ''), x['bars'], '+' if x['layer'] else '') for x in FM)
    info = dict(drums=drums, genre=genre, A=A['id'], B=B['id'], mels=[m_['id'] for m_ in MEL], bpm=bpm, key=A['key'] + ' ' + A['mode'], shift=shift, swing=round(sw / S16, 2), form=struct, kit={c: kit[c]['f'] for c in kit}, bassloop=use_bassloop)
    return dict(genre=genre, bpm=bpm, total=total, out=out_wav, tracks=tracks, info=info,
                drive=R.choice([.2, .3, .45, .6]) if genre not in ('rage', 'phonk') else R.choice([.6, .7, .85]), loud={'rage': -7, 'phonk': -7, 'drill': -8}.get(genre, -9), vox=bool(vox), q=round(60 / bpm, 5),
                sections=[(x['name'], x['bars']) for x in FM])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pack', required=True); ap.add_argument('--genre', default='auto'); ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--smp', default='app/smp'); ap.add_argument('--wav', required=True); ap.add_argument('--lua', required=True); ap.add_argument('--drums', default='kit'); ap.add_argument('--vox'); ap.add_argument('--voxvol', type=float, default=0.0); ap.add_argument('--info'); ap.add_argument('--solo')
    a = ap.parse_args(); Rp = build(a.pack, a.genre, a.seed, a.smp, a.wav, a.drums, a.vox, a.voxvol)
    if a.solo:   # замер: оставить только вокальные дорожки (vox) или только бит (beat)
        Rp['tracks'] = [t for t in Rp['tracks'] if (t['name'] in ('vox', 'voxhook', 'voxdbl', 'voxstk')) == (a.solo == 'vox')]
    open(a.lua, 'w').write('return ' + to_lua(Rp) + '\n')
    if a.info: json.dump(dict(bpm=Rp['bpm'], total=Rp['total'], info=Rp['info']), open(a.info, 'w'))
    print('пак → Reaper:', json.dumps(Rp['info'], ensure_ascii=False), 'дорожки:', [(t['name'], len(t['items'])) for t in Rp['tracks']])
