"""Бит «как в DAW»: проект из дорожек (сэмплы китов + JS-синты по MIDI + JS-эффекты на каждой дорожке + мастер),
два рендера из одного проекта:
  A) свой движок (numpy + jsfx_render на ysfx) — бесплатно;
  B) настоящий Reaper: пишем .rpp, Reaper рендерит (сравнение).
python3 daw.py --genre trap --seed 7 --smp app/smp --jsfx <папка JSFX> --bin jsfx_render --out a.wav [--rpp b.rpp]
"""
import os, sys, json, random, argparse, subprocess, tempfile, math
import numpy as np
import soundfile as sf

SR = 44100
SCALE = {'min': [0, 2, 3, 5, 7, 8, 10], 'maj': [0, 2, 4, 5, 7, 9, 11], 'phr': [0, 1, 3, 5, 7, 8, 10], 'harm': [0, 2, 3, 5, 7, 8, 11]}
# ходы (ступени) — из разбора 500 MIDI BLACKOUT (как в приложении: гармония стоит на 1–2 аккордах)
PROGS = {'min': [[0, 0, 0, 0], [0, 1, 0, 1], [0, 4, 0, 4], [0, 5, 0, 5], [0, 3, 0, 3], [0, 5, 1, 5], [0, 1, 0, 5], [0, 3, 0, 4], [0, 0, 6, 0], [0, 5, 3, 4]],
         'maj': [[0, 0, 0, 0], [0, 3, 0, 3], [0, 5, 0, 5], [0, 4, 0, 4], [0, 5, 3, 4], [0, 3, 4, 3]]}
MCELLS = [[0, 2, 4, 6, 8, 10, 12, 14], [0, 4, 8, 12], [0, 8, 12], [0, 4, 6, 8, 10, 12, 14], [0, 8], [0, 4, 6, 8, 10, 12], [0, 2, 4, 6, 8, 10, 12], [0, 4, 6, 8, 12], [0, 3, 6, 8, 10, 12], [0, 4, 8, 10, 12, 14]]
G = {   # жанр: темп, лад, кит, синт мелодии, рисунок
    'trap': dict(bpm=(136, 148), sc='min', kit=['trap', 'tsai'], syn='bell', drums='trap'),
    'drill': dict(bpm=(140, 144), sc='min', kit=['trap', 'tsai'], syn='piano', drums='drill'),
    'plugg': dict(bpm=(145, 155), sc='maj', kit=['tsai_plugg', 'trap'], syn='bell', drums='plugg'),
    'rage': dict(bpm=(155, 165), sc='min', kit=['tsai_rage', 'trap'], syn='saw', drums='rage'),
    'phonk': dict(bpm=(125, 135), sc='phr', kit=['tsai_phonk', 'trap'], syn='cowbell', drums='phonk'),
    'detroit': dict(bpm=(96, 104), sc='min', kit=['trap'], syn='piano', drums='detroit'),
}


def pat(genre, R, bar, role):
    """рисунок ударных на такт (16 шагов): kick, snare, hats(шаг, деление), perc"""
    k = {'trap': [0, 7, 10], 'drill': [0, 6, 10, 13], 'plugg': [0, 10], 'rage': [0, 3, 8, 11], 'phonk': [0, 6, 10], 'detroit': [0, 3, 7, 10, 13]}[genre]
    k = list(k)
    if R.random() < .35: k.append(R.choice([5, 9, 14, 15]))
    sn = [8] if genre != 'drill' else [8, R.choice([14, 15])] if bar % 2 else [8]
    if genre == 'detroit': sn = [4, 12]
    hats = []
    for s in range(0, 16, 2 if genre in ('detroit', 'phonk') and R.random() < .5 else 2):
        hats.append((s, 1))
    if genre in ('trap', 'plugg', 'rage', 'drill') and (bar % 2 == 1 or R.random() < .4):   # роллы в конце такта
        st = R.choice([12, 14]); hats = [h for h in hats if h[0] < st] + [(st, R.choice([2, 3, 4])), (st + 1 if st == 14 else 13, 2)]
    if genre in ('trap', 'rage', 'plugg') and R.random() < .5: hats += [(s, 1) for s in (1, 5, 9) if R.random() < .3]
    if role == 'intro': k, sn = [], []
    if role == 'outro' and bar % 4 == 3: k, hats = [0], []
    return sorted(set(k)), sn, hats


def load_kit(smp, genre, R):
    idx = json.load(open(os.path.join(smp, 'index.json')))
    want = G[genre]['kit']; out = {}; srcs = set(e.get('src') for v in idx.values() if isinstance(v, list) for e in v)
    src = next((s for s in want if s in srcs), 'trap')
    for c in ['kick', 'snare', 'clap', 'hat', 'ohat', 'perc', '808']:
        pool = [e for e in idx.get(c, []) if e.get('src') == src] or [e for e in idx.get(c, []) if e.get('src') in ('trap', 'tsai')]
        if c == '808': pool = [e for e in pool if e.get('hz') and 25 < e['hz'] < 90] or pool
        if pool: out[c] = R.choice(pool)
    return src, out


def build(genre='trap', seed=1, smp='app/smp'):
    """проект: дорожки с событиями (время в долях такта → секунды), FX-цепочки — абстрактные (имя эффекта + параметры)"""
    R = random.Random(seed); g = G[genre]
    bpm = R.randint(*g['bpm']); key = R.randint(0, 11); sc = g['sc']; S = SCALE[sc]
    step = 60 / bpm / 4; bar = step * 16
    form = [('intro', 4), ('hook', 8), ('verse', 8), ('hook', 8), ('outro', 4)]
    prog = R.choice(PROGS['maj' if sc == 'maj' else 'min']); cell = R.choice(MCELLS)
    src, kit = load_kit(smp, genre, R)
    T = {n: [] for n in ['kick', 'snare', 'clap', 'hat', 'ohat', 'perc', '808', 'melody', 'pad']}
    root = 48 + key   # C3+key
    note = lambda deg, base: base + S[deg % 7] + 12 * (deg // 7)
    mel_cells = [R.choice(MCELLS) for _ in range(2)]
    b0 = 0
    for role, nb in form:
        for b in range(nb):
            t0 = (b0 + b) * bar; deg = prog[b % 4]
            k, sn, hats = pat(g['drums'], R, b, role)
            for s in k: T['kick'].append(dict(t=t0 + s * step, v=1.0))
            for s in sn: T['snare'].append(dict(t=t0 + s * step, v=1.0)); T['clap'].append(dict(t=t0 + s * step, v=.7))
            for s, div in hats:
                for r in range(div): T['hat'].append(dict(t=t0 + (s + r / div) * step, v=.8 - .15 * (r > 0) + .1 * (s % 4 == 0)))
            if role in ('hook',) and b % 4 == 3: T['ohat'].append(dict(t=t0 + 14 * step, v=.6))
            if role != 'intro' and genre in ('phonk',): T['perc'] += [dict(t=t0 + s * step, v=.7, m=note(deg + R.choice([0, 2, 4, 7]), root + 24)) for s in (0, 3, 6, 10, 12) if R.random() < .7]
            # 808: на киках, нота — корень аккорда; держится до следующего удара
            if role != 'intro':
                for i, s in enumerate(k):
                    nxt = k[i + 1] if i + 1 < len(k) else 16
                    T['808'].append(dict(t=t0 + s * step, d=max(2, nxt - s) * step * .95, m=note(deg, root - 12), gl=R.random() < .12))
            # мелодия (синт по MIDI): ячейка ритма × ступени аккорда
            mc = mel_cells[(b // 2) % 2]
            for j, s in enumerate(mc):
                if role == 'outro' and b >= 2: break
                d = ((mc[j + 1] if j + 1 < len(mc) else 16) - s) * step
                T['melody'].append(dict(t=t0 + s * step, d=min(d, 4 * step) * .9, m=note(deg + [0, 2, 4, 2, 7, 4, 2, 0][(j + b) % 8], root + 12), v=.75 + .2 * (s % 8 == 0)))
            if b % 2 == 0: T['pad'].append(dict(t=t0, d=bar * 2 * .98, ms=[note(deg, root), note(deg + 2, root), note(deg + 4, root)], v=.5))
        b0 += nb
    total = b0 * bar + 2.0
    FX = {   # цепочки по дорожкам (имя → файл JSFX + слайдеры 1-based)
        'kick': [('bus_comp', {2: -10, 3: 2, 4: 3, 5: 1})],
        'snare': [('tanh_sat', {1: 3, 2: -1})],
        'clap': [('hadamverb', {})],
        'hat': [('stereo_width', {})] if False else [],
        '808': [('tanh_sat', {1: {'rage': 14, 'phonk': 12}.get(genre, 6), 2: -2, 8: 2})] + ([('megagrit', {1: 8, 8: .25, 9: .6})] if genre == 'phonk' else []),
        'melody': [('big_chorus', {1: 3, 3: 12, 4: 50, 9: .08})] + ([('crusher', {5: 4, 7: 12})] if genre == 'phonk' else []) + [('filthy_delay', {1: 2, 2: round(60000 / bpm * .75), 3: 25, 8: 100, 9: 22}), ('abyss', {})],
        'pad': [('ripple', {1: 4, 7: .25, 9: 10, 10: .4})],
        'master': [('bus_comp', {2: -8, 3: 1, 4: 3, 5: 6}), ('tanh_sat', {1: 2, 2: -0.8})],
    }
    LVL = {'kick': -13, 'snare': -17, 'clap': -24, 'hat': -25, 'ohat': -28, 'perc': -25, '808': -16, 'melody': -21, 'pad': -29}   # дБ RMS там, где дорожка звучит
    VOL = {'kick': 0.9, 'snare': 0.75, 'clap': 0.35, 'hat': 0.32, 'ohat': 0.25, 'perc': 0.35, '808': 0.85, 'melody': 0.32, 'pad': 0.12}
    PAN = {'hat': 0.15, 'ohat': -0.2, 'perc': -0.25, 'clap': 0.0}
    SYN = {'bell': 'js3osc', 'piano': 'wavesynth', 'saw': 'js3osc', 'cowbell': 'js3osc'}[g['syn']]
    return dict(genre=genre, seed=seed, bpm=bpm, key=key, scale=sc, kit_src=src, kit=kit, tracks=T, fx=FX, vol=VOL, pan=PAN, lvl=LVL, syn=SYN, total=total, bars=b0)


JS = {   # абстрактные имена → файлы (коллекции JSFX: MIT / свободные)
    'bus_comp': 'chkhld_jsfx/Dynamics/bus_comp.jsfx', 'tanh_sat': 'JoepVanlier_JSFX/Basics/Tanh_Saturator_AA.jsfx', 'hadamverb': 'ReaTeam_JSFX/Reverb/TiaR_HadamVerb.jsfx',
    'megagrit': 'ReaTeam_JSFX/Distortion/reincore_MegaGrit.jsfx', 'big_chorus': 'ReaTeam_JSFX/Modulation/TiaR_Ze Big Chorus.jsfx', 'crusher': 'chkhld_jsfx/Lo-Fi/signal_crusher.jsfx',
    'filthy_delay': 'chkhld_jsfx/FX/filthy_delay.jsfx', 'abyss': 'JoepVanlier_JSFX/Abyss/saike_abyss.jsfx', 'ripple': 'geraintluff_jsfx/Ripple.jsfx',
    'js3osc': 'ReaTeam_JSFX/Synth/tilr_JS3Osc.jsfx', 'wavesynth': 'ReaTeam_JSFX/Synth/tilr_JSWavesynth.jsfx', 'padsynth': 'geraintluff_jsfx/pad-synth.jsfx',
}


# ---------- A) свой движок ----------
def sample(path, rate=1.0):
    x, sr = sf.read(path, dtype='float32', always_2d=True); x = x.mean(1)
    if sr != SR or rate != 1.0:
        n = int(len(x) * SR / sr / rate); x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype('float32')
    return x


def jsfx(binp, lib, x, chain, bpm, midi=None, n=None):
    """прогон стерео x (N×2) через цепочку JSFX; midi — события (кадр, статус, d1, d2) для синта"""
    if not chain: return x
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, JSFX_BPM=str(bpm))
        if midi is not None:
            open(d + '/m.txt', 'w').write('\n'.join('%d %d %d %d' % e for e in sorted(midi))); env['JSFX_MIDI'] = d + '/m.txt'; env['JSFX_LEN'] = str(n)
        else: np.ascontiguousarray(x, dtype='<f4').tofile(d + '/i.f32')
        specs = [os.path.join(lib, JS[nm]) + ('@' + ','.join('%d=%g' % kv for kv in sl.items()) if sl else '') for nm, sl in chain]
        r = subprocess.run([binp, str(SR), d + '/i.f32', d + '/o.f32'] + specs, env=env, capture_output=True, text=True)
        if r.returncode: print(r.stderr[-500:], file=sys.stderr)
        y = np.fromfile(d + '/o.f32', '<f4').reshape(-1, 2)
    y[~np.isfinite(y)] = 0
    return y


def render_own(P, smp, lib, binp):
    N = int(P['total'] * SR); bpm = P['bpm']; stems = {}
    for name, ev in P['tracks'].items():
        if not ev: continue
        buf = np.zeros((N, 2), 'float32')
        if name in ('melody', 'pad'):
            midi = []
            for e in ev:
                for m in e.get('ms', [e.get('m')]):
                    f0 = int(e['t'] * SR); f1 = int((e['t'] + e['d']) * SR); v = int(100 * e.get('v', .8))
                    midi += [(f0, 144, m, v), (f1, 128, m, 0)]
            syn = P['syn'] if name == 'melody' else 'padsynth'
            buf = jsfx(binp, lib, None, [(syn, {})] + P['fx'].get(name, []), bpm, midi=midi, n=N)[:N]
        else:
            k = P['kit'].get(name)
            if not k: continue
            path = os.path.join(smp, k['f'])
            base = sample(path)
            for e in ev:
                if name == '808' and k.get('hz'):
                    hz = 440 * 2 ** ((e['m'] - 69) / 12); x = sample(path, hz / k['hz'])
                    dn = int(e['d'] * SR); x = x[:dn + int(.05 * SR)].copy(); fo = min(len(x), int(.05 * SR))
                    if fo: x[-fo:] *= np.linspace(1, 0, fo)
                elif name == 'perc' and e.get('m'): x = sample(path, 2 ** ((e['m'] - 72) / 12))
                else: x = base
                i = int(e['t'] * SR); j = min(N, i + len(x)); g = e.get('v', 1.0)
                pn = P['pan'].get(name, 0); buf[i:j, 0] += x[:j - i] * g * (1 - max(0, pn)); buf[i:j, 1] += x[:j - i] * g * (1 + min(0, pn))
            buf = jsfx(binp, lib, buf, P['fx'].get(name, []), bpm)[:N]
        m = np.abs(buf).max(1); act = m > m.max() * .05   # громкость по звучащим местам (RMS), а не по пику: 808 тянется и иначе задавит всё
        if act.any():
            r = np.sqrt((buf[act] ** 2).mean()); gn = 10 ** (P['lvl'][name] / 20) / (r + 1e-9); buf = buf * gn; P.setdefault('gain', {})[name] = float(gn)
        stems[name] = buf
    # сайдчейн: 808 и мелодия уступают кику
    if 'kick' in stems:
        e = np.abs(stems['kick']).max(1); env = np.zeros(N, 'float32'); a = 0.0; rl = math.exp(-1 / (.12 * SR))
        for i in range(0, N, 32):
            a = max(e[i:i + 32].max(), a * rl ** 32); env[i:i + 32] = a
        env /= env.max() + 1e-9
        for nm, dep in (('808', .45), ('melody', .25), ('pad', .35)):
            if nm in stems: stems[nm] *= (1 - dep * env)[:, None]
    mix = sum(stems.values())
    mix = jsfx(binp, lib, mix.astype('float32'), P['fx']['master'], bpm)[:N]
    pk = np.abs(mix).max(); P['mgain'] = float(0.89 / pk) if pk > 0 else 1.0; mix = mix * P['mgain']
    return mix, stems


# ---------- B) Reaper (.rpp) ----------
def rpp(P, smp, lib, out_wav):
    bpm = P['bpm']; L = []
    def fxchain(chain, inst=None):
        if not chain and not inst: return []
        s = ['<FXCHAIN', 'SHOW 0', 'LASTSEL 0', 'DOCKED 0']
        for nm, sl in ([(inst, {})] if inst else []) + list(chain):
            vals = ['-'] * 64
            for k, v in sl.items(): vals[k - 1] = '%g' % v
            s += ['BYPASS 0 0 0', '<JS "%s" ""' % os.path.join(lib, JS[nm]), ' '.join(vals), '>', 'FLOATPOS 0 0 0 0', 'WAK 0 0']
        return s + ['>']
    L += ['<REAPER_PROJECT 0.1 "7.0" 0', 'TEMPO %d 4 4' % bpm, 'SAMPLERATE %d 0 0' % SR,
          'RENDER_FILE "%s"' % out_wav, 'RENDER_PATTERN ""', 'RENDER_FMT 0 2 %d' % SR, 'RENDER_1X 0', 'RENDER_RANGE 1 0 0 18 1000', 'RENDER_RESAMPLE 3 0 1',
          'RENDER_ADDTOPROJ 0', 'RENDER_STEMS 0', 'RENDER_DITHER 0',
          '<RENDER_CFG', 'ZXZhdxgAAQ==', '>']
    L += ['MASTER_VOLUME %g 0 -1 -1 1' % P.get('mgain', 1.0)]
    L += ['<MASTERFXLIST'] + fxchain(P['fx']['master'])[1:-1] + ['>'] if P['fx']['master'] else []
    for name, ev in P['tracks'].items():
        if not ev: continue
        vol = P.get('gain', {}).get(name, P['vol'][name]); pan = P['pan'].get(name, 0)   # громкость дорожки — та же, что выставил свой движок по RMS
        L += ['<TRACK', 'NAME %s' % name, 'VOLPAN %g %g -1 -1 1' % (vol, pan)]
        if name in ('melody', 'pad'):
            L += fxchain(P['fx'].get(name, []), inst=P['syn'] if name == 'melody' else 'padsynth')
            tpq = 960; qn = 60 / bpm; evs = []
            for e in ev:
                for m in e.get('ms', [e.get('m')]):
                    evs += [(round(e['t'] / qn * tpq), '90', m, int(100 * e.get('v', .8))), (round((e['t'] + e['d']) / qn * tpq), '80', m, 0)]
            evs.sort(); last = 0; lines = []
            for tk, st, m, v in evs: lines.append('E %d %s %02x %02x' % (tk - last, st, m, v)); last = tk
            L += ['<ITEM', 'POSITION 0', 'LENGTH %g' % P['total'], '<SOURCE MIDI', 'HASDATA 1 %d QN' % tpq] + lines + ['>', '>']
        else:
            k = P['kit'].get(name)
            if not k: L += ['>']; continue
            L += fxchain(P['fx'].get(name, []))
            path = os.path.abspath(os.path.join(smp, k['f'])); ln = sf.info(path).duration
            for e in ev:
                rate = 1.0; d = ln
                if name == '808' and k.get('hz'): rate = 440 * 2 ** ((e['m'] - 69) / 12) / k['hz']; d = min(ln / rate, e['d'] + .05)
                elif name == 'perc' and e.get('m'): rate = 2 ** ((e['m'] - 72) / 12); d = ln / rate
                L += ['<ITEM', 'POSITION %.6f' % e['t'], 'LENGTH %.6f' % d, 'FADEOUT 1 0.02 0', 'VOLPAN %g 0 1 -1' % e.get('v', 1.0),
                      'PLAYRATE %.6f 0 0 -1 0 0.0025' % rate, '<SOURCE ' + {'.ogg': 'VORBIS', '.mp3': 'MP3', '.flac': 'FLAC'}.get(os.path.splitext(path)[1].lower(), 'WAVE'), 'FILE "%s"' % path, '>', '>']
        L += ['>']
    L += ['>']
    return '\n'.join(L)


if __name__ == '__main__' and '--hybrid' in sys.argv:
    pass
elif __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--genre', default='trap'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--smp', default='app/smp')
    ap.add_argument('--jsfx', required=True); ap.add_argument('--bin', default='jsfx_render'); ap.add_argument('--out'); ap.add_argument('--rpp'); ap.add_argument('--rppwav'); ap.add_argument('--rpplib', help='путь к JSFX внутри Effects Reaper (относительный)')
    a = ap.parse_args()
    P = build(a.genre, a.seed, a.smp)
    print('проект: %s %d bpm, тональность %d %s, кит %s, тактов %d' % (P['genre'], P['bpm'], P['key'], P['scale'], P['kit_src'], P['bars']), {k: len(v) for k, v in P['tracks'].items()})
    if a.out:
        mix, st = render_own(P, a.smp, a.jsfx, a.bin); sf.write(a.out, mix, SR); print('свой движок →', a.out)
    if a.rpp:
        open(a.rpp, 'w').write(rpp(P, a.smp, a.rpplib if a.rpplib is not None else a.jsfx, a.rppwav or a.rpp.replace('.rpp', '.wav'))); print('проект Reaper →', a.rpp)


# ---------- гибрид: мелодия от ИИ + наши ударные/808 по её сетке и нотам ----------
def analyze_ai(hy, g, bpm0):
    """темп, сдвиг сетки (где первая доля) и корень баса по тактам — из дорожек ИИ-бита"""
    import librosa
    dr, _ = librosa.load(f'{hy}/{g}_drums.mp3', sr=22050); bs, _ = librosa.load(f'{hy}/{g}_bass.mp3', sr=22050)
    tempo = float(librosa.feature.tempo(y=dr, sr=22050, start_bpm=bpm0)[0])
    for c in (tempo, tempo * 2, tempo / 2, tempo * 1.5, tempo / 1.5):
        if abs(c - bpm0) / bpm0 < .04: tempo = c; break
    else: tempo = bpm0
    # сдвиг: низ барабанов (кики) против сетки долей — фаза с наибольшим совпадением
    S = np.abs(librosa.stft(dr, n_fft=1024, hop_length=256)); fr = librosa.fft_frequencies(sr=22050, n_fft=1024)
    lo = S[fr < 150].sum(0); on = np.maximum(np.diff(lo, prepend=lo[0]), 0); hop = 256 / 22050
    beat = 60 / tempo; best, off = -1, 0.0
    for ph in np.arange(0, beat, .005):
        idx = ((ph + np.arange(0, len(on) * hop - ph, beat)) / hop).astype(int); idx = idx[idx < len(on)]
        sc = on[idx].sum()
        if sc > best: best, off = sc, ph
    # корни баса по тактам
    bar = beat * 4; nb = int(len(bs) / 22050 // bar); roots = []
    f0, vo, _ = librosa.pyin(bs, fmin=30, fmax=200, sr=22050, frame_length=4096, hop_length=1024)
    t = np.arange(len(f0)) * 1024 / 22050
    for b in range(nb):
        m = (t >= off + b * bar) & (t < off + (b + 1) * bar) & vo
        roots.append(int(round(np.median(librosa.hz_to_midi(f0[m])))) if m.sum() > 3 else None)
    return tempo, off, roots


def build_hybrid(genre, seed, smp, hy):
    import librosa
    R = random.Random(seed); g = G[genre]; meta = json.load(open(f'{hy}/meta.json'))[genre]
    tempo, off, roots = analyze_ai(hy, genre, meta['bpm'])
    step = 60 / tempo / 4; bar = step * 16; nb = len(roots)
    last = None
    for i in range(nb):   # дыры в басе — держим предыдущий корень
        if roots[i] is None: roots[i] = last
        last = roots[i]
    kroot = next((r for r in roots if r), 36)
    src, kit = load_kit(smp, genre, R)
    T = {n: [] for n in ['kick', 'snare', 'clap', 'hat', 'ohat', 'perc', '808', 'ai_melody']}
    T['ai_melody'].append(dict(t=0.0, f=f'{hy}/{genre}_other.mp3'))
    for b in range(nb):
        t0 = off + b * bar
        role = 'intro' if b < 2 else 'outro' if b >= nb - 1 else 'hook'
        k, sn, hats = pat(g['drums'], R, b, role)
        for s in k: T['kick'].append(dict(t=t0 + s * step, v=1.0))
        for s in sn: T['snare'].append(dict(t=t0 + s * step, v=1.0)); T['clap'].append(dict(t=t0 + s * step, v=.7))
        for s, div in hats:
            for r in range(div): T['hat'].append(dict(t=t0 + (s + r / div) * step, v=.8 - .15 * (r > 0) + .1 * (s % 4 == 0)))
        if b % 4 == 3 and role == 'hook': T['ohat'].append(dict(t=t0 + 14 * step, v=.6))
        rt = roots[b] or kroot
        while rt > 40: rt -= 12
        while rt < 28: rt += 12
        if role != 'intro':
            for i, s in enumerate(k):
                nxt = k[i + 1] if i + 1 < len(k) else 16
                T['808'].append(dict(t=t0 + s * step, d=max(2, nxt - s) * step * .95, m=rt, gl=False))
    total = librosa.get_duration(path=f'{hy}/{genre}_other.mp3') + 1.0
    FX = {'kick': [('bus_comp', {2: -10, 3: 2, 4: 3, 5: 1})], 'snare': [('tanh_sat', {1: 3, 2: -1})], 'clap': [('hadamverb', {})], 'hat': [],
          '808': [('tanh_sat', {1: {'rage': 14, 'phonk': 12}.get(genre, 6), 2: -2, 8: 2})] + ([('megagrit', {1: 8, 8: .25, 9: .6})] if genre == 'phonk' else []),
          'ai_melody': [('bus_comp', {2: -14, 3: 1, 4: 3, 5: 3})],
          'master': [('bus_comp', {2: -8, 3: 1, 4: 3, 5: 6}), ('tanh_sat', {1: 2, 2: -0.8})]}
    LVL = {'kick': -13, 'snare': -17, 'clap': -24, 'hat': -25, 'ohat': -28, 'perc': -25, '808': -16, 'ai_melody': -18}
    return dict(genre=genre, seed=seed, bpm=tempo, key=0, scale=g['sc'], kit_src=src, kit=kit, tracks=T, fx=FX, vol={k: 1 for k in LVL}, pan={'hat': .15, 'ohat': -.2},
                lvl=LVL, syn='js3osc', total=total, bars=nb, off=off, roots=roots)


def render_hybrid(P, smp, lib, binp):
    """как render_own, но мелодия — готовая дорожка ИИ (верх чуть приглушаем: там живёт «жужжание»)"""
    ev = P['tracks'].pop('ai_melody'); N = int(P['total'] * SR)
    x, sr = sf.read(ev[0]['f'], dtype='float32', always_2d=True)
    if sr != SR: x = np.stack([np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)), np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype('float32')
    m = np.zeros((N, 2), 'float32'); n = min(N, len(x)); m[:n] = x[:n, :2] if x.shape[1] > 1 else np.repeat(x[:n], 2, 1)
    Fm = np.fft.rfft(m, axis=0); fq = np.fft.rfftfreq(N, 1 / SR); Fm *= (1 - 0.35 * np.clip((fq - 7000) / 5000, 0, 1))[:, None]; m = np.fft.irfft(Fm, N, axis=0).astype('float32')
    m = jsfx(binp, lib, m, P['fx']['ai_melody'], P['bpm'])[:N]
    a = np.abs(m).max(1) > np.abs(m).max() * .05; m *= 10 ** (P['lvl']['ai_melody'] / 20) / (np.sqrt((m[a] ** 2).mean()) + 1e-9)
    P['fx'] = {k: v for k, v in P['fx'].items() if k != 'ai_melody'}
    Q = dict(P, fx=dict(P['fx'], master=[])); mix, stems = render_own(Q, smp, lib, binp)   # барабаны+808 без мастера
    mix = mix / Q.get('mgain', 1.0)   # вернуть уровень до нормализации
    # сайдчейн мелодии от кика
    if 'kick' in stems:
        e = np.abs(stems['kick']).max(1); env = np.zeros(N, 'float32'); acc = 0.0; rl = math.exp(-1 / (.12 * SR))
        for i in range(0, N, 32): acc = max(e[i:i + 32].max(), acc * rl ** 32); env[i:i + 32] = acc
        m *= (1 - .2 * env / (env.max() + 1e-9))[:, None]
    tot = mix[:N] + m
    tot = jsfx(binp, lib, tot.astype('float32'), P['fx']['master'], P['bpm'])[:N]
    pk = np.abs(tot).max(); return tot / pk * .89 if pk > 0 else tot


def main_hybrid():
    ap = argparse.ArgumentParser(); ap.add_argument('--hybrid', required=True); ap.add_argument('--genres', default='trap,phonk,plugg'); ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--smp', default='app/smp'); ap.add_argument('--jsfx', required=True); ap.add_argument('--bin', default='jsfx_render'); ap.add_argument('--outdir', default='.')
    a = ap.parse_args()
    for g in a.genres.split(','):
        P = build_hybrid(g, a.seed, a.smp, a.hybrid)
        print('гибрид %s: темп %.1f, сдвиг %.3f с, корни %s, кит %s' % (g, P['bpm'], P['off'], P['roots'][:8], P['kit_src']), flush=True)
        sf.write(f'{a.outdir}/hybrid_{g}.wav', render_hybrid(P, a.smp, a.jsfx, a.bin), SR)


if __name__ == '__main__' and '--hybrid' in sys.argv:
    main_hybrid()
