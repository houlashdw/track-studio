"""Полный трек одной функцией (работает и в облаке Modal, и на раннере):
бит из пака → Reaper → ACE поёт вокалоид-голос под него → подгонка к сетке/чистка → баланс до мастера → вокальная цепочка и мастер в Reaper → громкость и потолок.
make_track(pack_dir, genre, seed, work, sing) → (mp3_bytes, info). sing(beat_mp3_bytes, bpm) → bytes mp3 вокала (в облаке — вызов AceLego)."""
import os, sys, json, time, subprocess, shutil
import numpy as np, soundfile as sf, librosa

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pack as packmod, voxprep
REAPER = os.environ.get('TS_REAPER', '/opt/REAPER/reaper'); BUILD = os.environ.get('TS_BUILD_LUA', os.path.join(HERE, '..', 'reaper', 'build.lua'))


def reaper_render(lua, wav, log, mg=0.0, nomaster=False, timeout=200):
    """headless-рендер проекта build.lua: ждём строку «РЕНДЕР ГОТОВ» в логе, потом глушим Reaper"""
    for f in (wav, log):
        if os.path.exists(f): os.remove(f)
    os.makedirs(os.path.expanduser('~/.config/REAPER'), exist_ok=True)
    ini = os.path.expanduser('~/.config/REAPER/reaper.ini')
    if not os.path.exists(ini): open(ini, 'w').write('[REAPER]\nsplashwnd=0\nverchk=0\n')
    env = dict(os.environ, TS_PROJ=lua, TS_LOG=log, TS_MGAIN=str(mg))
    env.pop('TS_NOMASTER', None)
    if nomaster: env['TS_NOMASTER'] = '1'
    p = subprocess.Popen(['xvfb-run', '-a', REAPER, '-nosplash', '-newinst', BUILD], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(log) and 'РЕНДЕР ГОТОВ' in open(log, errors='replace').read(): break
        time.sleep(.5)
    time.sleep(1)
    subprocess.run(['pkill', '-f', REAPER], check=False); subprocess.run(['pkill', 'Xvfb'], check=False)
    try: p.wait(timeout=5)
    except Exception: p.kill()
    if not os.path.exists(wav):
        raise RuntimeError('Reaper не отрендерил %s: %s' % (os.path.basename(wav), open(log, errors='replace').read()[-600:] if os.path.exists(log) else 'нет лога'))
    return time.time() - t0


def lufs(path):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', path, '-af', 'ebur128', '-f', 'null', '-'], capture_output=True, text=True).stderr
    i = r.rfind('I:'); return float(r[i + 2:].split('LUFS')[0])


def project(pack_dir, genre, seed, smp, wav, lua, drums='kit', vox=None, voxvol=0.0, solo=None, secgain=None):
    R = packmod.build(pack_dir, genre, seed, smp, wav, drums, vox, voxvol, secgain)
    if solo: R['tracks'] = [t for t in R['tracks'] if (t['name'] in ('vox', 'voxhook', 'voxdbl', 'voxstk')) == (solo == 'vox')]
    open(lua, 'w').write('return ' + packmod.to_lua(R) + '\n'); return R


def fader(bpre, vpre, want=1.5):
    SR = 22050; H = int(.4 * SR); ld = lambda f: librosa.load(f, sr=SR, mono=True)[0]
    def wr(x): k = len(x) // H; return np.sqrt((x[:k * H].reshape(k, H) ** 2).mean(1) + 1e-12)
    rb, rv = wr(ld(bpre)), wr(ld(vpre))
    return float((20 * np.log10(rb[rb > rb.max() * .05].mean()) + want) - 20 * np.log10(rv[rv > rv.max() * .15].mean()))


NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KMAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]); KMIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def chroma(path, sr=22050, harm=True):
    x, _ = librosa.load(path, sr=sr, mono=True)
    if harm: x = librosa.effects.harmonic(x, margin=3)
    C = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=2048); e = C.sum(0)
    return C[:, e > np.percentile(e, 40)].mean(1) if C.shape[1] > 4 else C.mean(1)


def key_of(ch):
    """тональность по профилям Крумхансла → (тоника, 'major'|'minor', сила)"""
    best = (-2, 0, 'minor')
    for k in range(12):
        for prof, md in ((KMAJ, 'major'), (KMIN, 'minor')):
            r = float(np.corrcoef(ch, np.roll(prof, k))[0, 1])
            if r > best[0]: best = (r, k, md)
    return best[1], best[2], best[0]


def key_fix(vox_wav, ch_beat, k, md, kconf=1.0):
    """на сколько полутонов сдвинуть голос, чтобы он лёг в ноты бита (0 — уже в тональности или нельзя уверенно сказать).
    Сдвигаем только при уверенности: тональность бита читается (kconf), у голоса чёткие ноты (хрома не плоская), и оба признака — доля в гамме и прямое сходство с хромой бита — согласны."""
    S = np.roll(np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1.]) if md == 'major' else np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0.]), k)
    cv = chroma(vox_wav, harm=False); peak = float((cv.max() - cv.min()) / (cv.max() + 1e-9)); cv = cv / (cv.sum() + 1e-9)
    dc = {s_: float(np.corrcoef(np.roll(cv, s_), ch_beat)[0, 1]) for s_ in range(-5, 6)}
    sc = {s_: float((np.roll(cv, s_) * S).sum()) + .5 * dc[s_] for s_ in range(-5, 6)}
    s_ = max(sc, key=lambda z: (round(sc[z], 3), -abs(z))); bd = max(dc, key=dc.get)
    sure = kconf >= .5 and peak >= .45 and sc[s_] - sc[0] > .15 and dc[s_] - dc[0] > .3 and abs(bd - s_) <= 1
    return (s_ if s_ != 0 and sure else 0), round(sc[0], 2), round(sc[s_], 2), round(peak, 2)


def grid_phase(path, bpm, sr=22050, delta=.15):
    """средняя фаза атак относительно сетки 1/16 (мс) и её устойчивость R (0 — атаки как попало, 1 — все в одной фазе)"""
    x, _ = librosa.load(path, sr=sr, mono=True); on = librosa.onset.onset_detect(y=x, sr=sr, units='time', delta=delta)
    if len(on) < 8: return 0.0, 0.0
    st = 15 / bpm; z = np.exp(2j * np.pi * (on / st % 1)).mean()
    return float(np.angle(z) / (2 * np.pi) * st * 1000), float(abs(z))


def pshift(inp, out, semis):
    try: subprocess.run(['rubberband', '-q', '-p', str(semis), '-F', inp, out], check=True, capture_output=True)
    except Exception:
        x, sr = sf.read(inp); sf.write(out, librosa.effects.pitch_shift(x.T if x.ndim > 1 else x, sr=sr, n_steps=semis).T, sr)


TGT = {'intro': -4.0, 'verse': -2.5, 'hook': 0.0, 'break': -6.0, 'outro': -4.0}   # громкость секций относительно хука (дБ): подъём к припеву, провал в брейке


def sec_gain(wav, sections, bpm, sr=22050):
    """поправки громкости по секциям (дБ), чтобы бит шёл по кривой TGT; слоёный второй хук +0.5"""
    x, _ = librosa.load(wav, sr=sr, mono=True); bar = 240 / bpm; t = 0; L = []
    for nm, n in sections:
        a, b = int(t * bar * sr), int((t + n) * bar * sr); L.append(10 * np.log10((x[a:b] ** 2).mean() + 1e-12)); t += n
    hk = [l for (nm, _), l in zip(sections, L) if nm == 'hook']; ref = float(np.mean(hk)) if hk else max(L)
    nh = 0; g = []
    for (nm, _), l in zip(sections, L):
        tg = TGT.get(nm, -2.0)
        if nm == 'hook': tg += .5 * nh; nh += 1
        g.append(float(np.clip(tg - (l - ref), -8, 6)))
    return g, [round(l - ref, 1) for l in L]


def make_track(pack_dir, genre, seed, work, sing, smp, drums='kit', target=-10.0, log=print):
    os.makedirs(work, exist_ok=True); W = lambda n: os.path.join(work, n); T = {}; t0 = time.time()
    tick = lambda k: (T.__setitem__(k, round(time.time() - t0, 1)), log('  ⏱ %s: %.0f с' % (k, time.time() - t0)))
    # 1) бит
    R = project(pack_dir, genre, seed, smp, W('beat.wav'), W('beat.lua'), drums); bpm = R['bpm']
    reaper_render(W('beat.lua'), W('beat.wav'), W('beat.log'), nomaster=True)
    SG, lv0 = sec_gain(W('beat.wav'), R['sections'], bpm); log('  секции до: %s → поправка %s' % (lv0, [round(g, 1) for g in SG]))
    R = project(pack_dir, genre, seed, smp, W('beat.wav'), W('beat.lua'), drums, secgain=SG)
    reaper_render(W('beat.lua'), W('beat.wav'), W('beat.log')); subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', W('beat.wav'), '-b:a', '192k', W('beat.mp3')], check=True); tick('бит')
    # 2) голос
    chb = chroma(W('beat.wav')); k, md, ks = key_of(chb); key = '%s %s' % (NOTE[k], md); log('  тональность бита: %s (%.2f)' % (key, ks))
    try: raw = sing(open(W('beat.mp3'), 'rb').read(), bpm, key=key, sections=R.get('sections'))
    except TypeError: raw = sing(open(W('beat.mp3'), 'rb').read(), bpm)
    open(W('vox_raw.mp3'), 'wb').write(raw); tick('голос')
    x, _ = librosa.load(W('vox_raw.mp3'), sr=voxprep.SR, mono=True); y, vinfo = voxprep.prep(x, bpm, raw=True); sf.write(W('vox.wav'), y, voxprep.SR, subtype='PCM_24')
    sh, c0, c1, pk = key_fix(W('vox.wav'), chb, k, md, ks); vinfo.update(key=key, key_conf=round(ks, 2), key_fit=c0, shift=sh, key_fit_after=c1, vox_peak=pk)
    if sh: pshift(W('vox.wav'), W('vox_k.wav'), sh); os.replace(W('vox_k.wav'), W('vox.wav')); log('  голос не в тональности: сдвиг %+d пт (%.2f → %.2f)' % (sh, c0, c1))
    # голос в «карман» бита: сдвиг по времени, чтобы атаки голоса шли в той же фазе сетки 1/16, что и удары бита (не больше ±40 мс)
    bp, bR = grid_phase(W('beat.wav'), bpm, delta=.2); vp, vR = grid_phase(W('vox.wav'), bpm); st = 15000 / bpm
    dt = ((bp - vp + st / 2) % st) - st / 2; vinfo.update(beat_phase=round(bp), vox_phase=round(vp), vox_R=round(vR, 2))
    if vR > .25 and bR > .4 and 8 < abs(dt) <= 40:
        y, srv = sf.read(W('vox.wav')); n = int(round(abs(dt) / 1000 * srv)); z = np.zeros((n,) + y.shape[1:], y.dtype)
        y = np.concatenate([z, y[:-n]]) if dt > 0 else np.concatenate([y[n:], z]); sf.write(W('vox.wav'), y, srv, subtype='PCM_24'); vinfo['nudge_ms'] = round(dt); log('  голос в карман: %+.0f мс' % dt)
    tick('подгонка')
    # 3) баланс до мастера
    for nm, solo in (('bpre', 'beat'), ('vpre', 'vox')):
        project(pack_dir, genre, seed, smp, W(nm + '.wav'), W(nm + '.lua'), drums, W('vox.wav'), 0.0, solo, SG); reaper_render(W(nm + '.lua'), W(nm + '.wav'), W(nm + '.log'), nomaster=True)
    vv = fader(W('bpre.wav'), W('vpre.wav')); tick('баланс')
    # 4) сведение и мастер
    project(pack_dir, genre, seed, smp, W('mix.wav'), W('mix.lua'), drums, W('vox.wav'), vv, secgain=SG); reaper_render(W('mix.lua'), W('mix.wav'), W('mix.log'))
    g = target - lufs(W('mix.wav'))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', W('mix.wav'), '-af', 'volume=%.2fdB,alimiter=limit=0.966:attack=3:release=60:level=disabled' % g, '-b:a', '192k', W('mix.mp3')], check=True); tick('мастер')
    info = dict(R['info'], genre=R.get('genre', genre), sec_before=lv0, sec_gain=[round(g, 1) for g in SG], vox=vinfo, fader=round(vv, 1), loud=round(lufs(W('mix.mp3')), 1), t=T)
    return open(W('mix.mp3'), 'rb').read(), info
