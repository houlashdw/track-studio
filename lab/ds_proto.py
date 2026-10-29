# Прототип DiffSinger (Canary): строки текста + мелодия → wav. Пишет в ds-out/
import os, sys, json, time, re, zipfile, glob, numpy as np, onnxruntime as ort, yaml, wave
R = 'dl/canary/inner'; O = 'ds-out'; os.makedirs(O, exist_ok=True)
info = {}
for f in ['dsconfig.yaml', 'dsvocoder/vocoder.yaml', 'dsdur/dsconfig.yaml', 'dspitch/dsconfig.yaml']:
    try: info[f] = open(os.path.join(R, f)).read()
    except Exception as e: info[f] = str(e)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
ac = ort.InferenceSession(R + '/dsacoustic/acoustic.onnx', so)
vc = ort.InferenceSession(glob.glob(R + '/dsvocoder/*.onnx')[0], so)
info['ac_in'] = [(i.name, i.type, i.shape) for i in ac.get_inputs()]; info['ac_out'] = [(o.name, o.shape) for o in ac.get_outputs()]
info['vc_in'] = [(i.name, i.type, i.shape) for i in vc.get_inputs()]; info['vc_out'] = [(o.name, o.shape) for o in vc.get_outputs()]
json.dump(info, open(O + '/info.json', 'w'), indent=1, default=str); print(json.dumps(info, indent=1, default=str)[:3000])
cfg = yaml.safe_load(open(R + '/dsconfig.yaml'))
phs = [l.strip() for l in open(R + '/dsacoustic/phonemes.txt') if l.strip()]
pid = {p: i for i, p in enumerate(phs)}
D = yaml.safe_load(open(R + '/dsdur/dsdict-en.yaml'))
vowels = set(s['symbol'] for s in D.get('symbols', []) if s.get('type') == 'vowel')
words = {}
for e in D.get('entries', []): words.setdefault(str(e['grapheme']).lower(), e['phonemes'])
sr = cfg.get('sample_rate', 44100); hop = cfg.get('hop_size', 512); fps = sr / hop
emb = np.fromfile(R + '/dsacoustic/canary_arc.emb', dtype=np.float32)
LINES = [("city lights are calling out my name tonight", [64, 64, 67, 67, 69, 67, 64, 62, 64, 67, 69]),
         ("hold me close and never let me go", [62, 64, 66, 67, 69, 67, 66, 64, 62]),
         ("stars are falling i will sing tonight", [69, 71, 72, 71, 69, 67, 69, 71, 72, 74]),
         ("this is our song so sing it loud", [67, 69, 71, 72, 74, 72, 71, 69, 67])]
def mk(text, notes, spk, steps=10, depth=None, sylsec=0.42):
    seq = [('SP', 0.25, None)]; ni = 0
    for w in text.split():
        p = [x.split('/')[-1] for x in words.get(w, [])]
        if not p: continue
        # слоги: каждая гласная — нота; согласные до гласной — короткие
        cons = []
        for ph in p:
            if ph in vowels or ph.split('/')[-1] in vowels:
                for c in cons: seq.append((c, 0.06, notes[min(ni, len(notes) - 1)]))
                seq.append((ph, max(0.12, sylsec - 0.06 * len(cons)), notes[min(ni, len(notes) - 1)])); ni += 1; cons = []
            else: cons.append(ph)
        for c in cons: seq.append((c, 0.06, notes[max(0, min(ni - 1, len(notes) - 1))]))
    seq.append(('SP', 0.3, None))
    toks = [pid.get(s[0], pid.get('SP')) for s in seq]; dur = [max(1, round(s[1] * fps)) for s in seq]; T = sum(dur)
    f0 = np.zeros(T, np.float32); t = 0; last = 220.0
    for (ph, d, m), n in zip(seq, dur):
        hz = 440 * 2 ** (((m if m is not None else 69 + 12 * np.log2(last / 440)) - 69) / 12); last = hz; f0[t:t + n] = hz; t += n
    # плавные переходы + вибрато
    k = np.ones(9) / 9; f0 = np.convolve(np.pad(f0, 4, mode='edge'), k, 'valid').astype(np.float32)
    tt = np.arange(T) / fps; f0 *= 2 ** (0.25 * np.sin(2 * np.pi * 5.5 * tt) * np.clip((tt % 0.42) / 0.3, 0, 1) / 12)
    feeds = {}
    for i in ac.get_inputs():
        n = i.name
        if n == 'tokens': feeds[n] = np.array([toks], np.int64)
        elif n in ('durations', 'durs'): feeds[n] = np.array([dur], np.int64)
        elif n == 'f0': feeds[n] = f0[None]
        elif n == 'spk_embed': feeds[n] = np.tile(spk[None, None, :], (1, T, 1)).astype(np.float32)
        elif n == 'speedup': feeds[n] = np.array(max(1, 1000 // steps), np.int64)
        elif n == 'steps': feeds[n] = np.array(steps, np.int64)
        elif n == 'depth': feeds[n] = np.array(depth if depth is not None else 1.0, np.float32 if 'float' in i.type else np.int64)
        elif n == 'gender': feeds[n] = np.zeros((1, T), np.float32)
        elif n == 'velocity': feeds[n] = np.ones((1, T), np.float32)
        elif n in ('energy', 'breathiness', 'voicing', 'tension'): feeds[n] = np.zeros((1, T), np.float32)
        elif n == 'languages': feeds[n] = np.zeros((1, len(toks)), np.int64)
        else: print('неизвестный вход', n, i.type, i.shape)
    t0 = time.time(); mel = ac.run(None, feeds)[0]; ta = time.time() - t0
    vin = {vc.get_inputs()[0].name: mel}
    if len(vc.get_inputs()) > 1: vin[vc.get_inputs()[1].name] = f0[None]
    t0 = time.time(); wav = vc.run(None, vin)[0].reshape(-1); tv = time.time() - t0
    return wav, T / fps, ta, tv, [(s[0], d) for s, d in zip(seq, dur)]
res = []
for k, (txt, notes) in enumerate(LINES):
    for steps in (10,):
        try:
            w, dsec, ta, tv, seq = mk(txt, notes, emb, steps)
            x = np.clip(w, -1, 1); pcm = (x * 32767).astype(np.int16)
            with wave.open(f'{O}/line{k}_s{steps}.wav', 'wb') as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(pcm.tobytes())
            res.append({'line': txt, 'steps': steps, 'sec': round(dsec, 2), 't_acoustic': round(ta, 2), 't_vocoder': round(tv, 2), 'missing': [w for w in txt.split() if w not in words], 'seq': seq})
        except Exception as e: res.append({'line': txt, 'err': repr(e)[:500]})
        print(res[-1].get('err') or (txt, res[-1]['t_acoustic'], res[-1]['t_vocoder']))
json.dump(res, open(O + '/res.json', 'w'), indent=1)
