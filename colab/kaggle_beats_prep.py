"""Kaggle GPU: биты free for profit (выход ноутбука ts-yt) → убрать голосовые теги продюсеров (demucs) → разметка для ACE-Step.
Итог: /kaggle/working/ds/*.mp3 + dataset.json (вход для kaggle_ace_lora.py, TAG=tsbeat)."""
import os, sys, glob, json, subprocess
W = '/kaggle/working/ds'; os.makedirs(W, exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True)
sh(f'{sys.executable} -m pip install -q demucs librosa')
import librosa, numpy as np
mj = glob.glob('/kaggle/input/**/meta.json', recursive=True); assert mj, 'нет meta.json во входе'
meta = {}
for m in mj:   # несколько входов (старые и новые выкачки) — сливаем, повторы по id ролика отпадают
    for k, v in json.load(open(m)).items(): v = dict(v); v['file'] = os.path.join(os.path.dirname(m), v['file']); meta.setdefault(k, v)
src = ''; print('битов:', len(meta), {g: sum(1 for v in meta.values() if v['genre'] == g) for g in sorted(set(v['genre'] for v in meta.values()))}, flush=True)
DESC = {'trap': 'dark melodic trap beat, heavy 808, rolling hi-hats, hard kick, crisp snare', 'drill': 'UK drill beat, sliding 808 bass, syncopated hi-hats, dark piano',
        'plugg': 'plugg beat, bouncy 808, soft bells and synth leads, dreamy', 'rage': 'rage beat, distorted 808, aggressive supersaw synth lead',
        'phonk': 'phonk beat, cowbell melody, distorted 808, memphis drums', 'jerk': 'jerk beat, fast bouncy drums, synth bass, playful',
        'pluggnb': 'pluggnb beat, smooth chords, bouncy 808, melodic', 'detroit': 'detroit trap beat, hard punchy drums, bouncy 808, piano loop',
        'atlanta': 'Atlanta trap beat, hard 808, rolling triplet hi-hats, dark synths', 'lofi': 'lofi hip hop beat, dusty drums, warm keys, vinyl crackle'}
files = [m['file'] for m in meta.values() if os.path.exists(m['file'])]
for i in range(0, len(files), 40):
    subprocess.run([sys.executable, '-m', 'demucs', '--two-stems', 'vocals', '-n', 'htdemucs', '--mp3', '--mp3-bitrate', '192', '-o', '/tmp/sep'] + files[i:i + 40])
out = []
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]); MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
for vid, m in meta.items():
    b = os.path.splitext(os.path.basename(m['file']))[0]; nv = f'/tmp/sep/htdemucs/{b}/no_vocals.mp3'; voc = f'/tmp/sep/htdemucs/{b}/vocals.mp3'
    if not os.path.exists(nv): continue
    y, sr = librosa.load(nv, sr=22050, duration=120)
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0]); bpm = int(round(tempo * 2 if tempo < 90 and m['genre'] != 'lofi' else tempo))
    ch = librosa.feature.chroma_cqt(y=y, sr=sr).mean(1); sc = [(np.corrcoef(np.roll(MAJ, k), ch)[0, 1], k, 'major') for k in range(12)] + [(np.corrcoef(np.roll(MIN, k), ch)[0, 1], k, 'minor') for k in range(12)]
    _, k, mode = max(sc)
    vy, _ = librosa.load(voc, sr=22050, duration=120); vr = float(np.sqrt((vy ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-9))
    fn = f'{b}.mp3'; subprocess.run(['cp', nv, f'{W}/{fn}'])
    out.append({'filename': fn, 'caption': DESC.get(m['genre'], m['genre'] + ' beat') + ', instrumental, no vocals, modern hip-hop production, free for profit type beat',
                'lyrics': '[Instrumental]', 'bpm': bpm, 'keyscale': f'{NOTES[k]} {mode}', 'language': 'unknown', 'is_instrumental': True, 'genre': m['genre'], 'voc_ratio': round(vr, 3)})
    print(fn, bpm, NOTES[k], mode, 'voc %.2f' % vr, flush=True)
json.dump({'samples': out}, open(W + '/dataset.json', 'w'), ensure_ascii=False, indent=1); print('готово', len(out))
