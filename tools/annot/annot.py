"""Разметка треков для дообучения ACE-Step: Gemini слушает каждый трек и даёт текст песни с метками частей, описание (caption), bpm и тональность.
usage: annot.py <папка с mp3> <out.json>"""
import os, sys, json, glob, time, re
from google import genai
from google.genai import types
cl = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
names = []
for m in cl.models.list():
    n = m.name.split('/')[-1]; acts = getattr(m, 'supported_actions', None) or []
    if n.startswith('gemini') and ('generateContent' in acts or not acts) and not any(x in n for x in ['tts', 'image', 'embed', 'live', 'audio', 'lite', 'thinking-exp']): names.append(n)
def key(n):
    v = re.findall(r'(\d+)\.(\d+)', n); return (tuple(map(int, v[0])) if v else (0, 0), 'preview' not in n)
names.sort(key=key, reverse=True)
MODELS = [n for n in names if 'pro' in n][:2] + [n for n in names if 'flash' in n][:3]
print('модели:', MODELS, flush=True)
P = """Listen to this song carefully. It is sung in Russian by one female voice.
Return ONLY a JSON object with keys:
"lyrics": the exact Russian lyrics as actually sung, in Cyrillic, line by line, with section tags on their own lines like [Intro], [Verse 1], [Pre-Chorus], [Chorus], [Verse 2], [Bridge], [Outro]. Repeat sections as sung. No commentary.
"caption": 2-4 English sentences describing genre, tempo feel, instruments, vocal timbre and delivery, mood and production style.
"bpm": integer tempo,
"keyscale": like "A minor" or "C# major",
"timesignature": "4" or "3" or "6"."""
out = []; dead = set()
for f in sorted(glob.glob(sys.argv[1] + '/*.mp3')):
    up = cl.files.upload(file=f)
    for _ in range(30):
        st = cl.files.get(name=up.name).state.name
        if st == 'ACTIVE': break
        time.sleep(2)
    res = None
    for model in MODELS:
        if model in dead: continue
        try:
            r = cl.models.generate_content(model=model, contents=[up, P], config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0.2))
            res = json.loads(r.text); print('✅', os.path.basename(f), model, flush=True); break
        except Exception as e:
            print('⚠', model, str(e)[:160], flush=True)
            if '429' in str(e): dead.add(model)
    if res:
        res['filename'] = os.path.basename(f); res['language'] = 'ru'; res['is_instrumental'] = False; out.append(res)
    time.sleep(3)
json.dump({'samples': out}, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print('размечено', len(out), flush=True)
