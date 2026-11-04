"""Gemini слушает каждый mp3 из папки и пишет разбор. Промпт — как у звукорежиссёра, сравнение с уровнем Suno."""
import os, sys, glob, json, time
from google import genai
src, dst = sys.argv[1], sys.argv[2]
cl = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
note = os.environ.get('NOTE', '')
ctx = open(os.path.join(src, 'context.txt'), encoding='utf-8').read() if os.path.exists(os.path.join(src, 'context.txt')) else ''
P = """Ты — опытный звукорежиссёр и продюсер, слушаешь трек ушами как человек. Контекст: """ + ctx + """
Ответь по-русски, конкретно и с таймкодами (мм:сс). Разбери:
1) ГОЛОС: искажения (роботизация, металл, хрип, «двойной голос», щелчки, заикание) — где именно; естественность; эмоция; попадание в ноты и ритм; разборчивость слов (перечисли, какие слова не понял).
2) МЕЛОДИЯ И ФЛОУ голоса: интересна ли, повторяется ли, цепляет ли хук, как сидит на бите.
3) БИТ И АРАНЖИРОВКА: какие инструменты слышишь, звучат ли они реально или дёшево, есть ли развитие, переходы, что лишнее, что пусто.
4) СВЕДЕНИЕ: баланс голос/бит, грязь, громкость, низ, верх, пространство.
5) Сравни с типичным треком Suno v4/v5: что хуже, что лучше — по пунктам.
6) ТОП-5 конкретных исправлений по важности (что сделать в обработке/синтезе, чтобы звучало как у профи).
Оценки 1–10: голос, мелодия, бит, сведение, общее. В конце строка JSON: {"voice":..,"melody":..,"beat":..,"mix":..,"overall":..}""" + ("\nОтдельно проверь: " + note if note else '')
names = []
try:
    for m in cl.models.list():
        n = m.name.split('/')[-1]; acts = getattr(m, 'supported_actions', None) or []
        if n.startswith('gemini') and ('generateContent' in acts or not acts) and not any(x in n for x in ['tts', 'image', 'embed', 'live', 'audio', 'lite', 'thinking-exp']): names.append(n)
except Exception as e: print('list', e)
import re
def key(n):
    v = re.findall(r'gemini-(\d+(?:\.\d+)?)', n); ver = float(v[0]) if v else 0
    return (ver, 'pro' in n, 'preview' not in n and 'exp' not in n)
names.sort(key=key, reverse=True); print('models:', names[:8])
MODELS = [n for n in names if 'pro' in n and 'customtools' not in n][:2] + [n for n in names if 'flash' in n][:4] + [n for n in names if 'argon' in n] or ['gemini-3.8-flash']
open(os.path.join(dst, '_models.txt'), 'w').write('\n'.join(names))
PC = '''Ты — опытный звукорежиссёр, слушаешь ушами как человек. Контекст: ''' + ctx + '''
Два файла: ПЕРВЫЙ — старая версия (A), ВТОРОЙ — новая (B), тот же текст и бит. Сравни ТОЛЬКО честно, что слышишь, по-русски, с таймкодами:
1) Голос: искажения/роботизация, «двойной голос»/фазовость, ступеньки между нотами/щелчки, режущие 3 кГц и свист, «бурундук», сухость, громкость относительно бита, естественность и эмоция. Для каждого пункта: где лучше — A, B или одинаково.
2) Что в B стало хуже, если стало.
3) 3 самых заметных оставшихся проблемы голоса в B с таймкодами и как их исправить в синтезе/обработке.
В конце строка JSON: {"A":{"voice":..,"mix":..,"overall":..},"B":{"voice":..,"mix":..,"overall":..},"better":"A|B|same"}'''
def gen(contents):
    errs = []
    for att in range(3):
        for model in MODELS:
            try: r = cl.models.generate_content(model=model, contents=contents); return model, r.text
            except Exception as e: errs.append(model + ': ' + str(e)[:160])
        time.sleep(30)
    return 'none', 'ошибки:\n' + '\n'.join(errs)
def upl(f):
    up = cl.files.upload(file=f)
    while up.state.name == 'PROCESSING': time.sleep(2); up = cl.files.get(name=up.name)
    return up
pairs = sorted(set(f[:-6] for f in glob.glob(os.path.join(src, '*_A.mp3'))))
for b in pairs:
    if not os.path.exists(b + '_B.mp3'): continue
    a, bb = upl(b + '_A.mp3'), upl(b + '_B.mp3'); model, txt = gen([a, bb, PC])
    open(os.path.join(dst, os.path.basename(b) + '_AB.md'), 'w', encoding='utf-8').write('# ' + os.path.basename(b) + ' A vs B (' + model + ')\n\n' + txt); print(b, 'pair ok')
for f in sorted(glob.glob(os.path.join(src, '*.mp3'))):
    if f.endswith('_A.mp3') or f.endswith('_B.mp3'): continue
    up = cl.files.upload(file=f)
    while up.state.name == 'PROCESSING': time.sleep(2); up = cl.files.get(name=up.name)
    errs = []; txt = None
    for att in range(3):
        for model in MODELS:
            try: r = cl.models.generate_content(model=model, contents=[up, P]); txt = r.text; break
            except Exception as e: errs.append(model + ': ' + str(e)[:160])
        if txt: break
        time.sleep(30)
    if not txt: txt = 'ошибки:\n' + '\n'.join(errs)
    open(os.path.join(dst, os.path.basename(f)[:-4] + '.md'), 'w', encoding='utf-8').write('# ' + os.path.basename(f) + ' (' + model + ')\n\n' + txt)
    print(os.path.basename(f), 'ok')
