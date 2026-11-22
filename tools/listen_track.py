"""«Уши» для готовых треков: Gemini слушает каждый mp3 в папке и пишет разбор рядом (<имя>.md).
Фокус — на жалобах пользователя: голос и бит «не стакаются» (тональность, настроение, тембр), обрывы, плоский/ровный вокал, «под водой».
python3 listen_track.py <папка>   (секрет GEMINI_API_KEY в окружении)"""
import os, sys, glob, time, re
from google import genai

d = sys.argv[1]
cl = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
P = """Ты — опытный продюсер и звукорежиссёр хип-хопа (trap/plugg/rage/drill/phonk) и J-pop/вокалоид, слушаешь трек ушами как человек.
Трек собран автоматически: бит из лупов и сэмплов в Reaper, сверху — женский аниме/вокалоид-голос, который нейросеть спела под этот бит.
Ответь по-русски, коротко и конкретно, всегда с таймкодами мм:сс.
1) СВЯЗКА ГОЛОС↔БИТ: звучат ли они как одна песня или как два наложенных трека? Совпадает ли тональность (если голос фальшивит относительно бита — где)?
   Попадает ли голос в ритм и доли бита? Совпадает ли настроение и тембр? Что именно мешает «склеиться».
2) ОБРЫВЫ И ПЕРЕХОДЫ: каждое резкое обрывание/включение голоса или бита — таймкод и что именно оборвалось. Как звучит начало трека (первые 10 с).
3) ГОЛОС: ровный/монотонный или живой; есть ли развитие мелодии, паузы, дыхание; звук «под водой», глухой, фазовый, металлический — где.
4) БИТ: развитие по ходу трека (куплет/припев/брейк различимы?), что скучно, что лишнее.
5) СВЕДЕНИЕ: голос громче/тише бита, грязь в низах, перегруз, пространство.
6) ТОП-5 исправлений по важности, каждое — конкретно (что сделать с голосом/битом/сведением).
В конце ОДНА строка JSON: {"glue":..,"key_match":..,"rhythm":..,"cuts":..,"vocal_life":..,"beat":..,"mix":..,"overall":..}
(оценки 1–10, где 10 — как релиз; cuts: 10 = нет обрывов)."""


def models():
    names = []; allnames = set()
    try:
        for m in cl.models.list():
            n = m.name.split('/')[-1]; allnames.add(n); acts = getattr(m, 'supported_actions', None) or []
            if n.startswith('gemini') and 'flash' in n and ('generateContent' in acts or not acts) and not any(x in n for x in ['tts', 'image', 'embed', 'live', 'audio', 'lite']): names.append(n)
    except Exception as e: print('list', e)
    ver = lambda n: float((re.findall(r'gemini-(\d+(?:\.\d+)?)', n) or ['0'])[0])
    names.sort(key=lambda n: (ver(n), 'preview' not in n and 'exp' not in n), reverse=True)
    top = [n for n in ('gemini-4-neon', 'gemini-4-hydrogen') if n in allnames] or ['gemini-4-neon', 'gemini-4-hydrogen']   # Gemini 4: Neon (платный ключ), Hydrogen (есть бесплатно)
    return top + names[:3] or ['gemini-2.5-flash']


M = models(); print('модели:', M)
for f in sorted(glob.glob(os.path.join(d, '*.mp3'))):
    up = cl.files.upload(file=f)
    while up.state.name == 'PROCESSING': time.sleep(2); up = cl.files.get(name=up.name)
    txt, errs, dead = None, [], set()
    for att in range(4):
        for m in M:
            if m in dead: continue
            try: txt = cl.models.generate_content(model=m, contents=[up, P]).text; used = m; break
            except Exception as e:
                errs.append(m + ': ' + str(e)[:160])
                if any(c in str(e) for c in ('429', '403', '404', 'PERMISSION', 'NOT_FOUND')): dead.add(m)
        if txt or len(dead) == len(M): break
        time.sleep(30 + 15 * att)
    out = ('# %s (%s)\n\n%s' % (os.path.basename(f), used, txt)) if txt else 'ошибки:\n' + '\n'.join(errs)
    open(f[:-4] + '.md', 'w', encoding='utf-8').write(out); print(os.path.basename(f), 'ok' if txt else 'FAIL')
